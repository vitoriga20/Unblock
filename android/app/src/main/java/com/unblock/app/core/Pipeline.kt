package com.unblock.app.core

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import org.json.JSONObject
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.imgcodecs.Imgcodecs

data class Sample(
    val id: String,
    val baseAsset: String,
    val auxAsset: String,
    val baseQuad: Array<Point>,
    val auxQuad: Array<Point>
)

data class PipelineResult(
    val id: String,
    val restored: Bitmap,
    val coverage: Double,
    val blindRatio: Double,
    val seamRatio: Double,
    val seamStep: Double,
    val elapsedMs: Long,
    val steps: List<Pair<String, Long>>,
    val diag: String = "",
    val issues: List<String> = emptyList(),
    val alignMode: String = "none",
    val contentChange: Double = 0.0
)

data class PipelineConfig(
    val longSide: Int = 1920,
    val targetAspect: Double? = null,
    val align: AlignConfig = AlignConfig(),
    val occlusion: OcclusionConfig = OcclusionConfig(),
    val compose: ComposeConfig = ComposeConfig(),
    val screen: ScreenConfig = ScreenConfig(),
    val quality: QualityConfig = QualityConfig(),
    val baseSelect: BaseSelectConfig = BaseSelectConfig(),
    val fusion: FusionConfig = FusionConfig(),
    val modelAsset: String = "model/yolov8n-seg.onnx",
    val segmentInputSize: Int = 640,
    val segmentConf: Float = 0.1f,
    val usePersonMask: Boolean = true,
    val segmentClasses: IntArray = intArrayOf(0),
    val segmentObjectClasses: IntArray = intArrayOf(24, 25, 26, 39, 40, 41, 42, 43, 44, 45, 46,
        47, 49, 63, 65, 67, 73, 76, 78, 79),
    val dumpDebug: Boolean = false
)

private class SourceInfo(
    val index: Int,
    val h: Mat,
    val align: AlignmentResult,
    val corrected: Mat,
    val mask: Mat,
    val valid: Mat,
    val person: Mat?,
    val info: OcclusionInfo,
    val gainMean: Double,
    val colorApplied: Boolean,
    val residual: ResidualInfo?
)

class Pipeline(private val context: Context, private val cfg: PipelineConfig) {

    private var segmenter: Segmenter? = null
    private var baseInfo: String = ""

    @Volatile
    private var cancelRequested = false

    fun cancel() {
        cancelRequested = true
    }

    fun close() {
        segmenter?.close()
        segmenter = null
    }

    private fun segmenter(): Segmenter? {
        if (!cfg.usePersonMask) return null
        if (segmenter == null) {
            segmenter = try {
                Segmenter(context, cfg.modelAsset, cfg.segmentInputSize, cfg.segmentConf)
            } catch (e: Exception) {
                null
            }
        }
        return segmenter
    }

    private inline fun <T> step(name: String, block: () -> T): T = try {
        block()
    } catch (t: Throwable) {
        throw IllegalStateException("阶段[$name] " + (t.message ?: t.javaClass.simpleName), t)
    }

    fun loadSamples(): List<Sample> {
        val json = context.assets.open("config.json").use { it.readBytes() }.toString(Charsets.UTF_8)
        val root = JSONObject(json)
        val groups = root.getJSONArray("groups")
        val out = ArrayList<Sample>()
        for (i in 0 until groups.length()) {
            val g = groups.getJSONObject(i)
            val corners = g.getJSONObject("corners")
            out.add(
                Sample(
                    id = g.getString("id"),
                    baseAsset = "samples/" + g.getString("base"),
                    auxAsset = "samples/" + g.getString("aux"),
                    baseQuad = parseQuad(corners.getJSONArray("base")),
                    auxQuad = parseQuad(corners.getJSONArray("aux"))
                )
            )
        }
        return out
    }

    private fun parseQuad(arr: org.json.JSONArray): Array<Point> {
        val pts = ArrayList<Point>()
        for (i in 0 until arr.length()) {
            val p = arr.getJSONArray(i)
            pts.add(Point(p.getDouble(0), p.getDouble(1)))
        }
        return Geometry.orderQuad(pts)
    }

    private fun loadAsset(path: String): Mat {
        val bmp = context.assets.open(path).use { BitmapFactory.decodeStream(it) }
        val mat = MatTools.fromBitmap(bmp)
        if (mat.channels() == 4) {
            val bgr = Mat()
            Imgproc.cvtColor(mat, bgr, Imgproc.COLOR_RGBA2BGR)
            mat.release()
            return bgr
        }
        return mat
    }

    private fun meta(m: Mat): String = "${m.cols()}x${m.rows()} ch=${m.channels()} type=${m.type()}"

    private fun dump(name: String, m: Mat?) {
        if (!cfg.dumpDebug || m == null || m.empty()) return
        try {
            val dir = java.io.File(context.getExternalFilesDir(null), "debug")
            dir.mkdirs()
            val src = if (m.depth() == CvType.CV_8U) m else {
                val c = Mat()
                m.convertTo(c, CvType.CV_8U)
                c
            }
            Imgcodecs.imwrite(java.io.File(dir, name).absolutePath, src)
            if (src !== m) src.release()
        } catch (t: Throwable) {
            Log.w("Unblock", "dump $name 失败: ${t.message}")
        }
    }

    private fun pct(m: Mat?): Double = if (m == null || m.empty()) -1.0
    else 100.0 * Core.countNonZero(m) / (m.rows() * m.cols())

    private fun detectErrorPx(detected: Array<Point>?, reference: Array<Point>): Double {
        if (detected == null) return -1.0
        val a = Geometry.orderQuad(detected.toList())
        val b = Geometry.orderQuad(reference.toList())
        var sum = 0.0
        for (i in 0 until 4) sum += Math.hypot(a[i].x - b[i].x, a[i].y - b[i].y)
        return sum / 4.0
    }

    /** 平面偏亮（纸张/白板）时把手持物类一起召回；暗色黑板只取 person。 */
    private fun classesFor(img: Mat, quad: Array<Point>?): IntArray {
        val base = cfg.segmentClasses
        if (cfg.segmentObjectClasses.isEmpty() || quad == null) return base
        return try {
            val (small, scale) = MatTools.resizeTo(MatTools.toGray(img), 480)
            val mask = Mat.zeros(small.rows(), small.cols(), CvType.CV_8UC1)
            MatTools.fillQuad(mask, quad.map { Point(it.x * scale, it.y * scale) }.toTypedArray(), 255.0)
            if (Core.countNonZero(mask) < 100) return base
            val mean = Core.mean(small, mask).`val`[0]
            if (mean > 130.0) {
                (base.toSet() + cfg.segmentObjectClasses.toSet()).toIntArray()
            } else base
        } catch (t: Throwable) {
            base
        }
    }

    fun run(sample: Sample): PipelineResult {
        val first = step("decode-base") { loadAsset(sample.baseAsset) }
        val second = step("decode-aux") { loadAsset(sample.auxAsset) }
        try {
            // 样张也走与真实使用一致的自动选基，避免用"较差的那张"当底图来评估效果
            val baseIdx = selectBase(listOf(first, second))
            return if (baseIdx == 0) {
                runPair(sample.id, first, second, sample.baseQuad, sample.auxQuad)
            } else {
                val align = Geometry.estimateAlignment(second, first, cfg.align,
                    sample.baseQuad, sample.auxQuad)
                val h = align.h ?: throw IllegalStateException("阶段[align] 两张照片无法对齐")
                runMulti(sample.id, second, listOf(first), sample.auxQuad, listOf(align))
            }
        } finally {
            first.release()
            second.release()
        }
    }

    fun runPair(
        id: String,
        base: Mat,
        aux: Mat,
        baseQuad: Array<Point>,
        auxQuad: Array<Point>
    ): PipelineResult {
        val align = Geometry.estimateAlignment(base, aux, cfg.align, auxQuad, baseQuad)
        val h = align.h ?: throw IllegalStateException("阶段[align] 两张照片无法对齐")
        return runMulti(id, base, listOf(aux), baseQuad, listOf(align))
    }

    /** 自动选基：用平面置信度、遮挡面积、清晰度、正对程度、曝光五项打分。 */
    fun selectBase(images: List<Mat>): Int {
        if (!cfg.baseSelect.enable || images.size < 2) return 0
        val seg = segmenter()
        val metrics = ArrayList<BaseMetric>()
        for ((i, img) in images.withIndex()) {
            val cand = try {
                ScreenDetector.detectBest(img, cfg.screen)
            } catch (t: Throwable) {
                PlaneCandidate(Geometry.insetQuad(img.cols(), img.rows()), 0.0, "inset")
            }
            val classes = classesFor(img, cand.quad)
            val person = seg?.personMask(img, classes)
            val quad = cand.quad
            metrics.add(
                BaseMetric(
                    index = i,
                    name = "photo_$i",
                    occlusion = QualityGate.personRatio(person, img.rows(), img.cols()),
                    sharpness = QualityGate.sharpness(img),
                    frontal = QualityGate.frontalScore(quad),
                    clipped = QualityGate.clippedRatio(img),
                    planeScore = cand.score,
                    kind = cand.kind
                )
            )
        }
        val chosen = QualityGate.chooseBase(metrics, cfg.baseSelect)
        baseInfo = metrics.joinToString("; ") {
            String.format(java.util.Locale.US, "#%d occ=%.3f sharp=%.0f front=%.3f clip=%.3f %s",
                it.index, it.occlusion, it.sharpness, it.frontal, it.clipped, it.kind)
        } + " -> $chosen"
        return chosen
    }

    fun runMulti(
        id: String,
        base: Mat,
        auxes: List<Mat>,
        baseQuad: Array<Point>,
        alignList: List<AlignmentResult>? = null
    ): PipelineResult {
        require(auxes.isNotEmpty()) { "至少需要一张补图" }
        val steps = ArrayList<Pair<String, Long>>()
        val tStart = System.currentTimeMillis()
        cancelRequested = false

        fun mark(name: String) {
            if (cancelRequested) throw java.util.concurrent.CancellationException("已取消")
            val elapsed = System.currentTimeMillis() - tStart
            steps.add(name to elapsed)
            Log.d("Unblock", "[$id] $name @ ${elapsed}ms")
        }

        val baseSize = Size(base.cols().toDouble(), base.rows().toDouble())
        val seg = segmenter()
        val classes = classesFor(base, baseQuad)
        val screenMask = Occlusion.buildScreenMask(base.rows(), base.cols(), baseQuad, cfg.occlusion.borderTrimPx)
        val personBase = seg?.let { it.personMask(base, classes) }

        val sources = ArrayList<SourceInfo>()
        for ((i, aux) in auxes.withIndex()) {
            val align = alignList?.getOrNull(i)
                ?: Geometry.estimateAlignment(base, aux, cfg.align)
            val h = align.h ?: throw IllegalStateException("阶段[align] 第 ${i + 1} 张补图无法对齐")
            sources.add(blendWith(base, aux, h, align, baseSize, screenMask, personBase, classes, ::mark))
            mark("source-${i + 1}")
        }

        val hole = Mat.zeros(base.rows(), base.cols(), CvType.CV_8UC1)
        val validUnion = Mat.zeros(base.rows(), base.cols(), CvType.CV_8UC1)
        for (s in sources) {
            Core.max(hole, s.mask, hole)
            Core.max(validUnion, s.valid, validUnion)
        }

        val fusionCfg = cfg.fusion.copy(
            minAlpha = cfg.compose.minAlpha,
            edgeSoftPx = cfg.compose.edgeSoftPx
        )
        val fusionOut = Fusion.fuse(
            base, sources.map { it.corrected }, sources.map { it.valid },
            sources.map { it.person }, hole, screenMask, fusionCfg
        )
        val composed = fusionOut.first
        val fusionInfo = fusionOut.second
        mark("fusion")

        val coveredMat = Mat()
        Core.bitwise_and(hole, validUnion, coveredMat)
        val covered = Core.countNonZero(coveredMat)
        val total = Core.countNonZero(hole)
        val coverage = if (total > 0) covered.toDouble() / total else 1.0
        val blind = if (total > 0) 1.0 - coverage else 0.0

        val seam = Compose.seamRatio(composed, hole)
        val seamStep = Compose.seamStep(composed, hole, cfg.compose.seamBandPx)
        val restored = Geometry.rectify(composed, baseQuad, cfg.longSide, cfg.targetAspect)
        mark("rectify")

        val first = sources.first()
        val contentChange = sources.maxOf { it.info.contentChange }
        val qc = QualityGate.postcheck(
            coverage, blind, seamStep,
            first.align.mode, first.align.coverage, first.align.residual,
            contentChange, cfg.quality
        )

        dump("${id}_base.png", base)
        dump("${id}_personBase.png", personBase)
        dump("${id}_occl.png", hole)
        dump("${id}_composed.png", composed)

        val diag = String.format(
            java.util.Locale.US,
            "align %s(%.1fpx,%d) | personBase %.2f%% | occl %.2f%% | fillPx %d | diffRatio %.2f | " +
                "coverage %.1f%% | fusion %d/%d | gain %.2f | color %s | step %.2f | " +
                "resid %s(%d,%.1fpx) | base %s | issues %s",
            first.align.mode, first.align.residual, first.align.inliers,
            pct(personBase), pct(hole), first.info.fillPixels, first.info.diffFillRatio,
            coverage * 100.0, fusionInfo.filled, fusionInfo.regions,
            first.gainMean, if (first.colorApplied) "on" else "off", seamStep,
            if (first.residual?.applied == true) "on" else "off",
            first.residual?.matches ?: 0, first.residual?.maxShift ?: 0.0,
            baseInfo,
            if (qc.issues.isEmpty()) "none" else qc.issues.joinToString(",")
        )
        Log.d("Unblock", "[$id] diag: $diag")

        sources.forEach { s ->
            s.corrected.release()
            s.mask.release()
            s.valid.release()
            s.person?.release()
            s.h.release()
        }
        hole.release()
        validUnion.release()
        screenMask.release()
        personBase?.release()

        val restoredBitmap = step("to-bitmap") { MatTools.toBitmap(restored) }
        restored.release()
        composed.release()

        return PipelineResult(
            id = id,
            restored = restoredBitmap,
            coverage = coverage,
            blindRatio = blind,
            seamRatio = seam,
            seamStep = seamStep,
            elapsedMs = System.currentTimeMillis() - tStart,
            steps = steps,
            diag = diag,
            issues = qc.issues,
            alignMode = first.align.mode,
            contentChange = contentChange
        )
    }

    /**
     * 单张补图的一轮：warp -> 精修 -> 粗增益 -> 遮挡 -> 残差增益 -> 色彩对齐。
     * 粗增益放在遮挡检测之前，避免视角造成的幕面亮度差被误判成遮挡。
     */
    private fun blendWith(
        current: Mat,
        aux: Mat,
        h: Mat,
        align: AlignmentResult,
        baseSize: Size,
        screenMask: Mat,
        personCurrent: Mat?,
        classes: IntArray,
        mark: (String) -> Unit
    ): SourceInfo {
        val warpedPair = Geometry.warpToBase(aux, h, baseSize)
        var auxWarped = warpedPair.first
        var auxValid = warpedPair.second
        mark("warp")

        var residual: ResidualInfo? = null
        if (cfg.align.residualRefine) {
            val refined = Geometry.refineResidual(current, auxWarped, auxValid, cfg.align)
            if (refined.first !== auxWarped) auxWarped.release()
            auxWarped = refined.first
            auxValid = refined.second
            residual = refined.third
            mark("residual")
        }

        val seg = segmenter()
        val personAux = seg?.let { it.personMask(aux, classes) }
        val personAuxWarped = personAux?.let {
            val warped = Mat()
            Imgproc.warpPerspective(it, warped, h, baseSize, Imgproc.INTER_NEAREST)
            warped
        }
        mark("segment")

        // 遮挡检测前的块中位数粗增益
        val preRegion = Mat.zeros(current.size(), CvType.CV_8UC1)
        Core.bitwise_and(screenMask, auxValid, preRegion)
        val preGain = Compose.estimateGainMapRobust(current, auxWarped, preRegion, cfg.compose,
            cfg.compose.gainGrid)
        val auxPre = Compose.applyGain(auxWarped, preGain)
        preGain.release()
        auxWarped.release()
        mark("pre-gain")

        val occl = Occlusion.detect(current, auxPre, auxValid, screenMask, personCurrent,
            personAuxWarped, cfg.occlusion)
        mark("occlusion")

        val region = Mat.zeros(current.size(), CvType.CV_8UC1)
        val notMask = Mat()
        Core.bitwise_not(occl.first, notMask)
        Core.bitwise_and(notMask, screenMask, region)
        Core.bitwise_and(region, auxValid, region)
        if (cfg.compose.gainExcludePx > 0) {
            val ring = MatTools.dilate(occl.first, cfg.compose.gainExcludePx * 2 + 1)
            region.setTo(Scalar(0.0), ring)
            ring.release()
        }

        val gainMap = Compose.estimateGainMap(current, auxPre, region, cfg.compose)
        val auxGained = Compose.applyGain(auxPre, gainMap)
        val gainMeanValue = Core.mean(gainMap).`val`[0]
        gainMap.release()
        mark("gain")
        val colorApplied = cfg.compose.colorMatchEnable && Core.countNonZero(occl.first) > 0
        val auxCorrected = Compose.matchColorLab(auxGained, current, occl.first, auxValid,
            screenMask, cfg.compose)
        mark("color")

        region.release()
        notMask.release()
        auxPre.release()
        if (auxGained !== auxCorrected) auxGained.release()

        return SourceInfo(
            index = 0,
            h = h,
            align = align,
            corrected = auxCorrected,
            mask = occl.first,
            valid = auxValid,
            person = personAuxWarped,
            info = occl.second,
            gainMean = gainMeanValue,
            colorApplied = colorApplied,
            residual = residual
        )
    }
}

data class UiConfig(
    val estimateSeconds: Int = 4
)

/**
 * 从 assets/config.json 的 params 段读取全部算法参数，与 PC 端保持同一份口径。
 */
object ConfigLoader {

    fun loadUi(context: Context): UiConfig {
        val d = UiConfig()
        return try {
            val text = context.assets.open("config.json").use { it.readBytes().toString(Charsets.UTF_8) }
            val ui = JSONObject(text).optJSONObject("params")?.optJSONObject("ui") ?: return d
            d.copy(estimateSeconds = ui.optInt("estimate_seconds", d.estimateSeconds))
        } catch (t: Throwable) {
            d
        }
    }

    private fun intArray(o: JSONObject?, key: String, def: IntArray): IntArray {
        val a = o?.optJSONArray(key) ?: return def
        val out = IntArray(a.length()) { a.optInt(it, 0) }
        return if (out.isEmpty()) def else out
    }

    fun load(context: Context): PipelineConfig {
        val d = PipelineConfig()
        return try {
            val text = context.assets.open("config.json").use { it.readBytes().toString(Charsets.UTF_8) }
            val root = JSONObject(text)
            val p = root.optJSONObject("params") ?: return d
            val a = p.optJSONObject("align")
            val o = p.optJSONObject("occlusion")
            val c = p.optJSONObject("compose")
            val s = p.optJSONObject("segment")
            val r = p.optJSONObject("rectify")
            val g = p.optJSONObject("debug")
            val q = p.optJSONObject("quality")
            val b = p.optJSONObject("base_select")
            val f = p.optJSONObject("fusion")
            val sc = p.optJSONObject("screen")
            val br = sc?.optJSONObject("bright")

            d.copy(
                longSide = r?.optInt("target_long_side", d.longSide) ?: d.longSide,
                targetAspect = if (r != null && r.has("target_aspect") && !r.isNull("target_aspect"))
                    r.optDouble("target_aspect") else d.targetAspect,
                align = d.align.copy(
                    siftNfeatures = a?.optInt("sift_nfeatures", d.align.siftNfeatures) ?: d.align.siftNfeatures,
                    ratioTest = a?.optDouble("ratio_test", d.align.ratioTest) ?: d.align.ratioTest,
                    ransacThresh = a?.optDouble("ransac_thresh", d.align.ransacThresh) ?: d.align.ransacThresh,
                    minInliers = a?.optInt("min_inliers", d.align.minInliers) ?: d.align.minInliers,
                    minInlierCoverage = a?.optDouble("min_inlier_coverage", d.align.minInlierCoverage)
                        ?: d.align.minInlierCoverage,
                    featureMaxSide = a?.optInt("feature_max_side", d.align.featureMaxSide) ?: d.align.featureMaxSide,
                    residualRefine = a?.optBoolean("residual_refine", d.align.residualRefine) ?: d.align.residualRefine,
                    residualGrid = a?.optInt("residual_grid", d.align.residualGrid) ?: d.align.residualGrid,
                    residualMaxPx = a?.optDouble("residual_max_px", d.align.residualMaxPx) ?: d.align.residualMaxPx,
                    residualSmooth = a?.optInt("residual_smooth", d.align.residualSmooth) ?: d.align.residualSmooth,
                    residualMaxSide = a?.optInt("residual_max_side", d.align.residualMaxSide)
                        ?: d.align.residualMaxSide,
                    residualMinMatches = a?.optInt("residual_min_matches", d.align.residualMinMatches)
                        ?: d.align.residualMinMatches,
                    guardMaxShiftRatio = a?.optDouble("guard_max_shift_ratio", d.align.guardMaxShiftRatio)
                        ?: d.align.guardMaxShiftRatio,
                    eccEnable = a?.optBoolean("ecc_enable", d.align.eccEnable) ?: d.align.eccEnable,
                    eccMaxSide = a?.optInt("ecc_max_side", d.align.eccMaxSide) ?: d.align.eccMaxSide,
                    eccIterations = a?.optInt("ecc_iterations", d.align.eccIterations) ?: d.align.eccIterations,
                    eccEps = a?.optDouble("ecc_eps", d.align.eccEps) ?: d.align.eccEps,
                    eccMinCc = a?.optDouble("ecc_min_cc", d.align.eccMinCc) ?: d.align.eccMinCc,
                    phaseEnable = a?.optBoolean("phase_enable", d.align.phaseEnable) ?: d.align.phaseEnable,
                    phaseMinResponse = a?.optDouble("phase_min_response", d.align.phaseMinResponse)
                        ?: d.align.phaseMinResponse,
                    alignMaxResidualMedian = a?.optDouble("align_max_residual_median",
                        d.align.alignMaxResidualMedian) ?: d.align.alignMaxResidualMedian
                ),
                occlusion = d.occlusion.copy(
                    diffBlurKernel = o?.optInt("diff_blur_kernel", d.occlusion.diffBlurKernel)
                        ?: d.occlusion.diffBlurKernel,
                    diffThresh = o?.optInt("diff_thresh", d.occlusion.diffThresh) ?: d.occlusion.diffThresh,
                    morphOpenKernel = o?.optInt("morph_open_kernel", d.occlusion.morphOpenKernel)
                        ?: d.occlusion.morphOpenKernel,
                    morphCloseKernel = o?.optInt("morph_close_kernel", d.occlusion.morphCloseKernel)
                        ?: d.occlusion.morphCloseKernel,
                    dilateKernel = o?.optInt("dilate_kernel", d.occlusion.dilateKernel) ?: d.occlusion.dilateKernel,
                    personBlockKernel = o?.optInt("person_block_kernel", d.occlusion.personBlockKernel)
                        ?: d.occlusion.personBlockKernel,
                    minBlobAreaRatio = o?.optDouble("min_blob_area_ratio", d.occlusion.minBlobAreaRatio)
                        ?: d.occlusion.minBlobAreaRatio,
                    borderTrimPx = o?.optInt("border_trim_px", d.occlusion.borderTrimPx)
                        ?: d.occlusion.borderTrimPx,
                    shadowEnable = o?.optBoolean("shadow_enable", d.occlusion.shadowEnable)
                        ?: d.occlusion.shadowEnable,
                    shadowRatio = o?.optDouble("shadow_ratio", d.occlusion.shadowRatio) ?: d.occlusion.shadowRatio,
                    shadowDelta = o?.optDouble("shadow_delta", d.occlusion.shadowDelta) ?: d.occlusion.shadowDelta,
                    shadowMaxPx = o?.optInt("shadow_max_px", d.occlusion.shadowMaxPx) ?: d.occlusion.shadowMaxPx,
                    shadowMaxRatio = o?.optDouble("shadow_max_ratio", d.occlusion.shadowMaxRatio)
                        ?: d.occlusion.shadowMaxRatio,
                    interiorBlobs = o?.optBoolean("interior_blobs", d.occlusion.interiorBlobs)
                        ?: d.occlusion.interiorBlobs,
                    interiorMinAreaRatio = o?.optDouble("interior_min_area_ratio",
                        d.occlusion.interiorMinAreaRatio) ?: d.occlusion.interiorMinAreaRatio,
                    interiorMinSolidity = o?.optDouble("interior_min_solidity", d.occlusion.interiorMinSolidity)
                        ?: d.occlusion.interiorMinSolidity,
                    interiorMinThicknessRatio = o?.optDouble("interior_min_thickness_ratio",
                        d.occlusion.interiorMinThicknessRatio) ?: d.occlusion.interiorMinThicknessRatio,
                    interiorColorDelta = o?.optDouble("interior_color_delta", d.occlusion.interiorColorDelta)
                        ?: d.occlusion.interiorColorDelta,
                    maxDiffFillRatio = o?.optDouble("max_diff_fill_ratio", d.occlusion.maxDiffFillRatio)
                        ?: d.occlusion.maxDiffFillRatio,
                    personGrowKernel = o?.optInt("person_grow_kernel", d.occlusion.personGrowKernel)
                        ?: d.occlusion.personGrowKernel,
                    nearBlurKernel = o?.optInt("near_blur_kernel", d.occlusion.nearBlurKernel)
                        ?: d.occlusion.nearBlurKernel,
                    nearDiffThresh = o?.optInt("near_diff_thresh", d.occlusion.nearDiffThresh)
                        ?: d.occlusion.nearDiffThresh,
                    contentCheck = o?.optBoolean("content_check", d.occlusion.contentCheck)
                        ?: d.occlusion.contentCheck,
                    contentDiffThresh = o?.optInt("content_diff_thresh", d.occlusion.contentDiffThresh)
                        ?: d.occlusion.contentDiffThresh,
                    contentChangeRatio = o?.optDouble("content_change_ratio", d.occlusion.contentChangeRatio)
                        ?: d.occlusion.contentChangeRatio
                ),
                compose = d.compose.copy(
                    edgeSoftPx = c?.optInt("edge_soft_px", d.compose.edgeSoftPx) ?: d.compose.edgeSoftPx,
                    minAlpha = c?.optDouble("min_alpha", d.compose.minAlpha) ?: d.compose.minAlpha,
                    gainClipMin = c?.optDouble("gain_clip_min", d.compose.gainClipMin) ?: d.compose.gainClipMin,
                    gainClipMax = c?.optDouble("gain_clip_max", d.compose.gainClipMax) ?: d.compose.gainClipMax,
                    gainSmooth = c?.optInt("gain_smooth", d.compose.gainSmooth) ?: d.compose.gainSmooth,
                    gainGrid = c?.optInt("gain_grid", d.compose.gainGrid) ?: d.compose.gainGrid,
                    gainExcludePx = c?.optInt("gain_exclude_px", d.compose.gainExcludePx)
                        ?: d.compose.gainExcludePx,
                    seamBandPx = c?.optInt("seam_band_px", d.compose.seamBandPx) ?: d.compose.seamBandPx,
                    colorMatchEnable = c?.optBoolean("color_match_enable", d.compose.colorMatchEnable)
                        ?: d.compose.colorMatchEnable,
                    colorMatchRingPx = c?.optInt("color_match_ring_px", d.compose.colorMatchRingPx)
                        ?: d.compose.colorMatchRingPx,
                    colorMatchFlatGradMax = c?.optDouble("color_match_flat_grad_max",
                        d.compose.colorMatchFlatGradMax) ?: d.compose.colorMatchFlatGradMax,
                    colorMatchMinRingPx = c?.optInt("color_match_min_ring_px", d.compose.colorMatchMinRingPx)
                        ?: d.compose.colorMatchMinRingPx,
                    colorMatchStdClipMin = c?.optDouble("color_match_std_clip_min",
                        d.compose.colorMatchStdClipMin) ?: d.compose.colorMatchStdClipMin,
                    colorMatchStdClipMax = c?.optDouble("color_match_std_clip_max",
                        d.compose.colorMatchStdClipMax) ?: d.compose.colorMatchStdClipMax,
                    colorMatchAlignL = c?.optBoolean("color_match_align_l", d.compose.colorMatchAlignL)
                        ?: d.compose.colorMatchAlignL
                ),
                quality = d.quality.copy(
                    minSharpness = q?.optDouble("min_sharpness", d.quality.minSharpness)
                        ?: d.quality.minSharpness,
                    maxOcclusionRatio = q?.optDouble("max_occlusion_ratio", d.quality.maxOcclusionRatio)
                        ?: d.quality.maxOcclusionRatio,
                    minCoverage = q?.optDouble("min_coverage", d.quality.minCoverage) ?: d.quality.minCoverage,
                    maxBlindRatio = q?.optDouble("max_blind_ratio", d.quality.maxBlindRatio)
                        ?: d.quality.maxBlindRatio,
                    maxSeamStep = q?.optDouble("max_seam_step", d.quality.maxSeamStep) ?: d.quality.maxSeamStep,
                    maxAlignResidual = q?.optDouble("max_align_residual", d.quality.maxAlignResidual)
                        ?: d.quality.maxAlignResidual,
                    minInlierCoverage = q?.optDouble("min_inlier_coverage", d.quality.minInlierCoverage)
                        ?: d.quality.minInlierCoverage,
                    maxContentChange = q?.optDouble("max_content_change", d.quality.maxContentChange)
                        ?: d.quality.maxContentChange,
                    retryAltBase = q?.optBoolean("retry_alt_base", d.quality.retryAltBase)
                        ?: d.quality.retryAltBase
                ),
                baseSelect = d.baseSelect.copy(
                    enable = b?.optBoolean("enable", d.baseSelect.enable) ?: d.baseSelect.enable,
                    occlusionWeight = b?.optDouble("occlusion_weight", d.baseSelect.occlusionWeight)
                        ?: d.baseSelect.occlusionWeight,
                    sharpnessWeight = b?.optDouble("sharpness_weight", d.baseSelect.sharpnessWeight)
                        ?: d.baseSelect.sharpnessWeight,
                    perspectiveWeight = b?.optDouble("perspective_weight", d.baseSelect.perspectiveWeight)
                        ?: d.baseSelect.perspectiveWeight,
                    exposureWeight = b?.optDouble("exposure_weight", d.baseSelect.exposureWeight)
                        ?: d.baseSelect.exposureWeight
                ),
                fusion = d.fusion.copy(
                    enable = f?.optBoolean("enable", d.fusion.enable) ?: d.fusion.enable,
                    minRegionPx = f?.optInt("min_region_px", d.fusion.minRegionPx) ?: d.fusion.minRegionPx,
                    ringPx = f?.optInt("ring_px", d.fusion.ringPx) ?: d.fusion.ringPx,
                    maxLabDelta = f?.optDouble("max_lab_delta", d.fusion.maxLabDelta) ?: d.fusion.maxLabDelta,
                    maxPersonOverlap = f?.optDouble("max_person_overlap", d.fusion.maxPersonOverlap)
                        ?: d.fusion.maxPersonOverlap,
                    minValidCoverage = f?.optDouble("min_valid_coverage", d.fusion.minValidCoverage)
                        ?: d.fusion.minValidCoverage
                ),
                screen = d.screen.copy(
                    detectMaxSide = sc?.optInt("detect_max_side", d.screen.detectMaxSide) ?: d.screen.detectMaxSide,
                    minDarkRatio = sc?.optDouble("min_dark_ratio", d.screen.minDarkRatio) ?: d.screen.minDarkRatio,
                    maxMedianLuma = sc?.optDouble("max_median_luma", d.screen.maxMedianLuma)
                        ?: d.screen.maxMedianLuma,
                    minWidthRatio = sc?.optDouble("min_width_ratio", d.screen.minWidthRatio)
                        ?: d.screen.minWidthRatio,
                    minHeightRatio = sc?.optDouble("min_height_ratio", d.screen.minHeightRatio)
                        ?: d.screen.minHeightRatio,
                    minAreaRatio = sc?.optDouble("min_area_ratio", d.screen.minAreaRatio)
                        ?: d.screen.minAreaRatio,
                    maxAreaRatio = sc?.optDouble("max_area_ratio", d.screen.maxAreaRatio)
                        ?: d.screen.maxAreaRatio,
                    aspectMin = sc?.optDouble("aspect_min", d.screen.aspectMin) ?: d.screen.aspectMin,
                    aspectMax = sc?.optDouble("aspect_max", d.screen.aspectMax) ?: d.screen.aspectMax,
                    brightMedianLumaMin = br?.optDouble("median_luma_min", d.screen.brightMedianLumaMin)
                        ?: d.screen.brightMedianLumaMin,
                    brightMedianLumaMax = br?.optDouble("median_luma_max", d.screen.brightMedianLumaMax)
                        ?: d.screen.brightMedianLumaMax,
                    brightMinBrightRatio = br?.optDouble("min_bright_ratio", d.screen.brightMinBrightRatio)
                        ?: d.screen.brightMinBrightRatio,
                    brightMinAreaRatio = br?.optDouble("min_area_ratio", d.screen.brightMinAreaRatio)
                        ?: d.screen.brightMinAreaRatio,
                    brightMaxAreaRatio = br?.optDouble("max_area_ratio", d.screen.brightMaxAreaRatio)
                        ?: d.screen.brightMaxAreaRatio,
                    brightMinWidthRatio = br?.optDouble("min_width_ratio", d.screen.brightMinWidthRatio)
                        ?: d.screen.brightMinWidthRatio,
                    brightMinHeightRatio = br?.optDouble("min_height_ratio", d.screen.brightMinHeightRatio)
                        ?: d.screen.brightMinHeightRatio,
                    brightAspectMin = br?.optDouble("aspect_min", d.screen.brightAspectMin)
                        ?: d.screen.brightAspectMin,
                    brightAspectMax = br?.optDouble("aspect_max", d.screen.brightAspectMax)
                        ?: d.screen.brightAspectMax,
                    brightMinRectangularity = br?.optDouble("min_rectangularity", d.screen.brightMinRectangularity)
                        ?: d.screen.brightMinRectangularity,
                    brightContourEpsilon = br?.optDouble("contour_epsilon_ratio", d.screen.brightContourEpsilon)
                        ?: d.screen.brightContourEpsilon,
                    brightMinContrast = br?.optDouble("min_contrast", d.screen.brightMinContrast)
                        ?: d.screen.brightMinContrast
                ),
                modelAsset = s?.optString("model", d.modelAsset) ?: d.modelAsset,
                segmentInputSize = s?.optInt("input_size", d.segmentInputSize) ?: d.segmentInputSize,
                segmentConf = s?.optDouble("conf", d.segmentConf.toDouble())?.toFloat() ?: d.segmentConf,
                usePersonMask = s?.optBoolean("enabled", d.usePersonMask) ?: d.usePersonMask,
                segmentClasses = intArray(s, "classes", d.segmentClasses),
                segmentObjectClasses = intArray(s, "object_classes", d.segmentObjectClasses),
                dumpDebug = g?.optBoolean("dump", d.dumpDebug) ?: d.dumpDebug
            )
        } catch (t: Throwable) {
            d
        }
    }
}
