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
    val diag: String = ""
)

data class PipelineConfig(
    val longSide: Int = 1920,
    val targetAspect: Double? = null,
    val align: AlignConfig = AlignConfig(),
    val occlusion: OcclusionConfig = OcclusionConfig(),
    val compose: ComposeConfig = ComposeConfig(),
    val modelAsset: String = "model/yolov8n-seg.onnx",
    val segmentInputSize: Int = 640,
    val segmentConf: Float = 0.1f,
    val usePersonMask: Boolean = true,
    val dumpDebug: Boolean = false
)

class Pipeline(private val context: Context, private val cfg: PipelineConfig) {

    private var segmenter: Segmenter? = null

    /** 由界面置位，在阶段交界处生效；单步内部不打断。 */
    @Volatile
    private var cancelRequested = false

    fun cancel() {
        cancelRequested = true
    }

    /** 释放 ONNX 会话；进程退出或需要重建管线时调用。 */
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

    /** 把每一步包起来，出错时能直接指出是哪个阶段。 */
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
        // bitmapToMat 出来的是 RGBA 四通道，后面所有算法都按 BGR 三通道处理，这里统一转掉
        if (mat.channels() == 4) {
            val bgr = Mat()
            Imgproc.cvtColor(mat, bgr, Imgproc.COLOR_RGBA2BGR)
            mat.release()
            return bgr
        }
        return mat
    }

    private fun meta(m: Mat): String = "${m.cols()}x${m.rows()} ch=${m.channels()} type=${m.type()}"

    /** 调试期把中间结果落到外部私有目录，方便 adb pull 核对。 */
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

    /** 自动检测四角与人工标注四角的平均偏差（像素），检测失败返回 -1。 */
    private fun detectErrorPx(detected: Array<Point>?, reference: Array<Point>): Double {
        if (detected == null) return -1.0
        val a = Geometry.orderQuad(detected.toList())
        val b = Geometry.orderQuad(reference.toList())
        var sum = 0.0
        for (i in 0 until 4) sum += Math.hypot(a[i].x - b[i].x, a[i].y - b[i].y)
        return sum / 4.0
    }

    /** 跑内置样张：先解码再走同一套合成流程。 */
    fun run(sample: Sample): PipelineResult {
        val base = step("decode-base") { loadAsset(sample.baseAsset) }
        val aux = step("decode-aux") { loadAsset(sample.auxAsset) }
        return runPair(sample.id, base, aux, sample.baseQuad, sample.auxQuad)
    }

    /**
     * 外部图像入口：图像已在内存里，四角由调用方给定（通常来自四角编辑界面）。
     * hOverride 给出 aux -> base 的单应时跳过重复的特征匹配。
     */
    fun runPair(
        id: String,
        base: Mat,
        aux: Mat,
        baseQuad: Array<Point>,
        auxQuad: Array<Point>,
        hOverride: Mat? = null
    ): PipelineResult {
        val steps = ArrayList<Pair<String, Long>>()
        val tStart = System.currentTimeMillis()
        cancelRequested = false

        fun mark(name: String) {
            if (cancelRequested) throw java.util.concurrent.CancellationException("已取消")
            val elapsed = System.currentTimeMillis() - tStart
            steps.add(name to elapsed)
            Log.d("Unblock", "[$id] $name @ ${elapsed}ms")
        }

        Log.d("Unblock", "[$id] base=${meta(base)} aux=${meta(aux)}")

        val screen = step("screen-detect") { ScreenDetector.detect(base) }
        mark("detect")

        val h = step("homography") {
            hOverride ?: Geometry.homographyFromFeatures(base, aux, baseQuad, auxQuad, cfg.align)
        }
        mark("match")
        val baseSize = Size(base.cols().toDouble(), base.rows().toDouble())

        val seg = step("segment-init") { segmenter() }
        val personBase = seg?.let { step("segment-base") { it.personMask(base) } }
        val screenMask = step("screen-mask") {
            Occlusion.buildScreenMask(base.rows(), base.cols(), baseQuad, cfg.occlusion.borderTrimPx)
        }
        val out = step("blend-pair") {
            blendWith(base, aux, h, baseSize, screenMask, personBase, seg) { mark(it) }
        }
        val composed = out.composed
        val personAuxWarped = out.personAuxWarped
        val auxValid = out.auxValid
        val occl = out.mask to out.info
        mark("blend")

        val seam = step("seam") { Compose.seamRatio(composed, occl.first) }
        val seamStep = step("seam-step") { Compose.seamStep(composed, occl.first, cfg.compose.seamBandPx) }
        val restored = step("rectify") { Geometry.rectify(composed, baseQuad, cfg.longSide, cfg.targetAspect) }
        mark("rectify")

        dump("${id}_base.png", base)
        dump("${id}_personBase.png", personBase)
        dump("${id}_personAuxWarped.png", personAuxWarped)
        dump("${id}_occl.png", occl.first)
        dump("${id}_composed.png", composed)

        val diag = String.format(
            java.util.Locale.US,
            "detect %s | err %.1fpx | personBase %.2f%% | personAuxW %.2f%% | auxValid %.2f%% | occl %.2f%% | fillPx %d | borderPx %d | blockedPx %d | occlPx %d | gain %.2f | color %s | step %.2f | resid %s(%d, %.1fpx)",
            screen.message, detectErrorPx(screen.quad, baseQuad),
            pct(personBase), pct(personAuxWarped), pct(auxValid), pct(occl.first),
            occl.second.fillPixels, occl.second.borderPixels, occl.second.personAuxBlockedPixels,
            occl.second.maskPixels, out.gainMean, if (out.colorApplied) "on" else "off", seamStep,
            if (out.residual?.applied == true) "on" else "off",
            out.residual?.matches ?: 0, out.residual?.maxShift ?: 0.0
        )
        Log.d("Unblock", "[$id] diag: $diag")

        return PipelineResult(
            id = id,
            restored = step("to-bitmap") { MatTools.toBitmap(restored) },
            coverage = occl.second.coverage,
            blindRatio = occl.second.blindRatio,
            seamRatio = seam,
            seamStep = seamStep,
            elapsedMs = System.currentTimeMillis() - tStart,
            steps = steps,
            diag = diag
        )
    }

    private class BlendOut(
        val composed: Mat,
        val mask: Mat,
        val info: OcclusionInfo,
        val auxValid: Mat,
        val personAuxWarped: Mat?,
        val gainMean: Double,
        val residual: ResidualInfo? = null,
        val colorApplied: Boolean = false
    )

    /** 单次“把 aux 里真实可见的像素补进 current”的完整一轮：warp -> 精修 -> 分割 -> 遮挡 -> 增益 -> 融合。 */
    private fun blendWith(
        current: Mat,
        aux: Mat,
        h: Mat,
        baseSize: Size,
        screenMask: Mat,
        personCurrent: Mat?,
        seg: Segmenter?,
        mark: (String) -> Unit
    ): BlendOut {
        val warpedPair = Geometry.warpToBase(aux, h, baseSize)
        var auxWarped = warpedPair.first
        var auxValid = warpedPair.second
        mark("warp")

        // 全局单应之后的局部残余位移：网格化位移场 + remap 再补一次
        var residual: ResidualInfo? = null
        if (cfg.align.residualRefine) {
            val refined = Geometry.refineResidual(current, auxWarped, auxValid, cfg.align)
            if (refined.first !== auxWarped) auxWarped.release()
            auxWarped = refined.first
            auxValid = refined.second
            residual = refined.third
            mark("residual")
        }

        val personAux = seg?.let { it.personMask(aux) }
        val personAuxWarped = personAux?.let {
            val warped = Mat()
            Imgproc.warpPerspective(it, warped, h, baseSize, Imgproc.INTER_NEAREST)
            warped
        }
        personAux?.release()
        mark("segment")

        val occl = Occlusion.detect(current, auxWarped, auxValid, screenMask, personCurrent, personAuxWarped, cfg.occlusion)
        mark("occlusion")

        val region = Mat.zeros(current.size(), CvType.CV_8UC1)
        val notMask = Mat()
        Core.bitwise_not(occl.first, notMask)
        Core.bitwise_and(notMask, screenMask, region)
        Core.bitwise_and(region, auxValid, region)
        if (cfg.compose.gainExcludePx > 0) {
            // 接缝外侧一圈可能还带着人物残留/半影，参与增益估计会把整张图拉偏
            val ring = MatTools.dilate(occl.first, cfg.compose.gainExcludePx * 2 + 1)
            region.setTo(Scalar(0.0), ring)
            ring.release()
        }

        val gainMap = Compose.estimateGainMap(current, auxWarped, region, cfg.compose)
        val auxGained = Compose.applyGain(auxWarped, gainMap)
        val gainMeanValue = Core.mean(gainMap).`val`[0]
        gainMap.release()
        mark("gain")
        // 增益只补偿乘性亮度差；色度差用洞口环带的一阶统计再对齐一次（与 PC 端同口径）
        val colorApplied = cfg.compose.colorMatchEnable && Core.countNonZero(occl.first) > 0
        val auxCorrected = Compose.matchColorLab(auxGained, current, occl.first, auxValid, screenMask, cfg.compose)
        mark("color")

        val composed = Compose.blend(current, auxCorrected, occl.first, auxValid, cfg.compose)
        mark("blend")

        region.release()
        notMask.release()
        auxWarped.release()
        if (auxGained !== auxCorrected) auxGained.release()
        auxCorrected.release()

        return BlendOut(composed, occl.first, occl.second, auxValid, personAuxWarped,
            gainMeanValue, residual, colorApplied)
    }

    /**
     * 多图融合：底图 + 若干补图，逐张补进“该处没人”的可见像素。
     * 每轮都对当前合成图重新做人物分割，因此已经补好的区域不会再被当成遮挡。
     */
    fun runMulti(
        id: String,
        base: Mat,
        auxes: List<Mat>,
        baseQuad: Array<Point>,
        hList: List<Mat>? = null
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
        val screenMask = Occlusion.buildScreenMask(base.rows(), base.cols(), baseQuad, cfg.occlusion.borderTrimPx)

        var current = base
        var personCurrent = seg?.let { it.personMask(current) }
        var mask: Mat = Mat.zeros(base.rows(), base.cols(), CvType.CV_8UC1)
        var info: OcclusionInfo? = null
        var gainMean = 0.0
        var lastAuxValid: Mat? = null
        var lastResidual: ResidualInfo? = null
        var lastColorApplied = false

        for ((i, aux) in auxes.withIndex()) {
            val h = hList?.getOrNull(i)
                ?: Geometry.homographyFromFeatures(current, aux, baseQuad,
                    Geometry.insetQuad(aux.cols(), aux.rows()), cfg.align)
            val previousCurrent = current
            val previousMask = mask
            val previousPerson = personCurrent
            val previousAuxValid = lastAuxValid
            val out = blendWith(current, aux, h, baseSize, screenMask, personCurrent, seg) { mark(it) }
            current = out.composed
            mask = out.mask
            info = out.info
            gainMean = out.gainMean
            lastAuxValid = out.auxValid
            lastResidual = out.residual
            lastColorApplied = out.colorApplied
            if (previousCurrent !== base && previousCurrent !== current) previousCurrent.release()
            if (previousMask !== mask) previousMask.release()
            if (previousAuxValid != null && previousAuxValid !== lastAuxValid) previousAuxValid.release()
            personCurrent = seg?.let { it.personMask(current) }
            if (previousPerson != null && previousPerson !== personCurrent) previousPerson.release()
        }

        val seam = Compose.seamRatio(current, mask)
        val seamStep = Compose.seamStep(current, mask, cfg.compose.seamBandPx)
        val restored = step("rectify") { Geometry.rectify(current, baseQuad, cfg.longSide, cfg.targetAspect) }
        mark("rectify")

        dump("${id}_base.png", base)
        dump("${id}_composed.png", current)
        dump("${id}_occl.png", mask)

        val diag = String.format(
            java.util.Locale.US,
            "multi %d 张补图 | 最后一轮 occl %.2f%% | fillPx %d | gain %.2f | color %s | auxValid %.2f%% | step %.2f | resid %s(%d, %.1fpx)",
            auxes.size, pct(mask), info?.fillPixels ?: 0, gainMean,
            if (lastColorApplied) "on" else "off", pct(lastAuxValid), seamStep,
            if (lastResidual?.applied == true) "on" else "off",
            lastResidual?.matches ?: 0, lastResidual?.maxShift ?: 0.0
        )
        Log.d("Unblock", "[$id] diag: $diag")

        return PipelineResult(
            id = id,
            restored = MatTools.toBitmap(restored),
            coverage = info?.coverage ?: 0.0,
            blindRatio = info?.blindRatio ?: 0.0,
            seamRatio = seam,
            seamStep = seamStep,
            elapsedMs = System.currentTimeMillis() - tStart,
            steps = steps,
            diag = diag
        )
    }
}

/**
 * 界面参数：同样是 config.json 的一部分，避免把「大约几秒」这类产品语义写死在代码里。
 */
data class UiConfig(
    val estimateSeconds: Int = 4
)

/**
 * 从 assets/config.json 的 params 段读取全部算法参数，与 PC 端保持同一份口径。
 * 缺省值仍由各 data class 提供，便于局部覆盖。
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

            d.copy(
                longSide = r?.optInt("target_long_side", d.longSide) ?: d.longSide,
                targetAspect = if (r != null && r.has("target_aspect") && !r.isNull("target_aspect"))
                    r.optDouble("target_aspect") else d.targetAspect,
                align = d.align.copy(
                    siftNfeatures = a?.optInt("sift_nfeatures", d.align.siftNfeatures) ?: d.align.siftNfeatures,
                    ratioTest = a?.optDouble("ratio_test", d.align.ratioTest) ?: d.align.ratioTest,
                    ransacThresh = a?.optDouble("ransac_thresh", d.align.ransacThresh) ?: d.align.ransacThresh,
                    minInliers = a?.optInt("min_inliers", d.align.minInliers) ?: d.align.minInliers,
                    featureMaxSide = a?.optInt("feature_max_side", d.align.featureMaxSide) ?: d.align.featureMaxSide,
                    residualRefine = a?.optBoolean("residual_refine", d.align.residualRefine) ?: d.align.residualRefine,
                    residualGrid = a?.optInt("residual_grid", d.align.residualGrid) ?: d.align.residualGrid,
                    residualMaxPx = a?.optDouble("residual_max_px", d.align.residualMaxPx) ?: d.align.residualMaxPx,
                    residualSmooth = a?.optInt("residual_smooth", d.align.residualSmooth) ?: d.align.residualSmooth,
                    residualMaxSide = a?.optInt("residual_max_side", d.align.residualMaxSide) ?: d.align.residualMaxSide,
                    residualMinMatches = a?.optInt("residual_min_matches", d.align.residualMinMatches)
                        ?: d.align.residualMinMatches
                ),
                occlusion = d.occlusion.copy(
                    diffBlurKernel = o?.optInt("diff_blur_kernel", d.occlusion.diffBlurKernel) ?: d.occlusion.diffBlurKernel,
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
                    borderTrimPx = o?.optInt("border_trim_px", d.occlusion.borderTrimPx) ?: d.occlusion.borderTrimPx,
                    shadowEnable = o?.optBoolean("shadow_enable", d.occlusion.shadowEnable) ?: d.occlusion.shadowEnable,
                    shadowRatio = o?.optDouble("shadow_ratio", d.occlusion.shadowRatio) ?: d.occlusion.shadowRatio,
                    shadowDelta = o?.optDouble("shadow_delta", d.occlusion.shadowDelta) ?: d.occlusion.shadowDelta,
                    shadowMaxPx = o?.optInt("shadow_max_px", d.occlusion.shadowMaxPx) ?: d.occlusion.shadowMaxPx,
                    shadowMaxRatio = o?.optDouble("shadow_max_ratio", d.occlusion.shadowMaxRatio)
                        ?: d.occlusion.shadowMaxRatio
                ),
                compose = d.compose.copy(
                    edgeSoftPx = c?.optInt("edge_soft_px", d.compose.edgeSoftPx) ?: d.compose.edgeSoftPx,
                    minAlpha = c?.optDouble("min_alpha", d.compose.minAlpha) ?: d.compose.minAlpha,
                    gainClipMin = c?.optDouble("gain_clip_min", d.compose.gainClipMin) ?: d.compose.gainClipMin,
                    gainClipMax = c?.optDouble("gain_clip_max", d.compose.gainClipMax) ?: d.compose.gainClipMax,
                    gainSmooth = c?.optInt("gain_smooth", d.compose.gainSmooth) ?: d.compose.gainSmooth,
                    gainExcludePx = c?.optInt("gain_exclude_px", d.compose.gainExcludePx)
                        ?: d.compose.gainExcludePx,
                    seamBandPx = c?.optInt("seam_band_px", d.compose.seamBandPx) ?: d.compose.seamBandPx,
                    colorMatchEnable = c?.optBoolean("color_match_enable", d.compose.colorMatchEnable)
                        ?: d.compose.colorMatchEnable,
                    colorMatchRingPx = c?.optInt("color_match_ring_px", d.compose.colorMatchRingPx)
                        ?: d.compose.colorMatchRingPx,
                    colorMatchFlatGradMax = c?.optDouble("color_match_flat_grad_max", d.compose.colorMatchFlatGradMax)
                        ?: d.compose.colorMatchFlatGradMax,
                    colorMatchMinRingPx = c?.optInt("color_match_min_ring_px", d.compose.colorMatchMinRingPx)
                        ?: d.compose.colorMatchMinRingPx,
                    colorMatchStdClipMin = c?.optDouble("color_match_std_clip_min", d.compose.colorMatchStdClipMin)
                        ?: d.compose.colorMatchStdClipMin,
                    colorMatchStdClipMax = c?.optDouble("color_match_std_clip_max", d.compose.colorMatchStdClipMax)
                        ?: d.compose.colorMatchStdClipMax,
                    colorMatchAlignL = c?.optBoolean("color_match_align_l", d.compose.colorMatchAlignL)
                        ?: d.compose.colorMatchAlignL
                ),
                modelAsset = s?.optString("model", d.modelAsset) ?: d.modelAsset,
                segmentInputSize = s?.optInt("input_size", d.segmentInputSize) ?: d.segmentInputSize,
                segmentConf = s?.optDouble("conf", d.segmentConf.toDouble())?.toFloat() ?: d.segmentConf,
                usePersonMask = s?.optBoolean("enabled", d.usePersonMask) ?: d.usePersonMask,
                dumpDebug = g?.optBoolean("dump", d.dumpDebug) ?: d.dumpDebug
            )
        } catch (t: Throwable) {
            d
        }
    }
}
