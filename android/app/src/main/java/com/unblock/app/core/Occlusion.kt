package com.unblock.app.core

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

data class OcclusionConfig(
    val diffBlurKernel: Int = 9,
    val diffThresh: Int = 26,
    val morphOpenKernel: Int = 5,
    val morphCloseKernel: Int = 15,
    val dilateKernel: Int = 15,
    val personBlockKernel: Int = 81,
    val minBlobAreaRatio: Double = 0.004,
    val borderTrimPx: Int = 2,
    val shadowEnable: Boolean = true,
    val shadowRatio: Double = 0.94,
    val shadowDelta: Double = 4.0,
    val shadowMaxPx: Int = 600,
    val shadowMaxRatio: Double = 0.12,
    val interiorBlobs: Boolean = true,
    val interiorMinAreaRatio: Double = 0.002,
    val interiorMinSolidity: Double = 0.22,
    val interiorMinThicknessRatio: Double = 0.008,
    val interiorColorDelta: Double = 18.0,
    val maxDiffFillRatio: Double = 0.15,
    val personGrowKernel: Int = 31,
    val nearBlurKernel: Int = 3,
    val nearDiffThresh: Int = 18,
    val contentCheck: Boolean = true,
    val contentDiffThresh: Int = 45,
    val contentChangeRatio: Double = 0.45
)

data class OcclusionInfo(
    val maskPixels: Int,
    val fillablePixels: Int,
    val coverage: Double,
    val blindRatio: Double,
    val personBasePixels: Int = 0,
    val personAuxBlockedPixels: Int = 0,
    val borderPixels: Int = 0,
    val fillPixels: Int = 0,
    val diffFillRatio: Double = 0.0,
    val contentChange: Double = 0.0
)

object Occlusion {

    fun buildScreenMask(rows: Int, cols: Int, quad: Array<Point>, trimPx: Int): Mat {
        val mask = Mat.zeros(rows, cols, CvType.CV_8UC1)
        MatTools.fillQuad(mask, quad, 255.0)
        return if (trimPx > 0) MatTools.erode(mask, trimPx * 2 + 1) else mask
    }

    private fun fillHoles(mask: Mat): Mat {
        val flooded = mask.clone()
        val ffMask = Mat.zeros(mask.rows() + 2, mask.cols() + 2, CvType.CV_8UC1)
        Imgproc.floodFill(flooded, ffMask, Point(0.0, 0.0), Scalar(255.0))
        val holes = Mat()
        Core.bitwise_not(flooded, holes)
        val out = Mat()
        Core.bitwise_or(mask, holes, out)
        return out
    }

    /** 平面背景色：优先取平面内未发生差异的像素，样本不足时退回整个平面。 */
    private fun planeBackground(base: Mat, screen: Mat, binary: Mat): DoubleArray {
        val w = base.cols()
        val h = base.rows()
        val screenB = ByteArray(w * h)
        screen.get(0, 0, screenB)
        val binB = ByteArray(w * h)
        binary.get(0, 0, binB)
        val baseB = ByteArray(w * h * base.channels())
        base.get(0, 0, baseB)

        val bs = ArrayList<Double>()
        val gs = ArrayList<Double>()
        val rs = ArrayList<Double>()
        val step = maxOf(1, (w * h) / 20000)
        var i = 0
        while (i < w * h) {
            if ((screenB[i].toInt() and 0xFF) > 0 && (binB[i].toInt() and 0xFF) == 0) {
                val p = i * 3
                bs.add((baseB[p].toInt() and 0xFF).toDouble())
                gs.add((baseB[p + 1].toInt() and 0xFF).toDouble())
                rs.add((baseB[p + 2].toInt() and 0xFF).toDouble())
            }
            i += step
        }
        if (bs.size < 100) {
            i = 0
            bs.clear(); gs.clear(); rs.clear()
            while (i < w * h) {
                if ((screenB[i].toInt() and 0xFF) > 0) {
                    val p = i * 3
                    bs.add((baseB[p].toInt() and 0xFF).toDouble())
                    gs.add((baseB[p + 1].toInt() and 0xFF).toDouble())
                    rs.add((baseB[p + 2].toInt() and 0xFF).toDouble())
                }
                i += step
            }
        }
        if (bs.isEmpty()) return doubleArrayOf(0.0, 0.0, 0.0)
        val b = bs.sorted()[bs.size / 2]
        val g = gs.sorted()[gs.size / 2]
        val r = rs.sorted()[rs.size / 2]
        return doubleArrayOf(b, g, r)
    }

    /**
     * 遮挡物连通块：贴边 / 与前景语义相交 / 画面内部实心色块 三者之一。
     * 不再用大核膨胀弥合连通性，否则满屏文字会被连成一张网。
     */
    private fun keepOccluderBlobs(binary: Mat, minArea: Int, base: Mat, aux: Mat,
                                  cfg: OcclusionConfig, person: Mat?): Mat {
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        val count = Imgproc.connectedComponentsWithStats(binary, labels, stats, centroids, 8, CvType.CV_32S)
        if (count <= 1) return binary

        val w = binary.cols()
        val h = binary.rows()
        val bg = planeBackground(base, Mat.ones(h, w, CvType.CV_8UC1), binary)
        val longest = maxOf(h, w).toDouble()
        val delta = cfg.interiorColorDelta
        val minThickness = maxOf(4.0, cfg.interiorMinThicknessRatio * longest)
        val minInteriorArea = maxOf(minArea, (cfg.interiorMinAreaRatio * w * h).toInt())
        val personNear = person?.let { MatTools.dilate(it, maxOf(9, cfg.personGrowKernel)) }

        val keep = Mat.zeros(h, w, CvType.CV_8UC1)
        for (i in 1 until count) {
            val x = stats.get(i, Imgproc.CC_STAT_LEFT)[0].toInt()
            val y = stats.get(i, Imgproc.CC_STAT_TOP)[0].toInt()
            val bw = stats.get(i, Imgproc.CC_STAT_WIDTH)[0].toInt()
            val bh = stats.get(i, Imgproc.CC_STAT_HEIGHT)[0].toInt()
            if (bw <= 0 || bh <= 0 || x < 0 || y < 0 || x + bw > w || y + bh > h) continue

            val rect = Rect(x, y, bw, bh)
            val subLab = labels.submat(rect)
            val subBin = binary.submat(rect)
            val eq = Mat()
            Core.compare(subLab, Scalar(i.toDouble()), eq, Core.CMP_EQ)
            val comp = Mat()
            Core.bitwise_and(eq, subBin, comp)
            val compPx = Core.countNonZero(comp)
            if (compPx < minArea) {
                // 麦克风、笔、手这类细长物面积小，但紧贴人形，仍然要补
                if (personNear != null) {
                    val hit = Mat()
                    Core.bitwise_and(personNear.submat(rect), comp, hit)
                    if (Core.countNonZero(hit) >= maxOf(10, (0.3 * compPx).toInt())) {
                        keep.submat(rect).setTo(Scalar(255.0), comp)
                    }
                }
                continue
            }

            val touches = x <= 1 || y <= 1 || (x + bw) >= w - 1 || (y + bh) >= h - 1
            val comp1 = Mat()
            comp.convertTo(comp1, CvType.CV_8UC1, 1.0 / 255.0)
            val dist = Mat()
            Imgproc.distanceTransform(comp1, dist, Imgproc.DIST_L2, 5)
            val thick = Core.minMaxLoc(dist).maxVal >= minThickness

            var keepIt = false
            var hitPerson = false
            if (personNear != null) {
                val near = personNear.submat(rect)
                val hit = Mat()
                Core.bitwise_and(near, comp, hit)
                hitPerson = Core.countNonZero(hit) >= maxOf(20, (0.1 * compPx).toInt())
            }

            if (hitPerson) {
                keepIt = true
            } else if (touches && thick) {
                keepIt = true
            } else if (cfg.interiorBlobs && compPx >= minInteriorArea && thick) {
                val solidity = compPx.toDouble() / (bw * bh).toDouble()
                if (solidity >= cfg.interiorMinSolidity) {
                    val mb = Core.mean(base.submat(rect), comp)
                    val ma = Core.mean(aux.submat(rect), comp)
                    val db = Math.hypot(Math.hypot(mb.`val`[0] - bg[0], mb.`val`[1] - bg[1]),
                        mb.`val`[2] - bg[2])
                    val da = Math.hypot(Math.hypot(ma.`val`[0] - bg[0], ma.`val`[1] - bg[1]),
                        ma.`val`[2] - bg[2])
                    keepIt = db >= delta && (db - da) >= delta * 0.5
                }
            }

            if (keepIt) keep.submat(rect).setTo(Scalar(255.0), comp)
        }

        val kept = Mat()
        Core.bitwise_and(binary, keep, kept)
        return kept
    }

    /** 半影：相对全局增益偏暗、且只取 base 偏暗的方向。 */
    private fun shadowBand(gb: Mat, ga: Mat, allowed: Mat, person: Mat, cfg: OcclusionConfig): Mat {
        val b = Mat()
        gb.convertTo(b, CvType.CV_32FC1, 1.0, 1.0)
        val a = Mat()
        ga.convertTo(a, CvType.CV_32FC1, 1.0, 1.0)
        val ratio = Mat()
        Core.divide(b, a, ratio)

        val longSide = maxOf(person.rows(), person.cols()).toDouble()
        val maxPx = minOf(cfg.shadowMaxPx.toDouble(), maxOf(8.0, longSide * cfg.shadowMaxRatio))
        val personBin = Mat()
        Imgproc.threshold(person, personBin, 0.0, 1.0, Imgproc.THRESH_BINARY)
        val dist = Mat()
        Imgproc.distanceTransform(negate(personBin), dist, Imgproc.DIST_L2, 5)
        val around = Mat()
        Core.compare(dist, Scalar(maxPx), around, Core.CMP_LE)

        val reference = Mat()
        Core.bitwise_and(allowed, negate(around), reference)
        if (Core.countNonZero(reference) < 1000) return Mat.zeros(person.size(), CvType.CV_8UC1)

        val gain = Core.mean(ratio, reference).`val`[0]
        val thr = Mat()
        Core.multiply(a, Scalar(gain * cfg.shadowRatio), thr)
        val darker = Mat()
        Core.compare(ratio, thr, darker, Core.CMP_LT)

        val corr = Mat()
        Core.multiply(a, Scalar(gain), corr)
        Core.subtract(corr, b, corr)
        val enough = Mat()
        Core.compare(corr, Scalar(cfg.shadowDelta), enough, Core.CMP_GT)

        val shadow = Mat()
        Core.bitwise_and(darker, enough, shadow)
        Core.bitwise_and(shadow, around, shadow)
        Core.bitwise_and(shadow, allowed, shadow)
        Imgproc.morphologyEx(shadow, shadow, Imgproc.MORPH_OPEN, MatTools.ones(3))
        Imgproc.morphologyEx(shadow, shadow, Imgproc.MORPH_CLOSE, MatTools.ones(9))
        Core.bitwise_and(shadow, around, shadow)
        Core.bitwise_and(shadow, allowed, shadow)
        return shadow
    }

    /** 重叠区差异大的像素占比，判断两张是否还是同一份内容。 */
    private fun contentChange(base: Mat, auxWarped: Mat, auxValid: Mat, screen: Mat,
                              cfg: OcclusionConfig): Double {
        if (!cfg.contentCheck) return 0.0
        val gb = MatTools.gaussianBlur(MatTools.toGray(base), cfg.diffBlurKernel)
        val ga = MatTools.gaussianBlur(MatTools.toGray(auxWarped), cfg.diffBlurKernel)
        val diff = Mat()
        Core.absdiff(gb, ga, diff)
        val overlap = Mat()
        Core.bitwise_and(screen, auxValid, overlap)
        val total = Core.countNonZero(overlap)
        if (total < 1000) return 0.0
        val changed = Mat()
        Imgproc.threshold(diff, changed, cfg.contentDiffThresh.toDouble(), 255.0, Imgproc.THRESH_BINARY)
        Core.bitwise_and(changed, overlap, changed)
        return Core.countNonZero(changed).toDouble() / total
    }

    fun detect(
        base: Mat,
        auxWarped: Mat,
        auxValid: Mat,
        screenMask: Mat,
        personBase: Mat?,
        personAuxWarped: Mat?,
        cfg: OcclusionConfig
    ): Pair<Mat, OcclusionInfo> {
        val gb = MatTools.gaussianBlur(MatTools.toGray(base), cfg.diffBlurKernel)
        val ga = MatTools.gaussianBlur(MatTools.toGray(auxWarped), cfg.diffBlurKernel)
        val diff = Mat()
        Core.absdiff(gb, ga, diff)
        val invalidAux = Mat()
        Core.bitwise_not(auxValid, invalidAux)
        diff.setTo(Scalar(0.0), invalidAux)

        val binary = Mat()
        Imgproc.threshold(diff, binary, cfg.diffThresh.toDouble(), 255.0, Imgproc.THRESH_BINARY)
        Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_OPEN, MatTools.ones(cfg.morphOpenKernel))
        Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_CLOSE, MatTools.ones(cfg.morphCloseKernel))

        val minArea = (cfg.minBlobAreaRatio * base.rows() * base.cols()).toInt()
        var kept = keepOccluderBlobs(binary, minArea, base, auxWarped, cfg, personBase)

        if (personBase != null) {
            // 麦克风/笔这类低对比细长物会被大核模糊抹掉，单独用小模糊、低阈值补在人物附近
            val nb = MatTools.gaussianBlur(MatTools.toGray(base), cfg.nearBlurKernel)
            val na = MatTools.gaussianBlur(MatTools.toGray(auxWarped), cfg.nearBlurKernel)
            val nd = Mat()
            Core.absdiff(nb, na, nd)
            nd.setTo(Scalar(0.0), invalidAux)
            val near = MatTools.dilate(personBase, maxOf(9, cfg.personGrowKernel))
            val extra = Mat()
            Imgproc.threshold(nd, extra, cfg.nearDiffThresh.toDouble(), 255.0, Imgproc.THRESH_BINARY)
            Core.bitwise_and(extra, near, extra)
            Core.max(kept, extra, kept)
        }

        val screenPx = maxOf(1, Core.countNonZero(screenMask))
        val diffRatio = Core.countNonZero(kept).toDouble() / screenPx
        if (diffRatio > cfg.maxDiffFillRatio) {
            // 差异铺满平面时多半是视角/刷新造成的整体光度差，只保留紧贴前景的一圈
            if (personBase != null) {
                val near = MatTools.dilate(personBase, cfg.personGrowKernel)
                val restricted = Mat()
                Core.bitwise_and(kept, near, restricted)
                kept = restricted
            } else {
                kept.setTo(Scalar(0.0))
            }
        }

        val auxBlocked = Mat.zeros(base.rows(), base.cols(), CvType.CV_8UC1)
        if (personAuxWarped != null && !personAuxWarped.empty()) {
            MatTools.dilate(personAuxWarped, cfg.personBlockKernel).copyTo(auxBlocked)
        }
        val notBlocked = Mat()
        Core.bitwise_not(auxBlocked, notBlocked)

        val fill = Mat.zeros(base.rows(), base.cols(), CvType.CV_8UC1)
        if (personBase != null && !personBase.empty()) {
            val pbAllowed = Mat()
            Core.bitwise_and(personBase, notBlocked, pbAllowed)
            Core.bitwise_or(fill, pbAllowed, fill)
        }
        val borderAllowed = Mat()
        Core.bitwise_and(kept, notBlocked, borderAllowed)
        Core.bitwise_or(fill, borderAllowed, fill)

        if (personBase != null && !personBase.empty() && cfg.shadowEnable) {
            val allowed = Mat()
            Core.bitwise_and(screenMask, auxValid, allowed)
            Core.bitwise_and(allowed, notBlocked, allowed)
            val shadow = shadowBand(gb, ga, allowed, personBase, cfg)
            Core.bitwise_or(fill, shadow, fill)
        }

        val closedFill = Mat()
        Imgproc.morphologyEx(fill, closedFill, Imgproc.MORPH_CLOSE, MatTools.ones(cfg.morphCloseKernel))
        val dilated = MatTools.dilate(closedFill, cfg.dilateKernel)

        val screen = Mat.zeros(base.rows(), base.cols(), CvType.CV_8UC1)
        dilated.copyTo(screen, screenMask)
        val scoped = fillHoles(screen)
        scoped.setTo(Scalar(0.0), negate(screenMask))

        val coveredMat = Mat()
        Core.bitwise_and(scoped, auxValid, coveredMat)
        val covered = Core.countNonZero(coveredMat)
        val total = Core.countNonZero(scoped)
        val info = OcclusionInfo(
            maskPixels = total,
            fillablePixels = covered,
            coverage = if (total > 0) covered.toDouble() / total else 0.0,
            blindRatio = if (total > 0) 1.0 - covered.toDouble() / total else 0.0,
            personBasePixels = if (personBase != null) Core.countNonZero(personBase) else 0,
            personAuxBlockedPixels = Core.countNonZero(auxBlocked),
            borderPixels = Core.countNonZero(kept),
            fillPixels = Core.countNonZero(fill),
            diffFillRatio = diffRatio,
            contentChange = contentChange(base, auxWarped, auxValid, screenMask, cfg)
        )
        return scoped to info
    }

    private fun negate(m: Mat): Mat {
        val out = Mat()
        Core.bitwise_not(m, out)
        return out
    }
}
