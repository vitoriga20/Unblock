package com.unblock.app.core

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

data class FusionConfig(
    val enable: Boolean = true,
    val minRegionPx: Int = 400,
    val ringPx: Int = 41,
    val maxLabDelta: Double = 26.0,
    val maxPersonOverlap: Double = 0.2,
    val minValidCoverage: Double = 0.85,
    val minAlpha: Double = 0.5,
    val edgeSoftPx: Int = 5
)

data class FusionRegion(val region: Int, val area: Int, val source: Int, val delta: Double)

data class FusionInfo(
    val regions: Int,
    val filled: Int,
    val skipped: Int,
    val sources: List<FusionRegion> = emptyList()
)

/**
 * 逐区域多源融合：每个遮挡连通块单独挑最合适的源。
 * 颜色一致性只用于排序；只有一个可用源时不会因为色差把主遮挡物留下。
 */
object Fusion {

    private fun grayTo3(gray: Mat): Mat {
        val out = Mat()
        Imgproc.cvtColor(gray, out, Imgproc.COLOR_GRAY2BGR)
        return out
    }

    private fun labMean(img: Mat, mask: Mat, rect: Rect): DoubleArray {
        if (rect.width <= 0 || rect.height <= 0) return doubleArrayOf(0.0, 0.0, 0.0)
        val lab = Mat()
        Imgproc.cvtColor(img.submat(rect), lab, Imgproc.COLOR_BGR2Lab)
        val m = Core.mean(lab, mask.submat(rect))
        return doubleArrayOf(m.`val`[0], m.`val`[1], m.`val`[2])
    }

    private fun sharpness(img: Mat, mask: Mat, rect: Rect): Double {
        if (rect.width <= 0 || rect.height <= 0) return 0.0
        val gray = MatTools.toGray(img.submat(rect))
        val lap = Mat()
        Imgproc.Laplacian(gray, lap, CvType.CV_32F)
        val mean = org.opencv.core.MatOfDouble()
        val std = org.opencv.core.MatOfDouble()
        Core.meanStdDev(lap, mean, std, mask.submat(rect))
        val s = std.toArray()[0]
        return s * s
    }

    fun fuse(canvas: Mat, sources: List<Mat>, valids: List<Mat>, persons: List<Mat?>,
             hole: Mat, screen: Mat, cfg: FusionConfig): Pair<Mat, FusionInfo> {
        var regions = 0
        var filled = 0
        var skipped = 0
        val used = ArrayList<FusionRegion>()
        if (!cfg.enable || sources.isEmpty() || Core.countNonZero(hole) == 0) {
            return canvas to FusionInfo(0, 0, 0)
        }

        val out = canvas.clone()
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        val num = Imgproc.connectedComponentsWithStats(
            hole, labels, stats, centroids, 8, CvType.CV_32S
        )

        for (i in 1 until num) {
            val area = stats.get(i, Imgproc.CC_STAT_AREA)[0].toInt()
            if (area < cfg.minRegionPx) continue
            regions++

            val x = stats.get(i, Imgproc.CC_STAT_LEFT)[0].toInt()
            val y = stats.get(i, Imgproc.CC_STAT_TOP)[0].toInt()
            val bw = stats.get(i, Imgproc.CC_STAT_WIDTH)[0].toInt()
            val bh = stats.get(i, Imgproc.CC_STAT_HEIGHT)[0].toInt()
            if (bw <= 0 || bh <= 0) continue

            val comp = Mat.zeros(hole.rows(), hole.cols(), CvType.CV_8UC1)
            val eq = Mat()
            Core.compare(labels, Scalar(i.toDouble()), eq, Core.CMP_EQ)
            Core.bitwise_and(eq, hole, comp)
            val rect = Rect(x, y, bw, bh)

            val expanded = MatTools.dilate(comp, cfg.ringPx)
            val ring = Mat()
            Core.subtract(expanded, comp, ring)
            val holeInv = Mat()
            Core.bitwise_not(hole, holeInv)
            Core.bitwise_and(ring, holeInv, ring)
            Core.bitwise_and(ring, screen, ring)
            val ringRect = Imgproc.boundingRect(ring)
            val ringLab = if (ringRect.width > 0) labMean(out, ring, ringRect) else null

            var bestIdx = -1
            var bestScore = -1e9
            var bestDelta = 0.0
            var fallbackIdx = -1
            var fallbackDelta = 1e9

            for (idx in sources.indices) {
                val valid = valids.getOrNull(idx) ?: continue
                val validAndComp = Mat()
                Core.bitwise_and(valid.submat(rect), comp.submat(rect), validAndComp)
                val cov = Core.countNonZero(validAndComp).toDouble() / area
                if (cov < cfg.minValidCoverage) continue

                val person = persons.getOrNull(idx)
                if (person != null) {
                    val both = Mat()
                    Core.bitwise_and(person.submat(rect), comp.submat(rect), both)
                    val overlap = Core.countNonZero(both).toDouble() / area
                    if (overlap > cfg.maxPersonOverlap) continue
                }

                var delta = 0.0
                if (ringLab != null) {
                    val srcLab = labMean(sources[idx], comp, rect)
                    delta = Math.sqrt(
                        Math.pow(srcLab[0] - ringLab[0], 2.0) +
                            Math.pow(srcLab[1] - ringLab[1], 2.0) +
                            Math.pow(srcLab[2] - ringLab[2], 2.0)
                    )
                }
                if (delta < fallbackDelta) {
                    fallbackDelta = delta
                    fallbackIdx = idx
                }
                if (delta > cfg.maxLabDelta) continue

                val sharp = sharpness(sources[idx], comp, rect)
                val score = Math.log1p(sharp) - delta / 20.0
                if (score > bestScore) {
                    bestScore = score
                    bestIdx = idx
                    bestDelta = delta
                }
            }

            if (bestIdx < 0 && fallbackIdx >= 0) {
                bestIdx = fallbackIdx
                bestDelta = fallbackDelta
            }
            if (bestIdx < 0) {
                skipped++
                continue
            }

            blendInto(out, sources[bestIdx], comp, valids[bestIdx], rect, cfg)
            filled++
            used.add(FusionRegion(i, area, bestIdx, Math.round(bestDelta * 100.0) / 100.0))
        }
        return out to FusionInfo(regions, filled, skipped, used)
    }

    private fun blendInto(canvas: Mat, src: Mat, comp: Mat, valid: Mat, rect: Rect, cfg: FusionConfig) {
        val compRoi = comp.submat(rect)
        val validRoi = valid.submat(rect)
        val comp1 = Mat()
        compRoi.convertTo(comp1, CvType.CV_8UC1, 1.0 / 255.0)
        val dist = Mat()
        Imgproc.distanceTransform(comp1, dist, Imgproc.DIST_L2, 5)

        val ramp = Mat()
        Core.divide(dist, Scalar(cfg.edgeSoftPx.toDouble()), ramp)
        val one = Mat(ramp.size(), ramp.type(), Scalar(1.0))
        Core.min(ramp, one, ramp)

        val alpha = Mat()
        ramp.convertTo(alpha, CvType.CV_32FC1, 1.0 - cfg.minAlpha, cfg.minAlpha)
        val maskF = Mat()
        compRoi.convertTo(maskF, CvType.CV_32FC1, 1.0 / 255.0)
        Core.multiply(alpha, maskF, alpha)
        val validF = Mat()
        validRoi.convertTo(validF, CvType.CV_32FC1, 1.0 / 255.0)
        Core.multiply(alpha, validF, alpha)

        val alpha3 = grayTo3(alpha)
        val inv = Mat()
        Core.subtract(one, alpha, inv)
        val inv3 = grayTo3(inv)

        val cf = Mat()
        canvas.submat(rect).convertTo(cf, CvType.CV_32FC3)
        val sf = Mat()
        src.submat(rect).convertTo(sf, CvType.CV_32FC3)
        val a = Mat()
        val b = Mat()
        Core.multiply(cf, inv3, a)
        Core.multiply(sf, alpha3, b)
        val sum = Mat()
        Core.add(a, b, sum)
        sum.convertTo(canvas.submat(rect), CvType.CV_8UC3)

        listOf(dist, ramp, one, alpha, maskF, validF, alpha3, inv, inv3, cf, sf, a, b, sum)
            .forEach { it.release() }
    }
}
