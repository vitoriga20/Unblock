package com.unblock.app.core

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
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
    val minBlobAreaRatio: Double = 0.0015,
    val borderTrimPx: Int = 2,
    val shadowEnable: Boolean = true,
    val shadowRatio: Double = 0.94,
    val shadowDelta: Double = 4.0,
    val shadowMaxPx: Int = 600,
    val shadowMaxRatio: Double = 0.12
)

data class OcclusionInfo(
    val maskPixels: Int,
    val fillablePixels: Int,
    val coverage: Double,
    val blindRatio: Double,
    val personBasePixels: Int = 0,
    val personAuxBlockedPixels: Int = 0,
    val borderPixels: Int = 0,
    val fillPixels: Int = 0
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

    /** 只保留最终能延伸到画面边界的连通块，用来排除完全位于屏幕内部的图形元素。 */
    private fun keepBorderBlobs(binary: Mat, minArea: Int, bridge: Int): Mat {
        val merged = if (bridge > 1) MatTools.dilate(binary, bridge) else binary
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        val count = Imgproc.connectedComponentsWithStats(merged, labels, stats, centroids, 8, CvType.CV_32S)
        val h = binary.rows()
        val w = binary.cols()

        val keep = Mat.zeros(h, w, CvType.CV_8UC1)
        for (i in 1 until count) {
            val area = stats.get(i, Imgproc.CC_STAT_AREA)[0]
            if (area < minArea) continue
            val x = stats.get(i, Imgproc.CC_STAT_LEFT)[0].toInt()
            val y = stats.get(i, Imgproc.CC_STAT_TOP)[0].toInt()
            val bw = stats.get(i, Imgproc.CC_STAT_WIDTH)[0].toInt()
            val bh = stats.get(i, Imgproc.CC_STAT_HEIGHT)[0].toInt()
            val touches = x <= 1 || y <= 1 || (x + bw) >= w - 1 || (y + bh) >= h - 1
            if (!touches) continue
            val labelMask = Mat()
            Core.inRange(labels, Scalar(i.toDouble()), Scalar(i.toDouble()), labelMask)
            keep.setTo(Scalar(255.0), labelMask)
            labelMask.release()
        }

        val kept = Mat()
        Core.bitwise_and(binary, keep, kept)
        return kept
    }

    /**
     * 找出被人物半影压暗的区域。板前站人时会投出阴影，只按人物轮廓填充会留下这圈暗影，
     * 视觉上就是一圈“描边”。判据：相对全局增益明显偏暗，且只取 base 偏暗的方向。
     */
    private fun shadowBand(gb: Mat, ga: Mat, allowed: Mat, person: Mat, cfg: OcclusionConfig): Mat {
        val b = Mat()
        gb.convertTo(b, CvType.CV_32FC1, 1.0, 1.0)
        val a = Mat()
        ga.convertTo(a, CvType.CV_32FC1, 1.0, 1.0)
        val ratio = Mat()
        Core.divide(b, a, ratio)

        // 阴影可能离人物轮廓很远，用距离场而不是膨胀核限制范围
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

    /**
     * 判定基准图里"可以且应该被替换"的像素。
     * 核心规则：只有当另一张图在该处没有人时，才允许用它的像素填补。
     */
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
        // aux 无效（变换后落在画面外）的像素不能参与差分判断
        val invalidAux = Mat()
        Core.bitwise_not(auxValid, invalidAux)
        diff.setTo(Scalar(0.0), invalidAux)

        val binary = Mat()
        Imgproc.threshold(diff, binary, cfg.diffThresh.toDouble(), 255.0, Imgproc.THRESH_BINARY)
        val open = MatTools.odd(cfg.morphOpenKernel)
        val close = MatTools.odd(cfg.morphCloseKernel)
        val opened = Mat()
        Imgproc.morphologyEx(binary, opened, Imgproc.MORPH_OPEN, MatTools.ones(open))
        val closed = Mat()
        Imgproc.morphologyEx(opened, closed, Imgproc.MORPH_CLOSE, MatTools.ones(close))

        val minArea = (cfg.minBlobAreaRatio * base.rows() * base.cols()).toInt()
        val borderOnly = keepBorderBlobs(closed, minArea, maxOf(3, cfg.dilateKernel * 2))

        val auxBlocked = Mat.zeros(base.rows(), base.cols(), CvType.CV_8UC1)
        if (personAuxWarped != null && !personAuxWarped.empty()) {
            val blocked = MatTools.dilate(personAuxWarped, cfg.personBlockKernel)
            blocked.copyTo(auxBlocked)
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
        Core.bitwise_and(borderOnly, notBlocked, borderAllowed)
        Core.bitwise_or(fill, borderAllowed, fill)

        if (personBase != null && !personBase.empty() && cfg.shadowEnable) {
            val allowed = Mat()
            Core.bitwise_and(screenMask, auxValid, allowed)
            Core.bitwise_and(allowed, notBlocked, allowed)
            Core.bitwise_or(fill, shadowBand(gb, ga, allowed, personBase, cfg), fill)
        }

        val closedFill = Mat()
        Imgproc.morphologyEx(fill, closedFill, Imgproc.MORPH_CLOSE, MatTools.ones(close))
        val dilated = MatTools.dilate(closedFill, cfg.dilateKernel)

        val screen = Mat.zeros(base.rows(), base.cols(), CvType.CV_8UC1)
        dilated.copyTo(screen, screenMask)
        val scoped = fillHoles(screen)
        scoped.setTo(Scalar(0.0), negate(screenMask))

        val auxValidCount = Core.countNonZero(auxValid)
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
            borderPixels = Core.countNonZero(borderOnly),
            fillPixels = Core.countNonZero(fill)
        )
        return scoped to info
    }

    private fun negate(m: Mat): Mat {
        val out = Mat()
        Core.bitwise_not(m, out)
        return out
    }
}
