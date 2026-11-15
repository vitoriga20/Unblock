package com.unblock.app.core

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.RotatedRect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * 平面候选检测：暗色线段（黑板/投影幕）、亮色线段与亮色轮廓（白板/纸张）、内缩兜底。
 * 结果既可作为四角编辑器的初值，也可作为自动选基的平面置信度。
 */
data class ScreenConfig(
    val detectMaxSide: Int = 1600,
    val darkPercentile: Double = 35.0,
    val minDarkRatio: Double = 0.10,
    val maxMedianLuma: Double = 125.0,
    val minBrightRatio: Double = 0.004,
    val maxBrightRatio: Double = 0.45,
    val minWidthRatio: Double = 0.35,
    val minHeightRatio: Double = 0.30,
    val requireCenterInside: Boolean = true,
    val minAreaRatio: Double = 0.12,
    val maxAreaRatio: Double = 0.98,
    val quadAngleToleranceDeg: Double = 25.0,
    val aspectMin: Double = 1.15,
    val aspectMax: Double = 2.45,
    val topK: Int = 8,
    val brightMedianLumaMin: Double = 150.0,
    val brightMedianLumaMax: Double = 252.0,
    val brightMinBrightRatio: Double = 0.45,
    val brightMinAreaRatio: Double = 0.04,
    val brightMaxAreaRatio: Double = 0.95,
    val brightMinWidthRatio: Double = 0.18,
    val brightMinHeightRatio: Double = 0.15,
    val brightAspectMin: Double = 0.45,
    val brightAspectMax: Double = 3.0,
    val brightMinRectangularity: Double = 0.55,
    val brightContourEpsilon: Double = 0.02,
    val brightMinContrast: Double = 8.0
)

data class ScreenDetectResult(
    val quad: Array<Point>?,
    val score: Double,
    val message: String,
    val kind: String = "none"
)

data class PlaneCandidate(val quad: Array<Point>, val score: Double, val kind: String)

object ScreenDetector {

    private data class Seg(val rho: Double, val theta: Double, val length: Double)

    fun detect(img: Mat, cfg: ScreenConfig = ScreenConfig()): ScreenDetectResult =
        detect(img, cfg, "dark")

    fun detect(img: Mat, cfg: ScreenConfig, mode: String): ScreenDetectResult {
        val (small, scale) = MatTools.resizeTo(img, cfg.detectMaxSide)
        val w = small.cols()
        val h = small.rows()
        val gray = MatTools.toGray(small)
        val eq = Mat()
        Imgproc.createCLAHE(2.5, Size(8.0, 8.0)).apply(gray, eq)

        val segs = segments(eq, w, h)
        if (segs.size < 4) return ScreenDetectResult(null, 0.0, "线段太少(${segs.size})", mode)

        val tol = Math.toRadians(cfg.quadAngleToleranceDeg)
        val vert = ArrayList<Int>()
        val horz = ArrayList<Int>()
        for ((i, s) in segs.withIndex()) {
            val t = s.theta
            if (t <= tol || t >= Math.PI - tol) vert.add(i)
            else if (abs(t - Math.PI / 2.0) <= tol) horz.add(i)
        }
        if (vert.size < 2 || horz.size < 2) return ScreenDetectResult(null, 0.0, "可用直线不足", mode)

        val vx = DoubleArray(vert.size) { verticalX(segs[vert[it]], h / 2.0) }
        val hy = DoubleArray(horz.size) { horizontalY(segs[horz[it]], w / 2.0) }
        val vGroups = cluster1d(vx, DoubleArray(vert.size) { segs[vert[it]].length }, 0.020 * w, cfg.topK)
        val hGroups = cluster1d(hy, DoubleArray(horz.size) { segs[horz[it]].length }, 0.020 * h, cfg.topK)
        if (vGroups.size < 2 || hGroups.size < 2) return ScreenDetectResult(null, 0.0, "可用直线不足", mode)

        val vIdx = vGroups.map { vert[it[0]] }.sortedBy { verticalX(segs[it], h / 2.0) }
        val hIdx = hGroups.map { horz[it[0]] }.sortedBy { horizontalY(segs[it], w / 2.0) }

        val grayBytes = ByteArray(w * h)
        gray.get(0, 0, grayBytes)
        val darkThresh = percentile(grayBytes, cfg.darkPercentile)
        val grad = gradMagnitude(gray)
        val gradData = FloatArray(w * h)
        grad.get(0, 0, gradData)

        var bestScore = 0.0
        var bestQuad: Array<Point>? = null
        val cx = w / 2.0
        val cy = h / 2.0
        val darkMode = mode == "dark"

        val minWidth = if (darkMode) cfg.minWidthRatio else cfg.brightMinWidthRatio
        val minHeight = if (darkMode) cfg.minHeightRatio else cfg.brightMinHeightRatio
        val minArea = if (darkMode) cfg.minAreaRatio else cfg.brightMinAreaRatio
        val maxArea = if (darkMode) cfg.maxAreaRatio else cfg.brightMaxAreaRatio
        val aspectMin = if (darkMode) cfg.aspectMin else cfg.brightAspectMin
        val aspectMax = if (darkMode) cfg.aspectMax else cfg.brightAspectMax

        for (a in 0 until vIdx.size - 1) {
            for (b in a + 1 until vIdx.size) {
                for (c in 0 until hIdx.size - 1) {
                    for (d in c + 1 until hIdx.size) {
                        val pts = ArrayList<Point>(4)
                        var ok = true
                        for ((vi, hi) in listOf(vIdx[a] to hIdx[c], vIdx[b] to hIdx[c],
                            vIdx[b] to hIdx[d], vIdx[a] to hIdx[d])) {
                            val p = intersect(segs[vi], segs[hi]) ?: run { ok = false; null }
                            if (p == null) break
                            pts.add(p)
                        }
                        if (!ok) continue
                        val quad = Geometry.orderQuad(pts)
                        val xl = quad[0].x
                        val xr = quad[1].x
                        val yt = quad[0].y
                        val yb = quad[3].y
                        if (xr - xl < minWidth * w) continue
                        if (yb - yt < minHeight * h) continue
                        if (darkMode && cfg.requireCenterInside && !pointInPoly(quad, cx, cy)) continue
                        val areaRatio = ((xr - xl) * (yb - yt)) / (w.toDouble() * h)
                        if (areaRatio < minArea || areaRatio > maxArea) continue
                        val aspect = (xr - xl) / max(1e-6, yb - yt)
                        if (aspect < aspectMin || aspect > aspectMax) continue

                        val stats = sampleInside(quad, w, h, grayBytes, darkThresh)
                        val support = (edgeSupport(gradData, w, h, quad[0], quad[1]) +
                            edgeSupport(gradData, w, h, quad[1], quad[2]) +
                            edgeSupport(gradData, w, h, quad[2], quad[3]) +
                            edgeSupport(gradData, w, h, quad[3], quad[0])) / 4.0

                        if (darkMode) {
                            if (stats.darkRatio < cfg.minDarkRatio) continue
                            if (stats.medianLuma > cfg.maxMedianLuma) continue
                            if (stats.brightRatio < cfg.minBrightRatio || stats.brightRatio > cfg.maxBrightRatio) continue
                        } else {
                            val med = stats.medianLuma.toDouble()
                            if (med < cfg.brightMedianLumaMin || med > cfg.brightMedianLumaMax) continue
                            if (stats.brightRatio < cfg.brightMinBrightRatio) continue
                            val ring = ringMedian(quad, w, h, grayBytes)
                            if (ring >= 0 && med - ring < cfg.brightMinContrast) continue
                        }

                        val score = areaRatio * (0.15 + support / 255.0)
                        if (score > bestScore) {
                            bestScore = score
                            bestQuad = quad
                        }
                    }
                }
            }
        }

        if (bestQuad == null) return ScreenDetectResult(null, 0.0, "未找到满足约束的平面四边形", mode)
        val full = Geometry.orderQuad(bestQuad.map { Point(it.x / scale, it.y / scale) })
        return ScreenDetectResult(full, bestScore, "ok", mode)
    }

    /** 亮色轮廓：Otsu 取亮区，最大近似四边形即纸张/白板。 */
    private fun detectContour(img: Mat, cfg: ScreenConfig): ScreenDetectResult {
        val (small, scale) = MatTools.resizeTo(img, cfg.detectMaxSide)
        val gray = MatTools.toGray(small)
        val blurred = Mat()
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
        val th = Mat()
        Imgproc.threshold(blurred, th, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)
        val meanFg = Core.mean(blurred, th).`val`[0]
        if (meanFg < 128.0) Core.bitwise_not(th, th)
        Imgproc.morphologyEx(th, th, Imgproc.MORPH_CLOSE, MatTools.ones(11))
        Imgproc.morphologyEx(th, th, Imgproc.MORPH_OPEN, MatTools.ones(5))

        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(th, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        if (contours.isEmpty()) return ScreenDetectResult(null, 0.0, "没有亮色轮廓", "contour")

        val w = gray.cols()
        val h = gray.rows()
        val grad = gradMagnitude(gray)
        val gradData = FloatArray(w * h)
        grad.get(0, 0, gradData)
        val grayBytes = ByteArray(w * h)
        gray.get(0, 0, grayBytes)

        var bestScore = 0.0
        var bestQuad: Array<Point>? = null
        for (c in contours) {
            val area = Imgproc.contourArea(c)
            val areaRatio = area / (w.toDouble() * h)
            if (areaRatio < cfg.brightMinAreaRatio || areaRatio > cfg.brightMaxAreaRatio) continue
            val rect: RotatedRect = Imgproc.minAreaRect(org.opencv.core.MatOfPoint2f(*c.toArray()))
            val rw = rect.size.width
            val rh = rect.size.height
            val rectArea = rw * rh
            if (rectArea <= 1.0) continue
            val rectangularity = area / rectArea
            if (rectangularity < cfg.brightMinRectangularity) continue
            if (min(rw, rh) < cfg.brightMinHeightRatio * min(w, h)) continue
            val aspect = max(rw, rh) / max(1e-6, min(rw, rh))
            if (aspect < cfg.brightAspectMin || aspect > cfg.brightAspectMax) continue

            val approx = org.opencv.core.MatOfPoint2f()
            val curve = org.opencv.core.MatOfPoint2f(*c.toArray())
            Imgproc.approxPolyDP(curve, approx, cfg.brightContourEpsilon * Imgproc.arcLength(curve, true), true)
            val pts = if (approx.total() == 4L) {
                approx.toArray()
            } else {
                val box = org.opencv.core.MatOfPoint2f()
                Imgproc.boxPoints(rect, box)
                box.toArray()
            }
            if (pts.size != 4) continue
            val quad = Geometry.orderQuad(pts.toList())
            val stats = sampleInside(quad, w, h, grayBytes, percentile(grayBytes, cfg.darkPercentile))
            val med = stats.medianLuma.toDouble()
            if (med < cfg.brightMedianLumaMin || med > cfg.brightMedianLumaMax) continue
            if (stats.brightRatio < cfg.brightMinBrightRatio) continue
            val ring = ringMedian(quad, w, h, grayBytes)
            if (ring >= 0 && med - ring < cfg.brightMinContrast) continue

            val support = (edgeSupport(gradData, w, h, quad[0], quad[1]) +
                edgeSupport(gradData, w, h, quad[1], quad[2]) +
                edgeSupport(gradData, w, h, quad[2], quad[3]) +
                edgeSupport(gradData, w, h, quad[3], quad[0])) / 4.0
            val score = areaRatio * (0.15 + support / 255.0) * rectangularity
            if (score > bestScore) {
                bestScore = score
                bestQuad = quad
            }
        }
        if (bestQuad == null) return ScreenDetectResult(null, 0.0, "没有满足约束的亮色四边形", "contour")
        val full = Geometry.orderQuad(bestQuad.map { Point(it.x / scale, it.y / scale) })
        return ScreenDetectResult(full, bestScore, "ok", "contour")
    }

    /** 多策略候选；末尾一定带内缩兜底。 */
    fun detectCandidates(img: Mat, cfg: ScreenConfig, maxN: Int = 4): List<PlaneCandidate> {
        val list = ArrayList<PlaneCandidate>()
        for (mode in listOf("dark", "bright")) {
            try {
                val r = detect(img, cfg, mode)
                if (r.quad != null && r.score > 0) list.add(PlaneCandidate(r.quad, r.score, mode))
            } catch (t: Throwable) {
                // 单策略失败不影响其它策略
            }
        }
        try {
            val r = detectContour(img, cfg)
            if (r.quad != null && r.score > 0) list.add(PlaneCandidate(r.quad, r.score, "contour"))
        } catch (t: Throwable) {
        }

        val kept = ArrayList<PlaneCandidate>()
        for (c in list.sortedByDescending { it.score }) {
            if (kept.any { quadIoU(c.quad, it.quad) > 0.85 }) continue
            kept.add(c)
            if (kept.size >= maxN) break
        }
        kept.add(PlaneCandidate(Geometry.insetQuad(img.cols(), img.rows()), 0.0, "inset"))
        return kept
    }

    fun detectBest(img: Mat, cfg: ScreenConfig): PlaneCandidate = detectCandidates(img, cfg, 1).first()

    private fun quadIoU(a: Array<Point>, b: Array<Point>): Double {
        val all = a + b
        val x0 = all.minOf { it.x }
        val y0 = all.minOf { it.y }
        val x1 = all.maxOf { it.x }
        val y1 = all.maxOf { it.y }
        val side = 320.0 / max(1e-6, max(x1 - x0, y1 - y0))
        val w = max(2, ((x1 - x0) * side).roundToInt() + 2)
        val h = max(2, ((y1 - y0) * side).roundToInt() + 2)
        val ma = Mat.zeros(h, w, CvType.CV_8UC1)
        val mb = Mat.zeros(h, w, CvType.CV_8UC1)
        MatTools.fillQuad(ma, a.map { Point((it.x - x0) * side + 1, (it.y - y0) * side + 1) }.toTypedArray(), 255.0)
        MatTools.fillQuad(mb, b.map { Point((it.x - x0) * side + 1, (it.y - y0) * side + 1) }.toTypedArray(), 255.0)
        val inter = Mat()
        val union = Mat()
        Core.bitwise_and(ma, mb, inter)
        Core.bitwise_or(ma, mb, union)
        val u = Core.countNonZero(union)
        return if (u == 0) 0.0 else Core.countNonZero(inter).toDouble() / u
    }

    // ---- 线段提取与直线参数化 ----

    private fun segments(grayEq: Mat, w: Int, h: Int): List<Seg> {
        val lsd = Imgproc.createLineSegmentDetector(Imgproc.LSD_REFINE_STD)
        val lines = Mat()
        try {
            lsd.detect(grayEq, lines)
        } catch (t: Throwable) {
            return emptyList()
        }
        if (lines.empty()) return emptyList()
        val data = FloatArray(lines.rows() * 4)
        lines.get(0, 0, data)
        val minLen = max(0.10 * min(h, w), 40.0)
        val out = ArrayList<Seg>()
        for (i in 0 until lines.rows()) {
            val x1 = data[i * 4]
            val y1 = data[i * 4 + 1]
            val x2 = data[i * 4 + 2]
            val y2 = data[i * 4 + 3]
            val dx = (x2 - x1).toDouble()
            val dy = (y2 - y1).toDouble()
            val len = hypot(dx, dy)
            if (len < minLen) continue
            val nx = -dy / len
            val ny = dx / len
            var theta = atan2(ny, nx)
            if (theta < 0) theta += Math.PI
            var rho = nx * x1 + ny * y1
            if (rho < 0) {
                rho = -rho
                theta = (theta + Math.PI) % Math.PI
            }
            out.add(Seg(rho, theta, len))
        }
        return out
    }

    private fun verticalX(s: Seg, yRef: Double): Double =
        (s.rho - Math.sin(s.theta) * yRef) / Math.cos(s.theta)

    private fun horizontalY(s: Seg, xRef: Double): Double =
        (s.rho - Math.cos(s.theta) * xRef) / Math.sin(s.theta)

    private fun intersect(a: Seg, b: Seg): Point? {
        val det = Math.cos(a.theta) * Math.sin(b.theta) - Math.sin(a.theta) * Math.cos(b.theta)
        if (abs(det) < 1e-6) return null
        val x = (a.rho * Math.sin(b.theta) - b.rho * Math.sin(a.theta)) / det
        val y = (Math.cos(a.theta) * b.rho - Math.cos(b.theta) * a.rho) / det
        return Point(x, y)
    }

    private fun cluster1d(values: DoubleArray, weights: DoubleArray, tol: Double, topK: Int): List<List<Int>> {
        if (values.isEmpty()) return emptyList()
        val groups = ArrayList<MutableList<Int>>()
        val centers = ArrayList<Double>()
        val sums = ArrayList<Double>()
        for (idx in values.indices.sortedBy { values[it] }) {
            val v = values[idx]
            if (groups.isNotEmpty() && abs(v - centers.last()) <= tol) {
                val g = groups.last()
                g.add(idx)
                sums[sums.size - 1] += weights[idx]
                var acc = 0.0
                var wAcc = 0.0
                for (k in g) {
                    acc += values[k] * weights[k]
                    wAcc += weights[k]
                }
                centers[centers.size - 1] = if (wAcc > 0) acc / wAcc else v
            } else {
                groups.add(mutableListOf(idx))
                centers.add(v)
                sums.add(weights[idx])
            }
        }
        return groups.indices.sortedByDescending { sums[it] }.take(topK).map { groups[it] }
    }

    // ---- 区域内统计与边缘支持度 ----

    private class Stats(val darkRatio: Double, val brightRatio: Double, val medianLuma: Int)

    private fun sampleInside(quad: Array<Point>, w: Int, h: Int, gray: ByteArray, darkThresh: Int): Stats {
        val xMin = max(0.0, quad.minOf { it.x })
        val xMax = min(w - 1.0, quad.maxOf { it.x })
        val yMin = max(0.0, quad.minOf { it.y })
        val yMax = min(h - 1.0, quad.maxOf { it.y })
        if (xMax <= xMin || yMax <= yMin) return Stats(0.0, 0.0, 255)

        val steps = 32
        val lumas = ArrayList<Int>(steps * steps)
        var inside = 0
        var dark = 0
        var bright = 0
        for (i in 0..steps) {
            val x = xMin + (xMax - xMin) * i / steps
            for (j in 0..steps) {
                val y = yMin + (yMax - yMin) * j / steps
                if (!pointInPoly(quad, x, y)) continue
                val xi = x.roundToInt().coerceIn(0, w - 1)
                val yi = y.roundToInt().coerceIn(0, h - 1)
                val luma = gray[yi * w + xi].toInt() and 0xFF
                inside++
                lumas.add(luma)
                if (luma < darkThresh) dark++
                if (luma > 170) bright++
            }
        }
        if (inside == 0) return Stats(0.0, 0.0, 255)
        lumas.sort()
        return Stats(dark.toDouble() / inside, bright.toDouble() / inside, lumas[lumas.size / 2])
    }

    /** 平面外侧一圈的亮度中位数；采样不足返回 -1。 */
    private fun ringMedian(quad: Array<Point>, w: Int, h: Int, gray: ByteArray): Double {
        val mask = Mat.zeros(h, w, CvType.CV_8UC1)
        MatTools.fillQuad(mask, quad, 255.0)
        val outer = MatTools.dilate(mask, 11)
        val ring = Mat()
        Core.subtract(outer, mask, ring)
        val values = ArrayList<Int>()
        val idx = ByteArray(w * h)
        ring.get(0, 0, idx)
        for (i in idx.indices) {
            if ((idx[i].toInt() and 0xFF) > 0) values.add(gray[i].toInt() and 0xFF)
        }
        if (values.size < 100) return -1.0
        values.sort()
        return values[values.size / 2].toDouble()
    }

    private fun pointInPoly(q: Array<Point>, x: Double, y: Double): Boolean {
        var inside = false
        var j = 3
        for (i in 0 until 4) {
            val xi = q[i].x
            val yi = q[i].y
            val xj = q[j].x
            val yj = q[j].y
            if (((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) inside = !inside
            j = i
        }
        return inside
    }

    private fun gradMagnitude(gray: Mat): Mat {
        val gx = Mat()
        val gy = Mat()
        Imgproc.Sobel(gray, gx, CvType.CV_32F, 1, 0, 3)
        Imgproc.Sobel(gray, gy, CvType.CV_32F, 0, 1, 3)
        val gx2 = Mat()
        val gy2 = Mat()
        Core.multiply(gx, gx, gx2)
        Core.multiply(gy, gy, gy2)
        val mag = Mat()
        Core.add(gx2, gy2, mag)
        Core.sqrt(mag, mag)
        return mag
    }

    private fun edgeSupport(grad: FloatArray, w: Int, h: Int, p0: Point, p1: Point): Double {
        val samples = 60
        val radius = 3
        val acc = ArrayList<Double>(samples)
        for (i in 0..samples) {
            val t = i.toDouble() / samples
            val x = p0.x + (p1.x - p0.x) * t
            val y = p0.y + (p1.y - p0.y) * t
            val xi = x.roundToInt()
            val yi = y.roundToInt()
            if (xi < radius || yi < radius || xi >= w - radius || yi >= h - radius) continue
            var best = 0.0
            for (dy in -radius..radius) {
                val row = (yi + dy) * w
                for (dx in -radius..radius) {
                    val v = grad[row + xi + dx].toDouble()
                    if (v > best) best = v
                }
            }
            acc.add(best)
        }
        if (acc.isEmpty()) return 0.0
        acc.sort()
        return acc[acc.size / 2]
    }

    private fun percentile(gray: ByteArray, p: Double): Int {
        val step = max(1, gray.size / 20000)
        val samples = ArrayList<Int>(gray.size / step + 1)
        var i = 0
        while (i < gray.size) {
            samples.add(gray[i].toInt() and 0xFF)
            i += step
        }
        samples.sort()
        val idx = ((samples.size - 1) * p / 100.0).roundToInt().coerceIn(0, samples.size - 1)
        return samples[idx]
    }
}
