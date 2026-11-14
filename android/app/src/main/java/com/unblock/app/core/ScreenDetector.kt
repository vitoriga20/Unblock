package com.unblock.app.core

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** 屏幕四角自动粗定位的参数；默认值来自 PC 端 config.json 的 screen 段。 */
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
    val topK: Int = 8
)

data class ScreenDetectResult(
    val quad: Array<Point>?,
    val score: Double,
    val message: String
)

/**
 * 屏幕四边形粗定位：LSD 线段 -> (rho, theta) 直线 -> 按方向聚类 -> 交点求四边形 -> 约束筛选。
 * 结果只作为交互初值，允许有偏差，由用户拖动微调。
 */
object ScreenDetector {

    private data class Seg(val rho: Double, val theta: Double, val length: Double)

    fun detect(img: Mat, cfg: ScreenConfig = ScreenConfig()): ScreenDetectResult {
        val (small, scale) = MatTools.resizeTo(img, cfg.detectMaxSide)
        val w = small.cols()
        val h = small.rows()
        val gray = MatTools.toGray(small)
        val eq = Mat()
        Imgproc.createCLAHE(2.5, Size(8.0, 8.0)).apply(gray, eq)

        val segs = segments(eq, w, h)
        if (segs.size < 4) return ScreenDetectResult(null, 0.0, "线段太少(${segs.size})")

        val tol = Math.toRadians(cfg.quadAngleToleranceDeg)
        val vert = ArrayList<Int>()
        val horz = ArrayList<Int>()
        for ((i, s) in segs.withIndex()) {
            val t = s.theta
            if (t <= tol || t >= Math.PI - tol) vert.add(i)
            else if (abs(t - Math.PI / 2.0) <= tol) horz.add(i)
        }
        if (vert.size < 2 || horz.size < 2) return ScreenDetectResult(null, 0.0, "可用直线不足")

        // 竖线取中心行的 x 值，横线取中心列的 y 值，再做一维聚类
        val vx = DoubleArray(vert.size) { verticalX(segs[vert[it]], h / 2.0) }
        val hy = DoubleArray(horz.size) { horizontalY(segs[horz[it]], w / 2.0) }
        val vGroups = cluster1d(vx, DoubleArray(vert.size) { segs[vert[it]].length }, 0.020 * w, cfg.topK)
        val hGroups = cluster1d(hy, DoubleArray(horz.size) { segs[horz[it]].length }, 0.020 * h, cfg.topK)
        if (vGroups.size < 2 || hGroups.size < 2) return ScreenDetectResult(null, 0.0, "可用直线不足")

        // 越界（完全在画面外的估计）也允许，但排序后只配对左右/上下
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
                        if (xr - xl < cfg.minWidthRatio * w) continue
                        if (yb - yt < cfg.minHeightRatio * h) continue
                        if (cfg.requireCenterInside && !pointInPoly(quad, cx, cy)) continue
                        val areaRatio = ((xr - xl) * (yb - yt)) / (w.toDouble() * h)
                        if (areaRatio < cfg.minAreaRatio || areaRatio > cfg.maxAreaRatio) continue
                        val aspect = (xr - xl) / max(1e-6, yb - yt)
                        if (aspect < cfg.aspectMin || aspect > cfg.aspectMax) continue

                        val stats = sampleInside(quad, w, h, grayBytes, darkThresh)
                        if (stats.darkRatio < cfg.minDarkRatio) continue
                        if (stats.medianLuma > cfg.maxMedianLuma) continue
                        if (stats.brightRatio < cfg.minBrightRatio || stats.brightRatio > cfg.maxBrightRatio) continue

                        val support = (edgeSupport(gradData, w, h, quad[0], quad[1]) +
                            edgeSupport(gradData, w, h, quad[1], quad[2]) +
                            edgeSupport(gradData, w, h, quad[2], quad[3]) +
                            edgeSupport(gradData, w, h, quad[3], quad[0])) / 4.0
                        val score = areaRatio * (0.15 + support / 255.0)
                        if (score > bestScore) {
                            bestScore = score
                            bestQuad = quad
                        }
                    }
                }
            }
        }

        if (bestQuad == null) return ScreenDetectResult(null, 0.0, "未找到满足约束的屏幕四边形")
        val full = Geometry.orderQuad(bestQuad.map { Point(it.x / scale, it.y / scale) })
        return ScreenDetectResult(full, bestScore, "ok")
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

    /** 一维聚类，按支持度（线段总长）取前 k 组，返回每组的代表索引。 */
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

    /** 在四边形内按网格采样，统计暗像素占比、亮像素占比与亮度中位数。 */
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

    /** 沿边采样，取梯度幅值的中位数，用来评价这条边是否真的是屏幕边框。 */
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

    /** 取样估计灰度分位数，避免对整幅图排序。 */
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
