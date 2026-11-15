package com.unblock.app.core

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

data class QualityConfig(
    val minSharpness: Double = 0.0,
    val maxOcclusionRatio: Double = 0.55,
    val minCoverage: Double = 0.45,
    val maxBlindRatio: Double = 0.45,
    val maxSeamStep: Double = 45.0,
    val maxAlignResidual: Double = 18.0,
    val minInlierCoverage: Double = 0.2,
    val maxContentChange: Double = 0.45,
    val retryAltBase: Boolean = true
)

data class BaseSelectConfig(
    val enable: Boolean = true,
    val occlusionWeight: Double = 1.4,
    val sharpnessWeight: Double = 0.8,
    val perspectiveWeight: Double = 0.5,
    val exposureWeight: Double = 0.3
)

data class BaseMetric(
    val index: Int,
    val name: String,
    val occlusion: Double,
    val sharpness: Double,
    val frontal: Double,
    val clipped: Double,
    val planeScore: Double,
    val kind: String
)

data class QcResult(val pass: Boolean, val issues: List<String>)

object QualityGate {

    fun sharpness(img: Mat, quad: Array<Point>? = null, maxSide: Int = 1024): Double {
        val (small, scale) = MatTools.resizeTo(img, maxSide)
        val gray = MatTools.toGray(small)
        val mask = Mat.zeros(gray.rows(), gray.cols(), CvType.CV_8UC1)
        if (quad != null) {
            MatTools.fillQuad(mask, quad.map { Point(it.x * scale, it.y * scale) }.toTypedArray(), 255.0)
        }
        val lap = Mat()
        Imgproc.Laplacian(gray, lap, CvType.CV_64F)
        val mean = org.opencv.core.MatOfDouble()
        val std = org.opencv.core.MatOfDouble()
        if (quad != null && Core.countNonZero(mask) > 100) {
            Core.meanStdDev(lap, mean, std, mask)
        } else {
            Core.meanStdDev(lap, mean, std)
        }
        val v = std.toArray()[0]
        return v * v
    }

    fun clippedRatio(img: Mat): Double {
        val (small, _) = MatTools.resizeTo(img, 640)
        val gray = MatTools.toGray(small)
        val n = gray.rows() * gray.cols()
        val bytes = ByteArray(n)
        gray.get(0, 0, bytes)
        var clipped = 0
        for (b in bytes) {
            val v = b.toInt() and 0xFF
            if (v <= 3 || v >= 252) clipped++
        }
        return clipped.toDouble() / n
    }

    /** 四边形的正对程度：面积 / 外接框面积，越接近 1 越正。 */
    fun frontalScore(quad: Array<Point>): Double {
        var sum = 0.0
        for (i in 0 until 4) {
            val a = quad[i]
            val b = quad[(i + 1) % 4]
            sum += a.x * b.y - b.x * a.y
        }
        val area = Math.abs(sum) / 2.0
        val x0 = quad.minOf { it.x }
        val x1 = quad.maxOf { it.x }
        val y0 = quad.minOf { it.y }
        val y1 = quad.maxOf { it.y }
        val bbox = maxOf(1e-6, (x1 - x0) * (y1 - y0))
        return (area / bbox).coerceIn(0.0, 1.0)
    }

    /** 整图前景占比：不依赖平面检测结果，用于选基时稳定比较两张图。 */
    fun personRatio(person: Mat?, rows: Int, cols: Int): Double {
        if (person == null || person.empty()) return 0.0
        val total = maxOf(1, rows * cols)
        return Core.countNonZero(person).toDouble() / total
    }

    private fun normalize(values: List<Double>): List<Double> {
        val lo = values.minOrNull() ?: 0.0
        val hi = values.maxOrNull() ?: 0.0
        if (hi - lo < 1e-9) return values.map { 1.0 }
        return values.map { (it - lo) / (hi - lo) }
    }

    fun chooseBase(metrics: List<BaseMetric>, cfg: BaseSelectConfig): Int {
        if (metrics.isEmpty()) return 0
        val occ = normalize(metrics.map { it.occlusion })
        val sharp = normalize(metrics.map { it.sharpness })
        val persp = normalize(metrics.map { it.frontal })
        val expo = normalize(metrics.map { it.clipped })

        data class Ranked(val score: Double, val sharpness: Double, val occlusion: Double,
                          val name: String, val index: Int)

        val ranked = metrics.indices.map { i ->
            Ranked(
                round6(-cfg.occlusionWeight * occ[i] + cfg.sharpnessWeight * sharp[i] +
                    cfg.perspectiveWeight * persp[i] - cfg.exposureWeight * expo[i]),
                metrics[i].sharpness,
                metrics[i].occlusion,
                metrics[i].name,
                i
            )
        }.sortedWith(compareByDescending<Ranked> { it.score }
            .thenByDescending { it.sharpness }
            .thenBy { it.occlusion }
            .thenBy { it.name })
        return ranked.first().index
    }

    private fun round6(v: Double): Double = Math.round(v * 1e6) / 1e6

    fun postcheck(coverage: Double, blindRatio: Double, seamStep: Double,
                  alignMode: String, alignCoverage: Double, alignResidual: Double,
                  contentChange: Double, cfg: QualityConfig): QcResult {
        val issues = ArrayList<String>()
        if (coverage < cfg.minCoverage) issues.add("coverage_low")
        if (blindRatio > cfg.maxBlindRatio) issues.add("blind_high")
        if (seamStep > cfg.maxSeamStep) issues.add("seam_high")
        if (alignResidual > cfg.maxAlignResidual) issues.add("align_residual_high")
        if (alignMode == "sift" && alignCoverage < cfg.minInlierCoverage) issues.add("align_coverage_low")
        if (contentChange > cfg.maxContentChange) issues.add("content_changed")
        if (alignMode == "failed") issues.add("align_failed")
        return QcResult(issues.isEmpty(), issues)
    }

    fun qcScore(coverage: Double, blindRatio: Double, seamStep: Double): Double =
        coverage - 1.5 * blindRatio - Math.min(seamStep, 100.0) / 25.0
}
