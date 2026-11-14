package com.unblock.app.core

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

data class ComposeConfig(
    val edgeSoftPx: Int = 5,
    val minAlpha: Double = 0.5,
    val gainClipMin: Double = 0.75,
    val gainClipMax: Double = 1.35,
    val gainSmooth: Int = 45,
    val gainExcludePx: Int = 12,
    val seamBandPx: Int = 2,
    val colorMatchEnable: Boolean = true,
    val colorMatchRingPx: Int = 81,
    val colorMatchFlatGradMax: Double = 12.0,
    val colorMatchMinRingPx: Int = 2000,
    val colorMatchStdClipMin: Double = 0.6,
    val colorMatchStdClipMax: Double = 1.6,
    val colorMatchAlignL: Boolean = false
)

object Compose {

    private fun grayTo3(gray: Mat): Mat {
        val out = Mat()
        Imgproc.cvtColor(gray, out, Imgproc.COLOR_GRAY2BGR)
        return out
    }

    private fun scalarMat(size: Size, type: Int, v: Double): Mat =
        Mat(size, type, Scalar(v, v, v))

    /** 在两图都可见且未遮挡的区域估计空间平滑的逐通道增益。 */
    fun estimateGainMap(base: Mat, aux: Mat, region: Mat, cfg: ComposeConfig): Mat {
        val full = base.size()
        // 增益图本身是低频的，在 1/4 分辨率上估计再上采样，能把模糊与乘除的像素量降一个数量级
        val ds = 4
        val small = Size((base.cols() / ds).toDouble(), (base.rows() / ds).toDouble())
        val baseS = Mat()
        Imgproc.resize(base, baseS, small, 0.0, 0.0, Imgproc.INTER_AREA)
        val auxS = Mat()
        Imgproc.resize(aux, auxS, small, 0.0, 0.0, Imgproc.INTER_AREA)
        val regionS = Mat()
        Imgproc.resize(region, regionS, small, 0.0, 0.0, Imgproc.INTER_AREA)

        val baseF = Mat()
        baseS.convertTo(baseF, CvType.CV_32FC3, 1.0, 1.0)
        val auxF = Mat()
        auxS.convertTo(auxF, CvType.CV_32FC3, 1.0, 1.0)

        // 分母下限取 1：aux 在变换后的画面外是纯黑，直接相除会得到 inf/NaN 并污染整张增益图
        val auxSafe = Mat()
        Core.max(auxF, scalarMat(small, CvType.CV_32FC3, 1.0), auxSafe)
        val ratio = Mat()
        Core.divide(baseF, auxSafe, ratio)

        val weight = Mat()
        regionS.convertTo(weight, CvType.CV_32FC1, 1.0 / 255.0)
        val weight3 = grayTo3(weight)

        val weightedRatio = Mat()
        Core.multiply(ratio, weight3, weightedRatio)

        // 平滑核按降采样比例同步缩小，等效于全分辨率上的 gainSmooth
        val k = MatTools.odd(maxOf(3, MatTools.odd(cfg.gainSmooth) / ds))
        val blurSize = Size(k.toDouble(), k.toDouble())
        val num = Mat()
        Imgproc.blur(weightedRatio, num, blurSize)
        val den = Mat()
        Imgproc.blur(weight, den, blurSize)

        // 分母下限，避免平滑后权重趋零的区域再次除零
        val denSafe = Mat()
        Core.max(den, scalarMat(weight.size(), CvType.CV_32FC1, 1e-3), denSafe)
        val den3 = grayTo3(denSafe)

        val gm = Mat()
        Core.divide(num, den3, gm)

        // 没有样本的区域直接给 1，避免被钳位到极值后凭空提亮或压暗
        val noData = Mat()
        Imgproc.threshold(den, noData, 1e-3, 255.0, Imgproc.THRESH_BINARY_INV)
        val noData8 = Mat()
        noData.convertTo(noData8, CvType.CV_8UC1)
        gm.setTo(Scalar(1.0, 1.0, 1.0), noData8)

        val lo = scalarMat(small, gm.type(), cfg.gainClipMin)
        val hi = scalarMat(small, gm.type(), cfg.gainClipMax)
        val clamped = Mat()
        Core.max(gm, lo, clamped)
        Core.min(clamped, hi, clamped)

        // 上采样回全分辨率；插值不会超出已钳位的取值范围，因此不必再钳位一次
        val out = Mat()
        Imgproc.resize(clamped, out, full, 0.0, 0.0, Imgproc.INTER_LINEAR)
        return out
    }

    /**
     * 把补丁的 LAB 一阶统计对齐到“洞口环带”。与 PC 端 match_color_lab 同一套口径。
     *
     * 两图的差异不是乘性增益能表达的：实测 base/aux 的通道比随亮度从 0.44 单调爬到 0.98，
     * 属于两条色调曲线，浅色卡片会被过度压红而露成蓝块。这里只对齐色度 a/b（默认不动 L），
     * 用 256 级 LUT 实现，避免在 4096x3072 上生成整幅浮点图把端上内存打爆。
     */
    fun matchColorLab(aux: Mat, base: Mat, mask: Mat, auxValid: Mat, screen: Mat,
                      cfg: ComposeConfig): Mat {
        if (!cfg.colorMatchEnable || Core.countNonZero(mask) == 0) return aux

        val ring = Mat.zeros(mask.size(), CvType.CV_8UC1)
        val dilated = MatTools.dilate(mask, cfg.colorMatchRingPx)
        Core.subtract(dilated, mask, ring)
        Core.bitwise_and(ring, auxValid, ring)
        Core.bitwise_and(ring, screen, ring)

        // 只用平坦区估计统计量，避开文字/图标边缘把均值方差拉偏
        val gray = MatTools.toGray(base)
        val gx = Mat()
        val gy = Mat()
        val mag = Mat()
        Imgproc.Sobel(gray, gx, CvType.CV_32F, 1, 0, 3)
        Imgproc.Sobel(gray, gy, CvType.CV_32F, 0, 1, 3)
        Core.magnitude(gx, gy, mag)
        val flat = Mat()
        Imgproc.threshold(mag, flat, cfg.colorMatchFlatGradMax, 255.0, Imgproc.THRESH_BINARY_INV)
        flat.convertTo(flat, CvType.CV_8UC1)
        Core.bitwise_and(ring, flat, ring)

        val ringPx = Core.countNonZero(ring)
        if (ringPx < cfg.colorMatchMinRingPx) {
            listOf(dilated, ring, gray, gx, gy, mag, flat).forEach { it.release() }
            return aux
        }

        val labA = Mat()
        val labB = Mat()
        Imgproc.cvtColor(aux, labA, Imgproc.COLOR_BGR2Lab)
        Imgproc.cvtColor(base, labB, Imgproc.COLOR_BGR2Lab)
        val chA = ArrayList<Mat>()
        val chB = ArrayList<Mat>()
        Core.split(labA, chA)
        Core.split(labB, chB)

        val first = if (cfg.colorMatchAlignL) 0 else 1
        for (c in first..2) {
            val meanSrc = MatOfDouble()
            val stdSrc = MatOfDouble()
            val meanDst = MatOfDouble()
            val stdDst = MatOfDouble()
            Core.meanStdDev(chA[c], meanSrc, stdSrc, mask)
            Core.meanStdDev(chB[c], meanDst, stdDst, ring)
            val ms = meanSrc.toArray()[0]
            val md = meanDst.toArray()[0]
            val scale = ((stdDst.toArray()[0] + 1e-6) / (stdSrc.toArray()[0] + 1e-6))
                .coerceIn(cfg.colorMatchStdClipMin, cfg.colorMatchStdClipMax)

            val lut = Mat(1, 256, CvType.CV_8UC1)
            for (i in 0..255) {
                val v = ((i - ms) * scale + md).toInt().coerceIn(0, 255)
                lut.put(0, i, v.toDouble())
            }
            val out = Mat()
            Core.LUT(chA[c], lut, out)
            chA[c].release()
            chA[c] = out
            lut.release()
        }

        val merged = Mat()
        Core.merge(chA, merged)
        val aligned = Mat()
        Imgproc.cvtColor(merged, aligned, Imgproc.COLOR_Lab2BGR)

        val result = aux.clone()
        aligned.copyTo(result, mask)

        listOf(dilated, ring, gray, gx, gy, mag, flat, labA, labB, merged, aligned).forEach { it.release() }
        chA.forEach { it.release() }
        chB.forEach { it.release() }
        return result
    }

    fun applyGain(aux: Mat, gainMap: Mat): Mat {
        val auxF = Mat()
        aux.convertTo(auxF, CvType.CV_32FC3)
        val out = Mat()
        Core.multiply(auxF, gainMap, out)
        val result = Mat()
        out.convertTo(result, CvType.CV_8UC3)
        return result
    }

    /**
     * mask 内部采用 aux 的真实像素，边界按到边的距离线性羽化。
     * mask 已经比遮挡物外扩一圈，羽化带落在干净背景上，不会把人混回来，
     * 但能把“填充区边界”这条硬线抹掉。
     */
    fun blend(base: Mat, auxCorrected: Mat, mask: Mat, auxValid: Mat, cfg: ComposeConfig): Mat {
        // 距离变换要求 0/1 的 8U 输入
        val m8 = Mat()
        Imgproc.threshold(mask, m8, 0.0, 1.0, Imgproc.THRESH_BINARY)

        val dist = Mat()
        Imgproc.distanceTransform(m8, dist, Imgproc.DIST_L2, 5)

        val ramp = Mat()
        Core.divide(dist, Scalar(cfg.edgeSoftPx.toDouble()), ramp)
        Core.min(ramp, scalarMat(ramp.size(), ramp.type(), 1.0), ramp)

        // alpha = minAlpha + (1 - minAlpha) * ramp，再限制在 mask 内
        val alpha = Mat()
        ramp.convertTo(alpha, CvType.CV_32FC1, 1.0 - cfg.minAlpha, cfg.minAlpha)

        val maskF = Mat()
        mask.convertTo(maskF, CvType.CV_32FC1, 1.0 / 255.0)
        Core.multiply(alpha, maskF, alpha)

        val validF = Mat()
        auxValid.convertTo(validF, CvType.CV_32FC1, 1.0 / 255.0)
        Core.multiply(alpha, validF, alpha)

        val alpha3 = grayTo3(alpha)
        val ones = Mat.ones(mask.size(), CvType.CV_32FC1)
        val inv = Mat()
        Core.subtract(ones, alpha, inv)
        val inv3 = grayTo3(inv)

        val baseF = Mat()
        base.convertTo(baseF, CvType.CV_32FC3)
        val auxF = Mat()
        auxCorrected.convertTo(auxF, CvType.CV_32FC3)

        val a = Mat()
        Core.multiply(baseF, inv3, a)
        val b = Mat()
        Core.multiply(auxF, alpha3, b)
        val sum = Mat()
        Core.add(a, b, sum)

        val out = Mat()
        sum.convertTo(out, CvType.CV_8UC3)
        return out
    }

    /** mask 边界内外各 bandPx 的灰度均值差，量化“看得见的那条线”。 */
    fun seamStep(img: Mat, mask: Mat, bandPx: Int): Double {
        val k = MatTools.odd(bandPx * 2 + 1)
        val inner = MatTools.erode(mask, k)
        val outerAll = MatTools.dilate(mask, k)
        val outer = Mat()
        Core.bitwise_and(outerAll, negate(mask), outer)
        if (Core.countNonZero(inner) == 0 || Core.countNonZero(outer) == 0) return 0.0
        val gray = MatTools.toGray(img)
        val mi = Core.mean(gray, inner).`val`[0]
        val mo = Core.mean(gray, outer).`val`[0]
        return Math.abs(mi - mo)
    }

    private fun negate(m: Mat): Mat {
        val out = Mat()
        Core.bitwise_not(m, out)
        return out
    }

    /** 遮挡区边界梯度与全图平均梯度之比，越小说明接缝越不明显。 */
    fun seamRatio(img: Mat, mask: Mat): Double {
        if (Core.countNonZero(mask) == 0) return 0.0
        val grad = Mat()
        Imgproc.morphologyEx(mask, grad, Imgproc.MORPH_GRADIENT, MatTools.ones(9))
        val gray = MatTools.toGray(img)
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

        val onEdge = Core.mean(mag, grad).`val`[0]
        val overall = Core.mean(mag).`val`[0]
        return onEdge / (overall + 1e-6)
    }
}
