package com.unblock.app.core

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/** OpenCV Mat 与 Android Bitmap 的转换，以及一些反复用到的小工具。 */
object MatTools {

    fun odd(v: Int): Int = if (v % 2 == 1) v else v + 1

    fun ones(k: Int): Mat = Mat.ones(Size(k.toDouble(), k.toDouble()), CvType.CV_8UC1)

    fun fromBitmap(bitmap: Bitmap): Mat {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        return mat
    }

    fun toBitmap(mat: Mat): Bitmap {
        val bmp = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, bmp)
        return bmp
    }

    /** 等比缩小到长边不超过 maxSide；不放大。返回 (图像, 缩放系数)。 */
    fun resizeTo(src: Mat, maxSide: Int): Pair<Mat, Double> {
        val h = src.rows()
        val w = src.cols()
        val longest = maxOf(h, w)
        if (longest <= maxSide) return src.clone() to 1.0
        val scale = maxSide.toDouble() / longest
        val dst = Mat()
        Imgproc.resize(src, dst, Size(w * scale, h * scale), 0.0, 0.0, Imgproc.INTER_AREA)
        return dst to scale
    }

    fun toGray(src: Mat): Mat {
        if (src.channels() == 1) return src.clone()
        val gray = Mat()
        // Bitmap 转出来是 RGBA 四通道，直接按 BGR 转会报错
        val code = if (src.channels() == 4) Imgproc.COLOR_RGBA2GRAY else Imgproc.COLOR_BGR2GRAY
        Imgproc.cvtColor(src, gray, code)
        return gray
    }

    fun emptyMask(rows: Int, cols: Int): Mat = Mat.zeros(rows, cols, CvType.CV_8UC1)

    /** 用 4 个点填充凸四边形。 */
    fun fillQuad(mask: Mat, quad: Array<Point>, value: Double) {
        val pts = MatOfPointHelper.quadToMat(quad)
        Imgproc.fillConvexPoly(mask, pts, Scalar(value))
    }

    fun dilate(src: Mat, k: Int): Mat {
        val dst = Mat()
        Imgproc.dilate(src, dst, ones(odd(k)))
        return dst
    }

    fun erode(src: Mat, k: Int): Mat {
        val dst = Mat()
        Imgproc.erode(src, dst, ones(odd(k)))
        return dst
    }

    fun gaussianBlur(src: Mat, k: Int, sigma: Double = 0.0): Mat {
        val dst = Mat()
        Imgproc.GaussianBlur(src, dst, Size(odd(k).toDouble(), odd(k).toDouble()), sigma)
        return dst
    }
}

/** 四点多边形的小封装，避免各处重复写 MatOfPoint。 */
object MatOfPointHelper {
    fun quadToMat(quad: Array<Point>): org.opencv.core.MatOfPoint {
        val pts = org.opencv.core.MatOfPoint()
        pts.fromArray(*quad)
        return pts
    }
}
