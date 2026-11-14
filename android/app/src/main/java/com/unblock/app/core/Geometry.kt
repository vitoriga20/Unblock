package com.unblock.app.core

import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.SIFT
import org.opencv.imgproc.Imgproc

/** 对齐参数，全部来自 assets/config.json 的 params.align，不在代码里写死。 */
data class AlignConfig(
    val siftNfeatures: Int = 8000,
    val ratioTest: Double = 0.75,
    val ransacThresh: Double = 6.0,
    val minInliers: Int = 30,
    val featureMaxSide: Int = 1024,
    val residualRefine: Boolean = true,
    val residualGrid: Int = 6,
    val residualMaxPx: Double = 24.0,
    val residualSmooth: Int = 3,
    val residualMaxSide: Int = 1280,
    val residualMinMatches: Int = 40
)

data class ResidualInfo(val applied: Boolean, val matches: Int, val maxShift: Double)

/** 平面单应、透视变换与正视校正。 */
object Geometry {

    /** 统一四个角为：左上、右上、右下、左下。 */
    fun orderQuad(pts: List<Point>): Array<Point> {
        require(pts.size == 4) { "需要 4 个角点，实际 ${pts.size}" }
        val sorted = Array(4) { Point() }
        val sums = pts.map { it.x + it.y }
        val diffs = pts.map { it.y - it.x }
        sorted[0] = pts[sums.indexOf(sums.min())]
        sorted[2] = pts[sums.indexOf(sums.max())]
        sorted[1] = pts[diffs.indexOf(diffs.min())]
        sorted[3] = pts[diffs.indexOf(diffs.max())]
        return sorted
    }

    private fun toMatOfPoint2f(quad: Array<Point>): MatOfPoint2f {
        val m = MatOfPoint2f()
        m.fromArray(*quad)
        return m
    }

    /** aux 的四角映射到 base 的四角。 */
    fun homographyFromQuad(auxQuad: Array<Point>, baseQuad: Array<Point>): Mat {
        return Imgproc.getPerspectiveTransform(toMatOfPoint2f(orderQuad(auxQuad.toList())),
            toMatOfPoint2f(orderQuad(baseQuad.toList())))
    }

    /**
     * 用 SIFT 特征做 aux -> base 的单应估计。
     * 四角只是屏幕的近似外接框（可能落在画面外），真正决定对齐的是特征匹配；
     * 特征不足或结果不合理时退回四角单应。
     */
    fun homographyFromFeatures(
        base: Mat,
        aux: Mat,
        baseQuad: Array<Point>,
        auxQuad: Array<Point>,
        cfg: AlignConfig = AlignConfig()
    ): Mat {
        val quadH = homographyFromQuad(auxQuad, baseQuad)
        val diag = Math.hypot(base.cols().toDouble(), base.rows().toDouble())
        try {
            val baseGray = MatTools.toGray(base)
            val auxGray = MatTools.toGray(aux)
            // 特征检测在缩小图上做，速度可控，再把单应还原到原分辨率
            val (bs, s) = MatTools.resizeTo(baseGray, cfg.featureMaxSide)
            val (as_, _) = MatTools.resizeTo(auxGray, cfg.featureMaxSide)

            val sift = SIFT.create(cfg.siftNfeatures)
            val kb = MatOfKeyPoint()
            val db = Mat()
            val ka = MatOfKeyPoint()
            val da = Mat()
            sift.detectAndCompute(bs, Mat(), kb, db)
            sift.detectAndCompute(as_, Mat(), ka, da)
            if (db.empty() || da.empty()) return quadH

            val matcher = BFMatcher(Core.NORM_L2)
            val knn = ArrayList<MatOfDMatch>()
            matcher.knnMatch(da, db, knn, 2)

            val kaArr = ka.toArray()
            val kbArr = kb.toArray()
            val src = ArrayList<Point>()
            val dst = ArrayList<Point>()
            for (m in knn) {
                val arr = m.toArray()
                if (arr.size < 2) continue
                if (arr[0].distance < cfg.ratioTest * arr[1].distance) {
                    src.add(kaArr[arr[0].queryIdx].pt)
                    dst.add(kbArr[arr[0].trainIdx].pt)
                }
            }
            if (src.size < 12) return quadH

            val inlierMask = Mat()
            val hSmall = Calib3d.findHomography(
                MatOfPoint2f(*src.toTypedArray()),
                MatOfPoint2f(*dst.toTypedArray()),
                Calib3d.RANSAC, cfg.ransacThresh, inlierMask
            )
            if (hSmall.empty() || hSmall.cols() != 3) return quadH
            val inliers = Core.countNonZero(inlierMask)
            if (inliers < cfg.minInliers) return quadH

            // 小图单应 -> 原图单应：H_full = S^-1 * H_small * S
            val toSmall = Mat.eye(3, 3, CvType.CV_64F)
            toSmall.put(0, 0, s)
            toSmall.put(1, 1, s)
            val toFull = Mat()
            Core.invert(toSmall, toFull)
            val tmp = Mat()
            Core.gemm(hSmall, toSmall, 1.0, Mat(), 0.0, tmp)
            val hFull = Mat()
            Core.gemm(toFull, tmp, 1.0, Mat(), 0.0, hFull)
            if (Math.abs(hFull.get(2, 2)[0]) < 1e-9) return quadH
            Core.divide(hFull, Scalar(hFull.get(2, 2)[0]), hFull)

            // 合理性校验：把 aux 四角映射过去，落到 base 四角附近才接受
            val mapped = MatOfPoint2f()
            Core.perspectiveTransform(toMatOfPoint2f(orderQuad(auxQuad.toList())), mapped, hFull)
            val bq = orderQuad(baseQuad.toList())
            val mq = orderQuad(mapped.toArray().toList())
            var worst = 0.0
            for (i in 0 until 4) worst = maxOf(worst, Math.hypot(mq[i].x - bq[i].x, mq[i].y - bq[i].y))
            if (worst > 0.25 * diag) return quadH

            return hFull
        } catch (t: Throwable) {
            return quadH
        }
    }

    /** 把 aux 变换到 base 坐标系，同时返回有效区域掩码。 */
    fun warpToBase(aux: Mat, h: Mat, baseSize: Size): Pair<Mat, Mat> {
        val warped = Mat()
        Imgproc.warpPerspective(aux, warped, h, baseSize, Imgproc.INTER_LINEAR,
            Core.BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0))
        val valid = Mat.zeros(aux.rows(), aux.cols(), CvType.CV_8UC1)
        valid.setTo(Scalar(255.0))
        val validWarped = Mat()
        Imgproc.warpPerspective(valid, validWarped, h, baseSize, Imgproc.INTER_NEAREST,
            Core.BORDER_CONSTANT, Scalar(0.0))
        val eroded = Mat()
        Imgproc.erode(validWarped, eroded, MatTools.ones(3))
        return warped to eroded
    }

    /**
     * 全局单应之后仍有局部残余位移（镜头畸变、幕布不绝对平）。
     * 用网格化的特征残差估计一个平滑位移场，再 remap 一次补偿。
     */
    fun refineResidual(base: Mat, warped: Mat, valid: Mat, cfg: AlignConfig): Triple<Mat, Mat, ResidualInfo> {
        val none = ResidualInfo(false, 0, 0.0)
        try {
            val (bSmall, _) = MatTools.resizeTo(MatTools.toGray(base), cfg.residualMaxSide)
            val (aSmall, s) = MatTools.resizeTo(MatTools.toGray(warped), cfg.residualMaxSide)
            if (s <= 0.0) return Triple(warped, valid, none)

            val sift = SIFT.create(cfg.siftNfeatures)
            val kb = MatOfKeyPoint()
            val db = Mat()
            val ka = MatOfKeyPoint()
            val da = Mat()
            sift.detectAndCompute(bSmall, Mat(), kb, db)
            sift.detectAndCompute(aSmall, Mat(), ka, da)
            if (db.empty() || da.empty()) return Triple(warped, valid, none)

            val knn = ArrayList<MatOfDMatch>()
            BFMatcher(Core.NORM_L2).knnMatch(da, db, knn, 2)
            val kbArr = kb.toArray()
            val kaArr = ka.toArray()

            // x, y, dx, dy，坐标与位移都还原到全分辨率
            val pts = ArrayList<DoubleArray>()
            var maxShift = 0.0
            for (m in knn) {
                val arr = m.toArray()
                if (arr.size < 2) continue
                if (arr[0].distance >= cfg.ratioTest * arr[1].distance) continue
                val p = kaArr[arr[0].queryIdx].pt
                val q = kbArr[arr[0].trainIdx].pt
                val dx = (q.x - p.x) / s
                val dy = (q.y - p.y) / s
                val norm = Math.hypot(dx, dy)
                if (norm > cfg.residualMaxPx) continue
                maxShift = maxOf(maxShift, norm)
                pts.add(doubleArrayOf(p.x / s, p.y / s, dx, dy))
            }
            if (pts.size < cfg.residualMinMatches) return Triple(warped, valid, none)

            val rows = base.rows()
            val cols = base.cols()
            val g = maxOf(1, cfg.residualGrid)
            val sumX = Array(g) { DoubleArray(g) }
            val sumY = Array(g) { DoubleArray(g) }
            val cnt = Array(g) { IntArray(g) }
            for (p in pts) {
                val cx = (p[0] / cols * g).toInt().coerceIn(0, g - 1)
                val cy = (p[1] / rows * g).toInt().coerceIn(0, g - 1)
                sumX[cy][cx] += p[2]
                sumY[cy][cx] += p[3]
                cnt[cy][cx]++
            }

            val gx = Array(g) { DoubleArray(g) }
            val gy = Array(g) { DoubleArray(g) }
            for (y in 0 until g) for (x in 0 until g) {
                if (cnt[y][x] > 0) {
                    gx[y][x] = sumX[y][x] / cnt[y][x]
                    gy[y][x] = sumY[y][x] / cnt[y][x]
                } else {
                    // 空格用邻域均值补齐，避免插值出空洞
                    var ax = 0.0
                    var ay = 0.0
                    var n = 0
                    for (dy in -1..1) for (dx in -1..1) {
                        val yy = y + dy
                        val xx = x + dx
                        if (yy !in 0 until g || xx !in 0 until g) continue
                        if (cnt[yy][xx] == 0) continue
                        ax += gx[yy][xx]
                        ay += gy[yy][xx]
                        n++
                    }
                    if (n > 0) {
                        gx[y][x] = ax / n
                        gy[y][x] = ay / n
                    }
                }
            }

            val cell = Mat(g, g, CvType.CV_32FC2)
            for (y in 0 until g) for (x in 0 until g) {
                cell.put(y, x, gx[y][x], gy[y][x])
            }
            val smooth = MatTools.odd(cfg.residualSmooth)
            val smoothed = Mat()
            if (smooth > 1) Imgproc.blur(cell, smoothed, Size(smooth.toDouble(), smooth.toDouble()))
            else cell.copyTo(smoothed)

            val field = Mat()
            Imgproc.resize(smoothed, field, Size(cols.toDouble(), rows.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            val planes = ArrayList<Mat>()
            Core.split(field, planes)

            val xxRow = Mat(1, cols, CvType.CV_32FC1)
            for (x in 0 until cols) xxRow.put(0, x, x.toDouble())
            val xx = Mat()
            Core.repeat(xxRow, rows, 1, xx)

            val yyCol = Mat(rows, 1, CvType.CV_32FC1)
            for (y in 0 until rows) yyCol.put(y, 0, y.toDouble())
            val yy = Mat()
            Core.repeat(yyCol, 1, cols, yy)

            val mapX = Mat()
            Core.add(xx, planes[0], mapX)
            val mapY = Mat()
            Core.add(yy, planes[1], mapY)

            val refined = Mat()
            Imgproc.remap(warped, refined, mapX, mapY, Imgproc.INTER_LINEAR,
                Core.BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0))
            val validOut = Mat()
            Imgproc.remap(valid, validOut, mapX, mapY, Imgproc.INTER_NEAREST,
                Core.BORDER_CONSTANT, Scalar(0.0))
            return Triple(refined, validOut, ResidualInfo(true, pts.size, maxShift))
        } catch (t: Throwable) {
            return Triple(warped, valid, none)
        }
    }

    /** 按四角把屏幕拉成正矩形。 */
    fun rectify(img: Mat, quad: Array<Point>, longSide: Int, targetAspect: Double?): Mat {
        val q = orderQuad(quad.toList())
        val top = dist(q[0], q[1])
        val bottom = dist(q[3], q[2])
        val left = dist(q[0], q[3])
        val right = dist(q[1], q[2])
        val srcW = (top + bottom) / 2.0
        val srcH = (left + right) / 2.0
        val aspect = targetAspect ?: (srcW / srcH)

        val outW: Int
        val outH: Int
        if (aspect >= 1.0) {
            outW = longSide
            outH = (longSide / aspect).toInt()
        } else {
            outH = longSide
            outW = (longSide * aspect).toInt()
        }

        val dst = MatOfPoint2f(
            Point(0.0, 0.0),
            Point((outW - 1).toDouble(), 0.0),
            Point((outW - 1).toDouble(), (outH - 1).toDouble()),
            Point(0.0, (outH - 1).toDouble())
        )
        val m = Imgproc.getPerspectiveTransform(toMatOfPoint2f(q), dst)
        val out = Mat()
        Imgproc.warpPerspective(img, out, m, Size(outW.toDouble(), outH.toDouble()),
            Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0))
        return out
    }

    private fun dist(a: Point, b: Point): Double = Math.hypot(a.x - b.x, a.y - b.y)

    /** 内缩矩形，作为自动检测失败时的兜底初值。 */
    fun insetQuad(w: Int, h: Int, ratio: Double = 0.06): Array<Point> {
        val ix = w * ratio
        val iy = h * ratio
        return arrayOf(
            Point(ix, iy), Point(w - ix, iy),
            Point(w - ix, h - iy), Point(ix, h - iy)
        )
    }
}
