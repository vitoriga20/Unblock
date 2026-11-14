package com.unblock.app.ui

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.hypot
import kotlin.math.min

/**
 * 整图 + 四角可拖拽的编辑视图。
 * 四角坐标始终保存在“图片像素”坐标系里，绘制时才乘显示矩阵，避免缩放后坐标漂移。
 */
class QuadEditorView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var bitmap: Bitmap? = null
    /** x0, y0, x1, y1, x2, y2, x3, y3，顺序为左上、右上、右下、左下。 */
    private val quad = FloatArray(8)
    private val matrix = Matrix()
    private val inverse = Matrix()
    private val tmp = FloatArray(8)

    private val handleR = dip(12f)
    private val touchR = dip(34f)
    private var lastX = 0f
    private var lastY = 0f
    private var active = -1

    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FFC53D")
        style = Paint.Style.STROKE
        strokeWidth = dip(2f)
    }
    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#33FFC53D")
        style = Paint.Style.FILL
    }
    private val handlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FFC53D")
        style = Paint.Style.FILL
    }
    private val handleStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#1A1A1A")
        style = Paint.Style.STROKE
        strokeWidth = dip(2f)
    }
    private val path = Path()

    var onQuadChanged: ((FloatArray) -> Unit)? = null

    private fun dip(v: Float): Float = v * resources.displayMetrics.density

    fun setImage(bmp: Bitmap) {
        bitmap = bmp
        resetMatrix()
        invalidate()
    }

    fun resetQuadToInset(insetRatio: Float = 0.06f) {
        val bmp = bitmap ?: return
        val iw = bmp.width.toFloat()
        val ih = bmp.height.toFloat()
        val ix = iw * insetRatio
        val iy = ih * insetRatio
        setQuad(
            floatArrayOf(
                ix, iy, iw - ix, iy,
                iw - ix, ih - iy, ix, ih - iy
            )
        )
    }

    fun setQuad(values: FloatArray) {
        if (values.size < 8) return
        for (i in 0 until 8) quad[i] = values[i]
        invalidate()
        onQuadChanged?.invoke(quad.copyOf())
    }

    fun getQuad(): FloatArray = quad.copyOf()

    fun imageWidth(): Int = bitmap?.width ?: 0

    fun imageHeight(): Int = bitmap?.height ?: 0

    private fun resetMatrix() {
        val bmp = bitmap ?: return
        val vw = width.toFloat()
        val vh = height.toFloat()
        if (vw <= 0f || vh <= 0f) return
        val s = min(vw / bmp.width.toFloat(), vh / bmp.height.toFloat())
        matrix.setScale(s, s)
        matrix.postTranslate((vw - bmp.width * s) / 2f, (vh - bmp.height * s) / 2f)
        matrix.invert(inverse)
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        resetMatrix()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val bmp = bitmap ?: return
        canvas.drawBitmap(bmp, matrix, null)

        matrix.mapPoints(tmp, quad)

        path.reset()
        path.moveTo(tmp[0], tmp[1])
        path.lineTo(tmp[2], tmp[3])
        path.lineTo(tmp[4], tmp[5])
        path.lineTo(tmp[6], tmp[7])
        path.close()
        canvas.drawPath(path, fillPaint)
        canvas.drawPath(path, linePaint)

        for (i in 0 until 4) {
            canvas.drawCircle(tmp[i * 2], tmp[i * 2 + 1], handleR, handlePaint)
            canvas.drawCircle(tmp[i * 2], tmp[i * 2 + 1], handleR, handleStroke)
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (bitmap == null) return false
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                active = nearestHandle(event.x, event.y)
                lastX = event.x
                lastY = event.y
                return active >= 0
            }
            MotionEvent.ACTION_MOVE -> {
                if (active < 0) return false
                val dx = event.x - lastX
                val dy = event.y - lastY
                lastX = event.x
                lastY = event.y
                moveHandleBy(active, dx, dy)
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                active = -1
                return true
            }
        }
        return false
    }

    private fun nearestHandle(x: Float, y: Float): Int {
        matrix.mapPoints(tmp, quad)
        var best = -1
        var bestDist = touchR
        for (i in 0 until 4) {
            val d = hypot(x - tmp[i * 2], y - tmp[i * 2 + 1])
            if (d <= bestDist) {
                bestDist = d
                best = i
            }
        }
        return best
    }

    /** 把屏幕位移换算成图片像素位移，并限制在图像范围内。 */
    private fun moveHandleBy(index: Int, dxView: Float, dyView: Float) {
        val bmp = bitmap ?: return
        val pts = floatArrayOf(dxView, dyView)
        val delta = FloatArray(2)
        inverse.mapVectors(pts)
        delta[0] = pts[0]
        delta[1] = pts[1]
        quad[index * 2] = (quad[index * 2] + delta[0]).coerceIn(0f, bmp.width.toFloat())
        quad[index * 2 + 1] = (quad[index * 2 + 1] + delta[1]).coerceIn(0f, bmp.height.toFloat())
        invalidate()
        onQuadChanged?.invoke(quad.copyOf())
    }
}
