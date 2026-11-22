package com.unblock.app.ui

import android.app.Activity
import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.core.view.WindowInsetsControllerCompat
import com.unblock.app.R

/**
 * 一次性聚光引导：压暗全屏、挖出目标区域、描一圈细线，旁边给一句话。
 *
 * 只负责呈现与"点任意处继续"，不做任何业务判断；是否该出现、出现到第几步，
 * 由调用方结合 [com.unblock.app.core.TutorialPrefs] 决定。
 * 不强制用户完成真实操作——新手在引导里无法完成也不该被卡住。
 */
class GuideOverlayView(context: Context) : FrameLayout(context) {

    private val scrimPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.guide_scrim)
    }
    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = resources.getDimension(R.dimen.guide_stroke)
    }
    private val hole = RectF()
    private val radius = resources.getDimension(R.dimen.guide_radius)
    private val cardGap = resources.getDimension(R.dimen.guide_card_gap)
    private val cardPad = resources.getDimension(R.dimen.guide_card_pad)
    private val cardMaxW = resources.getDimension(R.dimen.guide_card_max_width)

    private var onNext: (() -> Unit)? = null
    private var finished = false
    /** 引导结束后要把状态栏图标恢复成的明暗状态 */
    private var restoreLightStatusBars: Boolean? = null

    init {
        isClickable = true
        isFocusable = true
        setWillNotDraw(false)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (hole.isEmpty) return
        // 全屏矩形减去高亮矩形 = 挖洞
        val path = Path().apply {
            fillType = Path.FillType.EVEN_ODD
            addRect(0f, 0f, width.toFloat(), height.toFloat(), Path.Direction.CW)
            addRoundRect(hole, radius, radius, Path.Direction.CW)
        }
        canvas.drawPath(path, scrimPaint)
        canvas.drawRoundRect(hole, radius, radius, strokePaint)
    }

    override fun onLayout(changed: Boolean, l: Int, t: Int, r: Int, b: Int) {
        super.onLayout(changed, l, t, r, b)
        val card = getChildAt(0) ?: return
        val maxWidth = minOf(width - (cardGap * 2).toInt(), cardMaxW.toInt())
        card.measure(
            MeasureSpec.makeMeasureSpec(maxWidth, MeasureSpec.AT_MOST),
            MeasureSpec.makeMeasureSpec(height, MeasureSpec.AT_MOST)
        )
        val cw = card.measuredWidth
        val ch = card.measuredHeight
        val left = (width - cw) / 2f
        val below = hole.bottom + cardGap
        val above = hole.top - cardGap - ch
        val top = if (below + ch + cardGap <= height) {
            below
        } else {
            maxOf(cardGap, above)
        }
        card.layout(left.toInt(), top.toInt(), (left + cw).toInt(), (top + ch).toInt())
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.actionMasked == MotionEvent.ACTION_UP) dismiss()
        return true
    }

    fun dismiss() {
        if (finished) return
        finished = true
        restoreLightStatusBars?.let { light ->
            val activity = context as? Activity
            if (activity != null && !activity.isFinishing && !activity.isDestroyed) {
                WindowInsetsControllerCompat(
                    activity.window,
                    activity.window.decorView
                ).isAppearanceLightStatusBars = light
            }
        }
        restoreLightStatusBars = null
        val callback = onNext
        onNext = null
        (parent as? ViewGroup)?.removeView(this)
        callback?.invoke()
    }

    private fun buildCard(
        activity: Activity,
        text: String,
        button: String,
        onDark: Boolean
    ): View {
        val corner = resources.getDimension(R.dimen.guide_card_corner)
        val card = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(cardPad.toInt(), cardPad.toInt(), cardPad.toInt(), cardPad.toInt())
            background = GradientDrawable().apply {
                setColor(
                    ContextCompat.getColor(
                        activity,
                        if (onDark) R.color.guide_card_on_dark else R.color.guide_card_on_light
                    )
                )
                cornerRadius = corner
            }
        }

        val label = TextView(activity).apply {
            this.text = text
            setTextColor(
                ContextCompat.getColor(
                    activity,
                    if (onDark) R.color.guide_text_on_dark else R.color.guide_text_on_light
                )
            )
            textSize = 14f
            setLineSpacing(0f, 1.35f)
        }
        card.addView(label, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT))

        val action = TextView(activity).apply {
            this.text = button
            gravity = Gravity.CENTER
            setTextColor(
                ContextCompat.getColor(
                    activity,
                    if (onDark) R.color.guide_button_text_on_dark
                    else R.color.guide_button_text_on_light
                )
            )
            textSize = 14f
            typeface = Typeface.DEFAULT_BOLD
            setPadding(0, dip(10f), 0, dip(10f))
            background = GradientDrawable().apply {
                setColor(
                    ContextCompat.getColor(
                        activity,
                        if (onDark) R.color.guide_button_on_dark else R.color.guide_button_on_light
                    )
                )
                cornerRadius = radius
            }
            setOnClickListener { dismiss() }
        }
        card.addView(
            action,
            LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply {
                topMargin = cardGap.toInt()
            }
        )
        return card
    }

    private fun dip(v: Float): Int = (v * resources.displayMetrics.density).toInt()

    companion object {

        /**
         * 把引导挂到当前页面上。[anchorScreen] 是目标控件的屏幕坐标矩形。
         * 返回 null 表示页面已在退出或不适合挂载，调用方无需处理。
         */
        fun attach(
            activity: Activity,
            anchorScreen: RectF,
            text: String,
            button: String,
            onDark: Boolean,
            onNext: () -> Unit
        ): GuideOverlayView? {
            if (activity.isFinishing || activity.isDestroyed) return null
            // 挂到 decorView 才能盖住状态栏；只盖 content 会在顶部留一条白边
            val root = activity.window?.decorView as? ViewGroup
                ?: activity.findViewById<ViewGroup>(android.R.id.content)
                ?: return null

            val overlay = GuideOverlayView(activity)
            overlay.onNext = onNext
            overlay.strokePaint.color = ContextCompat.getColor(
                activity,
                if (onDark) R.color.guide_stroke_on_dark else R.color.guide_stroke_on_light
            )
            // 遮罩压暗后状态栏变成深色，图标必须同时转白，否则在浅色页上看不见
            overlay.restoreLightStatusBars = !onDark
            WindowInsetsControllerCompat(
                activity.window,
                activity.window.decorView
            ).isAppearanceLightStatusBars = false

            // 用屏幕坐标换算，避免受 fitsSystemWindows 与嵌套层级影响
            val origin = IntArray(2)
            root.getLocationOnScreen(origin)
            val pad = activity.resources.getDimension(R.dimen.guide_pad)
            overlay.hole.set(anchorScreen)
            overlay.hole.offset(-origin[0].toFloat(), -origin[1].toFloat())
            overlay.hole.inset(-pad, -pad)

            root.addView(overlay, LayoutParams(MATCH_PARENT, MATCH_PARENT))
            overlay.addView(overlay.buildCard(activity, text, button, onDark))
            overlay.requestLayout()
            return overlay
        }

        /** 取控件在屏幕上的矩形。控件未布局时返回 null。 */
        fun screenRectOf(view: View): RectF? {
            val w = view.width
            val h = view.height
            if (w <= 0 || h <= 0) return null
            val loc = IntArray(2)
            view.getLocationOnScreen(loc)
            return RectF(
                loc[0].toFloat(),
                loc[1].toFloat(),
                (loc[0] + w).toFloat(),
                (loc[1] + h).toFloat()
            )
        }
    }
}
