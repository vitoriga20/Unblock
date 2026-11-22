package com.unblock.app.ui

import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.View
import android.view.animation.TranslateAnimation
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.ViewFlipper
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.WindowInsetsControllerCompat
import com.unblock.app.R
import com.unblock.app.core.PhotoStore
import kotlin.math.abs

/**
 * 示例页：三屏讲清"为什么最少要两张"和"边界在哪"。
 *
 * 素材是预录的，不现场跑合成——教程的目的是让人看懂原理，不是验证设备性能；
 * 现场跑还会因机型不同出现等待与失败，反而干扰理解。
 */
class TutorialActivity : AppCompatActivity() {

    companion object {
        private const val FLIP_DURATION = 220L
        private const val SWIPE_MIN_DP = 48f
        private const val SWIPE_VELOCITY = 220f
    }

    private data class Page(val asset: String, val textRes: Int)

    private val pages = listOf(
        Page("demo/demo_b.jpg", R.string.tutorial_demo_step1),
        Page("demo/demo_a.jpg", R.string.tutorial_demo_step2),
        Page("demo/demo_result.jpg", R.string.tutorial_demo_step3)
    )

    private lateinit var flipper: ViewFlipper
    private lateinit var dots: LinearLayout
    private val dotViews = ArrayList<View>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_tutorial)

        window.statusBarColor = getColor(R.color.viewer_bg)
        WindowInsetsControllerCompat(window, window.decorView).isAppearanceLightStatusBars = false

        flipper = findViewById(R.id.demoFlipper)
        dots = findViewById(R.id.demoDots)

        findViewById<View>(R.id.demoClose).setOnClickListener { finish() }

        buildPages()
        buildDots()
        updateDots()
        bindGestures()
    }

    private fun buildPages() {
        val inflater = layoutInflater
        for (page in pages) {
            val view = inflater.inflate(R.layout.item_demo_page, flipper, false)
            view.findViewById<TextView>(R.id.demoPageText).setText(page.textRes)
            flipper.addView(view)
        }
        loadBitmaps()
    }

    /** 三张图都不大，一次性解码到位；解码失败就保持空页，不影响文字说明。 */
    private fun loadBitmaps() {
        Thread {
            val loaded = pages.map { PhotoStore.loadEditorBitmap(this, it.asset) }
            runOnUiThread {
                if (isFinishing || isDestroyed) return@runOnUiThread
                loaded.forEachIndexed { index, bmp ->
                    if (bmp != null) {
                        flipper.getChildAt(index)
                            .findViewById<ImageView>(R.id.demoPageImage)
                            .setImageBitmap(bmp)
                    }
                }
            }
        }.start()
    }

    private fun buildDots() {
        val size = resources.getDimensionPixelSize(R.dimen.demo_dot)
        val gap = resources.getDimensionPixelSize(R.dimen.demo_dot_gap)
        for (i in pages.indices) {
            val dot = View(this)
            val lp = LinearLayout.LayoutParams(size, size).apply {
                if (i > 0) marginStart = gap
            }
            dot.layoutParams = lp
            dots.addView(dot)
            dotViews.add(dot)
        }
    }

    private fun updateDots() {
        val active = ContextCompat.getColor(this, R.color.on_viewer)
        val idle = ContextCompat.getColor(this, R.color.on_viewer_tertiary)
        dotViews.forEachIndexed { index, dot ->
            dot.background = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(if (index == flipper.displayedChild) active else idle)
            }
        }
    }

    private fun bindGestures() {
        val minDist = SWIPE_MIN_DP * resources.displayMetrics.density
        val detector = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onDown(e: MotionEvent): Boolean = true

            override fun onFling(
                e1: MotionEvent?,
                e2: MotionEvent,
                velocityX: Float,
                velocityY: Float
            ): Boolean {
                if (e1 == null) return false
                val dx = e2.x - e1.x
                if (abs(dx) > minDist && abs(velocityX) > SWIPE_VELOCITY) {
                    step(if (dx < 0) 1 else -1)
                    return true
                }
                return false
            }

            override fun onSingleTapUp(e: MotionEvent): Boolean {
                step(if (e.x < flipper.width / 2f) -1 else 1)
                return true
            }
        })
        flipper.setOnTouchListener { _, event ->
            detector.onTouchEvent(event)
            true
        }
    }

    private fun step(delta: Int) {
        val from = flipper.displayedChild
        val target = (from + delta).coerceIn(0, pages.size - 1)
        if (target == from) return
        val shift = flipper.width.toFloat() * if (target > from) 1f else -1f
        flipper.inAnimation = TranslateAnimation(shift, 0f, 0f, 0f).apply {
            duration = FLIP_DURATION
        }
        flipper.outAnimation = TranslateAnimation(0f, -shift, 0f, 0f).apply {
            duration = FLIP_DURATION
        }
        flipper.displayedChild = target
        updateDots()
    }
}
