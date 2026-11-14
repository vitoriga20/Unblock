package com.unblock.app.ui

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.unblock.app.R
import com.unblock.app.core.MatTools
import com.unblock.app.core.ScreenDetector
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.util.Locale

/**
 * 四角微调界面：自动粗定位给初值，用户拖动四个角确认。
 * 坐标约定：全部在“界面所用图片”的像素坐标系里，调用方拿到后按显示缩放比换算回原图。
 */
class QuadEditorActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_ASSET = "asset"
        const val EXTRA_FILE = "file"
        const val EXTRA_QUAD = "quad"
        const val EXTRA_IMAGE_W = "image_w"
        const val EXTRA_IMAGE_H = "image_h"
    }

    private lateinit var editor: QuadEditorView
    private lateinit var hint: TextView
    private lateinit var valueText: TextView

    private var detectMessage = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_quad)

        editor = findViewById(R.id.quadView)
        hint = findViewById(R.id.quadHint)
        valueText = findViewById(R.id.quadValue)

        findViewById<Button>(R.id.redetectButton).setOnClickListener { runDetect() }
        findViewById<Button>(R.id.resetButton).setOnClickListener {
            editor.resetQuadToInset()
            refreshValue()
        }
        findViewById<Button>(R.id.confirmButton).setOnClickListener { finishWithQuad() }

        editor.onQuadChanged = { refreshValue() }

        loadImage()
    }

    private fun loadImage() {
        val asset = intent.getStringExtra(EXTRA_ASSET)
        val file = intent.getStringExtra(EXTRA_FILE)
        hint.text = "正在载入图片…"

        Thread {
            val bmp = try {
                if (asset != null) decodeAsset(asset) else if (file != null) decodeFile(file) else null
            } catch (t: Throwable) {
                Log.w("Unblock", "载入图片失败: ${t.message}")
                null
            }
            if (bmp == null) {
                runOnUiThread { hint.text = "图片载入失败" }
                return@Thread
            }
            runOnUiThread {
                editor.setImage(bmp)
                editor.resetQuadToInset()
                runDetect()
            }
        }.start()
    }

    /** 自动粗定位；失败或不可信时退回内缩矩形，交给用户拖。 */
    private fun runDetect() {
        if (editor.imageWidth() == 0) return
        hint.text = "正在检测屏幕边框…"
        val mat = grayMat()

        Thread {
            val result = try {
                if (mat == null) null else ScreenDetector.detect(mat)
            } catch (t: Throwable) {
                Log.w("Unblock", "四角检测失败: ${t.message}")
                null
            }
            runOnUiThread {
                if (result?.quad != null) {
                    detectMessage = String.format(Locale.US, "自动检测：%s（score %.2f）", result.message, result.score)
                    val q = result.quad
                    editor.setQuad(
                        floatArrayOf(
                            q[0].x.toFloat(), q[0].y.toFloat(),
                            q[1].x.toFloat(), q[1].y.toFloat(),
                            q[2].x.toFloat(), q[2].y.toFloat(),
                            q[3].x.toFloat(), q[3].y.toFloat()
                        )
                    )
                } else {
                    detectMessage = "自动检测未命中（${result?.message ?: "异常"}），已放置内缩矩形，请拖动对齐"
                    editor.resetQuadToInset()
                }
                hint.text = detectMessage
                refreshValue()
            }
        }.start()
    }

    /** 检测只需要灰度，单独取一份，避免每帧都从位图重建。 */
    private fun grayMat(): Mat? {
        val bmp = cachedBitmap ?: return null
        val mat = MatTools.fromBitmap(bmp)
        val bgr = Mat()
        Imgproc.cvtColor(mat, bgr, Imgproc.COLOR_RGBA2BGR)
        return bgr
    }

    private var cachedBitmap: Bitmap? = null

    private fun finishWithQuad() {
        val q = editor.getQuad()
        val data = Intent().apply {
            putExtra(EXTRA_QUAD, q)
            putExtra(EXTRA_IMAGE_W, editor.imageWidth())
            putExtra(EXTRA_IMAGE_H, editor.imageHeight())
        }
        setResult(Activity.RESULT_OK, data)
        finish()
    }

    private fun refreshValue() {
        val q = editor.getQuad()
        valueText.text = String.format(
            Locale.US,
            "%dx%d  TL(%.0f,%.0f) TR(%.0f,%.0f) BR(%.0f,%.0f) BL(%.0f,%.0f)",
            editor.imageWidth(), editor.imageHeight(),
            q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]
        )
    }

    private fun decodeAsset(path: String): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        assets.open(path).use { BitmapFactory.decodeStream(it, null, bounds) }
        val opts = BitmapFactory.Options().apply {
            inSampleSize = sampleSize(bounds.outWidth, bounds.outHeight)
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val bmp = assets.open(path).use { BitmapFactory.decodeStream(it, null, opts) }
        cachedBitmap = bmp
        return bmp
    }

    private fun decodeFile(path: String): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        val opts = BitmapFactory.Options().apply {
            inSampleSize = sampleSize(bounds.outWidth, bounds.outHeight)
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val bmp = BitmapFactory.decodeFile(path, opts)
        cachedBitmap = bmp
        return bmp
    }

    private fun sampleSize(w: Int, h: Int): Int {
        var sample = 1
        val maxSide = maxOf(w, h)
        while (maxSide / sample > 2048) sample *= 2
        return sample
    }
}
