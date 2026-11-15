package com.unblock.app.ui

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.WindowInsetsControllerCompat
import com.unblock.app.R
import com.unblock.app.core.ConfigLoader
import com.unblock.app.core.MatTools
import com.unblock.app.core.PhotoStore
import com.unblock.app.core.ScreenDetector
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File

/**
 * 四角确认：自动粗定位给初值，用户拖动贴合黑板边缘。
 * 坐标约定：全部在“编辑器所用图片”的像素坐标系里，调用方按实际尺寸换算回缓存原图。
 */
class QuadEditorActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_ASSET = "asset"
        const val EXTRA_FILE = "file"
        const val EXTRA_QUAD = "quad"
        const val EXTRA_IMAGE_W = "image_w"
        const val EXTRA_IMAGE_H = "image_h"
        private const val TAG = "Unblock"
    }

    private lateinit var editor: QuadEditorView
    private lateinit var hint: TextView
    private var cachedBitmap: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_quad)

        window.statusBarColor = getColor(R.color.viewer_bg)
        WindowInsetsControllerCompat(window, window.decorView).isAppearanceLightStatusBars = false

        editor = findViewById(R.id.quadView)
        hint = findViewById(R.id.quadHint)

        findViewById<android.view.View>(R.id.quadBack).setOnClickListener { finishCancelled() }
        findViewById<android.view.View>(R.id.quadReselect).setOnClickListener { finishCancelled() }
        findViewById<android.view.View>(R.id.confirmButton).setOnClickListener { finishWithQuad() }

        loadImage()
    }

    private fun loadImage() {
        val asset = intent.getStringExtra(EXTRA_ASSET)
        val file = intent.getStringExtra(EXTRA_FILE)
        if (asset == null && file == null) {
            hint.text = getString(R.string.quad_load_failed)
            return
        }
        hint.text = getString(R.string.quad_loading)

        Thread {
            val bmp = try {
                if (file != null) PhotoStore.loadEditorBitmap(File(file))
                else PhotoStore.loadEditorBitmap(this, asset!!)
            } catch (t: Throwable) {
                Log.w(TAG, "载入图片失败: ${t.message}")
                null
            }
            runOnUiThread {
                if (bmp == null) {
                    hint.text = getString(R.string.quad_load_failed)
                    return@runOnUiThread
                }
                cachedBitmap = bmp
                editor.setImage(bmp)
                editor.resetQuadToInset()
                runDetect()
            }
        }.start()
    }

    /** 自动粗定位；失败时退回内缩矩形，交给用户拖。 */
    private fun runDetect() {
        if (editor.imageWidth() == 0) return
        hint.text = getString(R.string.quad_detecting)
        val gray = grayMat()
        if (gray == null) {
            hint.text = getString(R.string.quad_hint_manual)
            return
        }

        Thread {
            val quad = try {
                ScreenDetector.detectBest(gray, ConfigLoader.load(this).screen).quad
            } catch (t: Throwable) {
                Log.w(TAG, "四角检测失败: ${t.message}")
                null
            }
            gray.release()
            runOnUiThread {
                if (quad != null) {
                    editor.setQuad(
                        floatArrayOf(
                            quad[0].x.toFloat(), quad[0].y.toFloat(),
                            quad[1].x.toFloat(), quad[1].y.toFloat(),
                            quad[2].x.toFloat(), quad[2].y.toFloat(),
                            quad[3].x.toFloat(), quad[3].y.toFloat()
                        )
                    )
                    hint.text = getString(R.string.quad_hint_detected)
                } else {
                    editor.resetQuadToInset()
                    hint.text = getString(R.string.quad_hint_manual)
                }
            }
        }.start()
    }

    /** 检测只需要灰度，单独取一份。 */
    private fun grayMat(): Mat? {
        val bmp = cachedBitmap ?: return null
        val rgba = MatTools.fromBitmap(bmp)
        val bgr = Mat()
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        rgba.release()
        return bgr
    }

    private fun finishWithQuad() {
        if (editor.imageWidth() == 0) return
        val data = Intent().apply {
            putExtra(EXTRA_QUAD, editor.getQuad())
            putExtra(EXTRA_IMAGE_W, editor.imageWidth())
            putExtra(EXTRA_IMAGE_H, editor.imageHeight())
        }
        setResult(Activity.RESULT_OK, data)
        finish()
    }

    private fun finishCancelled() {
        setResult(Activity.RESULT_CANCELED)
        finish()
    }

    override fun onDestroy() {
        super.onDestroy()
        cachedBitmap?.recycle()
        cachedBitmap = null
    }
}
