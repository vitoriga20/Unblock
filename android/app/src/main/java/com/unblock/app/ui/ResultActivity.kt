package com.unblock.app.ui

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.core.view.WindowInsetsControllerCompat
import com.unblock.app.R
import com.unblock.app.core.GallerySaver
import java.io.File
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

/** 单张结果：全屏看图，顶部返回 / 分享，底部只有一个保存按钮。 */
class ResultActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_FILE = "file"
        const val EXTRA_SOURCES = "sources"
        const val EXTRA_TIME = "time"
        private const val MAX_VIEW_SIDE = 2048
        private const val PROVIDER_SUFFIX = ".fileprovider"
    }

    private lateinit var file: File
    private lateinit var saveButton: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)

        window.statusBarColor = getColor(R.color.viewer_bg)
        WindowInsetsControllerCompat(window, window.decorView).isAppearanceLightStatusBars = false

        val path = intent.getStringExtra(EXTRA_FILE)
        if (path == null || !File(path).exists()) {
            finish()
            return
        }
        file = File(path)

        saveButton = findViewById(R.id.saveButton)
        val sources = intent.getIntExtra(EXTRA_SOURCES, 0)
        val time = intent.getLongExtra(EXTRA_TIME, System.currentTimeMillis())
        findViewById<TextView>(R.id.resultMeta).text =
            getString(R.string.result_meta, sources, formatTime(time))

        findViewById<View>(R.id.backButton).setOnClickListener { finish() }
        findViewById<View>(R.id.shareButton).setOnClickListener { share() }
        saveButton.setOnClickListener { save() }

        loadImage()
    }

    private fun loadImage() {
        Thread {
            val bmp = decodeSampled(file, MAX_VIEW_SIDE)
            runOnUiThread {
                if (bmp != null) {
                    findViewById<ImageView>(R.id.resultImage).setImageBitmap(bmp)
                }
            }
        }.start()
    }

    private fun save() {
        val name = "unblock_" +
            SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + ".jpg"
        val where = GallerySaver.saveFile(this, file, name)
        if (where == null) {
            Toast.makeText(this, R.string.save_failed, Toast.LENGTH_SHORT).show()
            return
        }
        saveButton.text = getString(R.string.action_saved)
        saveButton.isEnabled = false
        saveButton.alpha = 0.6f
    }

    private fun share() {
        try {
            val uri = FileProvider.getUriForFile(this, packageName + PROVIDER_SUFFIX, file)
            val intent = Intent(Intent.ACTION_SEND).apply {
                type = "image/jpeg"
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(intent, getString(R.string.cd_share)))
        } catch (t: Throwable) {
            Toast.makeText(this, R.string.share_failed, Toast.LENGTH_SHORT).show()
        }
    }

    private fun formatTime(time: Long): String {
        val now = Calendar.getInstance()
        val target = Calendar.getInstance().apply { timeInMillis = time }
        val sameDay = now.get(Calendar.YEAR) == target.get(Calendar.YEAR) &&
            now.get(Calendar.DAY_OF_YEAR) == target.get(Calendar.DAY_OF_YEAR)
        val pattern = if (sameDay) "'今天' HH:mm" else "MM-dd HH:mm"
        return SimpleDateFormat(pattern, Locale.CHINA).format(Date(time))
    }

    private fun decodeSampled(file: File, maxSide: Int): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        var sample = 1
        val longest = maxOf(bounds.outWidth, bounds.outHeight)
        while (longest / sample > maxSide) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        return BitmapFactory.decodeFile(file.absolutePath, opts)
    }
}
