package com.unblock.app.ui

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.unblock.app.R
import com.unblock.app.core.GallerySaver
import com.unblock.app.core.TaskStore
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** 单张结果查看：可保存到系统相册。 */
class ResultActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_FILE = "file"
        const val EXTRA_INFO = "info"
    }

    private lateinit var file: File
    private lateinit var info: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)

        val path = intent.getStringExtra(EXTRA_FILE)
        if (path == null || !File(path).exists()) {
            finish()
            return
        }
        file = File(path)

        val image = findViewById<ImageView>(R.id.resultImage)
        info = findViewById(R.id.resultInfo)
        info.text = intent.getStringExtra(EXTRA_INFO) ?: file.name

        val bmp = decodeSampled(file)
        if (bmp != null) image.setImageBitmap(bmp)

        findViewById<Button>(R.id.saveButton).setOnClickListener {
            val name = "unblock_" +
                SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + ".jpg"
            val where = GallerySaver.saveFile(this, file, name)
            info.text = if (where == null) "保存失败" else "已保存到相册：$name"
        }
    }

    private fun decodeSampled(file: File): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        var sample = 1
        val maxSide = maxOf(bounds.outWidth, bounds.outHeight)
        while (maxSide / sample > 2048) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        return BitmapFactory.decodeFile(file.absolutePath, opts)
    }
}
