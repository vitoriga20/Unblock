package com.unblock.app.core

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import android.util.Log
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream

/**
 * 照片导入与落地。
 * 从相册/相机拿到的原始照片先统一成“长边不超过 MAX_SIDE、方向已摆正”的缓存文件，
 * 后续四角编辑与合成都用这同一份，避免不同环节各自解码导致分辨率或方向不一致。
 */
object PhotoStore {

    const val MAX_SIDE = 2400
    private const val DIR_NAME = "photos"

    fun dir(context: Context): File {
        val d = File(context.cacheDir, DIR_NAME)
        d.mkdirs()
        return d
    }

    fun clear(context: Context) {
        dir(context).listFiles()?.forEach { it.delete() }
    }

    /** 从相册 Uri 导入，返回落地后的缓存文件。 */
    fun importFromUri(context: Context, uri: Uri, index: Int): File? {
        return try {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            context.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
            val opts = BitmapFactory.Options().apply {
                inSampleSize = sampleSize(bounds.outWidth, bounds.outHeight)
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            val raw = context.contentResolver.openInputStream(uri)?.use {
                BitmapFactory.decodeStream(it, null, opts)
            } ?: return null
            val fixed = applyExif(context, uri, raw)
            save(context, fixed, index)
        } catch (t: Throwable) {
            Log.w("Unblock", "导入照片失败: ${t.message}")
            null
        }
    }

    /** 从 assets 导入，用于内置样张走同一套真实流程。 */
    fun importFromAsset(context: Context, assetPath: String, index: Int): File? {
        return try {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            context.assets.open(assetPath).use { BitmapFactory.decodeStream(it, null, bounds) }
            val opts = BitmapFactory.Options().apply {
                inSampleSize = sampleSize(bounds.outWidth, bounds.outHeight)
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            val raw = context.assets.open(assetPath).use { BitmapFactory.decodeStream(it, null, opts) }
                ?: return null
            save(context, raw, index)
        } catch (t: Throwable) {
            Log.w("Unblock", "导入内置样张失败: ${t.message}")
            null
        }
    }

    fun load(file: File): Mat? {
        val bmp = BitmapFactory.decodeFile(file.absolutePath) ?: return null
        val rgba = MatTools.fromBitmap(bmp)
        val bgr = Mat()
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        return bgr
    }

    private fun save(context: Context, bmp: Bitmap, index: Int): File {
        val file = File(dir(context), "photo_$index.jpg")
        FileOutputStream(file).use { bmp.compress(Bitmap.CompressFormat.JPEG, 92, it) }
        return file
    }

    private fun applyExif(context: Context, uri: Uri, bmp: Bitmap): Bitmap {
        val orientation = try {
            context.contentResolver.openInputStream(uri)?.use {
                ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
            } ?: ExifInterface.ORIENTATION_NORMAL
        } catch (t: Throwable) {
            ExifInterface.ORIENTATION_NORMAL
        }
        val m = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                m.postRotate(90f)
                m.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_TRANSVERSE -> {
                m.postRotate(270f)
                m.postScale(-1f, 1f)
            }
        }
        if (m.isIdentity) return bmp
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
    }

    private fun sampleSize(w: Int, h: Int): Int {
        if (w <= 0 || h <= 0) return 1
        var sample = 1
        val maxSide = maxOf(w, h)
        while (maxSide / sample > MAX_SIDE) sample *= 2
        return sample
    }
}
