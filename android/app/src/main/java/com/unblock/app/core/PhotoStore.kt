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
 *
 * 每次“导入 - 对齐 - 合成”是一轮独立会话：会话目录里放这一轮的缓存文件，
 * 新会话不会去动正在使用的那一份，避免上一版“先 clear 再导入”把正在读的图删掉。
 */
object PhotoStore {

    /** 合成链路可接受的长边上限。 */
    const val MAX_SIDE = 2400

    /** 四角编辑器的显示分辨率上限；回传的四角按实际图像尺寸换算，两处不再各写一套阈值。 */
    const val EDITOR_MAX_SIDE = 2048

    private const val SESSION_ROOT = "sessions"
    private const val PHOTO_PREFIX = "photo_"
    private const val JPEG_QUALITY = 92

    /** 新建一轮会话目录，同时清掉更早的残留会话。 */
    fun newSession(context: Context): File {
        val root = File(context.cacheDir, SESSION_ROOT).apply { mkdirs() }
        root.listFiles()?.forEach { if (it.isDirectory) it.deleteRecursively() }
        val dir = File(root, "s_" + System.currentTimeMillis())
        dir.mkdirs()
        return dir
    }

    fun discardSession(dir: File?) {
        if (dir == null) return
        try {
            dir.deleteRecursively()
        } catch (t: Throwable) {
            Log.w("Unblock", "清理会话失败: ${t.message}")
        }
    }

    /** 会话里下一个可用的照片序号，重复追加不会互相覆盖。 */
    fun nextIndex(dir: File): Int {
        val max = dir.listFiles { f -> f.name.startsWith(PHOTO_PREFIX) && f.name.endsWith(".jpg") }
            ?.mapNotNull { it.name.removePrefix(PHOTO_PREFIX).removeSuffix(".jpg").toIntOrNull() }
            ?.maxOrNull()
        return (max ?: -1) + 1
    }

    /** 相机直拍的落盘位置，交给系统相机写入。 */
    fun captureTarget(dir: File, index: Int): File = File(dir, "$PHOTO_PREFIX$index.jpg")

    /** 相机拍完的图带 EXIF 方向，需要缩到 MAX_SIDE 并原地摆正。 */
    fun normalizeCaptured(file: File): File? {
        return try {
            val bitmap = decodeFileSampled(file, MAX_SIDE) ?: return null
            val orientation = ExifInterface(file.absolutePath)
                .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
            saveBitmap(file, applyOrientation(bitmap, orientation))
            file
        } catch (t: Throwable) {
            Log.w("Unblock", "相机照片处理失败: ${t.message}")
            null
        }
    }

    /** 从相册 Uri 导入。 */
    fun importFromUri(context: Context, uri: Uri, dir: File, index: Int): File? {
        return try {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            context.contentResolver.openInputStream(uri)?.use {
                BitmapFactory.decodeStream(it, null, bounds)
            }
            val opts = BitmapFactory.Options().apply {
                inSampleSize = sampleSize(bounds.outWidth, bounds.outHeight, MAX_SIDE)
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            val raw = context.contentResolver.openInputStream(uri)?.use {
                BitmapFactory.decodeStream(it, null, opts)
            } ?: return null
            val orientation = readOrientation(context, uri)
            val file = File(dir, "$PHOTO_PREFIX$index.jpg")
            saveBitmap(file, applyOrientation(raw, orientation))
            file
        } catch (t: Throwable) {
            Log.w("Unblock", "导入照片失败: ${t.message}")
            null
        }
    }

    /** 从 assets 导入，用于内置样张走同一套真实流程。 */
    fun importFromAsset(context: Context, assetPath: String, dir: File, index: Int): File? {
        return try {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            context.assets.open(assetPath).use { BitmapFactory.decodeStream(it, null, bounds) }
            val opts = BitmapFactory.Options().apply {
                inSampleSize = sampleSize(bounds.outWidth, bounds.outHeight, MAX_SIDE)
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            val raw = context.assets.open(assetPath).use { BitmapFactory.decodeStream(it, null, opts) }
                ?: return null
            val file = File(dir, "$PHOTO_PREFIX$index.jpg")
            saveBitmap(file, raw)
            file
        } catch (t: Throwable) {
            Log.w("Unblock", "导入内置样张失败: ${t.message}")
            null
        }
    }

    /** 合成用的 BGR 三通道图像；位图用完立即回收，不再和 Mat 同时占两份内存。 */
    fun load(file: File): Mat? {
        val bmp = BitmapFactory.decodeFile(file.absolutePath) ?: return null
        val rgba = MatTools.fromBitmap(bmp)
        bmp.recycle()
        val bgr = Mat()
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        rgba.release()
        return bgr
    }

    /** 四角编辑器用：按 EDITOR_MAX_SIDE 解码，界面坐标与回传坐标系一一对应。 */
    fun loadEditorBitmap(file: File): Bitmap? = decodeFileSampled(file, EDITOR_MAX_SIDE)

    fun loadEditorBitmap(context: Context, assetPath: String): Bitmap? {
        return try {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            context.assets.open(assetPath).use { BitmapFactory.decodeStream(it, null, bounds) }
            val opts = BitmapFactory.Options().apply {
                inSampleSize = sampleSize(bounds.outWidth, bounds.outHeight, EDITOR_MAX_SIDE)
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            context.assets.open(assetPath).use { BitmapFactory.decodeStream(it, null, opts) }
        } catch (t: Throwable) {
            Log.w("Unblock", "解码样张失败: ${t.message}")
            null
        }
    }

    private fun decodeFileSampled(file: File, maxSide: Int): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        val opts = BitmapFactory.Options().apply {
            inSampleSize = sampleSize(bounds.outWidth, bounds.outHeight, maxSide)
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        return BitmapFactory.decodeFile(file.absolutePath, opts)
    }

    private fun saveBitmap(file: File, bmp: Bitmap) {
        FileOutputStream(file).use { bmp.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, it) }
    }

    private fun readOrientation(context: Context, uri: Uri): Int = try {
        context.contentResolver.openInputStream(uri)?.use {
            ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        } ?: ExifInterface.ORIENTATION_NORMAL
    } catch (t: Throwable) {
        ExifInterface.ORIENTATION_NORMAL
    }

    private fun applyOrientation(bmp: Bitmap, orientation: Int): Bitmap {
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

    private fun sampleSize(w: Int, h: Int, maxSide: Int): Int {
        if (w <= 0 || h <= 0) return 1
        var sample = 1
        val longest = maxOf(w, h)
        while (longest / sample > maxSide) sample *= 2
        return sample
    }
}
