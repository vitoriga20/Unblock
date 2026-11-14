package com.unblock.app.core

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import java.io.File
import java.io.FileOutputStream

/** 把还原图写进系统相册。 */
object GallerySaver {

    private const val ALBUM = "Unblock"

    /** 返回保存到的位置（API 29+ 是 content uri，以下为文件路径），失败返回 null。 */
    fun save(context: Context, bitmap: Bitmap, displayName: String): String? {
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/$ALBUM")
                }
                val uri = context.contentResolver
                    .insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return null
                context.contentResolver.openOutputStream(uri)?.use {
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it)
                }
                uri.toString()
            } else {
                // Android 9 及以下没有分区存储，直接写公共图片目录
                val dir = File(
                    Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
                    ALBUM
                )
                dir.mkdirs()
                val file = File(dir, displayName)
                FileOutputStream(file).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
                file.absolutePath
            }
        } catch (t: Throwable) {
            Log.w("Unblock", "保存到相册失败: ${t.message}")
            null
        }
    }

    /** 直接复制已有文件到相册，避免二次压缩。 */
    fun saveFile(context: Context, file: File, displayName: String): String? {
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/$ALBUM")
                }
                val uri = context.contentResolver
                    .insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return null
                context.contentResolver.openOutputStream(uri)?.use { out ->
                    file.inputStream().use { it.copyTo(out) }
                }
                uri.toString()
            } else {
                val dir = File(
                    Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
                    ALBUM
                )
                dir.mkdirs()
                val target = File(dir, displayName)
                file.copyTo(target, overwrite = true)
                target.absolutePath
            }
        } catch (t: Throwable) {
            Log.w("Unblock", "保存到相册失败: ${t.message}")
            null
        }
    }
}
