package com.unblock.app.core

import android.content.Context
import android.graphics.Bitmap
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

data class TaskItem(
    val image: File,
    val thumb: File,
    val time: Long,
    val width: Int,
    val height: Int,
    val coverage: Double,
    val elapsedMs: Long,
    val sources: Int
)

/** App 内的最近任务：图片与缩略图落在私有目录，元信息写在同名 .json 里。 */
object TaskStore {

    /** 列表缩略图的长边，方形裁切在界面层做，这里只保证解码够快。 */
    private const val THUMB_SIDE = 480
    private const val THUMB_QUALITY = 85

    private fun dir(context: Context): File {
        val d = File(context.filesDir, "tasks")
        d.mkdirs()
        return d
    }

    fun save(
        context: Context,
        bitmap: Bitmap,
        coverage: Double,
        elapsedMs: Long,
        sources: Int
    ): TaskItem? {
        return try {
            val stamp = System.currentTimeMillis()
            val image = File(dir(context), "task_$stamp.jpg")
            FileOutputStream(image).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
            val thumb = File(dir(context), "task_${stamp}_thumb.jpg")
            writeThumb(bitmap, thumb)
            val meta = JSONObject().apply {
                put("time", stamp)
                put("width", bitmap.width)
                put("height", bitmap.height)
                put("coverage", coverage)
                put("elapsed", elapsedMs)
                put("sources", sources)
            }
            File(dir(context), "task_$stamp.json").writeText(meta.toString(), Charsets.UTF_8)
            TaskItem(image, thumb, stamp, bitmap.width, bitmap.height, coverage, elapsedMs, sources)
        } catch (t: Throwable) {
            null
        }
    }

    fun list(context: Context): List<TaskItem> {
        val files = dir(context).listFiles { f -> f.name.endsWith(".json") } ?: return emptyList()
        val out = ArrayList<TaskItem>()
        for (meta in files) {
            try {
                val obj = JSONObject(meta.readText(Charsets.UTF_8))
                val stamp = obj.optLong("time")
                val image = File(dir(context), "task_$stamp.jpg")
                if (!image.exists()) continue
                val thumb = File(dir(context), "task_${stamp}_thumb.jpg")
                out.add(
                    TaskItem(
                        image,
                        if (thumb.exists()) thumb else image,
                        stamp,
                        obj.optInt("width"),
                        obj.optInt("height"),
                        obj.optDouble("coverage", 0.0),
                        obj.optLong("elapsed"),
                        obj.optInt("sources")
                    )
                )
            } catch (t: Throwable) {
                continue
            }
        }
        return out.sortedByDescending { it.time }
    }

    /** 删除一条记录：结果图、缩略图、元信息一起清掉；系统相册里的原图不受影响。 */
    fun delete(context: Context, item: TaskItem) {
        item.image.delete()
        item.thumb.delete()
        File(dir(context), "task_${item.time}.json").delete()
    }

    private fun writeThumb(bitmap: Bitmap, target: File) {
        val longest = max(bitmap.width, bitmap.height)
        val scale = min(1f, THUMB_SIDE.toFloat() / longest)
        val w = max(1, (bitmap.width * scale).toInt())
        val h = max(1, (bitmap.height * scale).toInt())
        val thumb = Bitmap.createScaledBitmap(bitmap, w, h, true)
        FileOutputStream(target).use { thumb.compress(Bitmap.CompressFormat.JPEG, THUMB_QUALITY, it) }
        if (thumb !== bitmap) thumb.recycle()
    }

    fun label(item: TaskItem): String {
        val fmt = SimpleDateFormat("MM-dd HH:mm", Locale.US)
        return fmt.format(Date(item.time))
    }
}
