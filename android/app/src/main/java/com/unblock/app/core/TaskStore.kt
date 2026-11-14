package com.unblock.app.core

import android.content.Context
import android.graphics.Bitmap
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class TaskItem(
    val image: File,
    val time: Long,
    val width: Int,
    val height: Int,
    val coverage: Double,
    val elapsedMs: Long,
    val sources: Int
)

/** App 内的最近任务：图片落在私有目录，元信息写在同名 .json 里。 */
object TaskStore {

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
            val meta = JSONObject().apply {
                put("time", stamp)
                put("width", bitmap.width)
                put("height", bitmap.height)
                put("coverage", coverage)
                put("elapsed", elapsedMs)
                put("sources", sources)
            }
            File(dir(context), "task_$stamp.json").writeText(meta.toString(), Charsets.UTF_8)
            TaskItem(image, stamp, bitmap.width, bitmap.height, coverage, elapsedMs, sources)
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
                out.add(
                    TaskItem(
                        image,
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

    fun label(item: TaskItem): String {
        val fmt = SimpleDateFormat("MM-dd HH:mm", Locale.US)
        return String.format(
            Locale.US,
            "%s   %d 张  %dx%d  覆盖 %.0f%%  %dms",
            fmt.format(Date(item.time)), item.sources, item.width, item.height,
            item.coverage * 100.0, item.elapsedMs
        )
    }
}
