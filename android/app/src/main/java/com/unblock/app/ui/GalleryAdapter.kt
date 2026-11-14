package com.unblock.app.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.LruCache
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import androidx.recyclerview.widget.RecyclerView
import com.unblock.app.R
import com.unblock.app.core.TaskItem

/**
 * 首页画廊：第 0 格固定是导入入口，之后按时间倒序排合成结果。
 * 缩略图在入库时已经生成，这里只负责按目标尺寸解码和做内存缓存。
 */
class GalleryAdapter(
    private val onAdd: () -> Unit,
    private val onOpen: (TaskItem) -> Unit,
    private val onDelete: (TaskItem) -> Unit
) : RecyclerView.Adapter<RecyclerView.ViewHolder>() {

    companion object {
        private const val TYPE_ENTRY = 0
        private const val TYPE_TASK = 1
        private const val THUMB_TARGET = 480
    }

    private var items: List<TaskItem> = emptyList()

    private val cache = object : LruCache<String, Bitmap>(
        (Runtime.getRuntime().maxMemory() / 1024 / 6).toInt()
    ) {
        override fun sizeOf(key: String, value: Bitmap): Int = value.byteCount / 1024
    }

    fun submit(list: List<TaskItem>) {
        items = list
        notifyDataSetChanged()
    }

    override fun getItemCount(): Int = items.size + 1

    override fun getItemViewType(position: Int): Int = if (position == 0) TYPE_ENTRY else TYPE_TASK

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val inflater = LayoutInflater.from(parent.context)
        return if (viewType == TYPE_ENTRY) {
            val view = inflater.inflate(R.layout.item_gallery_entry, parent, false)
            EntryHolder(view).apply {
                itemView.setOnClickListener { onAdd() }
            }
        } else {
            TaskHolder(inflater.inflate(R.layout.item_gallery_task, parent, false))
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        if (holder !is TaskHolder) return
        val item = items.getOrNull(position - 1) ?: return
        holder.bind(item)
        holder.itemView.setOnClickListener { onOpen(item) }
        holder.itemView.setOnLongClickListener {
            onDelete(item)
            true
        }
    }

    override fun onViewRecycled(holder: RecyclerView.ViewHolder) {
        if (holder is TaskHolder) holder.clear()
    }

    private class EntryHolder(view: View) : RecyclerView.ViewHolder(view)

    private inner class TaskHolder(view: View) : RecyclerView.ViewHolder(view) {
        private val image: ImageView = view.findViewById(R.id.thumbImage)

        fun bind(item: TaskItem) {
            val key = item.thumb.absolutePath + ":" + item.time
            val cached = cache.get(key)
            if (cached != null) {
                image.setImageBitmap(cached)
                return
            }
            val bmp = decode(item.thumb.absolutePath, THUMB_TARGET)
            if (bmp != null) {
                cache.put(key, bmp)
                image.setImageBitmap(bmp)
            } else {
                image.setImageDrawable(null)
            }
        }

        fun clear() {
            image.setImageDrawable(null)
        }
    }

    private fun decode(path: String, target: Int): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / sample > target) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        return BitmapFactory.decodeFile(path, opts)
    }
}
