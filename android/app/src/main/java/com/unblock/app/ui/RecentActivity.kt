package com.unblock.app.ui

import android.content.Intent
import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.ListView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.unblock.app.R
import com.unblock.app.core.TaskItem
import com.unblock.app.core.TaskStore

/** 最近任务列表，点一条进入结果页。 */
class RecentActivity : AppCompatActivity() {

    private var items: List<TaskItem> = emptyList()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_recent)

        val hint = findViewById<TextView>(R.id.recentHint)
        val list = findViewById<ListView>(R.id.recentList)

        items = TaskStore.list(this)
        if (items.isEmpty()) {
            hint.text = "还没有任务"
            return
        }
        hint.text = "最近任务（${items.size}）"

        list.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_list_item_1,
            items.map { TaskStore.label(it) }
        )
        list.setOnItemClickListener { _, _, position, _ ->
            val item = items[position]
            startActivity(
                Intent(this, ResultActivity::class.java)
                    .putExtra(ResultActivity.EXTRA_FILE, item.image.absolutePath)
                    .putExtra(ResultActivity.EXTRA_INFO, TaskStore.label(item))
            )
        }
    }
}
