package com.unblock.app

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.unblock.app.core.ConfigLoader
import com.unblock.app.core.Geometry
import com.unblock.app.core.PhotoStore
import com.unblock.app.core.Pipeline
import com.unblock.app.core.PipelineResult
import com.unblock.app.core.TaskStore
import com.unblock.app.ui.QuadEditorActivity
import com.unblock.app.ui.RecentActivity
import org.opencv.android.OpenCVLoader
import org.opencv.core.Point
import java.io.File
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var runButton: Button
    private lateinit var status: TextView
    private lateinit var image0: ImageView
    private lateinit var meta0: TextView
    private lateinit var image1: ImageView
    private lateinit var meta1: TextView

    /** 选好照片后待合成的缓存文件，第 0 张作为底图。 */
    private var pendingFiles: List<File> = emptyList()
    /** 四角编辑结束后是否直接进入合成。 */
    private var composeAfterEdit = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        runButton = findViewById(R.id.runButton)
        status = findViewById(R.id.status)
        image0 = findViewById(R.id.image0)
        meta0 = findViewById(R.id.meta0)
        image1 = findViewById(R.id.image1)
        meta1 = findViewById(R.id.meta1)

        val opencvReady = try {
            OpenCVLoader.initLocal()
        } catch (t: Throwable) {
            false
        }

        status.text = if (opencvReady) "OpenCV 就绪" else "OpenCV 初始化失败，无法继续"

        runButton.isEnabled = opencvReady
        runButton.setOnClickListener { runSamples() }

        findViewById<Button>(R.id.pickButton).setOnClickListener { pickPhotos() }
        findViewById<Button>(R.id.sampleFlowButton).setOnClickListener { importSamples() }
        findViewById<Button>(R.id.recentButton).setOnClickListener {
            startActivity(Intent(this, RecentActivity::class.java))
        }

        findViewById<Button>(R.id.quadButton).setOnClickListener {
            composeAfterEdit = false
            val intent = Intent(this, QuadEditorActivity::class.java)
                .putExtra(QuadEditorActivity.EXTRA_ASSET, "samples/0_0.jpg")
            quadLauncher.launch(intent)
        }
    }

    private fun pickPhotos() {
        picker.launch(androidx.activity.result.PickVisualMediaRequest(
            ActivityResultContracts.PickVisualMedia.ImageOnly))
    }

    private val picker = registerForActivityResult(ActivityResultContracts.PickMultipleVisualMedia(4)) { uris ->
        if (uris.isEmpty()) return@registerForActivityResult
        status.text = "正在导入 ${uris.size} 张照片…"
        Thread {
            PhotoStore.clear(this)
            val files = ArrayList<File>()
            uris.take(4).forEachIndexed { i, uri ->
                PhotoStore.importFromUri(this, uri, i)?.let { files.add(it) }
            }
            runOnUiThread {
                if (files.size < 2) {
                    status.text = "至少需要 2 张照片（成功导入 ${files.size} 张）"
                    return@runOnUiThread
                }
                pendingFiles = files
                status.text = "已导入 ${files.size} 张，先确认底图四角"
                openEditor(files[0])
            }
        }.start()
    }

    /** 内置样张走完整真实流程，用于在没有相册素材时回归验证。 */
    private fun importSamples() {
        status.text = "正在导入内置样张…"
        Thread {
            PhotoStore.clear(this)
            val a = PhotoStore.importFromAsset(this, "samples/0_0.jpg", 0)
            val b = PhotoStore.importFromAsset(this, "samples/0_1.jpg", 1)
            // 第三张用同一份补图的副本，用来验证多图循环本身不会退化
            val c = PhotoStore.importFromAsset(this, "samples/0_1.jpg", 2)
            runOnUiThread {
                if (a == null || b == null) {
                    status.text = "样张导入失败"
                    return@runOnUiThread
                }
                pendingFiles = listOfNotNull(a, b, c)
                openEditor(a)
            }
        }.start()
    }

    private fun openEditor(file: File) {
        composeAfterEdit = true
        val intent = Intent(this, QuadEditorActivity::class.java)
            .putExtra(QuadEditorActivity.EXTRA_FILE, file.absolutePath)
        quadLauncher.launch(intent)
    }

    private val quadLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode != Activity.RESULT_OK) {
            composeAfterEdit = false
            return@registerForActivityResult
        }
        val data = result.data ?: return@registerForActivityResult
        val q = data.getFloatArrayExtra(QuadEditorActivity.EXTRA_QUAD) ?: return@registerForActivityResult
        val w = data.getIntExtra(QuadEditorActivity.EXTRA_IMAGE_W, 0)
        val h = data.getIntExtra(QuadEditorActivity.EXTRA_IMAGE_H, 0)

        if (composeAfterEdit && pendingFiles.size >= 2) {
            composeAfterEdit = false
            compose(pendingFiles, q)
        } else {
            status.text = String.format(
                Locale.US,
                "四角已确认（编辑图 %dx%d）\nTL(%.0f,%.0f) TR(%.0f,%.0f)\nBR(%.0f,%.0f) BL(%.0f,%.0f)",
                w, h, q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]
            )
        }
    }

    /**
     * 用底图四角作为唯一人工输入：其余照片的四角由特征单应推出来，用户只需拖一次。
     */
    private fun compose(files: List<File>, quadValues: FloatArray) {
        status.text = "正在合成 ${files.size} 张…"
        Thread {
            try {
                val base = PhotoStore.load(files[0]) ?: throw IllegalStateException("第 1 张解码失败")
                val baseQuad = toPoints(quadValues)
                val auxes = files.drop(1).mapNotNull { PhotoStore.load(it) }
                if (auxes.isEmpty()) throw IllegalStateException("补图解码失败")
                // 各补图到基图的单应，后续每一轮合成共用
                val cfg = ConfigLoader.load(this)
                val hList = auxes.map { aux ->
                    Geometry.homographyFromFeatures(
                        base, aux, baseQuad, Geometry.insetQuad(aux.cols(), aux.rows()), cfg.align
                    )
                }
                val pipeline = Pipeline(this, cfg)
                val res = pipeline.runMulti("photo", base, auxes, baseQuad, hList)
                runOnUiThread { showSingle(res) }
            } catch (t: Throwable) {
                runOnUiThread { status.text = "合成失败：${t.message ?: t.javaClass.simpleName}" }
            }
        }.start()
    }

    private fun toPoints(values: FloatArray): Array<Point> = arrayOf(
        Point(values[0].toDouble(), values[1].toDouble()),
        Point(values[2].toDouble(), values[3].toDouble()),
        Point(values[4].toDouble(), values[5].toDouble()),
        Point(values[6].toDouble(), values[7].toDouble())
    )

    private fun showSingle(r: PipelineResult) {
        val saved = TaskStore.save(this, r.restored, r.coverage, r.elapsedMs, pendingFiles.size)
        val summary = StringBuilder()
        summary.append(
            String.format(
                Locale.US,
                "%s | %d 张 | 总耗时 %d ms | 覆盖 %.1f%% | 盲区 %.1f%% | 接缝比 %.2f | 接缝跳变 %.2f\n",
                r.id, pendingFiles.size, r.elapsedMs, r.coverage * 100.0, r.blindRatio * 100.0,
                r.seamRatio, r.seamStep
            )
        )
        for ((name, at) in r.steps) {
            summary.append(String.format(Locale.US, "  %-10s @ %d ms\n", name, at))
        }
        if (r.diag.isNotEmpty()) summary.append("  ").append(r.diag).append("\n")
        summary.append(if (saved == null) "已存入最近任务失败\n" else "已存入最近任务\n")
        status.text = summary.toString()

        image0.setImageBitmap(r.restored)
        meta0.text = String.format(Locale.US, "%s 输出 %dx%d", r.id, r.restored.width, r.restored.height)
        image1.setImageDrawable(null)
        meta1.text = ""
    }

    private fun runSamples() {
        runButton.isEnabled = false
        status.text = "运行中，请稍候（首次会加载模型）"

        Thread {
            val pipeline = Pipeline(this, ConfigLoader.load(this))
            val results = ArrayList<PipelineResult>()
            var error: String? = null
            try {
                val samples = pipeline.loadSamples()
                for (sample in samples) {
                    runOnUiThread { status.text = "正在处理 ${sample.id} ..." }
                    results.add(pipeline.run(sample))
                }
            } catch (t: Throwable) {
                error = t.message ?: t.javaClass.simpleName
            }

            runOnUiThread {
                if (error != null) {
                    status.text = "运行失败：$error"
                    runButton.isEnabled = true
                    return@runOnUiThread
                }
                bindResults(results)
                runButton.isEnabled = true
            }
        }.start()
    }

    private fun bindResults(results: List<PipelineResult>) {
        val summary = StringBuilder()
        summary.append("本轮共 ${results.size} 组\n")
        for (r in results) {
            summary.append(
                String.format(
                    Locale.US,
                    "%s | 总耗时 %d ms | 覆盖 %.1f%% | 盲区 %.1f%% | 接缝比 %.2f | 接缝跳变 %.2f\n",
                    r.id, r.elapsedMs, r.coverage * 100.0, r.blindRatio * 100.0, r.seamRatio, r.seamStep
                )
            )
            for ((name, at) in r.steps) {
                summary.append(String.format(Locale.US, "  %-10s @ %d ms\n", name, at))
            }
            if (r.diag.isNotEmpty()) {
                summary.append("  ").append(r.diag).append("\n")
            }
        }
        status.text = summary.toString()

        val first = results.getOrNull(0)
        val second = results.getOrNull(1)
        if (first != null) {
            image0.setImageBitmap(first.restored)
            meta0.text = String.format(Locale.US, "%s 输出 %dx%d", first.id, first.restored.width, first.restored.height)
        }
        if (second != null) {
            image1.setImageBitmap(second.restored)
            meta1.text = String.format(Locale.US, "%s 输出 %dx%d", second.id, second.restored.width, second.restored.height)
        }
    }
}
