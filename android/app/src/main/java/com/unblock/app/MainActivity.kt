package com.unblock.app

import android.animation.ValueAnimator
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.core.view.WindowInsetsControllerCompat
import androidx.recyclerview.widget.GridLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.unblock.app.core.ConfigLoader
import com.unblock.app.core.Geometry
import com.unblock.app.core.PhotoStore
import com.unblock.app.core.Pipeline
import com.unblock.app.core.PipelineProvider
import com.unblock.app.core.TaskItem
import com.unblock.app.core.TaskStore
import com.unblock.app.ui.GalleryAdapter
import com.unblock.app.ui.QuadEditorActivity
import com.unblock.app.ui.ResultActivity
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import org.opencv.core.Point
import java.io.File
import java.util.concurrent.CancellationException

/**
 * 首页即画廊：第一格是导入入口，之后是历次合成结果。
 * 选照片、拍照、四角确认、合成、入库都围绕同一轮「会话」推进，避免半路丢素材。
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val GRID_SPAN = 3
        private const val MIN_PHOTOS = 2
        private const val MAX_PHOTOS = 4
        private const val PROVIDER_SUFFIX = ".fileprovider"
        private const val PROGRESS_CEILING = 90
        private const val BUSY_THUMB_SIDE = 320
        private const val TAG = "Unblock"
    }

    private lateinit var adapter: GalleryAdapter
    private lateinit var busyOverlay: View
    private lateinit var busyThumb: ImageView
    private lateinit var busyTitle: TextView
    private lateinit var busyBar: ProgressBar
    private lateinit var busyHint: TextView
    private lateinit var failOverlay: View
    private lateinit var failMessage: TextView

    private val uiConfig by lazy { ConfigLoader.loadUi(this) }

    private var progressAnimator: ValueAnimator? = null

    /** 当前一轮会话：目录 + 已加入的照片。 */
    private var sessionDir: File? = null
    private val photos = ArrayList<File>()

    private var editorTarget: File? = null
    private var captureTarget: File? = null
    private var lastQuad: FloatArray? = null

    private var activePipeline: Pipeline? = null
    @Volatile
    private var abandoned = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        window.statusBarColor = getColor(R.color.paper)
        WindowInsetsControllerCompat(window, window.decorView).isAppearanceLightStatusBars = true

        busyOverlay = findViewById(R.id.busyOverlay)
        busyThumb = findViewById(R.id.busyThumb)
        busyTitle = findViewById(R.id.busyTitle)
        busyBar = findViewById(R.id.busyBar)
        busyHint = findViewById(R.id.busyHint)
        failOverlay = findViewById(R.id.failOverlay)
        failMessage = findViewById(R.id.failMessage)

        val gallery = findViewById<RecyclerView>(R.id.gallery)
        gallery.layoutManager = GridLayoutManager(this, GRID_SPAN)
        gallery.setHasFixedSize(true)
        adapter = GalleryAdapter(
            onAdd = { showImportSheet() },
            onOpen = { openResult(it) },
            onDelete = { confirmDelete(it) }
        )
        gallery.adapter = adapter

        findViewById<View>(R.id.busyCancel).setOnClickListener { cancelCompose() }
        findViewById<View>(R.id.retryButton).setOnClickListener { retryCompose() }
        findViewById<View>(R.id.swapButton).setOnClickListener {
            failOverlay.visibility = View.GONE
            discardSession()
            showImportSheet()
        }
        // 开发回归入口：长按标题跑内置样张，正式界面不出现任何按钮
        findViewById<View>(R.id.homeTitle).setOnLongClickListener {
            confirmSamples()
            true
        }

        try {
            OpenCVLoader.initLocal()
        } catch (t: Throwable) {
            Log.w(TAG, "OpenCV 初始化失败: ${t.message}")
        }

        refreshTasks()
    }

    override fun onDestroy() {
        super.onDestroy()
        progressAnimator?.cancel()
    }

    // ---------- 首页数据 ----------

    private fun refreshTasks() {
        Thread {
            val list = TaskStore.list(this)
            runOnUiThread { adapter.submit(list) }
        }.start()
    }

    private fun openResult(item: TaskItem) {
        startActivity(
            Intent(this, ResultActivity::class.java)
                .putExtra(ResultActivity.EXTRA_FILE, item.image.absolutePath)
                .putExtra(ResultActivity.EXTRA_SOURCES, item.sources)
                .putExtra(ResultActivity.EXTRA_TIME, item.time)
        )
    }

    private fun confirmDelete(item: TaskItem) {
        val dialog = AlertDialog.Builder(this)
            .setTitle(R.string.dialog_delete_title)
            .setMessage(R.string.dialog_delete_message)
            .setNegativeButton(R.string.action_cancel, null)
            .setPositiveButton(R.string.action_delete) { _, _ ->
                Thread {
                    TaskStore.delete(this, item)
                    runOnUiThread { refreshTasks() }
                }.start()
            }
            .create()
        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                .setTextColor(getColor(R.color.destructive))
        }
        dialog.show()
    }

    // ---------- 导入 ----------

    private fun showImportSheet() {
        val options = arrayOf(
            getString(R.string.action_take_photo),
            getString(R.string.action_pick_album)
        )
        AlertDialog.Builder(this)
            .setTitle(R.string.title_import)
            .setItems(options) { _, which ->
                if (which == 0) startCamera() else startAlbum()
            }
            .setNegativeButton(R.string.action_cancel, null)
            .show()
    }

    private fun startAlbum() {
        picker.launch(
            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
        )
    }

    private val picker = registerForActivityResult(
        ActivityResultContracts.PickMultipleVisualMedia(MAX_PHOTOS)
    ) { uris ->
        if (uris.isNullOrEmpty()) return@registerForActivityResult
        val dir = ensureSession() ?: return@registerForActivityResult
        val room = MAX_PHOTOS - photos.size
        showBusy(getString(R.string.busy_importing), indeterminate = true)
        Thread {
            val added = ArrayList<File>()
            for (uri in uris.take(room)) {
                val index = PhotoStore.nextIndex(dir)
                PhotoStore.importFromUri(this, uri, dir, index)?.let { added.add(it) }
            }
            runOnUiThread {
                hideBusy()
                if (added.isEmpty()) {
                    showFailure(getString(R.string.fail_import))
                    return@runOnUiThread
                }
                photos.addAll(added)
                onPhotosReady()
            }
        }.start()
    }

    private fun startCamera() {
        val dir = ensureSession() ?: return
        val target = PhotoStore.captureTarget(dir, PhotoStore.nextIndex(dir))
        try {
            val uri = FileProvider.getUriForFile(this, packageName + PROVIDER_SUFFIX, target)
            captureTarget = target
            camera.launch(uri)
        } catch (t: Throwable) {
            Log.w(TAG, "相机启动失败: ${t.message}")
            showFailure(getString(R.string.fail_import))
        }
    }

    private val camera = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        val target = captureTarget
        captureTarget = null
        if (!ok || target == null || !target.exists() || target.length() == 0L) {
            target?.delete()
            return@registerForActivityResult
        }
        showBusy(getString(R.string.busy_importing), indeterminate = true)
        Thread {
            val normalized = PhotoStore.normalizeCaptured(target) ?: target
            runOnUiThread {
                hideBusy()
                photos.add(normalized)
                onPhotosReady()
            }
        }.start()
    }

    private fun onPhotosReady() {
        if (photos.size >= MIN_PHOTOS) {
            openEditor(photos[0])
        } else {
            promptAddMore()
        }
    }

    private fun promptAddMore() {
        AlertDialog.Builder(this)
            .setTitle(R.string.dialog_need_more_title)
            .setMessage(R.string.dialog_need_more_message)
            .setPositiveButton(R.string.action_take_photo) { _, _ -> startCamera() }
            .setNegativeButton(R.string.action_pick_album) { _, _ -> startAlbum() }
            .setNeutralButton(R.string.action_cancel) { _, _ -> discardSession() }
            .show()
    }

    // ---------- 四角 ----------

    private fun openEditor(file: File) {
        editorTarget = file
        quadLauncher.launch(
            Intent(this, QuadEditorActivity::class.java)
                .putExtra(QuadEditorActivity.EXTRA_FILE, file.absolutePath)
        )
    }

    private val quadLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val target = editorTarget
        if (result.resultCode != Activity.RESULT_OK || target == null) {
            discardSession()
            return@registerForActivityResult
        }
        val data = result.data ?: return@registerForActivityResult
        val quad = data.getFloatArrayExtra(QuadEditorActivity.EXTRA_QUAD)
            ?: return@registerForActivityResult
        val editorWidth = data.getIntExtra(QuadEditorActivity.EXTRA_IMAGE_W, 0)
        val scaled = scaleQuadToFile(quad, editorWidth, target) ?: run {
            showFailure(getString(R.string.fail_decode))
            return@registerForActivityResult
        }
        lastQuad = scaled
        compose(scaled)
    }

    /**
     * 四角坐标是在编辑器分辨率下给的，合成用的是原始缓存图。
     * 两处分辨率可能不同，必须按实际宽高换算，不能直接拿来用。
     */
    private fun scaleQuadToFile(quad: FloatArray, editorWidth: Int, file: File): FloatArray? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
        if (editorWidth <= 0) return quad.copyOf()
        val scale = bounds.outWidth.toFloat() / editorWidth
        return if (scale == 1f) quad.copyOf() else FloatArray(quad.size) { quad[it] * scale }
    }

    // ---------- 合成 ----------

    private fun compose(quad: FloatArray) {
        val files = photos.toList()
        if (files.size < MIN_PHOTOS) {
            promptAddMore()
            return
        }
        abandoned = false
        showBusy(getString(R.string.busy_composing, files.size), indeterminate = false)

        val pipeline = PipelineProvider.get(this)
        activePipeline = pipeline

        Thread {
            var base: Mat? = null
            val auxes = ArrayList<Mat>()
            try {
                val cfg = ConfigLoader.load(this)
                val baseMat = PhotoStore.load(files[0])
                    ?: throw IllegalStateException(getString(R.string.fail_decode))
                base = baseMat
                for (i in 1 until files.size) {
                    PhotoStore.load(files[i])?.let { auxes.add(it) }
                }
                if (auxes.isEmpty()) throw IllegalStateException(getString(R.string.fail_decode))

                val baseQuad = toPoints(quad)
                val hList = auxes.map { aux ->
                    Geometry.homographyFromFeatures(
                        baseMat, aux, baseQuad,
                        Geometry.insetQuad(aux.cols(), aux.rows()), cfg.align
                    )
                }
                val result = pipeline.runMulti("photo", baseMat, auxes, baseQuad, hList)
                if (abandoned) return@Thread

                val saved = TaskStore.save(
                    this, result.restored, result.coverage, result.elapsedMs, files.size
                )
                runOnUiThread {
                    hideBusy()
                    if (saved == null) {
                        showFailure(getString(R.string.fail_save))
                    } else {
                        discardSession()
                        refreshTasks()
                        openResult(saved)
                    }
                }
            } catch (c: CancellationException) {
                runOnUiThread { hideBusy() }
            } catch (t: Throwable) {
                Log.w(TAG, "合成失败: ${t.message}")
                runOnUiThread {
                    hideBusy()
                    showFailure(t.message ?: getString(R.string.fail_message))
                }
            } finally {
                base?.release()
                auxes.forEach { it.release() }
                activePipeline = null
            }
        }.start()
    }

    private fun retryCompose() {
        failOverlay.visibility = View.GONE
        val quad = lastQuad
        if (quad != null) compose(quad) else onPhotosReady()
    }

    private fun cancelCompose() {
        abandoned = true
        activePipeline?.cancel()
        hideBusy()
    }

    private fun toPoints(values: FloatArray): Array<Point> = arrayOf(
        Point(values[0].toDouble(), values[1].toDouble()),
        Point(values[2].toDouble(), values[3].toDouble()),
        Point(values[4].toDouble(), values[5].toDouble()),
        Point(values[6].toDouble(), values[7].toDouble())
    )

    // ---------- 遮罩 ----------

    private fun showBusy(message: String, indeterminate: Boolean) {
        busyTitle.text = message
        busyThumb.setImageBitmap(baseThumbnail())
        busyOverlay.visibility = View.VISIBLE
        progressAnimator?.cancel()
        busyBar.isIndeterminate = indeterminate
        // 预估耗时只在合成时才有意义，导入阶段不显示
        busyHint.visibility = if (indeterminate) View.GONE else View.VISIBLE
        if (!indeterminate) {
            val estimateMs = uiConfig.estimateSeconds * 1000L
            busyHint.text = getString(R.string.busy_approx, uiConfig.estimateSeconds)
            busyBar.progress = 0
            progressAnimator = ValueAnimator.ofInt(0, PROGRESS_CEILING).apply {
                duration = estimateMs
                addUpdateListener { busyBar.progress = it.animatedValue as Int }
                start()
            }
        }
    }

    private fun hideBusy() {
        progressAnimator?.cancel()
        progressAnimator = null
        busyOverlay.visibility = View.GONE
    }

    private fun baseThumbnail(): Bitmap? {
        val file = photos.firstOrNull() ?: return null
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        if (bounds.outWidth <= 0) return null
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / sample > BUSY_THUMB_SIDE) sample *= 2
        return BitmapFactory.decodeFile(
            file.absolutePath,
            BitmapFactory.Options().apply { inSampleSize = sample }
        )
    }

    private fun showFailure(message: String) {
        hideBusy()
        failMessage.text = message
        failOverlay.visibility = View.VISIBLE
    }

    // ---------- 会话 ----------

    private fun ensureSession(): File? {
        sessionDir?.let { return it }
        return try {
            PhotoStore.newSession(this).also { sessionDir = it }
        } catch (t: Throwable) {
            Log.w(TAG, "创建会话失败: ${t.message}")
            null
        }
    }

    private fun discardSession() {
        PhotoStore.discardSession(sessionDir)
        sessionDir = null
        photos.clear()
        lastQuad = null
        editorTarget = null
    }

    // ---------- 开发回归 ----------

    private fun confirmSamples() {
        AlertDialog.Builder(this)
            .setTitle(R.string.samples_title)
            .setMessage(R.string.samples_message)
            .setNegativeButton(R.string.action_cancel, null)
            .setPositiveButton(R.string.action_run) { _, _ -> runSamples() }
            .show()
    }

    private fun runSamples() {
        showBusy(getString(R.string.samples_title), indeterminate = true)
        Thread {
            try {
                val pipeline = PipelineProvider.get(this)
                val samples = pipeline.loadSamples()
                var count = 0
                for (sample in samples) {
                    val result = pipeline.run(sample)
                    if (TaskStore.save(
                            this, result.restored, result.coverage, result.elapsedMs, 2
                        ) != null
                    ) {
                        count++
                    }
                }
                runOnUiThread {
                    hideBusy()
                    refreshTasks()
                    Toast.makeText(
                        this, getString(R.string.samples_done, count), Toast.LENGTH_SHORT
                    ).show()
                }
            } catch (t: Throwable) {
                Log.w(TAG, "样张运行失败: ${t.message}")
                runOnUiThread {
                    hideBusy()
                    showFailure(t.message ?: getString(R.string.fail_message))
                }
            }
        }.start()
    }
}
