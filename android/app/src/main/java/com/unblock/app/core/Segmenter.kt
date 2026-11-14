package com.unblock.app.core

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/**
 * YOLOv8-seg 人物分割（ONNX Runtime，纯端上推理）。
 * 输入 letterbox 到 640x640，输出检测框与 32 通道 mask 原型，重构后用 0.5 阈值二值化。
 */
class Segmenter(
    context: Context,
    modelAsset: String,
    private val inputSize: Int = 640,
    private val confThresh: Float = 0.1f,
    private val iouThresh: Float = 0.45f
) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    init {
        val bytes = context.assets.open(modelAsset).use { it.readBytes() }
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(4)
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        session = env.createSession(bytes, opts)
        inputName = session.inputNames.first()
    }

    /** 返回与原图同尺寸的 0/255 人物掩码。 */
    fun personMask(img: Mat): Mat {
        val h = img.rows()
        val w = img.cols()
        val scale = min(inputSize.toDouble() / w, inputSize.toDouble() / h)
        val newW = (w * scale).toInt()
        val newH = (h * scale).toInt()
        val padX = (inputSize - newW) / 2
        val padY = (inputSize - newH) / 2

        val resized = Mat()
        Imgproc.resize(img, resized, Size(newW.toDouble(), newH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        val canvas = Mat(inputSize, inputSize, CvType.CV_8UC3, Scalar(114.0, 114.0, 114.0))
        val roi = canvas.submat(Rect(padX, padY, newW, newH))
        resized.copyTo(roi)

        // BGR -> RGB，再转 float 并归一化到 0..1，最后排成 CHW
        val rgb = Mat()
        Imgproc.cvtColor(canvas, rgb, Imgproc.COLOR_BGR2RGB)
        val rgbF = Mat()
        rgb.convertTo(rgbF, CvType.CV_32FC3, 1.0 / 255.0)
        val data = FloatArray(3 * inputSize * inputSize)
        rgbF.get(0, 0, data)
        val chw = FloatArray(3 * inputSize * inputSize)
        val plane = inputSize * inputSize
        for (c in 0 until 3) {
            for (i in 0 until plane) {
                chw[c * plane + i] = data[i * 3 + c]
            }
        }

        val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), shape)
        val outputs = session.run(mapOf(inputName to inputTensor))

        val detTensor = outputs.get(0) as OnnxTensor
        val protoTensor = outputs.get(1) as OnnxTensor
        val det = detTensor.floatBuffer
        val proto = protoTensor.floatBuffer

        val numCandidates = 8400
        val numChannels = 116
        val numClasses = 80
        val numMasks = 32
        val protoH = inputSize / 4
        val protoW = inputSize / 4

        // 解析候选框
        val boxes = ArrayList<FloatArray>()   // cx, cy, w, h
        val scores = ArrayList<Float>()
        val coeffs = ArrayList<FloatArray>()
        for (i in 0 until numCandidates) {
            val score = det.get(4 * numCandidates + i) // person 类索引 0 -> 通道 4
            if (score < confThresh) continue
            val cx = det.get(0 * numCandidates + i)
            val cy = det.get(1 * numCandidates + i)
            val bw = det.get(2 * numCandidates + i)
            val bh = det.get(3 * numCandidates + i)
            val coeff = FloatArray(numMasks)
            for (k in 0 until numMasks) {
                coeff[k] = det.get((4 + numClasses + k) * numCandidates + i)
            }
            boxes.add(floatArrayOf(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
            scores.add(score)
            coeffs.add(coeff)
        }

        val keep = nms(boxes, scores, iouThresh)
        val mask = Mat.zeros(h, w, CvType.CV_8UC1)
        if (keep.isEmpty()) {
            outputs.close()
            inputTensor.close()
            return mask
        }

        val protoPlane = protoH * protoW
        for (idx in keep) {
            val coeff = coeffs[idx]
            val box = boxes[idx]
            val raw = FloatArray(protoPlane)
            for (y in 0 until protoH) {
                for (x in 0 until protoW) {
                    var acc = 0.0f
                    for (k in 0 until numMasks) {
                        acc += coeff[k] * proto.get(k * protoPlane + y * protoW + x)
                    }
                    raw[y * protoW + x] = sigmoid(acc)
                }
            }
            val protoMask = Mat(protoH, protoW, CvType.CV_32FC1)
            protoMask.put(0, 0, raw)

            // 从 letterbox 坐标退回原图坐标
            val x0 = max(0.0, (box[0] - padX) / scale)
            val y0 = max(0.0, (box[1] - padY) / scale)
            val x1 = min(w.toDouble(), (box[2] - padX) / scale)
            val y1 = min(h.toDouble(), (box[3] - padY) / scale)
            if (x1 <= x0 || y1 <= y0) continue

            val boxW = (x1 - x0).toInt()
            val boxH = (y1 - y0).toInt()
            // proto 栅格以 letterbox 坐标系为基准：先按 scale 缩放，再补上 padding，最后除以 4
            val mx0 = max(0, ((x0 * scale + padX) / 4.0).toInt())
            val my0 = max(0, ((y0 * scale + padY) / 4.0).toInt())
            val mx1 = min(protoW, ((x1 * scale + padX) / 4.0).toInt())
            val my1 = min(protoH, ((y1 * scale + padY) / 4.0).toInt())
            if (mx1 <= mx0 || my1 <= my0) continue

            val crop = protoMask.submat(Rect(mx0, my0, mx1 - mx0, my1 - my0))
            val full = Mat()
            Imgproc.resize(crop, full, Size(boxW.toDouble(), boxH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)

            val bin = Mat()
            Imgproc.threshold(full, bin, 0.5, 255.0, Imgproc.THRESH_BINARY)
            val bin8 = Mat()
            bin.convertTo(bin8, CvType.CV_8UC1)
            val target = mask.submat(Rect(x0.toInt(), y0.toInt(), boxW, boxH))
            Core.bitwise_or(target, bin8, target)
        }

        outputs.close()
        inputTensor.close()
        return mask
    }

    private fun sigmoid(x: Float): Float = 1.0f / (1.0f + Math.exp(-x.toDouble()).toFloat())

    /** 释放 ONNX 会话。 */
    fun close() {
        session.close()
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val ix0 = max(a[0], b[0])
        val iy0 = max(a[1], b[1])
        val ix1 = min(a[2], b[2])
        val iy1 = min(a[3], b[3])
        val iw = max(0.0f, ix1 - ix0)
        val ih = max(0.0f, iy1 - iy0)
        val inter = iw * ih
        val areaA = (a[2] - a[0]) * (a[3] - a[1])
        val areaB = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (areaA + areaB - inter + 1e-6f)
    }

    private fun nms(boxes: List<FloatArray>, scores: List<Float>, thresh: Float): List<Int> {
        val order = scores.indices.sortedByDescending { scores[it] }.toMutableList()
        val keep = ArrayList<Int>()
        while (order.isNotEmpty()) {
            val i = order.removeAt(0)
            keep.add(i)
            val remain = ArrayList<Int>()
            for (j in order) {
                if (iou(boxes[i], boxes[j]) < thresh) remain.add(j)
            }
            order.clear()
            order.addAll(remain)
        }
        return keep
    }
}
