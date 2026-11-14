package com.unblock.app.core

import android.content.Context

/**
 * 管线单例：ONNX 模型只加载一次。
 * 早期每次合成都会 new 一个 Pipeline，等于每次重建一个分割会话，模型加载耗时和内存都被重复支付。
 */
object PipelineProvider {

    @Volatile
    private var instance: Pipeline? = null

    fun get(context: Context): Pipeline {
        instance?.let { return it }
        return synchronized(this) {
            instance ?: Pipeline(
                context.applicationContext,
                ConfigLoader.load(context.applicationContext)
            ).also { instance = it }
        }
    }

    fun close() {
        synchronized(this) {
            instance?.close()
            instance = null
        }
    }
}
