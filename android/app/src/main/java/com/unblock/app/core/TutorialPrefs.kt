package com.unblock.app.core

import android.content.Context

/**
 * 教程进度。0=未开始，1=首页已引导，2=四角页已引导，3=全部完成。
 * 只记录进度，不放任何业务数据。
 */
object TutorialPrefs {

    private const val FILE = "tutorial"
    private const val KEY_STEP = "step"
    private const val KEY_PICK_HINT = "pick_hint"
    private const val STEP_DONE = 3

    private fun prefs(context: Context) =
        context.getSharedPreferences(FILE, Context.MODE_PRIVATE)

    /** 已经引导到第几步。 */
    fun step(context: Context): Int = prefs(context).getInt(KEY_STEP, 0)

    fun mark(context: Context, step: Int) {
        prefs(context).edit().putInt(KEY_STEP, step).apply()
    }

    fun isDone(context: Context): Boolean = step(context) >= STEP_DONE

    fun pickHintShown(context: Context): Boolean =
        prefs(context).getBoolean(KEY_PICK_HINT, false)

    fun markPickHint(context: Context) {
        prefs(context).edit().putBoolean(KEY_PICK_HINT, true).apply()
    }
}
