package com.unblock.app.ui

import android.content.Context
import android.util.AttributeSet
import android.widget.FrameLayout

/** 强制 1:1 的方格容器，画廊网格用它保证每格高度等于宽度。 */
class SquareLayout @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : FrameLayout(context, attrs) {

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        super.onMeasure(widthMeasureSpec, widthMeasureSpec)
        setMeasuredDimension(measuredWidth, measuredWidth)
    }
}
