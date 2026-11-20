package com.formlite

import kotlin.math.hypot

internal object Motion {
    const val WINDOW = 32
    const val STRIDE = 16
    const val WIDTH = 182
    const val SAMPLE_MS = 100L
    val exerciseIds = listOf("side_plank", "squat", "reverse_crunches", "air_jump_rope")
    val modelClasses = listOf("side plank", "squats", "reverse crunches", "air jump rope")
    val names = listOf("Side plank", "Squats", "Reverse crunches", "Air jump rope")
    private const val JOINTS = 22
    private val hips = intArrayOf(12, 13)
    private val shoulders = intArrayOf(0, 1)

    data class Window(val values: FloatArray, val mask: BooleanArray)

    private fun visible(frame: FloatArray, index: Int): Boolean {
        val at = index * 4
        return frame[at + 3] >= .5f && (0..3).all { frame[at + it].isFinite() }
    }

    private fun anchor(frame: FloatArray, ids: IntArray): DoubleArray {
        val total = DoubleArray(3)
        var count = 0
        for (i in ids) if (visible(frame, i)) {
            for (axis in 0..2) total[axis] += frame[i * 4 + axis].toDouble()
            count++
        }
        if (count > 0) for (axis in 0..2) total[axis] = total[axis] / count
        return total
    }

    private fun torso(frame: FloatArray, root: DoubleArray): Double {
        if ((0 until JOINTS).count { visible(frame, it) } < 12 ||
            hips.none { visible(frame, it) } || shoulders.none { visible(frame, it) }) return 0.0
        val top = anchor(frame, shoulders)
        return hypot(root[0] - top[0], root[1] - top[1])
    }

    fun usable(frame: FloatArray?): Boolean =
        frame != null && torso(frame, anchor(frame, hips)) > 1e-4

    fun bodyOnly(read: (Int, Int) -> Float): FloatArray =
        FloatArray(JOINTS * 4) { read(11 + it / 4, it % 4) }

    fun features(frames: List<FloatArray?>): Window? {
        require(frames.size <= WINDOW)
        val roots = frames.map { frame -> frame?.let { anchor(it, hips) } ?: DoubleArray(3) }
        val lengths = frames.mapIndexed { index, frame -> frame?.let { torso(it, roots[index]) } ?: 0.0 }
        val valid = BooleanArray(WINDOW) { it < frames.size && lengths[it] > 1e-4 }
        val usableLengths = lengths.filter { it > 1e-4 }.sorted()
        if (usableLengths.size < 20) return null
        val scale = (usableLengths[usableLengths.size / 2] +
            usableLengths[(usableLengths.size - 1) / 2]) / 2.0
        val origin = roots[valid.indexOfFirst { it }]
        val values = FloatArray(WINDOW * WIDTH)
        for (i in frames.indices) {
            if (!valid[i]) continue
            val frame = checkNotNull(frames[i])
            val base = i * WIDTH
            val previous = base - WIDTH
            for (joint in 0 until JOINTS) {
                if (!visible(frame, joint)) continue
                val channel = joint * 8
                values[base + channel + 6] = 1f
                val hasDelta = i > 0 && values[previous + channel + 6] == 1f
                if (hasDelta) values[base + channel + 7] = 1f
                for (axis in 0..2) {
                    val position = (frame[joint * 4 + axis] - roots[i][axis]) / scale
                    values[base + channel + axis] = position.toFloat()
                    if (hasDelta) {
                        val old = (checkNotNull(frames[i - 1])[joint * 4 + axis] - roots[i - 1][axis]) / scale
                        values[base + channel + 3 + axis] = (position - old).toFloat()
                    }
                }
            }
            val rootChannel = JOINTS * 8
            val hasRootDelta = i > 0 && valid[i - 1]
            for (axis in 0..1) {
                values[base + rootChannel + axis] = ((roots[i][axis] - origin[axis]) / scale).toFloat()
                if (hasRootDelta) {
                    values[base + rootChannel + 2 + axis] =
                        ((roots[i][axis] - roots[i - 1][axis]) / scale).toFloat()
                }
            }
            values[base + rootChannel + 4] = 1f
            if (hasRootDelta) values[base + rootChannel + 5] = 1f
        }
        return Window(values, valid)
    }
}
