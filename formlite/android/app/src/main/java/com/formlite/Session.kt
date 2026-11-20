package com.formlite

import java.util.UUID

internal data class Segment(val id: String, val exerciseId: String, val start: Long, val end: Long)
internal data class Workout(
    val id: String, val started: Long, val ended: Long, val duration: Long, val segments: List<Segment>,
)

internal class Session(private val now: () -> Long, private val utc: () -> Long) {
    private var id: String? = null
    private var started = 0L
    private var interval: Long? = null
    private var accumulated = 0L
    private var labelStart = 0L
    private var pending: Int? = null
    private val segments = mutableListOf<Segment>()
    var label: Int? = null
        private set

    fun begin() {
        if (id == null) {
            id = UUID.randomUUID().toString()
            started = utc()
        }
        check(interval == null)
        interval = now()
    }

    fun elapsed(): Long = accumulated + (interval?.let { (now() - it).coerceAtLeast(0) } ?: 0)

    fun score(index: Int?) {
        if (interval == null) return
        if (index == null) {
            lose()
            return
        }
        if (label != null && label != index) closeLabel(elapsed())
        if (label == null && pending == index) {
            label = index
            labelStart = elapsed()
        }
        pending = index
    }

    fun lose() {
        closeLabel(elapsed())
        pending = null
    }

    private fun closeLabel(at: Long) {
        val previous = label
        if (previous != null && at > labelStart) {
            segments.add(Segment(UUID.randomUUID().toString(), Motion.exerciseIds[previous], labelStart, at))
        }
        label = null
    }

    fun pause() {
        if (interval == null) return
        val endedAt = elapsed()
        closeLabel(endedAt)
        pending = null
        accumulated = endedAt
        interval = null
    }

    fun finish(): Workout {
        val sessionId = checkNotNull(id)
        pause()
        return Workout(sessionId, started, utc(), accumulated, segments.toList())
    }
}
