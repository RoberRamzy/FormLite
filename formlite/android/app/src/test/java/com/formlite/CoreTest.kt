package com.formlite

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.DataInputStream

class CoreTest {
    @Test fun sessionClosesSegmentsAndFreezesElapsedTime() {
        var clock = 20L
        var wall = 1000L
        val session = Session({ clock }, { wall })
        session.begin()
        clock = 1600
        session.score(0)
        assertNull(session.label)
        clock = 3200
        session.score(0)
        assertEquals(0, session.label)
        clock = 4300
        session.score(1)
        assertNull(session.label)
        clock = 6000
        session.score(1)
        clock = 7000
        session.pause()
        clock = 9000
        assertEquals(6980L, session.elapsed())
        session.begin()
        clock = 10950
        session.lose()
        wall = 500
        val snapshot = session.finish()
        assertEquals(8930L, snapshot.duration)
        assertEquals(1000L, snapshot.started)
        assertEquals(500L, snapshot.ended)
        assertEquals(listOf("side_plank", "squat"), snapshot.segments.map { it.exerciseId })
        assertEquals(2100L, snapshot.segments.sumOf { it.end - it.start })
        assertEquals(snapshot, session.finish())
    }

    @Test fun bodyFeaturesMatchPythonFixtures() {
        val input = javaClass.getResourceAsStream("/motion.bin")
        assumeTrue("Generate the local motion fixture with prepare_test_fixtures.py", input != null)
        DataInputStream(checkNotNull(input)).use { fixture ->
            val tests = fixture.readInt()
            assertEquals(8, tests)
            repeat(tests) {
                val raw = Array(32) { FloatArray(33 * 4) { fixture.readFloat() } }
                val frames = raw.map { frame -> Motion.bodyOnly { i, j -> frame[i * 4 + j] } }
                val expected = FloatArray(32 * 182) { fixture.readFloat() }
                val mask = BooleanArray(32) { fixture.readBoolean() }
                val features = checkNotNull(Motion.features(frames))
                assertArrayEquals(expected, features.values, 1e-5f)
                assertTrue(mask.contentEquals(features.mask))
                raw.forEach { frame -> (0 until 44).forEach { frame[it] = 99f } }
                val changed = raw.map { frame -> Motion.bodyOnly { i, j -> frame[i * 4 + j] } }
                assertArrayEquals(features.values, checkNotNull(Motion.features(changed)).values, 0f)
            }
        }
    }

    @Test fun rejectsMissingPose() {
        assertNull(Motion.features(List(32) { null }))
    }
}
