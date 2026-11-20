package com.formlite

import android.database.sqlite.SQLiteDatabase
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.exp

@RunWith(AndroidJUnit4::class)
class AcceptanceTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext

    @Test fun roomStoresOneSessionAndDeletesItsSegments() = runBlocking {
        val name = "acceptance.db"
        context.deleteDatabase(name)
        val history = History(context, name)
        val session = Workout("test-id", 234, 230, 12000, listOf(
            Segment("a", "side_plank", 2200, 8000), Segment("b", "squat", 10000, 11500),
        ))
        try {
            val invalid = session.copy(id = "invalid", segments = session.segments +
                Segment("overlap", "squat", 4400, 7300))
            assertTrue(runCatching { history.save(invalid) }.exceptionOrNull() is IllegalArgumentException)
            history.save(session)
            history.save(session)
            assertEquals(1, history.list().size)
            val (_, durations, summary) = checkNotNull(history.detail(session.id))
            assertEquals(mapOf("side_plank" to 5800L, "squat" to 1500L), durations.associate { it.exerciseId to it.duration })
            assertEquals(12000L, summary.duration)
            assertEquals(7300L, summary.recognized)
            assertEquals(4700L, summary.unknown)
            history.delete(session.id)
            assertNull(history.detail(session.id))
        } finally {
            history.close()
        }
        SQLiteDatabase.openDatabase(context.getDatabasePath(name).path, null, SQLiteDatabase.OPEN_READONLY).use { db ->
            db.rawQuery("SELECT COUNT(*) FROM recognition_segment", null).use { rows ->
                assertTrue(rows.moveToFirst())
                assertEquals(0, rows.getInt(0))
            }
        }
        context.deleteDatabase(name)
        Unit
    }

    @Test fun replayHeldOutClipsThroughAndroidPoseAndOnnx() {
        val assets = instrumentation.context.assets
        assumeTrue("Generate local replay assets with prepare_test_fixtures.py",
            assets.list("")?.contains("replay.json") == true)
        val manifest = JSONArray(assets.open("replay.json").bufferedReader().use { it.readText() })
        Inference(context).use { engine ->
            val report = JSONArray()
            repeat(manifest.length()) { index ->
                val row = manifest.getJSONObject(index)
                val clip = File.createTempFile("formlite-replay-", ".mp4", context.cacheDir)
                try {
                    assets.open("replay-$index.mp4").use { input -> clip.outputStream().use { input.copyTo(it) } }
                    engine.reset()
                    val decoder = MediaMetadataRetriever()
                    try {
                        decoder.setDataSource(clip.absolutePath)
                        val probabilities = DoubleArray(4)
                        var windows = 0
                        val rolling = RollingPose()
                        val count = row.getInt("sample_frames")
                        for (slot in 0 until count) {
                            val frame = decoder.getFrameAtTime(slot * 100000L + 50000L, MediaMetadataRetriever.OPTION_CLOSEST)
                            val body = frame?.let { try { engine.body(it, 0, slot * 100L) } finally { it.recycle() } }
                            val window = rolling.add(slot.toLong(), body)
                            if (window != null) {
                                val logits = engine.logits(window)
                                val peak = logits.max()
                                val total = logits.sumOf { exp((it - peak).toDouble()) }
                                for (c in logits.indices) probabilities[c] += exp((logits[c] - peak).toDouble()) / total
                                windows++
                            }
                        }
                        assertTrue("No usable pose window for ${row.getString("filename")}", windows > 0)
                        val best = probabilities.indices.maxBy { probabilities[it] }
                        report.put(row.put("android_prediction", Motion.modelClasses[best])
                            .put("android_score", probabilities[best] / windows).put("android_windows", windows))
                    } finally { decoder.release() }
                } finally { clip.delete() }
            }
            Log.i("FormLiteReplay", report.toString())
            repeat(report.length()) { index ->
                val row = report.getJSONObject(index)
                assertEquals(row.getString("desktop_prediction"), row.getString("android_prediction"))
            }
        }
    }

    @Test fun rejectsSyntheticLowConfidenceAndBlankInput() {
        Inference(context).use { engine ->
            val uncertain = Motion.Window(FloatArray(Motion.WINDOW * Motion.WIDTH),
                BooleanArray(Motion.WINDOW) { true })
            assertNull(engine.predict(uncertain))
            val blank = Bitmap.createBitmap(384, 384, Bitmap.Config.ARGB_8888)
            try {
                engine.reset()
                assertNull(engine.body(blank, 0, 0))
            } finally { blank.recycle() }
        }
    }
}
