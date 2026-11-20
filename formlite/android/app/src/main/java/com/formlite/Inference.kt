package com.formlite

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.exp

internal class Inference(private val context: Context) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val model: OrtSession
    private var pose: PoseLandmarker? = null
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    init {
        val metadata = JSONObject(context.assets.open("model.json").bufferedReader().use { it.readText() })
        val ids = metadata.getJSONArray("classes")
        check(Motion.modelClasses.indices.all { ids.getString(it) == Motion.modelClasses[it] })
        val classMapping = metadata.getJSONArray("exercise_ids")
        check(Motion.exerciseIds.indices.all { classMapping.getString(it) == Motion.exerciseIds[it] })
        val schema = metadata.getJSONObject("schema")
        check(schema.getInt("version") == 1 && schema.getInt("window") == Motion.WINDOW &&
            schema.getInt("feature_count") == Motion.WIDTH && metadata.getDouble("score_threshold") == 0.60)
        val bytes = context.assets.open("exercise.onnx").use { it.readBytes() }
        check(digest(bytes) == metadata.getString("onnx_sha256"))
        val poseAsset = context.assets.open(POSE_FILE).use { it.readBytes() }
        check(digest(poseAsset) == metadata.getString("landmarker_sha256"))
        model = OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(2)
            env.createSession(bytes, options)
        }
    }

    fun warmup() {
        try { landmarker() } catch (failure: Throwable) { close(); throw failure }
    }

    fun reset() {
        pose?.close()
        pose = null
    }

    private fun landmarker(): PoseLandmarker = pose ?: PoseLandmarker.createFromOptions(
        context,
        PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath(POSE_FILE).build())
            .setRunningMode(RunningMode.VIDEO).setNumPoses(1).build(),
    ).also { pose = it }

    fun body(bitmap: Bitmap, rotation: Int, timeMs: Long): FloatArray? {
        val rotated = if (rotation == 0) bitmap else Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height, Matrix().apply { postRotate(rotation.toFloat()) }, true,
        )
        try {
            val ratio = 384f / maxOf(rotated.width, rotated.height)
            val w = rotated.width * ratio
            val h = rotated.height * ratio
            // MPImage owns the bitmap and recycles it when closed.
            val square = Bitmap.createBitmap(384, 384, Bitmap.Config.ARGB_8888)
            val image = try {
                Canvas(square).apply {
                    drawColor(Color.BLACK)
                    drawBitmap(rotated, null,
                        RectF((384 - w) / 2, (384 - h) / 2, (384 + w) / 2, (384 + h) / 2), paint)
                }
                BitmapImageBuilder(square).build()
            } catch (failure: Throwable) {
                square.recycle()
                throw failure
            }
            try {
                val points = landmarker().detectForVideo(image, timeMs).landmarks().firstOrNull() ?: return null
                if (points.size != 33) return null
                return Motion.bodyOnly { index, axis ->
                    val point = points[index]
                    when (axis) {
                        0 -> point.x()
                        1 -> point.y()
                        2 -> point.z()
                        else -> point.visibility().orElse(0f)
                    }
                }
            } finally {
                image.close()
            }
        } finally {
            if (rotated !== bitmap) rotated.recycle()
        }
    }

    fun logits(window: Motion.Window): FloatArray {
        val features = ByteBuffer.allocateDirect(window.values.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        features.put(window.values).rewind()
        val mask = ByteBuffer.allocateDirect(window.mask.size).order(ByteOrder.nativeOrder())
        window.mask.forEach { mask.put(if (it) 1.toByte() else 0.toByte()) }
        mask.rewind()
        OnnxTensor.createTensor(env, features, longArrayOf(1, 32, 182)).use { input ->
            OnnxTensor.createTensor(env, mask, longArrayOf(1, 32), OnnxJavaType.BOOL).use { validity ->
                model.run(mapOf("features" to input, "mask" to validity)).use { result ->
                    return (result[0].value as Array<*>).first() as FloatArray
                }
            }
        }
    }

    fun predict(window: Motion.Window): Int? {
        val raw = logits(window)
        if (raw.any { !it.isFinite() }) return null
        val best = raw.indices.maxBy { raw[it] }
        val score = 1 / raw.sumOf { exp((it - raw[best]).toDouble()) }
        return best.takeIf { score >= .60 }
    }

    override fun close() {
        reset()
        model.close()
    }

    companion object {
        private const val POSE_FILE = "pose_landmarker_lite.task"
        private fun digest(bytes: ByteArray) = MessageDigest.getInstance("SHA-256")
            .digest(bytes).joinToString("") { "%02x".format(it.toInt() and 255) }
    }
}

internal class RollingPose {
    private val frames = ArrayDeque<FloatArray?>()
    private var last = -1L
    var missed = 0L
        private set

    fun reset() {
        frames.clear()
        last = -1
        missed = 0
    }

    fun add(slot: Long, body: FloatArray?): Motion.Window? {
        require(slot > last)
        val missing = slot - last - 1
        if (missing >= Motion.WINDOW) frames.clear()
        repeat(missing.coerceAtMost(Motion.WINDOW.toLong()).toInt()) { append(null) }
        append(body)
        missed = if (Motion.usable(body)) 0 else missed + missing + 1
        val previous = last
        last = slot
        if (frames.size == Motion.WINDOW && scheduled(slot, previous)) {
            return Motion.features(frames.toList())
        }
        return null
    }

    fun scheduled(slot: Long, previous: Long) =
        slot >= Motion.WINDOW - 1 && (slot - (Motion.WINDOW - 1)) / Motion.STRIDE >
            ((previous - (Motion.WINDOW - 1)).takeIf { it >= 0 }?.div(Motion.STRIDE) ?: -1)

    private fun append(body: FloatArray?) {
        if (frames.size == Motion.WINDOW) frames.removeFirst()
        frames.addLast(body)
    }
}
