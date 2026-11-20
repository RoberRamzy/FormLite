package com.formlite

import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.widget.FrameLayout
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.doOnLayout
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.LifecycleOwner
import com.facebook.react.ReactPackage
import com.facebook.react.bridge.LifecycleEventListener
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.modules.core.DeviceEventManagerModule
import com.facebook.react.uimanager.SimpleViewManager
import com.facebook.react.uimanager.ThemedReactContext
import com.facebook.react.uimanager.ViewManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.Executors

class RecognitionPackage : ReactPackage {
    private var module: RecognitionModule? = null
    @Synchronized
    private fun recognition(context: ReactApplicationContext) =
        module ?: RecognitionModule(context).also { module = it }

    override fun createNativeModules(context: ReactApplicationContext): List<NativeModule> =
        listOf(recognition(context))

    override fun createViewManagers(context: ReactApplicationContext): List<ViewManager<*, *>> =
        listOf(CameraManager(recognition(context)))
}

private class CameraHost(context: ThemedReactContext) : FrameLayout(context) {
    val preview = PreviewView(context).apply {
        implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        scaleType = PreviewView.ScaleType.FIT_CENTER
    }
    private var needsLayout = false

    init {
        addView(preview, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
    }

    override fun requestLayout() {
        super.requestLayout()
        if (needsLayout || !isAttachedToWindow) return
        needsLayout = true
        post {
            needsLayout = false
            if (isAttachedToWindow && width > 0 && height > 0) {
                measure(MeasureSpec.makeMeasureSpec(width, MeasureSpec.EXACTLY),
                    MeasureSpec.makeMeasureSpec(height, MeasureSpec.EXACTLY))
                layout(left, top, right, bottom)
            }
        }
    }
}

private class CameraManager(private val module: RecognitionModule) : SimpleViewManager<CameraHost>() {
    override fun getName() = "FormLiteCamera"
    override fun createViewInstance(context: ThemedReactContext) = CameraHost(context).also { host ->
        host.addOnAttachStateChangeListener(object : View.OnAttachStateChangeListener {
            override fun onViewAttachedToWindow(view: View) {
                host.doOnLayout { if (host.isAttachedToWindow) module.attach(host.preview) }
            }

            override fun onViewDetachedFromWindow(view: View) {
                module.detach(host.preview)
            }
        })
    }

    override fun onDropViewInstance(view: CameraHost) {
        module.detach(view.preview)
        super.onDropViewInstance(view)
    }
}

class RecognitionModule(private val context: ReactApplicationContext) :
    ReactContextBaseJavaModule(context), LifecycleEventListener {
    private val main = Handler(Looper.getMainLooper())
    private val work = Executors.newSingleThreadExecutor()
    private val scope = CoroutineScope(Dispatchers.Main.immediate + SupervisorJob())
    private val history by lazy { History(context) }
    private lateinit var engine: Inference
    private val rolling = RollingPose()
    private var preview: PreviewView? = null
    private var provider: ProcessCameraProvider? = null
    @Volatile private var cameraGeneration = 0
    @Volatile private var run = 0L
    @Volatile private var origin: Long? = null
    @Volatile private var lastFrame = 0L
    @Volatile private var lastPose = 0L
    private var lastSlot = -1L
    private var modelReady = false
    private var loading = false
    private var rear = true
    private var foreground = true
    private var resumeWhenReady = false
    private var countdown = 0L
    private var phase = "idle"
    private var message: String? = null
    private var error: String? = null
    private var lastPublished: String? = null
    private var current: Session? = null
    private var draft: Workout? = null

    init { context.addLifecycleEventListener(this) }
    override fun getName() = "FormLiteRecognition"

    @ReactMethod fun addListener(event: String) = Unit
    @ReactMethod fun removeListeners(count: Int) = Unit
    @ReactMethod fun insets(promise: Promise) = main.post {
        val root = context.currentActivity?.window?.decorView
        val metrics = root?.let(ViewCompat::getRootWindowInsets)
        val safe = metrics?.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
        val density = context.resources.displayMetrics.density
        promise.resolve(safe?.let {
            JSONObject().put("top", it.top / density).put("left", it.left / density)
                .put("bottom", it.bottom / density).put("right", it.right / density).toString()
        } ?: "null")
    }.let { Unit }
    @ReactMethod fun state(promise: Promise) = main.post { promise.resolve(status().toString()) }.let { Unit }

    @ReactMethod fun openSetup() = main.post {
        if (phase == "idle" || phase == "saved") {
            draft = null
            current = null
            phase = "setup"
        }
        if (phase == "setup") {
            error = null
            loadModel()
            bind()
            update()
        }
    }.let { Unit }

    @ReactMethod fun closeSetup() = main.post {
        if (phase == "setup" || phase == "countdown") {
            phase = if (current == null) "idle" else "paused"
            resumeWhenReady = false
            releaseCamera()
            update()
        }
    }.let { Unit }

    @ReactMethod fun switchCamera() = main.post {
        if (phase == "setup" && !resumeWhenReady) {
            rear = !rear
            error = null
            bind()
            update()
        }
    }.let { Unit }

    @ReactMethod fun start() = main.post {
        if (phase == "setup" && ready()) startCountdown()
    }.let { Unit }

    @ReactMethod fun pause() = main.post { pauseActive() }.let { Unit }

    @ReactMethod fun resume() = main.post {
        if (phase == "paused") {
            phase = "setup"
            error = null
            message = null
            resumeWhenReady = true
            bind()
            update()
        }
    }.let { Unit }

    @ReactMethod fun finish() = main.post {
        if (current != null && phase in setOf("active", "paused", "setup", "countdown")) {
            draft = current!!.finish()
            current = null
            phase = "summary"
            resumeWhenReady = false
            message = null
            error = null
            releaseCamera()
            update()
        }
    }.let { Unit }

    @ReactMethod fun discard() = main.post {
        if (phase == "summary") {
            draft = null
            phase = "idle"
            update()
        }
    }.let { Unit }

    @ReactMethod fun save(promise: Promise) = database(promise) {
        val value = checkNotNull(draft) { "There is no session to save." }
        history.save(value)
        if (draft?.id == value.id) {
            phase = "saved"
            update()
        }
        workout(value).toString()
    }

    @ReactMethod fun sessions(promise: Promise) = database(promise) {
        JSONArray(history.list().map { (row, totals) ->
            saved(row, totals, row.duration - totals.sumOf { it.duration })
        }).toString()
    }

    @ReactMethod fun details(id: String, promise: Promise) = database(promise) {
        val (row, totals, summary) = checkNotNull(history.detail(id)) { "This session has been deleted." }
        saved(row, totals, summary.unknown).toString()
    }

    @ReactMethod fun deleteSession(id: String?, promise: Promise) = database(promise) {
        history.delete(id)
        null
    }

    private fun database(promise: Promise, call: suspend () -> String?) {
        scope.launch {
            try { promise.resolve(call()) }
            catch (error: Exception) { promise.reject("FORM_LITE_STORAGE", error.message, error) }
        }
    }

    fun attach(view: PreviewView) {
        preview = view
        if (phase in CAMERA_PHASES) bind()
    }

    fun detach(view: PreviewView) {
        if (preview !== view) return
        preview = null
        if (phase == "active") pauseActive() else releaseCamera()
    }

    private fun loadModel() {
        if (loading || modelReady) return
        loading = true
        work.execute {
            try {
                engine = Inference(context).also { it.warmup() }
                main.post { modelReady = true; loading = false; update() }
            } catch (failure: Throwable) {
                Log.e("FormLite", "Could not initialize on-device recognition", failure)
                main.post {
                    loading = false
                    error = "Recognition could not load. Reinstall FormLite or retry."
                    update()
                }
            }
        }
    }

    private fun bind() {
        val view = preview ?: return
        val owner = context.currentActivity as? LifecycleOwner ?: return
        if (!foreground || phase !in CAMERA_PHASES) return
        releaseCamera()
        val binding = cameraGeneration
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            if (binding != cameraGeneration || preview !== view || phase !in CAMERA_PHASES) return@addListener
            try {
                val cameraProvider = future.get()
                provider = cameraProvider
                val preferred = if (rear) CameraSelector.DEFAULT_BACK_CAMERA else CameraSelector.DEFAULT_FRONT_CAMERA
                val alternative = if (rear) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
                val selector = if (cameraProvider.hasCamera(preferred)) preferred else alternative
                if (selector != preferred) rear = !rear
                val feed = Preview.Builder().build().also { it.setSurfaceProvider(view.surfaceProvider) }
                val analyzer = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
                analyzer.setAnalyzer(work) { analyze(it, binding) }
                val camera = cameraProvider.bindToLifecycle(owner, selector, feed, analyzer)
                camera.cameraInfo.cameraState.observe(owner) { state ->
                    if (binding == cameraGeneration && state.error != null) {
                        Log.w("FormLite", "CameraX state error ${state.error?.code} on ${view.width}x${view.height}", state.error?.cause)
                        cameraFailure(binding)
                    }
                }
                update()
            } catch (failure: Exception) {
                Log.e("FormLite", "Could not bind the camera", failure)
                cameraFailure(binding)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    private fun analyze(image: ImageProxy, binding: Int) {
        try {
            if (binding != cameraGeneration) return
            val time = SystemClock.elapsedRealtime()
            lastFrame = time
            val start = origin ?: return
            val generation = run
            val slot = (time - start) / Motion.SAMPLE_MS
            if (slot <= lastSlot || slot < 0) return
            val prior = lastSlot
            lastSlot = slot
            val bitmap = image.toBitmap()
            val body = try { engine.body(bitmap, image.imageInfo.rotationDegrees, slot * Motion.SAMPLE_MS) }
                finally { bitmap.recycle() }
            if (binding != cameraGeneration || generation != run) return
            if (Motion.usable(body)) lastPose = time
            val window = rolling.add(slot, body)
            val lost = rolling.missed >= 3 || slot - prior >= 4
            val predicted = window?.let(engine::predict)
            main.post {
                if (generation != run || phase != "active") return@post
                if (lost) {
                    current?.lose()
                    message = "Keep your whole body in view."
                } else if (rolling.scheduled(slot, prior)) {
                    current?.score(predicted)
                    message = if (predicted == null) "Not sure yet" else null
                }
            }
        } catch (failure: Throwable) {
            Log.e("FormLite", "Camera-frame analysis failed", failure)
            val generation = run
            main.post {
                if (phase == "active" && generation == run) {
                    pauseActive("Recognition stopped. Resume to retry.")
                }
            }
        } finally {
            image.close()
        }
    }

    private fun cameraFailure(binding: Int) {
        if (binding != cameraGeneration) return
        val detail = "Camera is unavailable. Close other camera apps and retry."
        if (phase == "active") pauseActive(detail)
        else {
            phase = if (current == null) "setup" else "paused"
            error = detail
            resumeWhenReady = false
            releaseCamera()
            update()
        }
    }

    private fun ready() = modelReady && lastFrame != 0L &&
        SystemClock.elapsedRealtime() - lastFrame <= when (phase) {
            "active" -> 1700L
            "countdown" -> 2500L
            else -> 700L
        }

    private fun startCountdown() {
        Log.d("FormLite", "Starting session countdown")
        phase = "countdown"
        countdown = SystemClock.elapsedRealtime() + 3000
        resumeWhenReady = false
        update()
    }

    private fun beginActive() {
        Log.d("FormLite", "Starting active session time")
        if (current == null) current = Session(SystemClock::elapsedRealtime, System::currentTimeMillis)
        val generation = ++run
        val start = SystemClock.elapsedRealtime()
        current!!.begin()
        lastPose = start
        phase = "active"
        message = null
        work.execute {
            if (generation == run) {
                lastSlot = -1L
                rolling.reset()
                engine.reset()
                if (generation == run) origin = start
            }
        }
    }

    private fun pauseActive(reason: String? = null) {
        if (phase !in setOf("active", "countdown") && !(phase == "setup" && current != null)) return
        Log.d("FormLite", "Pausing camera session from $phase")
        current?.pause()
        phase = if (current == null) "setup" else "paused"
        error = reason
        message = null
        resumeWhenReady = false
        releaseCamera()
        update()
    }

    private fun releaseCamera() {
        val hadFrames = origin != null
        origin = null
        run++
        cameraGeneration++
        lastFrame = 0
        provider?.unbindAll()
        if (hadFrames) work.execute {
            rolling.reset()
            if (::engine.isInitialized) engine.reset()
        }
    }

    private fun status(): JSONObject {
        val label = current?.label?.let { Motion.names[it] }
        return JSONObject().put("phase", phase).put("ready", ready()).put("elapsed", current?.elapsed() ?: 0)
            .put("countdown", if (phase == "countdown") ((countdown - SystemClock.elapsedRealtime() + 999) / 1000)
                .coerceIn(1, 3) else 0)
            .put("label", label ?: JSONObject.NULL).put("message", message ?: JSONObject.NULL)
            .put("error", error ?: JSONObject.NULL).put("canFinish", current != null)
            .put("draft", draft?.let(::workout) ?: JSONObject.NULL)
    }

    private fun update() {
        if (!context.hasActiveReactInstance()) return
        val payload = status().toString()
        if (payload == lastPublished) return
        lastPublished = payload
        context.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
            .emit("FormLiteState", payload)
    }

    private val ticker = object : Runnable {
        override fun run() {
            when (phase) {
                "setup" -> if (resumeWhenReady && ready()) startCountdown()
                "countdown" -> if (SystemClock.elapsedRealtime() >= countdown) {
                    if (lastFrame != 0L && SystemClock.elapsedRealtime() - lastFrame < 5000L) {
                        beginActive()
                    } else {
                        phase = if (current == null) "setup" else "paused"
                        error = "Camera feed was interrupted."
                        releaseCamera()
                    }
                }
                "active" -> {
                    val now = SystemClock.elapsedRealtime()
                    if (now - lastFrame > 1700) {
                        pauseActive("Camera feed was interrupted.")
                    } else if (now - lastPose >= 300) {
                        current?.lose()
                        message = "Keep your whole body in view."
                    }
                }
            }
            if (phase in CAMERA_PHASES || phase == "paused") update()
            main.postDelayed(this, 150)
        }
    }

    override fun initialize() {
        super.initialize()
        main.post(ticker)
    }

    override fun onHostResume() {
        foreground = true
        if (phase == "setup") bind()
        update()
    }

    override fun onHostPause() {
        foreground = false
        if (phase == "setup" && current == null) releaseCamera() else pauseActive()
    }

    override fun onHostDestroy() = onHostPause()

    override fun invalidate() {
        main.removeCallbacks(ticker)
        context.removeLifecycleEventListener(this)
        main.post {
            releaseCamera()
            work.execute { if (::engine.isInitialized) engine.close() }
            work.shutdown()
        }
        scope.cancel()
        super.invalidate()
    }

    companion object {
        private val CAMERA_PHASES = setOf("setup", "countdown", "active")
    }
}

private fun workout(value: Workout): JSONObject {
    val totals = value.segments.groupBy { it.exerciseId }.mapValues { (_, segments) ->
        segments.sumOf { it.end - it.start }
    }
    return renderSummary(value.id, value.started, value.duration, totals, value.duration - totals.values.sum())
}

private fun saved(row: SessionRow, totals: List<ExerciseDuration>, unknown: Long) =
    renderSummary(row.id, row.started, row.duration, totals.associate { it.exerciseId to it.duration }, unknown)

private fun renderSummary(id: String, started: Long, duration: Long, totals: Map<String, Long>, unknown: Long) =
    JSONObject().put("id", id).put("started", started).put("duration", duration).put("unclassified", unknown)
        .put("exercises", JSONArray(Motion.exerciseIds.mapIndexedNotNull { index, key ->
            totals[key]?.let { JSONObject().put("id", key).put("name", Motion.names[index]).put("duration", it) }
        }))
