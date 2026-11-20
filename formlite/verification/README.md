# Android acceptance results

- PyTorch and ONNX predicted the same class on all 128 held-out feature windows. The largest absolute logit error was 0.00001144 against the 0.0001 tolerance. See [export validation](../model-validation.json).
- Kotlin matched the Python feature values and validity masks on eight fixtures covering all four exercises and missing frames. Replacing the facial coordinates and visibility values left the Kotlin features identical.
- Android MediaPipe and ONNX matched the desktop prediction on one held-out clip per exercise. The Android tests also rejected a blank frame and a synthetic low-confidence window. See [Android instrumentation](instrumentation.txt). Per-clip replay results are stored locally in `android-replay.json`.
- The API 36 emulator exercised airplane-mode launch, camera denial and Settings, front and rear previews, countdown, no pose, pause, resume, backgrounding, review, save, restart, deletion and discard. The Room test covered idempotent save, exact duration reconciliation, overlapping segments and cascading deletion. See [emulator smoke test](emulator-smoke.txt) and [additional emulator checks](emulator-extra.txt).
- All eight screens were reviewed at 390 × 844. At 340 × 720, Home scrolls to reveal the navigation buttons, and History fits the viewport. See [reference Home](home-reference.png), [small Home](home-small-scrolled.png) and [small History](history-small.png).

Physical-phone camera accuracy, latency and battery measurements were deferred. The low-confidence case used a synthetic window on Android and a React Native rendering test.
