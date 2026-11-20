# FormLite Android

FormLite recognizes side plank, squats, reverse crunches, and air jump rope through the Android camera. MediaPipe and the trained classifier run on the phone. Finish a session to review detected time and unclassified time, then save it to local history or discard it.

## Build and install

Use Node 22 or later, a full JDK 17, and Android SDK 36 with Build Tools 36.0.0, NDK 27.1.12297006, CMake 3.22.1, and platform tools. Set `JAVA_HOME` and `ANDROID_HOME` to their installation directories, then run from this project directory.

```bash
npm ci
cd android
./gradlew :app:assembleRelease
"$ANDROID_HOME/platform-tools/adb" install -r app/build/outputs/apk/release/app-release.apk
```

The release APK uses the Android debug key for local installation and includes the JavaScript bundle, classifier, pose model, and seed database. Open **FormLite**, tap **Start recognition**, grant camera access, and start a session with your body in frame. The release app needs neither Metro nor the laptop.

## Verify and regenerate artifacts

```bash
npm test -- --runInBand
npx tsc --noEmit
cd android
./gradlew :app:testReleaseUnitTest :app:connectedReleaseAndroidTest
```

Use an x86_64 Android API 36 emulator for the connected tests. Use the app offline to check permission, countdown, pause, resume, review, save, and history. The replay instrumentation test processes four local holdout clips through Android MediaPipe and ONNX and logs the decisions under `FormLiteReplay`. Replay and Python/Kotlin parity tests are skipped until the local fixtures are generated. Phone camera accuracy, latency, and battery performance need a physical-device test.

The model export script uses the existing desktop checkpoint and validates its scores with `onnx` and `onnxruntime`. The fixture script uses the existing training workspace and MediaPipe task to regenerate the Kotlin parity and Android replay fixtures. The SQLite seed is generated from the reference schema with the model’s exercise catalog:

```bash
python3 scripts/export_model.py
python3 scripts/prepare_test_fixtures.py
python3 scripts/seed_database.py ../exercise-recognition-design/schema.sql android/app/src/main/assets/formlite.db
```

The database stores session dates, active durations, and recognized intervals. Camera frames and landmarks stay in memory. The summary attributes time only after a stable label is accepted, so countdown and pauses are excluded and time spent acquiring or uncertain is unclassified.

`scripts/emulator_smoke.py` runs the offline UI smoke test on a disposable emulator and captures reference-sized and small-screen images. The script clears FormLite data on the supplied emulator before testing.

```bash
python3 scripts/emulator_smoke.py --adb "$ANDROID_HOME/platform-tools/adb" --serial emulator-5554
```
