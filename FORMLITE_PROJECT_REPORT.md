# FormLite exercise recognition project report

**Project status as of 20 November 2026**

Paths shown in code formatting identify local run artifacts. Per-clip data, cached features, and build outputs are generated locally.

FormLite recognizes four exercises from body posture and motion. The completed pipeline covers archive extraction, exploratory data analysis (EDA), clip selection and cleaning, MediaPipe landmark extraction, PyTorch CUDA training, a local MP4 upload demo, and a standalone Android app. The trained model achieved **98.06% clip-level accuracy and 97.34% macro-F1 on 464 locally held-out clips**. The Android build and emulator checks are complete. Physical-phone camera accuracy, latency, and battery use remain unmeasured.

```mermaid
flowchart LR
    A["QEVD Part 1 MP4s and full annotation JSON"] --> B["EDA and filename matching"]
    B --> C["Four-class selection and video cleaning"]
    C --> D["Grouped train, validation, and test split"]
    D --> E["MediaPipe body landmarks and motion windows"]
    E --> F["Temporal CNN trained on CUDA"]
    F --> G["Local MP4 upload demo on CUDA"]
    F --> H["ONNX export"]
    H --> I["Offline Android camera recognition and Room history"]
```

## 1. Dataset and EDA

The local source was a three-volume QEVD-FIT-300K Part 1 archive, consisting of `.z01`, `.z02`, and `.zip` files. The extracted directory contains 76,000 MP4s. The separately supplied `fine_grained_labels.json` covers the full dataset. Each JSON entry has a video path, an official split, and a list of fine-grained labels. The EDA joined entries to local videos by MP4 basename after normalizing path separators.

The Python standard-library script [qevd_eda.py](data/qevd_eda.py) reports two scopes. It counts each exact label at most once per clip and preserves spelling. A clip contributes to every distinct label it carries, so label percentages can sum to more than 100%. In each scope, every clip or annotation entry contributes to the denominator, including entries with empty labels. The script splits the first ` - ` into exercise and attribute and treats an unsplit label as an exercise name. It also counts per-clip exercise and attribute presence, labels per clip, and unordered label pairs. The HTML shows each scope's top 30 labels and long tail. The CSVs retain every ranked label and its official train and test counts.

| EDA measure | Local Part 1 | Full annotation JSON |
| --- | ---: | ---: |
| Clips or annotation entries | 76,000 | 298,089 |
| Official train | 76,000 | 281,660 |
| Official test | 0 | 16,429 |
| Nonempty label lists | 75,986 | 297,912 |
| Empty label lists | 14 | 177 |
| Distinct fine-grained labels | 1,800 | 1,851 |
| Distinct exercise names | 266 | 269 |
| Distinct attributes | 658 | 661 |
| Distinct label co-occurrence pairs | 3,336 | 3,900 |
| Labels seen on exactly one clip | 54 | 17 |
| Labels seen on at most five clips | 232 | 58 |

All 76,000 local clips matched exactly one annotation. The EDA found zero missing local annotations and zero ambiguous annotation filenames. The other 222,089 entries refer to videos outside the local folder and contribute to annotation-only statistics. One exact label was present on 64.90% of local clips and 66.46% of full JSON entries. About one-third of each scope had two or more labels, with observed maxima of seven and eight.

- **Most frequent exact label.** Part 1 had `side plank - elbow below shoulder` on 1,320 clips (1.74%). The full JSON had `fire hydrant - head straight` on 3,377 entries (1.13%).
- **Most frequent exercise and attribute.** Part 1's leading exercise was `glutes bridge` with 2,536 clips. The full JSON's was `elbow plank` with 6,477 entries. The leading attribute in both scopes was `average`, appearing 5,345 and 23,588 times.
- **Co-occurrence.** The most frequent pair in both scopes was `fire hydrant - head straight` with `fire hydrant - knee on ground`, on 734 local clips and 3,112 full JSON entries.

### Video quality and metadata

FFprobe successfully inspected all 76,000 local clips for duration, resolution, codec, frame rate, and frame count. All were H.264 at 30 FPS and 640 pixels wide. The most common resolutions were 640 × 360 (58,468) and 640 × 480 (16,931). Median duration was 5.83 s, mean duration was 5.66 s, and the range was 1.53–9.97 s. Median frame count was 175 and median file size was 260,454 bytes. SHA-256 hashing found zero byte-identical groups in the extracted Part 1 videos. FFmpeg fully decoded a reproducible sample of **300 clips with seed 42**, with zero failures. The HTML contains preview frames from the sample.

Artifacts include the self-contained HTML report (`data/qevd_eda/report.html`), [Part 1 label rankings](data/qevd_eda/part1_label_frequency.csv), [full JSON label rankings](data/qevd_eda/full_json_label_frequency.csv), per-video metadata (`data/qevd_eda/video_metadata.csv`), [numeric summary](data/qevd_eda/video_numeric_summary.csv), decode sample (`data/qevd_eda/decode_sample.csv`), and quality issues (`data/qevd_eda/quality_issues.csv`).

## 2. Four-class dataset and cleaning

The subset script used the exercise name in the original annotation to assign one class per clip. It required an unambiguous annotation, one exercise name, and valid video metadata. It blocked `not visible`, `getting into position`, and `starting late` for all classes, plus `hold` and `not moving` for squats, `not moving` for reverse crunches, and `not moving` and `no jump` for air jump rope. A sustained posture remained valid for side plank. It fully decoded selected candidates, copied them into class folders, and verified each copy against the EDA checksum. The original extracted videos were preserved.

| Class ID | Exercise | Full JSON mentions | Local mentions | Annotation exclusions | Collected |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | Side plank | 3,078 | 1,540 | 19 | 1,521 |
| 1 | Squats | 4,308 | 621 | 74 | 547 |
| 2 | Reverse crunches | 3,395 | 1,892 | 257 | 1,635 |
| 3 | Air jump rope | 2,625 | 1,031 | 194 | 837 |
| | **Total** | **13,406** | **5,084** | **544** | **4,540** |

The 8,322 additional annotations for these exercises were logged as unavailable locally. The 4,540 collected clips all came from the official training split. The [collection script](data/prepare_exercise_subset.py), manifest (`data/exercise_recognition_4/manifest.csv`), exclusion list (`data/exercise_recognition_4/excluded.csv`), unavailable list (`data/exercise_recognition_4/unavailable.csv`), and [summary](data/exercise_recognition_4/summary.json) record the selection.

### Video-level cleaning and split leakage

- **Readability and black frames.** The preparation stage decoded videos, rechecked their SHA-256 hashes, and sampled grayscale frames at 2 FPS. A frame is nearly black when at least 98% of its pixels have intensity at most 16 on a 0–255 scale. A clip is excluded when at least 90% of its sampled frames meet that condition. This stage recorded zero near-black or unreadable exclusions in the collected set. Synthetic black and corrupt clips exercised both rejection paths.
- **Near duplicates.** The script compared eight evenly spaced 64 × 64 grayscale frames per clip. Candidate durations had to differ by at most 10% of the longer duration. At least seven corresponding frame pairs had to pass both a 64-bit perceptual difference-hash Hamming distance of at most six and a mean absolute pixel difference of at most eight. It logged 1,717 near-duplicate pair matches. Pair matches were joined into connected families. Same-class families kept the longest clip, with ties broken by filename. Every member of a conflicting-label family was excluded.
- **Group split.** Before generating windows, the script tried to link numeric filename blocks of 1,000 IDs via duplicate families. The links connected all 76 blocks into one group. The completed run used seven stratified group folds of filename blocks with seed 42, reserved fold 0 for test and fold 1 for validation, and used the other folds for training. It then removed the surviving representatives of any duplicate family that spanned provisional splits. The final manifests were checked for shared filename groups or detected duplicate families between splits. Participant IDs were unavailable, so a participant may still occur in multiple splits.

| Stage | Excluded at this stage | Clips remaining |
| --- | ---: | ---: |
| Collected single-exercise clips | | 4,540 |
| Redundant videos and conflicting duplicate groups | 799 + 47 | 3,694 |
| Representatives of cross-split duplicate families | 191 | 3,503 |
| Low pose coverage | 240 | 3,263 |

The final dataset contains **9,444 windows**. Each row in the window index records the clip, class, split, group, start time, and usable frame count.

| Split | Side plank | Squats | Reverse crunches | Air jump rope | Total clips | Windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 838 | 268 | 836 | 381 | 2,323 | 6,722 |
| Validation | 170 | 56 | 181 | 69 | 476 | 1,385 |
| Local held-out test | 170 | 45 | 180 | 69 | 464 | 1,337 |

Mean usable-frame coverage among retained clips was 97.25% in training, 97.09% in validation, and 96.80% in test. Pose coverage removed 183, 25, and 32 clips from those splits, respectively. See the [dataset summary](data/exercise_motion_run/dataset_summary.json), exclusions (`data/exercise_motion_run/exclusions.csv`), duplicate pairs (`data/exercise_motion_run/duplicate_pairs.csv`), split manifest (`data/exercise_motion_run/split_manifest.csv`), and window index (`data/exercise_motion_run/window_index.csv`).

## 3. MediaPipe and the feature representation

FFmpeg samples video at **10 FPS** and fits each RGB frame into **384 × 384** pixels with padding to preserve aspect ratio. MediaPipe Tasks **Pose Landmarker Lite** runs on CPU in `VIDEO` mode, tracking one person and receiving monotonically increasing timestamps at 100 ms intervals. It produces 33 pose landmarks. The pipeline immediately keeps only **indices 11–32**, covering shoulders, arms, hands, hips, legs, and feet. Indices 0–10, which cover the nose, eyes, ears, and mouth, are discarded before quality checks, feature computation, or classification.

For a frame to be usable, it needs at least 12 finite body landmarks with visibility ≥ 0.50, including at least one shoulder and one hip. The hip and shoulder anchors are the means of the visible points in each pair. A usable frame also needs a positive 2D shoulder-to-hip distance above the numerical tolerance. The desktop pipeline uses the median 2D torso length over usable frames in the clip as its scale.

| Inputs per frame | Channels | Information captured |
| --- | ---: | --- |
| Hip-centered body positions | 3 × 22 | Normalized MediaPipe `x, y, z` relative to the hip anchor. These represent posture, including a held side plank. |
| Body position changes | 3 × 22 | Current minus previous normalized position for joints valid in both adjacent samples. |
| Body validity flags | 2 × 22 | A position-valid flag and a delta-valid flag for each joint. |
| Hip displacement and changes | 4 | Torso-normalized hip `x, y` relative to the first usable frame, plus adjacent hip `dx, dy`. These preserve whole-body motion that hip centering would remove. |
| Hip validity flags | 2 | Current-frame usable flag and a flag requiring adjacent usable frames for the hip delta. |
| **Total** | **182** | **22 × 8 + 6 float32 channels**, with an additional boolean timestep mask supplied to the network. |

Features with insufficient observations are zero-filled, and validity flags tell the model which values are present. Changes across missing frames are masked. Each window spans **32 samples (3.2 s)** with a **16-sample stride (1.6 s)** and needs at least **20 usable samples**. Desktop preprocessing also includes a final end-aligned window when needed and pads eligible short clips to 32 samples. The feature dataset is saved as features.npz (`data/exercise_motion_run/features.npz`) with arrays shaped `[9444, 32, 182]` and `[9444, 32]`, plus filenames, class IDs, and split names. The pose cache keys include the video SHA-256, feature schema version, and Pose Landmarker asset SHA-256. See [features.py](data/exercise_motion/features.py) and [video.py](data/exercise_motion/video.py).

## 4. Neural network and CUDA training

The custom classifier is a **temporal 1D convolutional neural network** with **99,908 trainable parameters**. It convolves across time, using the 182 engineered features as channels. The output order is side plank, squats, reverse crunches, and air jump rope.

| Layer | Output shape for batch size B | Configuration |
| --- | --- | --- |
| Input and transpose | `[B, 182, 32]` | Start with features `[B, 32, 182]` and boolean mask `[B, 32]` |
| Temporal convolution 1 | `[B, 64, 32]` | Kernel 5, padding 2, ReLU |
| Temporal convolution 2 | `[B, 128, 32]` | Kernel 5, padding 2, ReLU |
| Masked average pooling | `[B, 128]` | Sum and divide by the number of usable timesteps |
| Dropout | `[B, 128]` | Rate 0.20 during training |
| Linear classifier | `[B, 4]` | Four raw logits |

The timestep mask zeros missing inputs and is applied again after each convolution. Training uses shuffled windows, batch size 128, seed 42, AdamW with learning rate 0.001 and weight decay 0.0001, and per-window cross-entropy. Each window's loss is weighted by the inverse frequency of its class among training clips and divided by the number of windows from its source clip. This addresses class imbalance and limits the influence of clips that yield more windows.

Training, validation, held-out testing, and desktop upload inference ran on **`cuda:0` with an NVIDIA GeForce RTX 4070 Laptop GPU**, using Python 3.11.18 and PyTorch 2.6.0+cu124. The entry points require a successful CUDA tensor operation. FFmpeg and landmark extraction run on CPU. Validation and testing use softmax per window and average those probabilities by clip. The checkpoint is chosen using validation clip-level macro-F1, with a 50-epoch cap and early stopping after eight epochs without improvement. The best epoch was **14** at **98.37% validation macro-F1**, and training stopped after **22 epochs**. The [classifier source](data/exercise_motion/classifier.py) and [checkpoint](data/exercise_motion_run/model.pt) preserve the architecture, model weights, class order, and feature schema.

### Local held-out evaluation

| Metric | Result |
| --- | ---: |
| Correct predictions | 455 / 464 |
| Clip-level accuracy | 98.06% |
| Macro-F1 | 97.34% |
| Majority baseline accuracy, always predicting the training majority side plank | 36.64% |
| Majority baseline macro-F1 | 13.41% |
| Coverage at the desktop score threshold of 0.60 | 98.92% (459 / 464) |
| Accuracy among clips accepted at that threshold | 98.69% (453 / 459) |

| Exercise | Precision | Recall | F1 | Test clips |
| --- | ---: | ---: | ---: | ---: |
| Side plank | 100.00% | 99.41% | 99.71% | 170 |
| Squats | 95.65% | 97.78% | 96.70% | 45 |
| Reverse crunches | 97.79% | 98.33% | 98.06% | 180 |
| Air jump rope | 95.59% | 94.20% | 94.89% | 69 |

The confusion matrix shows actual labels in rows and predicted labels in columns.

| Actual exercise | Side plank | Squats | Reverse crunches | Air jump rope |
| --- | ---: | ---: | ---: | ---: |
| Side plank | 169 | 0 | 1 | 0 |
| Squats | 0 | 44 | 0 | 1 |
| Reverse crunches | 0 | 1 | 177 | 2 |
| Air jump rope | 0 | 1 | 3 | 65 |

These metrics apply to usable, labeled clips drawn from the local source. Continuous-session recognition, coverage on unfiltered camera input, and accuracy on new participants require separate evaluation. The official QEVD test videos were outside the local Part 1 dataset. The classifier's label space contains only the four exercises. It handles uncertainty with a confidence threshold. See [full metrics and epoch history](data/exercise_motion_run/metrics.json).

## 5. Desktop upload demo

A local **Flask** page at `127.0.0.1:8765` accepts one MP4 up to 100 MB and 30 seconds. It applies the same clip filtering, MediaPipe asset, body features, and windowing used during preparation. The GPU classifier averages window probabilities and returns the highest-scoring exercise, or **Uncertain** below 0.60. It gives specific errors for nearly black clips, invalid video, and insufficient pose coverage. Temporary uploads are removed after processing.

The acceptance run correctly classified one held-out clip per exercise on CUDA and exercised the three expected rejection cases. It also verified that cached and fresh features matched, the upload and direct predictions agreed, and changing facial landmarks left features and CUDA predictions unchanged. See the [desktop app](data/exercise_motion/app.py), [acceptance verifier](data/exercise_motion/verify_run.py), and acceptance results (`data/exercise_motion_run/acceptance.json`).

## 6. Offline FormLite Android MVP

The app implements the eight FormLite Figma screens in **React Native 0.83.1 and TypeScript**. **Kotlin** owns CameraX, pose processing, inference, timing, and persistence. **CameraX 1.4.2**, **MediaPipe Tasks Vision 0.10.32**, **ONNX Runtime Android 1.23.2**, and **Room 2.7.2** are the native dependencies. The APK targets Android API 36, requires API 26 or newer, and includes the JavaScript bundle, database seed, feature metadata, trained classifier, and the same Pose Landmarker Lite asset. Both Android models execute on the device CPU. See [the original Figma design](https://www.figma.com/design/HL3W14YY6vdjHtmRem0Unb/FormLite-%E2%80%94-Exercise-Recognition-MVP?node-id=0-1) and [Android build instructions](formlite/README.md).

### Model export and live input

The trained checkpoint was exported to FP32 **ONNX opset 17** with `features` shaped `[1, 32, 182]`, boolean `mask` shaped `[1, 32]`, and `logits` shaped `[1, 4]`. The bundled asset metadata preserves class order, feature schema, confidence threshold, and file hashes. ONNX and PyTorch gave identical top classes on all **128 checked feature windows**, with maximum absolute logit error of **0.00001144**, within the requested **0.0001** tolerance. See [export validation](formlite/model-validation.json) and [export code](formlite/scripts/export_model.py).

CameraX uses a live preview and keeps only the latest analysis frame under backpressure. Kotlin rotates frames upright, fits them into the same 384-pixel square, samples on a 100 ms monotonic grid, and treats missed samples as invalid. It ports the same 182 feature channels and mask rules. Live recognition computes torso scale and initial hip position within each rolling window, because the full session is unavailable while filming. The desktop path computes them over the submitted clip. Android scores complete windows at the fixed stride while the desktop path can add a final end-aligned window. These differences explain why Android and desktop scores and window counts can differ for the same clip.

A new label needs two consecutive agreeing windows with top probability ≥ 0.60. A conflicting or uncertain result, 300 ms of sustained pose loss, pause, or finish closes the recognized interval. Camera interruption or backgrounding pauses the session. Resume starts a new countdown and resets the recognition window. During acquisition, transitions, and uncertainty, active time is unclassified. The three-second countdown and paused time are outside active time.

### Session screens and local storage

The experience covers Home, Exercise Guide, Camera Setup, Live Recognition, Paused, Session Summary, Saved Summary, and History. Setup handles camera permission, permanent denial with an Android Settings shortcut, and switching between front and rear cameras. Finish opens an unsaved duration table with one row per recognized exercise plus unclassified time. Save commits the session, and discard or deletion requires confirmation. Saved entries are listed newest first and survive restarts.

Room implements the [SQLite reference schema](exercise-recognition-design/schema.sql) with `exercise`, `session`, and `recognition_segment` tables, foreign keys, indexes, and duration aggregation views. Class IDs 0–3 map to `side_plank`, `squat`, `reverse_crunches`, and `air_jump_rope`. History timestamps are UTC milliseconds. Active time and segment offsets use a monotonic clock. A stable UUID supports idempotent saves, and one transaction saves the session and its segments. The repository checks segment bounds and overlap, and deleting a session cascades to its segments. The report derives unclassified time by subtracting recognized time from active time. Only session metadata and segments persist. Camera frames, landmarks, and unfinished or unsaved sessions stay in memory.

### Verification and current limits

- Kotlin features and masks matched Python across eight fixtures covering all four exercises, missing observations, and altered facial values. Facial changes left the features identical.
- Android MediaPipe plus ONNX matched all four desktop classes on replay, one held-out clip per exercise. A blank frame and a synthetic low-confidence window exercised rejection. This replay verifies the model port on those clips.
- Android API 36 emulator checks covered offline launch, permission denial and Settings, both previews, countdown, no pose, pause and resume, backgrounding, review, save, history after restart, deletion, and discard. Room tests covered exact duration reconciliation, duplicate saves, overlap rejection, and cascading deletion. UI review covered all eight screens at 390 × 844 and navigation on a smaller 340 × 720 viewport.
- The desktop emulator was launched with a simulated camera for these checks. Webcam discovery identified the laptop camera as `webcam0`. Real-camera accuracy, session-level timing agreement, latency, and battery use remain open.

See the [Android acceptance report](formlite/verification/README.md), clip replay results (`formlite/verification/android-replay.json`), and debug-signed standalone APK (`formlite/android/app/build/outputs/apk/release/app-release.apk`).

## 7. Reproducing the pipeline

From the project's `data` directory, use a new EDA output directory and a new collection directory to preserve the completed run.

```bash
python3 qevd_eda.py --videos QEVD-FIT-300k-Part-1 \
  --labels fine_grained_labels.json --out qevd_eda_rerun
python3 prepare_exercise_subset.py --metadata qevd_eda_rerun/video_metadata.csv \
  --out exercise_recognition_4_rerun

# With the Python 3.11 environment, CUDA PyTorch, and Pose Landmarker asset ready
python -m exercise_motion prepare --manifest exercise_recognition_4_rerun/manifest.csv \
  --output exercise_motion_rerun
python -m exercise_motion train --output exercise_motion_rerun
python -m exercise_motion.verify_run --manifest exercise_recognition_4_rerun/manifest.csv \
  --output exercise_motion_rerun

# Serve the existing validated run locally
python -m exercise_motion serve
```

The [desktop README](data/exercise_motion/README.md) covers setup, and the [Python dependencies](data/exercise_motion/requirements.txt) pin Flask, MediaPipe, NumPy, and scikit-learn. To regenerate the app assets, run the export and database seed commands in the [Android README](formlite/README.md). The export script reads `exercise_motion_run/model.pt`, `exercise_motion_run/features.npz`, and the Pose Landmarker asset under its `--data` root. To export a rerun, stage those files in that layout under a separate root and use `--assets` to choose a separate output folder.
