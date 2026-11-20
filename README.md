# FormLite

FormLite recognizes side plank, squats, reverse crunches, and air jump rope from body landmarks. The project includes exploratory analysis of QEVD-FIT-300K Part 1, a CUDA-trained temporal CNN, a desktop MP4 demo, and an offline Android app with session history.

| Path | Contents |
| --- | --- |
| [Project report](FORMLITE_PROJECT_REPORT.md) | Dataset analysis, cleaning rules, feature and network shapes, training results, and Android verification |
| [data/](data/) | EDA and dataset preparation scripts, aggregate tables, CUDA training code, and the PyTorch checkpoint |
| [formlite/](formlite/) | React Native Android app, Kotlin recognition and storage, bundled inference models, and build instructions |
| [exercise-recognition-design/schema.sql](exercise-recognition-design/schema.sql) | SQLite reference schema and exercise catalog |

## Run FormLite

Build and install the offline app using the [Android instructions](formlite/README.md). The four-class ONNX classifier and Pose Landmarker Lite asset are bundled for CPU inference. The app stores explicitly saved sessions on the device.

To run EDA, replay the validation clips, or train, place your own QEVD-FIT-300K Part 1 clips and annotation JSON under `data/`, then follow the [pipeline commands](FORMLITE_PROJECT_REPORT.md#7-reproducing-the-pipeline). The desktop upload demo uses the bundled checkpoint and runs on CUDA. Setup is in the [desktop guide](data/exercise_motion/README.md). Raw data, videos, extracted feature arrays, and generated dataset-dependent test fixtures stay local.

The model scored 98.06% accuracy and 97.34% macro-F1 on 464 locally held-out clips. The split used filename groups and near-duplicate checks. Participant overlap is possible. Physical-phone recognition accuracy, latency, and battery use remain unmeasured.
