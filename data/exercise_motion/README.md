# Four-exercise video recognition

From the data directory, run the local upload page with the environment prepared on this machine.

```bash
.exercise-motion-prep-venv/bin/python -m exercise_motion serve
```

Open [the local upload page](http://127.0.0.1:8765). Choose one MP4 of up to 100 MB and 30 seconds, with one person and their shoulders and hips in view. The page returns side plank, squats, reverse crunches, air jump rope, or **Uncertain**. The score is the model’s average window probability. Processing runs locally, and temporary uploads are removed after inference.

## Reproduce the run

Use Python 3.11, FFmpeg and FFprobe on PATH, and a host with a working NVIDIA driver. Install the requirements and CUDA PyTorch in a fresh environment.

```bash
python3.11 -m venv .exercise-motion-local
source .exercise-motion-local/bin/activate
python -m pip install -r exercise_motion/requirements.txt
python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.8.0+cu128'
mkdir -p .exercise-motion-assets
curl -fL -o .exercise-motion-assets/pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
python -m exercise_motion prepare
python -m exercise_motion train
python -m exercise_motion.verify_run
python -m exercise_motion serve
```

For upload inference only, the repository includes the trained checkpoint. Set up the environment and download the Pose Landmarker asset as above, then run `python -m exercise_motion serve`.

The default input is `exercise_recognition_4/manifest.csv`. Set `--output` consistently across commands to use another run directory. The default run directory contains exclusions, duplicate pairs, split and window manifests, cached features, the model, evaluation metrics, and upload acceptance results. `prepare` can resume from the per-clip cache.

MediaPipe decodes pose on CPU. Preparation and upload use body landmarks 11–32, hip-centered positions, changes between consecutive frames, and hip displacement. Windows use a 16-frame stride with a final window aligned to the end of the clip. CUDA is required for training, validation, held-out predictions, and upload inference.

Local evaluation groups neighboring filename blocks. When duplicate links collapse those blocks into too few groups, preparation assigns folds by filename block, then removes duplicate families spanning splits. The local holdout can contain participants represented elsewhere in the dataset because the source annotations lack participant IDs.
