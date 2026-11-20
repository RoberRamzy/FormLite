"""Generate Kotlin parity fixtures and held-out Android replay inputs."""
import csv
import json
import shutil
import struct

import torch
import sys
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1]
DATA = APP.parent / 'data'
sys.path.insert(0, str(DATA))
from exercise_motion.classifier import load_model
from exercise_motion.features import body_windows, detect_body, motion_features
from exercise_motion.prepare import CLASSES
from exercise_motion.video import ClipError


def main():
    asset = DATA / '.exercise-motion-assets/pose_landmarker_lite.task'
    run = DATA / 'exercise_motion_run'
    files = APP / 'android/app/src/androidTest/assets'
    parity = APP / 'android/app/src/test/resources/motion.bin'
    files.mkdir(parents=True, exist_ok=True)
    parity.parent.mkdir(parents=True, exist_ok=True)
    with (run / 'split_manifest.csv').open() as source:
        rows = [row for row in csv.DictReader(source) if row['split'] == 'test' and row['status'] == 'usable']
    model = load_model(run / 'model.pt', asset, torch.device('cpu'))
    records = []
    with parity.open('wb') as binary:
        binary.write(struct.pack('>i', len(CLASSES) * 2))
        for class_id, name in enumerate(CLASSES):
            for row in rows:
                if int(row['class_id']) != class_id:
                    continue
                source = DATA / 'exercise_recognition_4/clips' / name.replace(' ', '_') / row['filename']
                body = detect_body(source, asset)
                selected = None
                for start in range(0, len(body) - 31, 16):
                    poses = body[start:start + 32].copy()
                    with_gap = poses.copy()
                    with_gap[6:9] = 0
                    try:
                        examples = [(sample, *motion_features(sample)) for sample in (poses, with_gap)]
                    except ClipError:
                        continue
                    selected = examples
                    break
                if selected is None:
                    continue
                shutil.copyfile(source, files / f'replay-{class_id}.mp4')
                with torch.inference_mode():
                    windows, masks, _, _ = body_windows(body)
                    scores = model(torch.from_numpy(windows), torch.from_numpy(masks)).softmax(1).mean(0)
                for poses, values, mask in selected:
                    raw = np.zeros((32, 33, 4), dtype=np.float32)
                    raw[..., 11:, :] = poses
                    binary.write(raw.astype('>f4').tobytes())
                    binary.write(values.astype('>f4').tobytes())
                    binary.write(mask.tobytes())
                records.append({'class_id': class_id, 'filename': row['filename'], 'class_name': name,
                                'desktop_prediction': CLASSES[int(scores.argmax())], 'sample_frames': len(body)})
                break
            else:
                raise ValueError(f'No usable 32-frame fixture found for {name}')
    (files / 'replay.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
