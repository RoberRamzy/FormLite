"""Export and validate the trained FormLite classifier for Android inference."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / 'data'))
from exercise_motion.classifier import load_model
from exercise_motion.features import SCHEMA
from exercise_motion.prepare import CLASSES
from exercise_motion.video import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=WORKSPACE / 'formlite/android/app/src/main/assets')
    parser.add_argument('--data', type=Path, default=WORKSPACE / 'data')
    args = parser.parse_args()
    data = args.data
    output = args.assets
    checkpoint = data / 'exercise_motion_run/model.pt'
    features = data / 'exercise_motion_run/features.npz'
    landmark_model = data / '.exercise-motion-assets/pose_landmarker_lite.task'
    output.mkdir(parents=True, exist_ok=True)
    model = load_model(checkpoint, landmark_model, torch.device('cpu'))
    with np.load(features, allow_pickle=False) as source:
        windows, masks, labels, splits = (source[k] for k in ('windows', 'masks', 'labels', 'splits'))
    indices = np.concatenate([np.flatnonzero((splits == 'test') & (labels == i))[:32] for i in range(4)])
    batch = torch.from_numpy(windows[indices])
    mask = torch.from_numpy(masks[indices])

    graph = output / 'exercise.onnx'
    torch.onnx.export(model, (batch[:1], mask[:1]), str(graph), input_names=['features', 'mask'],
                      output_names=['logits'], opset_version=17, dynamo=False)
    onnx.checker.check_model(str(graph))
    session = ort.InferenceSession(str(graph), providers=['CPUExecutionProvider'])
    with torch.inference_mode():
        expected = model(batch, mask).numpy()
    actual = np.concatenate([
        session.run(['logits'], {'features': batch[i:i+1].numpy(), 'mask': mask[i:i+1].numpy()})[0]
        for i in range(len(indices))
    ])
    np.testing.assert_array_equal(actual.argmax(axis=1), expected.argmax(axis=1))
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
    shutil.copyfile(landmark_model, output / landmark_model.name)
    metadata = {'classes': list(CLASSES), 'exercise_ids': ['side_plank','squat','reverse_crunches','air_jump_rope'],
                'schema': SCHEMA, 'score_threshold': 0.60, 'checkpoint_sha256': sha256(checkpoint),
                'landmarker_sha256': sha256(landmark_model), 'onnx_sha256': sha256(graph)}
    (output / 'model.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    report = {'windows': len(indices), 'matching_predictions': len(indices),
              'max_absolute_logit_error': float(np.max(np.abs(actual - expected)))}
    print(json.dumps(report))


if __name__ == '__main__':
    main()
