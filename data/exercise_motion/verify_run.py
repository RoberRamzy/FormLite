"""Verify the trained CUDA pipeline against the actual dataset and upload endpoint."""

import argparse
import csv
import io
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .app import create_app
from .classifier import load_model, predict_clip, require_cuda
from .features import SCHEMA, body_windows, extract_features, select_body
from .prepare import CLASSES, Groups, load_manifest, write_json
from .test_pipeline import fake_landmarks, solid_clip
from .video import sha256


def rows(path):
    with path.open(newline='', encoding='utf-8') as source:
        return list(csv.DictReader(source))


def verify(manifest, output, task):
    clips = load_manifest(manifest)
    by_name = {clip.filename: (index, clip) for index, clip in enumerate(clips)}
    split_rows = rows(output / 'split_manifest.csv')
    duplicates = Groups(len(clips))
    for row in rows(output / 'duplicate_pairs.csv'):
        duplicates.merge(by_name[row['filename_a']][0], by_name[row['filename_b']][0])
    split_per_family = {}
    split_per_block = {}
    chosen = {}
    for row in split_rows:
        index, clip = by_name[row['filename']]
        split, group = row['split'], row['group']
        assert split_per_block.setdefault(group, split) == split
        assert split_per_family.setdefault(duplicates.find(index), split) == split
        if split == 'test' and row['status'] == 'usable':
            chosen.setdefault(clip.class_id, clip)
    assert len(chosen) == len(CLASSES)

    device = require_cuda()
    model = load_model(output / 'model.pt', task, device)
    face_a = body_windows(np.stack([select_body(fake_landmarks(0, frame)) for frame in range(32)]))
    face_b = body_windows(np.stack([select_body(fake_landmarks(100, frame)) for frame in range(32)]))
    assert predict_clip(model, *face_a[:2], device) == predict_clip(model, *face_b[:2], device)

    api = create_app(output / 'model.pt', task).test_client()
    assert api.get('/').status_code == 200
    checks = []
    cache = output / 'cache' / f'pose-v{SCHEMA["version"]}-{sha256(task)}'

    def submit(path):
        with path.open('rb') as stream:
            return api.post('/predict', data={'video': (stream, path.name)})

    for class_id, clip in sorted(chosen.items()):
        windows, masks, starts, usable = extract_features(clip.path, task)
        with np.load(cache / f'{clip.digest}.npz', allow_pickle=False) as stored:
            for key, expected in zip(('windows', 'masks', 'starts', 'usable'), (windows, masks, starts, usable)):
                np.testing.assert_array_equal(stored[key], expected)
        reply = submit(clip.path)
        assert reply.status_code == 200, reply.json
        expected = predict_clip(model, windows, masks, device)
        assert expected['exercise'] == reply.json['exercise']
        assert expected['device'] == reply.json['device'] == 'cuda:0'
        for label, score in expected['scores'].items():
            assert abs(reply.json['scores'][label] - score) <= 1e-6
        checks.append({'filename': clip.filename, 'actual': CLASSES[class_id], **reply.json})

    with TemporaryDirectory(prefix='exercise-acceptance-') as directory:
        black, no_pose = (Path(directory) / filename for filename in ('black.mp4', 'no_pose.mp4'))
        solid_clip(black, 'black')
        solid_clip(no_pose, 'white')
        for response, code in (
            (submit(black), 'near_black_video'), (submit(no_pose), 'low_pose_coverage'),
            (api.post('/predict', data={'video': (io.BytesIO(b'invalid'), 'corrupt.mp4')}), 'unreadable_video'),
        ):
            assert response.status_code == 422 and response.json['code'] == code, response.json
            checks.append({'expected_error': code, **response.json})
    write_json(output / 'acceptance.json', checks)
    for row in checks:
        if 'actual' in row:
            print(f'{row["actual"]} → {row["exercise"]} ({row["score"]:.3f}, {row["device"]})')
        else:
            print(f'Handled {row["code"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('exercise_recognition_4/manifest.csv'))
    parser.add_argument('--output', type=Path, default=Path('exercise_motion_run'))
    parser.add_argument('--pose-model', type=Path, default=Path('.exercise-motion-assets/pose_landmarker_lite.task'))
    args = parser.parse_args()
    verify(args.manifest, args.output, args.pose_model)


if __name__ == '__main__':
    main()
