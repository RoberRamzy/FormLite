"""Focused regression checks for filtering, duplicate handling, and body features."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .features import BODY_COUNT, FEATURE_COUNT, body_windows, select_body
from .prepare import Clip, deduplicate
from .video import ClipError, Fingerprint, difference_hash, inspect_video


def solid_clip(path, color):
    subprocess.run([
        'ffmpeg', '-hide_banner', '-nostdin', '-v', 'error', '-f', 'lavfi',
        '-i', f'color={color}:size=128x128:rate=30:duration=3', '-c:v', 'libx264',
        '-threads', '1', str(path),
    ], check=True)


def fake_landmarks(face_offset, time=0):
    return [
        SimpleNamespace(x=face_offset, y=face_offset, z=face_offset, visibility=float(bool(face_offset)))
        if i < 11 else SimpleNamespace(x=0.1 + i * .02 + time * .001, y=0.2 + i * .01,
                                       z=0.05 * i, visibility=0.99)
        for i in range(33)
    ]


class PipelineTests(unittest.TestCase):
    def test_body_only_features_and_adjacent_deltas(self):
        body_a = np.stack([select_body(fake_landmarks(0, i)) for i in range(36)])
        body_b = np.stack([select_body(fake_landmarks(100, i)) for i in range(36)])
        self.assertEqual(body_a.shape, (36, BODY_COUNT, 4))
        for left, right in zip(body_windows(body_a), body_windows(body_b)):
            np.testing.assert_array_equal(left, right)
        body_a[7, 2, 3] = 0.2
        windows, masks, starts, _ = body_windows(body_a)
        self.assertEqual(windows.shape, (2, 32, FEATURE_COUNT))
        self.assertEqual((windows[0, 6:10, 2 * 8 + 7]).tolist(), [1, 0, 0, 1])
        self.assertEqual((windows[0, 6:10, -1]).tolist(), [1, 1, 1, 1])
        self.assertTrue(masks.all())
        np.testing.assert_array_equal(starts, [0, 4])

    def test_short_window_and_nonvisible_pose(self):
        body = np.stack([select_body(fake_landmarks(0))] * 22)
        _, masks, starts, _ = body_windows(body)
        self.assertEqual(masks.sum(), 22)
        self.assertEqual(starts.tolist(), [0])
        body[:, :, 3] = 0
        with self.assertRaisesRegex(ClipError, 'Too little'):
            body_windows(body)

    def test_black_and_corrupt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            black, bright, corrupt = (root / name for name in ('black.mp4', 'white.mp4', 'broken.mp4'))
            solid_clip(black, 'black')
            solid_clip(bright, 'white')
            corrupt.write_bytes(b'invalid file')
            self.assertEqual(inspect_video(bright).frames.shape, (8, 64, 64))
            for video, code in ((black, 'near_black_video'), (corrupt, 'unreadable_video')):
                with self.assertRaises(ClipError) as failure:
                    inspect_video(video)
                self.assertEqual(failure.exception.code, code)

    def test_dedup_with_conflicting_classes(self):
        images = np.full((8, 64, 64), 80, dtype=np.uint8)
        different = np.full_like(images, 150)
        clips = [
            Clip(Path(f'{i}.mp4'), f'{i:05d}.mp4', label, duration, digest)
            for i, (label, duration, digest) in enumerate(((0, 3, 'a'), (0, 3, 'a'), (0, 3.1, 'b'),
                                                            (1, 3, 'c'), (2, 3, 'd'), (3, 6, 'e')))
        ]
        signatures = [
            Fingerprint(clip.duration, clip.digest, frames, difference_hash(frames))
            for clip, frames in zip(clips, [images] * 3 + [different] * 3)
        ]
        kept, _, matches, rejected = deduplicate(clips, signatures)
        self.assertEqual(kept, [2, 5])
        self.assertIn((0, 1, 'exact_duplicate'), matches)
        self.assertIn((0, 2, 'near_duplicate'), matches)
        self.assertEqual(sum(reason == 'conflicting_duplicate_group' for _, _, reason, _ in rejected), 2)


if __name__ == '__main__':
    unittest.main()
