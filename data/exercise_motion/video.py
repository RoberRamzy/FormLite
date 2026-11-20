"""MP4 validation, black frame detection and visual fingerprints."""

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FINGERPRINT_FRAMES = 8
FINGERPRINT_SIDE = 64
FPS = 10
MAX_DURATION = 30
MAX_UPLOAD_BYTES = 100_000_000


class ClipError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Fingerprint:
    duration: float
    digest: str
    frames: np.ndarray
    hashes: np.ndarray


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run_media(args, timeout=90):
    try:
        result = subprocess.run(args, capture_output=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise ClipError('unreadable_video', 'Video decoding timed out.') from error
    except FileNotFoundError as error:
        raise RuntimeError(f'{args[0]} is required on PATH.') from error
    if result.returncode:
        detail = result.stderr.decode(errors='replace').strip()[-300:]
        raise ClipError('unreadable_video', 'The MP4 could not be decoded. ' + detail)
    return result


def video_duration(path):
    response = run_media([
        'ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe', '-f', 'mov',
        '-select_streams', 'v:0', '-show_entries', 'stream=duration:format=duration',
        '-of', 'json', str(path),
    ], timeout=15)
    try:
        probe = json.loads(response.stdout)
        stream = probe['streams'][0]
        durations = [float(value) for value in (stream.get('duration'), probe.get('format', {}).get('duration'))
                     if value not in (None, 'N/A')]
        if not durations or any(not math.isfinite(d) or d <= 0 for d in durations):
            raise ValueError
        return max(durations)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ClipError('unreadable_video', 'The MP4 has no readable video track with a valid duration.') from error


def decoder(path):
    return [
        'ffmpeg', '-hide_banner', '-nostdin', '-v', 'error', '-xerror', '-threads', '1',
        '-protocol_whitelist', 'file,pipe', '-f', 'mov', '-i', str(path),
    ]


def difference_hash(frames):
    """Compute a 64-bit perceptual difference hash for each grayscale frame."""
    rows = np.linspace(0, FINGERPRINT_SIDE - 1, 8, dtype=int)
    columns = np.linspace(0, FINGERPRINT_SIDE - 1, 9, dtype=int)
    small = frames[:, rows[:, None], columns]
    differences = small[:, :, 1:] > small[:, :, :-1]
    packed = np.packbits(differences.reshape(-1, 64), axis=1)
    return np.ascontiguousarray(packed).view(np.uint64).ravel()


def inspect_video(path: Path, duration=None, digest=None):
    if duration is None:
        duration = video_duration(path)
    if duration > MAX_DURATION:
        raise ClipError('video_too_long', f'Choose an MP4 of {MAX_DURATION} seconds or less.')
    graph = (
        '[0:v:0]split=2[dark][sample];'
        '[dark]fps=2,format=gray,blackframe=amount=0:threshold=17,'
        r'metadata=mode=print:key=lavfi.blackframe.pblack:file=pipe\\:2,nullsink;'
        f'[sample]fps={FINGERPRINT_FRAMES}/{duration:.9f},'
        f'scale={FINGERPRINT_SIDE}:{FINGERPRINT_SIDE}:flags=area,format=gray[out]'
    )
    result = run_media(decoder(path) + [
        '-filter_complex_threads', '1', '-filter_complex', graph, '-map', '[out]',
        '-threads', '1', '-f', 'rawvideo', 'pipe:1',
    ])
    black_percent = [int(value) for value in re.findall(rb'lavfi\.blackframe\.pblack=(\d+)', result.stderr)]
    if not black_percent or not result.stdout:
        raise ClipError('unreadable_video', 'The MP4 has too few decodable frames.')
    if 10 * sum(percent >= 98 for percent in black_percent) >= 9 * len(black_percent):
        raise ClipError('near_black_video', 'The video is nearly black. Try a well-lit clip.')
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, FINGERPRINT_SIDE, FINGERPRINT_SIDE)
    if len(frames) != FINGERPRINT_FRAMES:
        frames = frames[np.linspace(0, len(frames) - 1, FINGERPRINT_FRAMES).round().astype(int)]
    return Fingerprint(duration, digest or sha256(path), frames, difference_hash(frames))


def pose_frames(path):
    # Fit and pad into a square while preserving aspect ratio. Square input
    # gives x, y and z a consistent image scale without stretching the pose.
    side = 384
    result = run_media(decoder(path) + [
        '-map', '0:v:0', '-filter_threads', '1', '-vf',
        f'fps={FPS}:start_time=0,scale={side}:{side}:force_original_aspect_ratio=decrease,'
        f'pad={side}:{side}:(ow-iw)/2:(oh-ih)/2,setsar=1',
        '-frames:v', str(MAX_DURATION * FPS + 1),
        '-pix_fmt', 'rgb24', '-threads', '1', '-f', 'rawvideo', 'pipe:1',
    ])
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, side, side, 3)
