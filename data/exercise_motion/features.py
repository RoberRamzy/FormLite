"""Shared MediaPipe and body-motion feature processing."""

import numpy as np

from .video import FPS, ClipError, pose_frames


BODY_IDS = tuple(range(11, 33))
BODY_COUNT = len(BODY_IDS)
SHOULDERS = (0, 1)
HIPS = (12, 13)
WINDOW = 32
STRIDE = 16
MIN_USABLE = 20
FEATURE_COUNT = BODY_COUNT * 8 + 6
SCHEMA = {
    'version': 1,
    'body_ids': list(BODY_IDS),
    'fps': FPS,
    'window': WINDOW,
    'stride': STRIDE,
    'min_usable_frames': MIN_USABLE,
    'visibility': 0.5,
    'min_visible_landmarks': 12,
    'feature_count': FEATURE_COUNT,
    'landmark_channels': ['x', 'y', 'z', 'dx', 'dy', 'dz', 'valid', 'delta_valid'],
    'hip_channels': ['x', 'y', 'dx', 'dy', 'valid', 'delta_valid'],
}


def select_body(landmarks):
    """Serialize only the body portion of a MediaPipe result."""
    return np.asarray([
        (point.x, point.y, point.z, point.visibility)
        for point in landmarks[11:33]
    ], dtype=np.float32)


def detect_body(path, task_path):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    frames = pose_frames(path)
    if len(frames) < MIN_USABLE:
        raise ClipError('low_pose_coverage', 'The video needs at least two seconds of visible exercise.')
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task_path), delegate=BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
    )
    observations = np.zeros((len(frames), BODY_COUNT, 4), dtype=np.float32)
    with vision.PoseLandmarker.create_from_options(options) as tracker:
        for index, frame in enumerate(frames):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = tracker.detect_for_video(image, index * (1000 // FPS))
            if result.pose_landmarks:
                observations[index] = select_body(result.pose_landmarks[0])
    return observations


def anchor(coordinates, visible, indices):
    points = coordinates[:, indices]
    valid = visible[:, indices, None]
    return (points * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)


def motion_features(body):
    coordinates = body[..., :3]
    visible = np.isfinite(body).all(axis=2) & (body[..., 3] >= SCHEMA['visibility'])
    coordinates = np.where(visible[..., None], coordinates, 0)
    hips = anchor(coordinates, visible, HIPS)
    shoulders = anchor(coordinates, visible, SHOULDERS)
    torso = np.linalg.norm(shoulders[:, :2] - hips[:, :2], axis=1)
    usable = (
        (visible.sum(axis=1) >= SCHEMA['min_visible_landmarks'])
        & visible[:, SHOULDERS].any(axis=1) & visible[:, HIPS].any(axis=1) & (torso > 1e-4)
    )
    if np.count_nonzero(usable) < MIN_USABLE:
        raise ClipError('low_pose_coverage', 'Too little of the body was visible. Keep shoulders and hips in frame.')
    scale = np.median(torso[usable])
    valid = visible & usable[:, None]
    position = np.where(valid[..., None], (coordinates - hips[:, None]) / scale, 0)
    delta_valid = np.zeros_like(valid)
    delta_valid[1:] = valid[1:] & valid[:-1]
    delta = np.zeros_like(position)
    delta[1:] = np.where(delta_valid[1:, :, None], position[1:] - position[:-1], 0)
    landmarks = np.concatenate((position, delta, valid[..., None], delta_valid[..., None]), axis=2)

    first_hip = hips[np.flatnonzero(usable)[0], :2]
    hip = np.where(usable[:, None], (hips[:, :2] - first_hip) / scale, 0)
    hip_delta_valid = np.zeros_like(usable)
    hip_delta_valid[1:] = usable[1:] & usable[:-1]
    hip_delta = np.zeros_like(hip)
    hip_delta[1:] = np.where(hip_delta_valid[1:, None], hip[1:] - hip[:-1], 0)
    frames = np.column_stack((
        landmarks.reshape(len(body), -1), hip, hip_delta, usable, hip_delta_valid,
    )).astype(np.float32)
    return frames, usable


def body_windows(body):
    frames, usable = motion_features(body)
    last_start = max(0, len(frames) - WINDOW)
    starts = list(range(0, last_start + 1, STRIDE))
    if starts[-1] != last_start:
        starts.append(last_start)
    accepted = [start for start in starts if np.count_nonzero(usable[start:start + WINDOW]) >= MIN_USABLE]
    if not accepted:
        raise ClipError('low_pose_coverage', 'No window contained enough continuous, visible body movement or posture.')

    windows = np.zeros((len(accepted), WINDOW, FEATURE_COUNT), dtype=np.float32)
    masks = np.zeros((len(accepted), WINDOW), dtype=bool)
    for index, start in enumerate(accepted):
        view = frames[start:start + WINDOW]
        windows[index, :len(view)] = view
        masks[index, :len(view)] = usable[start:start + WINDOW]
    return windows, masks, np.asarray(accepted, dtype=np.int32), usable


def extract_features(path, task_path):
    return body_windows(detect_body(path, task_path))
