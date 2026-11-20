"""Loopback-only video upload and CUDA inference."""

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from .classifier import load_model, predict_clip, require_cuda
from .features import extract_features
from .prepare import CLASSES
from .video import MAX_DURATION, MAX_UPLOAD_BYTES, ClipError, inspect_video


def classify_video(path, task, model, device):
    inspect_video(path)
    windows, masks, _, usable = extract_features(path, task)
    result = predict_clip(model, windows, masks, device)
    result['pose_coverage'] = float(usable.mean())
    return result


def create_app(checkpoint, task):
    device = require_cuda()
    model = load_model(checkpoint, task, device)
    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES + 1024 * 1024,
        MAX_FORM_PARTS=4,
        TRUSTED_HOSTS=['127.0.0.1', 'localhost'],
    )
    prediction_lock = Lock()

    @app.get('/')
    def index():
        return render_template('index.html', classes=CLASSES, max_duration=MAX_DURATION,
                               max_bytes=MAX_UPLOAD_BYTES)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        return jsonify(error='Choose an MP4 file of 100 MB or less.', code='file_too_large'), 413

    @app.post('/predict')
    def upload():
        origin = request.headers.get('Origin')
        if (origin and origin.rstrip('/') != request.host_url.rstrip('/')) or (
            request.headers.get('Sec-Fetch-Site') == 'cross-site'
        ):
            return jsonify(error='Open the local page before submitting a video.', code='invalid_origin'), 403
        videos = request.files.getlist('video')
        if len(videos) != 1 or not videos[0].filename or Path(videos[0].filename).suffix.lower() != '.mp4':
            return jsonify(error='Choose one MP4 file.', code='invalid_upload'), 400
        if not prediction_lock.acquire(blocking=False):
            return jsonify(error='Another video is being analyzed. Try again in a moment.', code='busy'), 429
        try:
            with TemporaryDirectory(prefix='exercise-upload-') as scratch:
                path = Path(scratch) / 'clip.mp4'
                videos[0].save(path)
                if path.stat().st_size > MAX_UPLOAD_BYTES:
                    raise RequestEntityTooLarge()
                return jsonify(classify_video(path, task, model, device))
        except ClipError as error:
            message = ('The MP4 could not be read. Try exporting the clip as a new MP4.'
                       if error.code == 'unreadable_video' else str(error))
            return jsonify(error=message, code=error.code), 422
        finally:
            prediction_lock.release()

    return app
