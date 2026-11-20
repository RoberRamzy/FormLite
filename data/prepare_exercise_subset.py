#!/usr/bin/env python3
"""Collect four labeled exercise classes from the local QEVD clips."""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from qevd_eda import decode_video, read_annotations, write_csv


CLASSES = ('side plank', 'squats', 'reverse crunches', 'air jump rope')
SLUGS = ('side_plank', 'squats', 'reverse_crunches', 'air_jump_rope')
CLASS_IDS = {name: index for index, name in enumerate(CLASSES)}
BLOCKED_ATTRIBUTES = {
    name: {'not visible', 'getting into position', 'starting late'}
    for name in CLASSES
}
BLOCKED_ATTRIBUTES['squats'].update({'hold', 'not moving'})
BLOCKED_ATTRIBUTES['reverse crunches'].add('not moving')
BLOCKED_ATTRIBUTES['air jump rope'].update({'not moving', 'no jump'})


def exercise_names(annotation):
    return {label.partition(' - ')[0] for label in annotation.labels}


def collect_candidates(annotations, local_names, metadata):
    candidates = []
    unavailable = []
    excluded = []
    seen = Counter(row.filename for row in annotations)
    for row in sorted(annotations, key=lambda item: item.filename):
        exercises = exercise_names(row)
        targets = sorted(exercises & CLASS_IDS.keys(), key=CLASS_IDS.get)
        if not targets:
            continue
        category = ', '.join(targets)
        tags = sorted(row.labels)
        attributes = {label.partition(' - ')[2] for label in tags}
        blocked = sorted(set().union(*(BLOCKED_ATTRIBUTES[target] for target in targets)) & attributes)
        if row.filename not in local_names:
            unavailable.append((row.filename, category, row.split, json.dumps(tags), json.dumps(blocked)))
            continue

        record = metadata.get(row.filename)
        if seen[row.filename] != 1 or len(exercises) != 1:
            reason = 'ambiguous exercise or annotation'
        elif blocked:
            reason = 'excluded attribute: ' + ', '.join(blocked)
        elif not usable_metadata(record):
            reason = 'missing or invalid video metadata'
        else:
            candidates.append((row, category, record))
            continue
        excluded.append((row.filename, category, reason, json.dumps(tags)))
    return candidates, unavailable, excluded


def usable_metadata(record):
    if not record or record['probe_ok'] != 'True' or not record['sha256']:
        return False
    try:
        values = [float(record[field]) for field in ('width', 'height', 'fps', 'duration_s', 'size_bytes')]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(value) and value > 0 for value in values)


def verify_and_copy(candidate, videos, output):
    annotation, category, metadata = candidate
    source = videos / annotation.filename
    _, error = decode_video(source)
    if error:
        return None, f'decode failure: {error}'

    class_id = CLASS_IDS[category]
    relative = Path('clips') / SLUGS[class_id] / annotation.filename
    target = output / relative
    shutil.copyfile(source, target)
    with target.open('rb') as copied:
        digest = hashlib.file_digest(copied, 'sha256').hexdigest()
    if digest != metadata['sha256']:
        target.unlink()
        return None, 'video contents differ from the EDA checksum'

    return (
        relative.as_posix(), class_id, category, annotation.filename, annotation.split,
        json.dumps(sorted(annotation.labels), ensure_ascii=False),
        metadata['duration_s'], metadata['fps'], metadata['frames'], metadata['width'],
        metadata['height'], digest,
    ), ''


def prepare(args, output):
    annotations = read_annotations(args.labels)
    with args.metadata.open(encoding='utf-8', newline='') as source:
        metadata = {row['filename']: row for row in csv.DictReader(source)}
    local_names = {p.name for p in args.videos.iterdir() if p.is_file() and p.suffix.lower() == '.mp4'}
    candidates, unavailable, excluded = collect_candidates(annotations, local_names, metadata)

    hash_groups = defaultdict(list)
    for candidate in candidates:
        hash_groups[candidate[2]['sha256']].append(candidate)
    pending = []
    for matches in hash_groups.values():
        conflicting = len({category for _, category, _ in matches}) > 1
        if not conflicting:
            pending.append(matches[0])
        for row, category, _ in matches if conflicting else matches[1:]:
            reason = 'identical video has conflicting categories' if conflicting else 'byte-identical duplicate'
            excluded.append((row.filename, category, reason, json.dumps(sorted(row.labels))))

    for slug in SLUGS:
        (output / 'clips' / slug).mkdir(parents=True)
    print(f'Decoding and collecting {len(pending):,} eligible clips', flush=True)
    manifest = []
    worker = partial(verify_and_copy, videos=args.videos, output=output)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for index, (candidate, result) in enumerate(zip(pending, pool.map(worker, pending)), 1):
            record, error = result
            if error:
                row, category, _ = candidate
                excluded.append((row.filename, category, error, json.dumps(sorted(row.labels))))
            else:
                manifest.append(record)
            if index % 1000 == 0 or index == len(pending):
                print(f'Checked {index:,}/{len(pending):,} clips', flush=True)
    manifest.sort(key=lambda row: (row[1], row[3]))
    write_csv(
        output / 'manifest.csv',
        ('video_path', 'class_id', 'class_name', 'source_filename', 'official_split', 'fine_grained_labels',
         'duration_s', 'fps', 'frame_count', 'width', 'height', 'sha256'),
        manifest,
    )
    write_csv(output / 'excluded.csv', ('filename', 'class_name', 'reason', 'fine_grained_labels'),
              sorted(excluded))
    write_csv(output / 'unavailable.csv',
              ('filename', 'class_name', 'official_split', 'fine_grained_labels', 'blocked_attributes'),
              unavailable)
    counts = Counter(record[2] for record in manifest)
    excluded_counts = Counter(row[1] for row in excluded)
    missing_counts = Counter((row[1], row[2]) for row in unavailable)
    summary = {
        'included': len(manifest), 'excluded': len(excluded), 'unavailable': len(unavailable),
        'classes': [
            {
                'class_id': CLASS_IDS[name], 'name': name, 'directory': SLUGS[CLASS_IDS[name]],
                'included': counts[name], 'excluded': excluded_counts[name],
                'unavailable_official_train': missing_counts[name, 'train'],
                'unavailable_official_test': missing_counts[name, 'test'],
                'blocked_attributes': sorted(BLOCKED_ATTRIBUTES[name]),
            }
            for name in CLASSES
        ],
        'included_official_splits': Counter(record[4] for record in manifest),
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', type=Path, default=Path('QEVD-FIT-300k-Part-1'))
    parser.add_argument('--labels', type=Path, default=Path('fine_grained_labels.json'))
    parser.add_argument('--metadata', type=Path, default=Path('qevd_eda/video_metadata.csv'))
    parser.add_argument('--out', type=Path, default=Path('exercise_recognition_4'))
    parser.add_argument('--jobs', type=int, default=min(12, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    if not shutil.which('ffmpeg'):
        parser.error('ffmpeg is required on PATH')
    if args.out.exists():
        parser.error('--out must be a new directory')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.videos = args.videos.absolute()
    with TemporaryDirectory(prefix='.exercise_subset.', dir=args.out.parent) as temporary:
        stage = Path(temporary) / 'result'
        summary = prepare(args, stage)
        stage.rename(args.out)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    sys.exit(main())
