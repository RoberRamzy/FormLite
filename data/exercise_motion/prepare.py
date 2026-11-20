"""Filter the manifest, assign split groups, and build the body-motion dataset."""

import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from .features import SCHEMA, extract_features
from .video import FPS, ClipError, Fingerprint, inspect_video, sha256


CLASSES = ('side plank', 'squats', 'reverse crunches', 'air jump rope')
SEED = 42
SPLITS = ('train', 'validation', 'test')


@dataclass(frozen=True)
class Clip:
    path: Path
    filename: str
    class_id: int
    duration: float
    digest: str


class Groups:
    def __init__(self, count):
        self.parents = list(range(count))
        self.sizes = [1] * count

    def find(self, index):
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def merge(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.sizes[left] < self.sizes[right]:
            left, right = right, left
        self.parents[right] = left
        self.sizes[left] += self.sizes[right]


def load_manifest(path):
    root = path.resolve().parent
    with path.open(newline='', encoding='utf-8') as stream:
        clips = []
        for row in csv.DictReader(stream):
            clip_path = (root / row['video_path']).resolve()
            name = row['source_filename']
            class_id = int(row['class_id'])
            if (
                not clip_path.is_relative_to(root) or clip_path.name != name or not Path(name).stem.isdecimal()
                or class_id not in range(len(CLASSES)) or row['class_name'] != CLASSES[class_id]
            ):
                raise ValueError(f'Invalid manifest entry for {name}.')
            clips.append(Clip(clip_path, name, class_id, float(row['duration_s']), row['sha256']))
    if not clips or len({clip.filename for clip in clips}) != len(clips):
        raise ValueError('The manifest must contain unique video filenames.')
    return sorted(clips, key=lambda clip: clip.filename)


def scan_clip(clip, scan_cache):
    try:
        actual_digest = sha256(clip.path)
        if actual_digest != clip.digest:
            raise ClipError('checksum_mismatch', 'The clip differs from its manifest checksum.')
        cache_path = scan_cache / f'{actual_digest}-{clip.duration:.6f}.npz'
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cached:
                return Fingerprint(clip.duration, actual_digest, cached['frames'], cached['hashes']), None
        signature = inspect_video(clip.path, duration=clip.duration, digest=actual_digest)
        atomic_npz(cache_path, frames=signature.frames, hashes=signature.hashes)
        return signature, None
    except OSError as error:
        return None, exclusion(clip, 'unreadable_video', str(error))
    except ClipError as error:
        return None, exclusion(clip, error.code, str(error))


def atomic_npz(path, **arrays):
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(dir=path.parent, prefix='feature-', suffix='.npz', delete=False) as stream:
        staging = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            staging.replace(path)
        finally:
            staging.unlink(missing_ok=True)


def exclusion(clip, reason, detail=''):
    return (clip.filename, CLASSES[clip.class_id], reason, detail)


def duplicate_edges(signatures):
    """Yield exact and visual duplicate edges in filename order."""
    durations = np.asarray([row.duration for row in signatures])
    hashes = np.stack([row.hashes for row in signatures])
    frames = np.stack([row.frames for row in signatures])
    first_seen = {}
    for left, signature in enumerate(signatures):
        previous = first_seen.setdefault(signature.digest, left)
        if previous != left:
            yield previous, left, 'exact_duplicate'

        candidates = np.flatnonzero(
            np.abs(durations[left + 1:] - signature.duration)
            <= 0.1 * np.maximum(durations[left + 1:], signature.duration)
        ) + left + 1
        close_hashes = np.bitwise_count(hashes[candidates] ^ hashes[left]) <= 6
        likely = np.flatnonzero(close_hashes.sum(axis=1) >= 7)
        for batch in range(0, len(likely), 32):
            indices = likely[batch:batch + 32]
            matches = candidates[indices]
            differences = np.abs(frames[matches].astype(np.int16) - frames[left]).mean(axis=(2, 3))
            close_frames = ((differences <= 8) & close_hashes[indices]).sum(axis=1)
            for other in matches[close_frames >= 7]:
                if signatures[other].digest != signature.digest:
                    yield left, int(other), 'near_duplicate'


def deduplicate(clips, signatures):
    duplicates = Groups(len(clips))
    linked_blocks = Groups(len(clips))
    matches = list(duplicate_edges(signatures))
    first_in_block = {}
    for index, clip in enumerate(clips):
        block = int(Path(clip.filename).stem) // 1000
        linked_blocks.merge(first_in_block.setdefault(block, index), index)
    for left, right, _ in matches:
        duplicates.merge(left, right)
        linked_blocks.merge(left, right)

    grouped = defaultdict(list)
    for index in range(len(clips)):
        grouped[duplicates.find(index)].append(index)
    kept, excluded = [], []
    for indices in grouped.values():
        names = ', '.join(clips[index].filename for index in indices)
        if len({clips[index].class_id for index in indices}) > 1:
            excluded.extend(exclusion(clips[index], 'conflicting_duplicate_group', names) for index in indices)
        else:
            best = min(indices, key=lambda i: (-clips[i].duration, clips[i].filename))
            kept.append(best)
            excluded.extend(
                exclusion(clips[index], 'redundant_video', f'Kept {clips[best].filename}')
                for index in indices if index != best
            )
    return sorted(kept), linked_blocks, matches, excluded


def fold_assignments(clips, kept, linked_blocks):
    from sklearn.model_selection import StratifiedGroupKFold

    labels = [clips[index].class_id for index in kept]
    groups = [linked_blocks.find(index) for index in kept]
    if len(set(groups)) < 7:
        raise ValueError(f'Found only {len(set(groups))} independent groups for seven folds.')
    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    by_group = {}
    for fold, (_, held) in enumerate(splitter.split(np.zeros(len(kept)), labels, groups)):
        split = {0: 'test', 1: 'validation'}.get(fold, 'train')
        for offset in held:
            by_group[groups[offset]] = split
    verify_split_labels((clips[index].class_id, by_group[linked_blocks.find(index)]) for index in kept)
    return by_group


def assign_splits(clips, kept, linked, edges):
    try:
        assignments = fold_assignments(clips, kept, linked)
    except ValueError as error:
        print(f'Linked block split unavailable ({error}). Checking for cross-split duplicates.', flush=True)
    else:
        return kept, {index: (assignments[linked.find(index)], linked.find(index)) for index in kept}, []

    blocks = Groups(len(clips))
    duplicates = Groups(len(clips))
    first_in_block = {}
    for index, clip in enumerate(clips):
        block = int(Path(clip.filename).stem) // 1000
        blocks.merge(first_in_block.setdefault(block, index), index)
    assignments = fold_assignments(clips, kept, blocks)
    # Assign a fold even to blocks that only contain removed duplicates.
    for index in range(len(clips)):
        assignments.setdefault(blocks.find(index), 'train')
    for left, right, _ in edges:
        duplicates.merge(left, right)
    family_splits = defaultdict(set)
    for index in range(len(clips)):
        family_splits[duplicates.find(index)].add(assignments[blocks.find(index)])
    surviving, issues, by_clip = [], [], {}
    for index in kept:
        if len(family_splits[duplicates.find(index)]) > 1:
            issues.append(exclusion(clips[index], 'cross_split_duplicate_family',
                                    'Duplicate family links different provisional splits.'))
        else:
            surviving.append(index)
            group = blocks.find(index)
            by_clip[index] = assignments[group], group
    verify_split_labels((clips[index].class_id, by_clip[index][0]) for index in surviving)
    return surviving, by_clip, issues


def verify_split_labels(rows):
    labels = defaultdict(set)
    for class_id, split in rows:
        labels[split].add(class_id)
    for split in SPLITS:
        missing = set(range(len(CLASSES))) - labels[split]
        if missing:
            raise ValueError(f'{split} is missing classes: {", ".join(CLASSES[index] for index in sorted(missing))}.')


def extract_cached(clip, cache, task):
    path = cache / f'{clip.digest}.npz'
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            return tuple(data[key] for key in ('windows', 'masks', 'starts', 'usable')), None
    try:
        windows, masks, starts, usable = extract_features(clip.path, task)
    except ClipError as error:
        return None, exclusion(clip, error.code, str(error))
    atomic_npz(path, windows=windows, masks=masks, starts=starts, usable=usable)
    return (windows, masks, starts, usable), None


def write_csv(path, columns, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def prepare(manifest, task, output, jobs, scan_only=False):
    clips = load_manifest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    scan_cache = output / 'cache' / 'video-v1'
    scan_cache.mkdir(parents=True, exist_ok=True)
    print(f'Inspecting {len(clips):,} clips for black frames and redundancies', flush=True)
    good, signatures, exclusions = [], [], []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        scans = pool.map(partial(scan_clip, scan_cache=scan_cache), clips)
        for step, (clip, (signature, issue)) in enumerate(zip(clips, scans), 1):
            if issue:
                exclusions.append(issue)
            else:
                good.append(clip)
                signatures.append(signature)
            if step % 500 == 0 or step == len(clips):
                print(f'Inspected {step:,}/{len(clips):,} clips', flush=True)

    if not good:
        write_csv(output / 'exclusions.csv', ('filename', 'class_name', 'reason', 'detail'), exclusions)
        raise ValueError('No readable, nonblack clips remain.')
    kept, linked, edges, duplicate_exclusions = deduplicate(good, signatures)
    exclusions.extend(duplicate_exclusions)
    write_csv(output / 'duplicate_pairs.csv', ('filename_a', 'filename_b', 'match'),
              ((good[a].filename, good[b].filename, kind) for a, b, kind in edges))
    write_csv(output / 'exclusions.csv', ('filename', 'class_name', 'reason', 'detail'), exclusions)
    print(f'Retained {len(kept):,} after video filtering. Detected {len(edges):,} duplicate pairs.', flush=True)
    if scan_only:
        return
    del signatures

    kept, splits, split_exclusions = assign_splits(good, kept, linked, edges)
    exclusions.extend(split_exclusions)
    if split_exclusions:
        print(f'Excluded {len(split_exclusions):,} representatives of cross-split duplicate families.', flush=True)
        write_csv(output / 'exclusions.csv', ('filename', 'class_name', 'reason', 'detail'), sorted(exclusions))
    task_hash = sha256(task)
    feature_cache = output / 'cache' / f'pose-v{SCHEMA["version"]}-{task_hash}'
    feature_cache.mkdir(parents=True, exist_ok=True)
    windows, masks, items, index = [], [], [], []
    print(f'Extracting CPU MediaPipe body landmarks from {len(kept):,} clips', flush=True)
    retained = [good[i] for i in kept]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = pool.map(partial(extract_cached, cache=feature_cache, task=task), retained)
        for step, (source_index, (result, issue)) in enumerate(zip(kept, results), 1):
            clip = good[source_index]
            split, group = splits[source_index]
            if issue:
                exclusions.append(issue)
                items.append((clip.filename, clip.class_id, split, group, 'excluded', 0, ''))
            else:
                samples, valid, starts, usable = result
                windows.extend(samples)
                masks.extend(valid)
                items.append((clip.filename, clip.class_id, split, group, 'usable', len(starts), float(usable.mean())))
                for start, mask in zip(starts, valid):
                    index.append((clip.filename, clip.class_id, split, group, int(start) / FPS, int(mask.sum())))
            if step % 250 == 0 or step == len(kept):
                print(f'Processed pose for {step:,}/{len(kept):,} clips', flush=True)

    write_csv(output / 'exclusions.csv', ('filename', 'class_name', 'reason', 'detail'), sorted(exclusions))
    write_csv(output / 'split_manifest.csv',
              ('filename', 'class_id', 'split', 'group', 'status', 'windows', 'pose_coverage'), items)
    write_csv(output / 'window_index.csv',
              ('filename', 'class_id', 'split', 'group', 'start_seconds', 'usable_frames'), index)
    verify_split_labels((row[1], row[2]) for row in items if row[4] == 'usable')
    atomic_npz(output / 'features.npz', windows=np.stack(windows), masks=np.stack(masks),
               filenames=np.asarray([row[0] for row in index]), labels=np.asarray([row[1] for row in index]),
               splits=np.asarray([row[2] for row in index]))
    counts = Counter((row[2], row[1]) for row in items if row[4] == 'usable')
    summary = {
        'input_clips': len(clips),
        'video_eligible': len(kept),
        'split_strategy': 'purged_filename_blocks' if split_exclusions else 'duplicate_linked_filename_blocks',
        'usable_clips': sum(counts.values()),
        'windows': len(index),
        'exclusions': Counter(row[2] for row in exclusions),
        'clips_per_split': {split: {name: counts[split, i] for i, name in enumerate(CLASSES)}
                            for split in SPLITS},
        'pose_model_sha256': task_hash,
        'schema': SCHEMA,
    }
    write_json(output / 'dataset_summary.json', summary)
    print(f'Saved {len(index):,} windows in {output}', flush=True)
