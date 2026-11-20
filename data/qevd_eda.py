#!/usr/bin/env python3
"""Analyze QEVD clip annotations and local MP4s, then write HTML and CSV reports."""

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from html import escape
from itertools import combinations
from pathlib import Path, PurePosixPath


SAMPLE_SIZE = 300
SAMPLE_SEED = 42
PREVIEW_COUNT = 12
METRICS = ('duration_s', 'fps', 'frames', 'width', 'height', 'size_bytes')
METADATA_FIELDS = (
    'filename', 'split', 'label_count', 'probe_ok', 'duration_s', 'resolution',
    'width', 'height', 'codec', 'fps', 'frames', 'size_bytes', 'sha256',
    'decode_status',
)


@dataclass(frozen=True)
class Annotation:
    path: str
    filename: str
    split: str
    labels: frozenset[str]


@dataclass(frozen=True)
class Issue:
    scope: str
    check: str
    filename: str
    detail: str = ''


def read_annotations(path):
    with path.open(encoding='utf-8-sig') as source:
        rows = json.load(source)
    if not isinstance(rows, list):
        raise ValueError('Expected a JSON array of annotation objects')
    annotations = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f'Annotation {index} must be an object')
        video_path, split, labels = (row.get(key) for key in ('video_path', 'split', 'labels'))
        if (
            not isinstance(video_path, str)
            or not isinstance(split, str)
            or not isinstance(labels, list)
            or any(not isinstance(label, str) for label in labels)
        ):
            raise ValueError(f'Invalid video_path, split, or labels in annotation {index}')
        filename = PurePosixPath(video_path.replace('\\', '/')).name
        if not filename.lower().endswith('.mp4'):
            raise ValueError(f'Expected an MP4 filename in annotation {index}')
        annotations.append(Annotation(video_path, filename, split, frozenset(labels)))
    return annotations


def match_annotations(annotations, local_names, issues):
    by_name = defaultdict(list)
    for annotation in annotations:
        by_name[annotation.filename].append(annotation)
        if not annotation.labels:
            issues.append(Issue('full_json', 'empty_labels', annotation.filename, annotation.path))
        if annotation.filename not in local_names:
            issues.append(Issue('full_json', 'outside_local_subset', annotation.filename, annotation.path))
    for name, matches in sorted(by_name.items()):
        if len(matches) > 1:
            detail = f'{len(matches)} entries: ' + ', '.join(row.path for row in matches)
            issues.append(Issue('full_json', 'ambiguous_annotation_filename', name, detail))

    matched = {}
    for name in sorted(local_names):
        matches = by_name.get(name, [])
        if len(matches) == 1:
            matched[name] = matches[0]
            if not matches[0].labels:
                issues.append(Issue('part1', 'empty_labels', name))
        elif matches:
            issues.append(Issue('part1', 'ambiguous_annotation_filename', name, f'{len(matches)} entries'))
        else:
            issues.append(Issue('part1', 'missing_annotation', name))
    return matched


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as target:
        writer = csv.writer(target)
        writer.writerow(fields)
        writer.writerows(rows)


def percent(count, total):
    return f'{100 * count / total:.6f}' if total else '0.000000'


def ranked(counter):
    return sorted(counter, key=lambda item: (-counter[item], item))


def write_label_analysis(output, scope, annotations, total):
    counts = Counter()
    per_split = defaultdict(Counter)
    exercises = Counter()
    attributes = Counter()
    cooccurrences = Counter()
    cardinality = Counter()
    splits = Counter()
    for annotation in annotations:
        labels = sorted(annotation.labels)
        splits[annotation.split] += 1
        cardinality[len(labels)] += 1
        counts.update(labels)
        per_split[annotation.split].update(labels)
        parsed = [label.partition(' - ') for label in labels]
        exercises.update({exercise for exercise, _, _ in parsed})
        attributes.update({attribute for _, sep, attribute in parsed if sep})
        cooccurrences.update(combinations(labels, 2))

    ordered = ranked(counts)
    write_csv(
        output / f'{scope}_label_frequency.csv',
        ('rank', 'label', 'clip_count', 'percent_of_scope', 'train_count', 'test_count'),
        (
            (rank, label, counts[label], percent(counts[label], total),
             per_split['train'][label], per_split['test'][label])
            for rank, label in enumerate(ordered, start=1)
        ),
    )
    card_rows = list(sorted(cardinality.items()))
    if len(annotations) < total:
        card_rows.append(('Unknown', total - len(annotations)))
    write_csv(
        output / f'{scope}_labels_per_clip.csv',
        ('label_count', 'clip_count', 'percent_of_scope'),
        ((size, count, percent(count, total)) for size, count in card_rows),
    )
    for name, counter in (('exercise', exercises), ('attribute', attributes)):
        write_csv(
            output / f'{scope}_{name}_frequency.csv',
            (name, 'clip_count', 'percent_of_scope'),
            ((item, counter[item], percent(counter[item], total)) for item in ranked(counter)),
        )
    ordered_pairs = ranked(cooccurrences)
    write_csv(
        output / f'{scope}_label_cooccurrence.csv',
        ('label_a', 'label_b', 'clip_count', 'percent_of_scope'),
        ((*pair, cooccurrences[pair], percent(cooccurrences[pair], total)) for pair in ordered_pairs),
    )

    bins = [('1', 1, 1), ('2–5', 2, 5), ('6–10', 6, 10), ('11–50', 11, 50),
            ('51–100', 51, 100), ('101–500', 101, 500), ('501+', 501, math.inf)]
    tail = [(label, sum(low <= n <= high for n in counts.values())) for label, low, high in bins]
    return {
        'total': total, 'labeled': sum(n for k, n in cardinality.items() if k),
        'empty': cardinality[0], 'distinct': len(counts), 'splits': splits,
        'top': [(label, counts[label]) for label in ordered[:30]], 'tail': tail,
        'cardinality': card_rows,
        'cooccurrences': [(*pair, cooccurrences[pair]) for pair in ordered_pairs[:15]],
    }


def run_media(command, timeout):
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, str(error)
    stderr = result.stderr.decode('utf-8', errors='replace').strip()
    if result.returncode or stderr:
        return None, stderr[:2000] or f'Exit status {result.returncode}'
    return result.stdout, ''


def parse_number(parser, value):
    try:
        number = parser(value)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def video_metadata(path):
    properties = {'filename': path.name, 'probe_ok': False, 'decode_status': 'not_sampled'}
    issues = []
    try:
        with path.open('rb') as source:
            properties['size_bytes'] = os.fstat(source.fileno()).st_size
            properties['sha256'] = hashlib.file_digest(source, 'sha256').hexdigest()
    except OSError as error:
        issues.append(Issue('part1', 'hash_failure', path.name, str(error)))

    probe, error = run_media([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,avg_frame_rate,nb_frames,duration:format=duration',
        '-of', 'json', str(path),
    ], timeout=30)
    if probe is not None:
        try:
            result = json.loads(probe)
            stream = result.get('streams', [])[0]
            width = parse_number(int, stream.get('width'))
            height = parse_number(int, stream.get('height'))
            duration = parse_number(float, stream.get('duration'))
            if duration is None:
                duration = parse_number(float, result.get('format', {}).get('duration'))
            fps = parse_number(lambda value: float(Fraction(value)), stream.get('avg_frame_rate'))
            properties.update(
                probe_ok=True, duration_s=duration, width=width, height=height,
                resolution=f'{width}×{height}' if width and height else None,
                codec=stream.get('codec_name'), fps=fps, frames=parse_number(int, stream.get('nb_frames')),
            )
        except (IndexError, ValueError) as failure:
            error = f'No usable video stream: {failure}'
    if error:
        issues.append(Issue('part1', 'probe_failure', path.name, error))
    return properties, issues


def decode_video(path):
    _, error = run_media([
        'ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-threads', '1', '-i', str(path),
        '-map', '0:v:0', '-f', 'null', '-',
    ], timeout=120)
    return path.name, error


def preview_frame(path, duration):
    frame, error = run_media([
        'ffmpeg', '-nostdin', '-v', 'error', '-ss', str((duration or 0) / 2),
        '-threads', '1', '-i', str(path), '-map', '0:v:0', '-frames:v', '1',
        '-vf', 'scale=320:-2', '-c:v', 'mjpeg', '-threads', '1',
        '-f', 'image2pipe', '-',
    ], timeout=30)
    if error or not frame:
        return None, Issue('part1', 'preview_failure', path.name, error or 'No frame returned')
    return base64.b64encode(frame).decode('ascii'), None


def analyze_videos(paths, matched, jobs, issues):
    metadata = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for start in range(0, len(paths), 512):
            for row, failures in pool.map(video_metadata, paths[start:start + 512]):
                annotation = matched.get(row['filename'])
                row['split'] = annotation.split if annotation else None
                row['label_count'] = len(annotation.labels) if annotation else None
                metadata[row['filename']] = row
                issues.extend(failures)
            done = min(start + 512, len(paths))
            if done % 5120 == 0 or done == len(paths):
                print(f'Probed and checked hashes for {done:,}/{len(paths):,} clips', flush=True)

        sample = random.Random(SAMPLE_SEED).sample(paths, min(SAMPLE_SIZE, len(paths)))
        for name, error in pool.map(decode_video, sample):
            metadata[name]['decode_status'] = 'failed' if error else 'passed'
            if error:
                issues.append(Issue('part1', 'decode_failure', name, error))
    previews = []
    for path in sample:
        if metadata[path.name]['decode_status'] != 'passed':
            continue
        encoded, issue = preview_frame(path, metadata[path.name].get('duration_s'))
        if issue:
            issues.append(issue)
        if encoded:
            annotation = matched.get(path.name)
            previews.append((path.name, sorted(annotation.labels) if annotation else [], encoded))
            if len(previews) == PREVIEW_COUNT:
                break
    return metadata, sample, previews


def duplicate_groups(metadata, matched, issues):
    hashes = defaultdict(list)
    for name, row in metadata.items():
        if row.get('sha256'):
            hashes[row['sha256']].append(name)
    groups = []
    for digest, names in sorted(hashes.items()):
        if len(names) < 2:
            continue
        conflicting = len({matched[name].labels for name in names if name in matched}) > 1
        groups.append((digest, names, conflicting))
        if conflicting:
            issues.append(Issue('part1', 'duplicate_group_label_conflict', names[0], ', '.join(names)))
    return groups


def numeric_summary(metadata):
    summary = []
    histograms = {}
    for metric in METRICS:
        values = sorted(row[metric] for row in metadata.values() if row.get(metric) is not None)
        if not values:
            summary.append((metric, 0, len(metadata), '', '', '', '', '', '', ''))
            continue
        quartiles = statistics.quantiles(values, n=4, method='inclusive') if len(values) > 1 else values * 3
        summary.append((
            metric, len(values), len(metadata) - len(values),
            round(statistics.fmean(values), 6), round(statistics.pstdev(values), 6),
            values[0], *(round(q, 6) for q in quartiles), values[-1],
        ))
        if metric in ('duration_s', 'fps', 'frames', 'size_bytes'):
            histograms[metric] = histogram(values)
    return summary, histograms


def histogram(values, bin_count=12):
    low, high = values[0], values[-1]
    if low == high:
        return [(f'{low:g}', len(values))]
    width = (high - low) / bin_count
    counts = [0] * bin_count
    for value in values:
        counts[min(bin_count - 1, int((value - low) / width))] += 1
    return [(f'{low + i * width:.4g}–{low + (i + 1) * width:.4g}', n) for i, n in enumerate(counts)]


def table(headers, rows):
    heading = ''.join(f'<th scope="col">{escape(str(cell))}</th>' for cell in headers)
    body = ''.join(
        '<tr>' + ''.join(f'<td>{escape(str(cell))}</td>' for cell in row) + '</tr>'
        for row in rows
    )
    return f'<div class="table"><table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>'


def bars(rows, label_title, count_title):
    maximum = max((count for _, count in rows), default=1) or 1
    body = ''.join(
        f'<tr><th scope="row">{escape(str(label))}</th><td class="number">{count:,}</td>'
        f'<td class="bar"><span style="width:{100 * count / maximum:.3f}%"></span></td></tr>'
        for label, count in rows
    )
    return (
        f'<div class="table"><table><thead><tr><th scope="col">{escape(label_title)}</th>'
        f'<th scope="col">{escape(count_title)}</th><th scope="col">Relative size</th></tr></thead>'
        f'<tbody>{body}</tbody></table></div>'
    )


def annotation_section(title, scope, report):
    splits = ', '.join(f'{split} {count:,}' for split, count in sorted(report['splits'].items())) or 'none'
    return f"""
    <section id="{scope}">
    <h2>{escape(title)}</h2>
    <p><strong>{report['total']:,}</strong> in scope · <strong>{report['labeled']:,}</strong> with labels ·
       <strong>{report['empty']:,}</strong> empty label lists · <strong>{report['distinct']:,}</strong> distinct labels</p>
    <p>Annotation splits among entries counted here: {escape(splits)}.</p>
    <p><a href="{scope}_label_frequency.csv" download>Download all label frequencies</a></p>
    <h3>Top 30 labels</h3>{bars(report['top'], 'Label', 'Clips')}
    <div class="grid">
    <div><h3>Label frequency distribution</h3>{bars(report['tail'], 'Clips per label', 'Labels')}</div>
    <div><h3>Labels per clip</h3>{bars(report['cardinality'], 'Labels', 'Clips')}</div>
    </div>
    <h3>Most frequent label pairs</h3>{table(('Label A', 'Label B', 'Clips'), report['cooccurrences'])}
    </section>"""


def write_report(output, local, full, metadata, sample, previews, summary, histograms, issues, groups):
    codec = Counter(row['codec'] for row in metadata.values() if row.get('codec'))
    resolution = Counter(row['resolution'] for row in metadata.values() if row.get('resolution'))
    quality = Counter((issue.scope, issue.check) for issue in issues)
    quality_checks = {
        'part1': ('missing_annotation', 'ambiguous_annotation_filename', 'empty_labels',
                  'probe_failure', 'hash_failure', 'decode_failure', 'preview_failure',
                  'duplicate_group_label_conflict'),
        'full_json': ('empty_labels', 'ambiguous_annotation_filename', 'outside_local_subset'),
    }
    quality_rows = (
        (scope, check.replace('_', ' '), f'{quality[scope, check]:,}')
        for scope, checks in quality_checks.items() for check in checks
    )
    sample_html = ''.join(
        f'<figure><img src="data:image/jpeg;base64,{encoded}" alt="Sample frame from {escape(name)}">'
        f'<figcaption><strong>{escape(name)}</strong><br>{escape(", ".join(labels) or "No label available")}'
        '</figcaption></figure>'
        for name, labels, encoded in previews
    )
    csv_links = ''.join(
        f'<li><a href="{escape(path.name)}" download>{escape(path.name)}</a></li>'
        for path in sorted(output.glob('*.csv'))
    )
    charts = ''.join(
        f'<div><h3>{escape(metric)}</h3>{bars(rows, "Range", "Clips")}</div>'
        for metric, rows in histograms.items()
    )
    decode_failures = quality['part1', 'decode_failure']
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>QEVD-FIT-300K dataset analysis</title>
<style>
:root {{color-scheme:light; color:#132b30; background:#f1f5f3; font:16px/1.5 system-ui,sans-serif}}
body {{max-width:1120px; margin:0 auto; padding:24px; box-sizing:border-box}}
header,section {{background:white; border:1px solid #d5e2de; border-radius:12px; padding:clamp(16px,3vw,32px); margin-bottom:20px}}
h1,h2,h3 {{line-height:1.2; margin-top:0}}
h1 {{font-size:clamp(1.9rem,5vw,3rem)}}
h3 {{margin-top:20px}}
a {{color:#126352; text-underline-offset:3px}}
nav {{display:flex; flex-wrap:wrap; gap:8px 24px}}
.grid,.frames {{display:grid; gap:12px 22px; grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr))}}
.table {{overflow:auto}}
table {{width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums}}
td,th {{text-align:left; padding:6px 8px; border-bottom:1px solid #e5eeeb}}
thead {{background:#edf6f2}}
tbody th {{font-weight:500}}
.number {{text-align:right; white-space:nowrap}}
.bar {{width:25%; min-width:65px}}
.bar span {{display:block; background:#19816a; height:9px; border-radius:4px}}
figure {{margin:0; padding:8px; border:1px solid #d5e2de; border-radius:8px}}
figure img {{width:100%; aspect-ratio:16/10; background:#132b30; object-fit:contain}}
figcaption {{font-size:.9rem; overflow-wrap:anywhere}}
</style></head>
<body><header><h1>QEVD-FIT-300K dataset analysis</h1>
<nav aria-label="Report sections"><a href="#part1">Part 1 labels</a><a href="#full_json">Full JSON labels</a>
<a href="#videos">Video metadata</a><a href="#quality">Data quality</a><a href="#downloads">CSV downloads</a></nav>
<p>Label counts measure clips carrying each exact string. Each clip can contribute to multiple labels.
Percentages use every item in the selected scope. Full JSON counts use one record per annotation entry.
Local counts use uniquely matched annotation entries and all local clips as the denominator.</p></header>
<main>
{annotation_section('Part 1 labels', 'part1', local)}
{annotation_section('Full JSON labels', 'full_json', full)}
<section id="videos"><h2>Video metadata</h2>
<p>{sum(row['probe_ok'] for row in metadata.values()):,} of {len(metadata):,} local MP4s were probed successfully.</p>
{table(('Metric', 'Count', 'Missing', 'Mean', 'Std. dev.', 'Min', 'P25', 'Median', 'P75', 'Max'), summary)}
<div class="grid">
<div><h3>Video codecs</h3>{bars([(key, codec[key]) for key in ranked(codec)], 'Codec', 'Clips')}</div>
<div><h3>Video resolutions</h3>{bars([(key, resolution[key]) for key in ranked(resolution)], 'Resolution', 'Clips')}</div>
{charts}
</div></section>
<section id="quality"><h2>Data quality</h2>
<p>Full decode sample: <strong>{len(sample):,}</strong> clips (seed {SAMPLE_SEED}) ·
{len(sample) - decode_failures:,} passed · {decode_failures:,} failed.</p>
<p>Exact duplicates: {len(groups):,} groups covering {sum(len(names) for _, names, _ in groups):,} clips.
{sum(conflict for _, _, conflict in groups):,} groups have different label sets among matched clips.</p>
<p>Annotations outside the local subset refer to clips absent from this run's video folder.</p>
{table(('Scope', 'Check', 'Count'), quality_rows)}
<h3>Sample frames</h3><div class="frames">{sample_html or '<p>No sample frames available.</p>'}</div></section>
<section id="downloads"><h2>CSV downloads</h2><ul>{csv_links}</ul></section>
</main></body></html>"""
    (output / 'report.html').write_text(report, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', type=Path, required=True, help='Directory containing local MP4 files')
    parser.add_argument('--labels', type=Path, required=True, help='Path to fine_grained_labels.json')
    parser.add_argument('--out', type=Path, required=True, help='New or empty report directory')
    parser.add_argument('--jobs', type=int, default=min(12, os.cpu_count() or 1),
                        help='Number of parallel media workers (default: up to 12)')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    for tool in ('ffprobe', 'ffmpeg'):
        if not shutil.which(tool):
            parser.error(f'{tool} is required on PATH')
    if not args.videos.is_dir():
        parser.error('--videos must be a directory')
    paths = sorted(
        path.absolute() for path in args.videos.iterdir()
        if path.is_file() and path.suffix.lower() == '.mp4'
    )
    if not paths:
        parser.error('--videos must contain at least one MP4')
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error('--out must be a new or empty directory')
    try:
        annotations = read_annotations(args.labels)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=True)

    issues = []
    matched = match_annotations(annotations, {path.name for path in paths}, issues)
    print(f'Matched {len(matched):,}/{len(paths):,} local clips against {len(annotations):,} annotation entries', flush=True)
    local = write_label_analysis(args.out, 'part1', list(matched.values()), len(paths))
    full = write_label_analysis(args.out, 'full_json', annotations, len(annotations))

    metadata, sample, previews = analyze_videos(paths, matched, args.jobs, issues)
    groups = duplicate_groups(metadata, matched, issues)
    summary, histograms = numeric_summary(metadata)
    write_csv(
        args.out / 'video_metadata.csv', METADATA_FIELDS,
        (tuple(row.get(field) for field in METADATA_FIELDS) for row in metadata.values()),
    )
    write_csv(
        args.out / 'video_numeric_summary.csv',
        ('metric', 'count', 'missing', 'mean', 'std_dev', 'min', 'p25', 'median', 'p75', 'max'),
        summary,
    )
    write_csv(
        args.out / 'decode_sample.csv', ('filename', 'decode_status'),
        ((path.name, metadata[path.name]['decode_status']) for path in sample),
    )
    write_csv(
        args.out / 'exact_duplicates.csv',
        ('sha256', 'group_size', 'conflicting_labels', 'filename'),
        ((digest, len(names), conflict, name) for digest, names, conflict in groups for name in names),
    )
    write_csv(
        args.out / 'quality_issues.csv', ('scope', 'check', 'filename', 'detail'),
        ((issue.scope, issue.check, issue.filename, issue.detail) for issue in issues),
    )
    write_report(args.out, local, full, metadata, sample, previews, summary, histograms, issues, groups)
    print(f'Report written to {(args.out / "report.html").resolve()}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
