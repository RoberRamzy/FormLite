"""Train and evaluate the temporal exercise classifier on CUDA."""

import csv
import json
from pathlib import Path
from platform import python_version

import numpy as np
import torch
from torch import nn

from .features import FEATURE_COUNT, SCHEMA
from .prepare import CLASSES, SEED, write_json
from .video import sha256


MIN_SCORE = 0.60
BATCH_SIZE = 128


def require_cuda():
    message = (
        'CUDA is required for training, testing and upload predictions. '
        'Run on the NVIDIA host with a working driver and CUDA-enabled PyTorch.'
    )
    if not torch.cuda.is_available():
        raise RuntimeError(message)
    device = torch.device('cuda:0')
    try:
        probe = torch.ones((2, 2), device=device)
        if (probe @ probe).sum().item() != 8:
            raise RuntimeError('CUDA smoke test produced an unexpected result.')
    except RuntimeError as error:
        raise RuntimeError(f'{message} {error}') from error
    return device


class ExerciseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Conv1d(FEATURE_COUNT, 64, kernel_size=5, padding=2)
        self.second = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(128, len(CLASSES))

    def forward(self, windows, masks):
        valid = masks.unsqueeze(1)
        values = windows.transpose(1, 2) * valid
        values = self.first(values).relu() * valid
        values = self.second(values).relu() * valid
        pooled = values.sum(dim=2) / valid.sum(dim=2).clamp_min(1)
        return self.head(self.dropout(pooled))


def clip_layout(filenames, labels):
    _, first, inverse, windows_per_clip = np.unique(
        filenames, return_index=True, return_inverse=True, return_counts=True,
    )
    return labels[first], inverse, windows_per_clip


def predict_clip(model, windows, masks, device):
    with torch.inference_mode():
        scores = model(
            torch.as_tensor(windows, device=device), torch.as_tensor(masks, device=device),
        ).softmax(dim=1).mean(dim=0).tolist()
    class_id = max(range(len(scores)), key=scores.__getitem__)
    return {
        'exercise': CLASSES[class_id] if scores[class_id] >= MIN_SCORE else 'Uncertain',
        'top_class': CLASSES[class_id], 'score': scores[class_id],
        'scores': dict(zip(CLASSES, scores)), 'windows': len(windows),
        'device': str(device), 'gpu': torch.cuda.get_device_name(device),
    }


def clip_scores(model, windows, masks, filenames, labels, device):
    clip_labels, inverse, counts = clip_layout(filenames, labels)
    all_scores = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(windows), BATCH_SIZE):
            batch = slice(start, start + BATCH_SIZE)
            all_scores.append(model(windows[batch], masks[batch]).softmax(dim=1))
        scores = torch.cat(all_scores)
        totals = torch.zeros((len(counts), len(CLASSES)), device=device)
        totals.index_add_(0, torch.as_tensor(inverse, device=device), scores)
        average = totals / torch.as_tensor(counts, device=device).unsqueeze(1)
    return clip_labels, average.cpu().numpy()


def split_tensors(features, split, device):
    subset = features['splits'] == split
    return (
        torch.as_tensor(features['windows'][subset], device=device),
        torch.as_tensor(features['masks'][subset], device=device),
        features['filenames'][subset], features['labels'][subset],
    )


def read_dataset(output):
    summary = json.loads((output / 'dataset_summary.json').read_text(encoding='utf-8'))
    if summary['schema'] != SCHEMA:
        raise ValueError('Dataset feature schema differs from the classifier. Run prepare again.')
    with np.load(output / 'features.npz', allow_pickle=False) as source:
        features = {key: source[key] for key in ('windows', 'masks', 'filenames', 'labels', 'splits')}
    if features['windows'].shape[2] != FEATURE_COUNT or len(features['windows']) != summary['windows']:
        raise ValueError('The saved features do not match the dataset summary.')
    return summary, features


def classification_metrics(truth, predicted):
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    support = matrix.sum(axis=1)
    selected = matrix.sum(axis=0)
    correct = matrix.diagonal()
    precision = correct / np.maximum(selected, 1)
    recall = correct / np.maximum(support, 1)
    f1 = 2 * correct / np.maximum(support + selected, 1)
    return {
        'accuracy': float(correct.sum() / len(truth)),
        'macro_f1': float(f1.mean()),
        'confusion_matrix': matrix.tolist(),
        'per_class': {
            name: {'precision': float(precision[index]), 'recall': float(recall[index]),
                   'f1-score': float(f1[index]), 'support': int(support[index])}
            for index, name in enumerate(CLASSES)
        },
    }


def train(output, epochs=50):
    device = require_cuda()
    gpu = torch.cuda.get_device_name(device)
    print(f'Training and evaluating on {device} ({gpu})', flush=True)
    torch.manual_seed(SEED)
    summary, features = read_dataset(output)
    windows, masks, names, labels = split_tensors(features, 'train', device)
    validation = split_tensors(features, 'validation', device)
    test = split_tensors(features, 'test', device)

    train_labels, clip_ids, window_counts = clip_layout(names, labels)
    train_counts = np.bincount(train_labels, minlength=len(CLASSES))
    class_weights = len(train_labels) / (len(CLASSES) * train_counts)
    window_weights = torch.as_tensor(class_weights[labels] / window_counts[clip_ids], dtype=torch.float32,
                                     device=device)
    targets = torch.as_tensor(labels, device=device)
    model = ExerciseCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(reduction='none')
    best_score = -1.0
    best_epoch = 0
    state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(windows), device=device)
        total_losses = torch.zeros((), device=device)
        for batch in order.split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            losses = criterion(model(windows[batch], masks[batch]), targets[batch])
            weights = window_weights[batch]
            loss = (losses * weights).sum() / weights.sum()
            loss.backward()
            optimizer.step()
            total_losses += loss.detach() * len(batch)

        truth, scores = clip_scores(model, *validation, device)
        score = classification_metrics(truth, scores.argmax(axis=1))['macro_f1']
        mean_loss = (total_losses / len(windows)).item()
        history.append({'epoch': epoch, 'training_loss': mean_loss, 'validation_macro_f1': score})
        print(f'Epoch {epoch:02d}  loss {mean_loss:.4f}  validation macro-F1 {score:.4f}', flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= 8:
            break

    model.load_state_dict(state)
    truth, scores = clip_scores(model, *test, device)
    predicted = scores.argmax(axis=1)
    baseline = np.full(len(truth), train_counts.argmax())
    evaluation = classification_metrics(truth, predicted)
    baseline_metrics = classification_metrics(truth, baseline)
    accepted = scores.max(axis=1) >= MIN_SCORE
    report = {
        'evaluation': 'Local held-out clip evaluation. Participant overlap is possible.',
        'device': str(device), 'gpu': gpu, 'torch': str(torch.__version__),
        'python': python_version(),
        'training_epochs': len(history), 'best_epoch': best_epoch, 'validation_macro_f1': best_score,
        'test_clips': len(truth), 'test_accuracy': evaluation['accuracy'],
        'test_macro_f1': evaluation['macro_f1'], 'test_per_class': evaluation['per_class'],
        'class_order': list(CLASSES),
        'test_confusion_matrix': evaluation['confusion_matrix'],
        'upload_score_threshold': MIN_SCORE, 'test_coverage_at_upload_threshold': float(accepted.mean()),
        'test_accuracy_at_upload_threshold': float((truth[accepted] == predicted[accepted]).mean())
        if accepted.any() else None,
        'majority_baseline': {
            'class': CLASSES[train_counts.argmax()], 'accuracy': baseline_metrics['accuracy'],
            'macro_f1': baseline_metrics['macro_f1'],
        },
        'dataset': summary, 'pose_coverage': pose_coverage(output), 'history': history,
    }
    checkpoint = {'weights': state, 'schema': SCHEMA, 'classes': list(CLASSES),
                  'pose_model_sha256': summary['pose_model_sha256'], 'score_threshold': MIN_SCORE}
    temporary = output / '.model.pt.tmp'
    torch.save(checkpoint, temporary)
    temporary.replace(output / 'model.pt')
    write_json(output / 'metrics.json', report)
    print(f'Test accuracy {report["test_accuracy"]:.4f}  macro-F1 {report["test_macro_f1"]:.4f}', flush=True)


def pose_coverage(output):
    with (output / 'split_manifest.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    return {
        split: {
            'mean_usable_frame_fraction': float(np.mean(
                [float(row['pose_coverage']) for row in rows if row['split'] == split and row['pose_coverage']]
            )),
            'excluded_clips': sum(row['split'] == split and row['status'] != 'usable' for row in rows),
        }
        for split in ('train', 'validation', 'test')
    }


def load_model(path: Path, task: Path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint['schema'] != SCHEMA or checkpoint['classes'] != list(CLASSES)
            or checkpoint['score_threshold'] != MIN_SCORE or checkpoint['pose_model_sha256'] != sha256(task)):
        raise ValueError('The model, class order, or landmark asset does not match this pipeline.')
    model = ExerciseCNN().to(device)
    model.load_state_dict(checkpoint['weights'])
    model.eval()
    return model
