#!/usr/bin/env python3
"""
TAIF / TC-AdaptFormer 训练脚本

支持：
- 窗口级轨迹序列输入
- forward_with_aux() 的联合损失训练
- raw / engineered 两种轨迹特征模式
"""

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-agri-mbt')

import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from dataset import AgriMultimodalDataset, CLASS_NAMES, PROJECT_ROOT, split_dataset
from models import TCAdaptFormer


def parse_args():
    parser = argparse.ArgumentParser(description='Train TAIF / TC-AdaptFormer')
    parser.add_argument('--csv-path', default='data/aligned_output/aligned_data.csv')
    parser.add_argument('--train-csv', default='', help='显式训练集 CSV；设置后优先于 --csv-path 内部划分')
    parser.add_argument('--val-csv', default='', help='显式验证集 CSV')
    parser.add_argument('--test-csv', default='', help='显式测试集 CSV')
    parser.add_argument('--window-size', type=int, default=5)
    parser.add_argument('--feature-mode', choices=['raw', 'engineered'], default='engineered')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lambda-align', type=float, default=0.1)
    parser.add_argument('--lambda-proto', type=float, default=0.05)
    parser.add_argument('--lambda-balance', type=float, default=0.05)
    parser.add_argument('--save-dir', default='experiments/taif_runs')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--all-gpus', action='store_true', help='使用当前可见的所有 CUDA GPU')
    parser.add_argument('--gpu-ids', default='', help='逗号分隔的 GPU id，例如 0,1,2；为空则使用所有可见 GPU')
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--max-train-batches', type=int, default=0)
    parser.add_argument('--max-eval-batches', type=int, default=0)
    parser.add_argument('--quiet-warnings', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--no-plots', action='store_true', help='不生成训练可视化图')
    return parser.parse_args()


def configure_warnings(quiet: bool):
    if not quiet:
        return
    warnings.filterwarnings('ignore', message='.*imbalance between your GPUs.*')
    warnings.filterwarnings('ignore', message='.*torch.cuda.amp.autocast.*')
    warnings.filterwarnings('ignore', message='.*Was asked to gather along dimension 0.*')
    warnings.filterwarnings('ignore', message='.*Attempting to run cuBLAS.*')
    warnings.filterwarnings('ignore', category=FutureWarning, module='torch.nn.parallel.parallel_apply')
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.parallel')
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.linear')


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_class_weights(dataset: AgriMultimodalDataset, train_subset: Subset, device: str):
    labels = [int(dataset.df.iloc[dataset.windows[i][1]]['分类']) for i in train_subset.indices]
    counts = torch.bincount(torch.tensor(labels), minlength=11).float()
    counts = torch.clamp(counts, min=1.0)
    weights = 1.0 / torch.sqrt(counts)
    weights = weights / weights.sum() * len(weights)
    return weights.to(device)


def load_class_weights_from_dataset(dataset: AgriMultimodalDataset, device: str):
    labels = [int(dataset.df.iloc[mid_idx]['分类']) for _, mid_idx in dataset.windows]
    counts = torch.bincount(torch.tensor(labels), minlength=11).float()
    counts = torch.clamp(counts, min=1.0)
    weights = 1.0 / torch.sqrt(counts)
    weights = weights / weights.sum() * len(weights)
    return weights.to(device)


def build_loaders(args):
    save_dir = PROJECT_ROOT / args.save_dir
    gnss_stats_path = str(save_dir / 'gnss_normalization.json')

    if args.train_csv or args.val_csv or args.test_csv:
        assert args.train_csv and args.val_csv and args.test_csv, (
            '使用显式划分时，必须同时提供 --train-csv / --val-csv / --test-csv'
        )

        train_dataset = AgriMultimodalDataset(
            csv_path=args.train_csv,
            window_size=args.window_size,
            normalize_gnss=True,
            gnss_stats_path=gnss_stats_path,
            feature_mode=args.feature_mode,
        )
        mean, std = train_dataset._compute_gnss_stats_from_windows()
        train_dataset.set_gnss_stats(mean, std)
        train_dataset._save_gnss_stats(mean, std)

        val_dataset = AgriMultimodalDataset(
            csv_path=args.val_csv,
            window_size=args.window_size,
            normalize_gnss=True,
            gnss_stats_path=gnss_stats_path,
            gnss_mean=mean,
            gnss_std=std,
            feature_mode=args.feature_mode,
        )
        test_dataset = AgriMultimodalDataset(
            csv_path=args.test_csv,
            window_size=args.window_size,
            normalize_gnss=True,
            gnss_stats_path=gnss_stats_path,
            gnss_mean=mean,
            gnss_std=std,
            feature_mode=args.feature_mode,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return train_dataset, train_dataset, train_loader, val_loader, test_loader

    dataset = AgriMultimodalDataset(
        csv_path=args.csv_path,
        window_size=args.window_size,
        normalize_gnss=True,
        gnss_stats_path=gnss_stats_path,
        feature_mode=args.feature_mode,
    )
    split_path = str((save_dir / 'split_indices.json').relative_to(PROJECT_ROOT))
    train_set, val_set, test_set = split_dataset(dataset, seed=args.seed, save_path=split_path)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, train_set, train_loader, val_loader, test_loader


class TrainWrapper(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, video, gnss, labels):
        return self.base_model.forward_with_aux(video, gnss, labels)


def unwrap_base_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if isinstance(model, TrainWrapper):
        return model.base_model
    return model


def reduce_loss_tensor(value):
    if torch.is_tensor(value) and value.ndim > 0:
        return value.mean()
    return value


def run_epoch(model, loader, optimizer, criterion, device, train: bool, args):
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    max_batches = args.max_train_batches if train else args.max_eval_batches
    for batch_idx, (video, gnss, labels) in enumerate(loader, start=1):
        video = video.to(device, non_blocking=True)
        gnss = gnss.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits, aux = model(video, gnss, labels)
            cls_loss = criterion(logits, labels)
            alignment_loss = reduce_loss_tensor(aux['alignment_loss'])
            prototype_loss = reduce_loss_tensor(aux['prototype_loss'])
            balance_loss = reduce_loss_tensor(aux['balance_loss'])
            loss = (
                cls_loss
                + args.lambda_align * alignment_loss
                + args.lambda_proto * prototype_loss
                + args.lambda_balance * balance_loss
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

        if max_batches and batch_idx >= max_batches:
            break

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    if all_labels:
        labels_np = torch.cat(all_labels).numpy()
        preds_np = torch.cat(all_preds).numpy()
        macro_f1 = f1_score(labels_np, preds_np, average='macro', zero_division=0)
        weighted_f1 = f1_score(labels_np, preds_np, average='weighted', zero_division=0)
        per_class_recall = recall_score(
            labels_np,
            preds_np,
            average=None,
            labels=list(range(len(CLASS_NAMES))),
            zero_division=0,
        ).tolist()
        conf_mat = confusion_matrix(labels_np, preds_np, labels=list(range(len(CLASS_NAMES)))).tolist()
    else:
        macro_f1 = 0.0
        weighted_f1 = 0.0
        per_class_recall = [0.0 for _ in CLASS_NAMES]
        conf_mat = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    metrics = {
        'loss': avg_loss,
        'acc': avg_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_recall': per_class_recall,
        'confusion_matrix': conf_mat,
    }
    return metrics


def evaluate(model, loader, criterion, device, args):
    return run_epoch(model, loader, None, criterion, device, train=False, args=args)


def save_checkpoint(save_dir: Path, model, optimizer, epoch, best_val_acc, args, feature_dim):
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        'epoch': epoch,
        'model_state': unwrap_base_model(model).state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'args': vars(args),
        'feature_dim': feature_dim,
    }
    torch.save(ckpt, save_dir / 'best.pt')


def flatten_history(history):
    rows = []
    for item in history:
        row = {'epoch': item['epoch']}
        for split in ['train', 'val']:
            for metric in ['loss', 'acc', 'macro_f1', 'weighted_f1']:
                row[f'{split}_{metric}'] = item[split][metric]
        rows.append(row)
    return rows


def save_training_plots(save_dir: Path, summary: dict):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    save_dir.mkdir(parents=True, exist_ok=True)
    rows = flatten_history(summary['history'])
    metrics_csv = save_dir / 'metrics.csv'
    pd.DataFrame(rows).to_csv(metrics_csv, index=False, encoding='utf-8-sig')

    if not rows:
        return

    epochs = [row['epoch'] for row in rows]
    best_epoch = max(summary['history'], key=lambda x: x['val']['macro_f1'])['epoch']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=160)
    plot_specs = [
        ('loss', 'Loss'),
        ('acc', 'Accuracy'),
        ('macro_f1', 'Macro F1'),
    ]
    for ax, (metric, title) in zip(axes, plot_specs):
        ax.plot(epochs, [row[f'train_{metric}'] for row in rows], label='train', linewidth=2)
        ax.plot(epochs, [row[f'val_{metric}'] for row in rows], label='val', linewidth=2)
        ax.axvline(best_epoch, color='gray', linestyle='--', linewidth=1, label='best val' if metric == 'loss' else None)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / 'training_curves.png')
    plt.close(fig)

    conf_mat = np.asarray(summary['test']['confusion_matrix'], dtype=np.float32)
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    conf_norm = np.divide(conf_mat, np.maximum(row_sums, 1.0))

    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    im = ax.imshow(conf_norm, cmap='Blues', vmin=0.0, vmax=1.0)
    ax.set_title('Test Confusion Matrix (row-normalized)')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(range(len(CLASS_NAMES)))
    ax.set_yticklabels(range(len(CLASS_NAMES)))
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            if conf_mat[i, j] > 0:
                ax.text(j, i, str(int(conf_mat[i, j])), ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_dir / 'confusion_matrix.png')
    plt.close(fig)

    recalls = summary['test']['per_class_recall']
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=160)
    ax.bar(range(len(CLASS_NAMES)), recalls, color='#3b82f6')
    ax.set_title('Test Recall by Class')
    ax.set_xlabel('Class ID')
    ax.set_ylabel('Recall')
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_dir / 'per_class_recall.png')
    plt.close(fig)

    final_train = summary['history'][-1]['train']['macro_f1']
    final_val = summary['history'][-1]['val']['macro_f1']
    best_val = summary['best_val_macro_f1']
    overfit_gap = final_train - final_val
    report = {
        'best_val_macro_f1': best_val,
        'best_epoch': best_epoch,
        'final_train_macro_f1': final_train,
        'final_val_macro_f1': final_val,
        'final_train_val_macro_f1_gap': overfit_gap,
        'test_macro_f1': summary['test']['macro_f1'],
        'test_weighted_f1': summary['test']['weighted_f1'],
        'test_acc': summary['test']['acc'],
        'artifacts': [
            'metrics.csv',
            'training_curves.png',
            'confusion_matrix.png',
            'per_class_recall.png',
        ],
    }
    with open(save_dir / 'training_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    configure_warnings(args.quiet_warnings)
    set_seed(args.seed)
    device = torch.device(args.device)

    dataset, train_set, train_loader, val_loader, test_loader = build_loaders(args)
    base_model = TCAdaptFormer(
        num_classes=11,
        gnss_input_dim=dataset.feature_dim,
        pretrained=args.pretrained,
    )
    model = TrainWrapper(base_model).to(device)

    use_multi_gpu = args.all_gpus and device.type == 'cuda' and torch.cuda.device_count() > 1
    if use_multi_gpu:
        device_ids = None
        if args.gpu_ids:
            device_ids = [int(item) for item in args.gpu_ids.split(',') if item.strip()]
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        visible = device_ids if device_ids is not None else list(range(torch.cuda.device_count()))
        print(f"[Train] 使用 DataParallel GPU: {visible}")
    else:
        print(f"[Train] 使用设备: {device}")

    if isinstance(train_set, Subset):
        class_weights = load_class_weights(dataset, train_set, device)
    else:
        class_weights = load_class_weights_from_dataset(train_set, device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = -1.0
    save_dir = PROJECT_ROOT / args.save_dir
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, True, args)
        val_metrics = evaluate(model, val_loader, criterion, device, args)
        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
        })

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics['macro_f1'] > best_val_acc:
            best_val_acc = val_metrics['macro_f1']
            save_checkpoint(save_dir, model, optimizer, epoch, best_val_acc, args, dataset.feature_dim)

    test_metrics = evaluate(model, test_loader, criterion, device, args)
    summary = {
        'best_val_macro_f1': best_val_acc,
        'test': test_metrics,
        'feature_dim': dataset.feature_dim,
        'feature_names': dataset.feature_names,
        'feature_mode': args.feature_mode,
        'class_names': CLASS_NAMES,
        'history': history,
    }
    print(
        f"[Test] loss={test_metrics['loss']:.4f} acc={test_metrics['acc']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f} weighted_f1={test_metrics['weighted_f1']:.4f}"
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    per_class_rows = [
        {
            'class_id': idx,
            'class_name': CLASS_NAMES[idx],
            'recall': test_metrics['per_class_recall'][idx],
        }
        for idx in range(len(CLASS_NAMES))
    ]
    with open(save_dir / 'per_class_recall.json', 'w', encoding='utf-8') as f:
        json.dump(per_class_rows, f, ensure_ascii=False, indent=2)
    with open(save_dir / 'confusion_matrix.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'class_names': CLASS_NAMES,
                'matrix': test_metrics['confusion_matrix'],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if not args.no_plots:
        save_training_plots(save_dir, summary)
        print(f"[Plots] saved to {save_dir}")


if __name__ == '__main__':
    main()
