import argparse
import numpy as np
import pandas as pd
import os
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from dataloader.av_data import AV_Dataset, TRAJ_COLS
from models.visual_model import AVmodel


def parse_options():
    parser = argparse.ArgumentParser(description="Multimodal Bottleneck Attention — Trajectory + Video")

    # Training dynamics
    parser.add_argument('--gpu_id',     type=str,   default="cuda:0")
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--num_epochs', type=int,   default=15)
    parser.add_argument('--seed',       type=int,   default=1111)

    # Model
    parser.add_argument('--mode',         type=str, default='multimodal',
                        choices=['multimodal', 'trajectory_only', 'image_only'],
                        help='Experiment mode: multimodal (traj+img), trajectory_only, image_only')
    parser.add_argument('--adapter_dim',  type=int, default=8,  help='AdaptFormer bottleneck dim')
    parser.add_argument('--num_latent',   type=int, default=4,  help='MBT latent tokens')
    parser.add_argument('--num_classes',  type=int, default=11, help='number of activity classes')

    # Data
    parser.add_argument('--csv_file',  type=str, default='../../data/aligned_output/aligned_data.csv',
                        help='path to aligned_data.csv')
    parser.add_argument('--data_dir',  type=str, default='../../',
                        help='base directory prepended to frame_path column')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='fraction of data held out for validation')
    parser.add_argument('--output_dir', type=str, default='../../experiments',
                        help='directory to save experiment results')

    opts = parser.parse_args()
    torch.manual_seed(opts.seed)
    opts.device = torch.device(opts.gpu_id if torch.cuda.is_available() else 'cpu')
    return opts


# ── Training / Validation loops ───────────────────────────────────────────────

def train_one_epoch(loader, model, optimizer, loss_fn, device):
    epoch_loss, correct, total = [], 0, 0
    model.train()
    for traj, imgs, labels in loader:
        traj   = traj.to(device)
        imgs   = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(traj, imgs)
        loss  = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        correct += (torch.argmax(preds, dim=1) == labels).sum().item()
        total   += len(labels)

    return np.mean(epoch_loss), round(correct / total, 5) * 100


def val_one_epoch(loader, model, loss_fn, device, return_predictions=False):
    """
    Validate one epoch.

    Args:
        return_predictions: If True, return all predictions and labels for metrics calculation
    """
    epoch_loss, correct, total = [], 0, 0
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for traj, imgs, labels in loader:
            traj   = traj.to(device)
            imgs   = imgs.to(device)
            labels = labels.to(device)

            preds = model(traj, imgs)
            loss  = loss_fn(preds, labels)

            epoch_loss.append(loss.item())
            pred_labels = torch.argmax(preds, dim=1)
            correct += (pred_labels == labels).sum().item()
            total   += len(labels)

            if return_predictions:
                all_preds.extend(pred_labels.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    if return_predictions:
        return np.mean(epoch_loss), round(correct / total, 5) * 100, np.array(all_preds), np.array(all_labels)
    return np.mean(epoch_loss), round(correct / total, 5) * 100


def compute_detailed_metrics(y_true, y_pred, num_classes):
    """
    Compute precision, recall, F1-score for each class and overall.

    Returns:
        dict: Metrics including macro/micro averages and per-class scores
    """
    # Overall metrics (macro and micro average)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100

    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0) * 100
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0) * 100
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0) * 100

    # Weighted average (accounts for class imbalance)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0) * 100
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0) * 100
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0) * 100

    # Build per-class dict
    per_class_metrics = {}
    for i in range(num_classes):
        per_class_metrics[f'class_{i}'] = {
            'precision': round(float(precision_per_class[i]), 2),
            'recall': round(float(recall_per_class[i]), 2),
            'f1_score': round(float(f1_per_class[i]), 2)
        }

    return {
        'macro_avg': {
            'precision': round(float(precision_macro), 2),
            'recall': round(float(recall_macro), 2),
            'f1_score': round(float(f1_macro), 2)
        },
        'micro_avg': {
            'precision': round(float(precision_micro), 2),
            'recall': round(float(recall_micro), 2),
            'f1_score': round(float(f1_micro), 2)
        },
        'weighted_avg': {
            'precision': round(float(precision_weighted), 2),
            'recall': round(float(recall_weighted), 2),
            'f1_score': round(float(f1_weighted), 2)
        },
        'per_class': per_class_metrics
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def train_test(args):

    print(f"\n{'='*70}")
    print(f"Experiment: {args.mode}")
    print(f"{'='*70}")

    # ── Load & split data ─────────────────────────────────────────────────
    full_df = pd.read_csv(args.csv_file)
    print(f"Total samples: {len(full_df)}")
    print(f"Class distribution:\n{full_df['分类'].value_counts().sort_index()}")

    # Filter out rows whose frame file does not exist on disk
    import os as _os
    mask = full_df['frame_path'].apply(
        lambda p: _os.path.exists(_os.path.join(args.data_dir, p))
    )
    if (~mask).sum() > 0:
        print(f"Skipping {(~mask).sum()} rows with missing frame files")
        full_df = full_df[mask].reset_index(drop=True)
        print(f"Remaining samples: {len(full_df)}")

    full_df = full_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    split   = int(len(full_df) * (1 - args.test_size))
    train_df = full_df.iloc[:split]
    test_df  = full_df.iloc[split:]

    # Compute normalisation stats from training data only
    traj_vals = train_df[TRAJ_COLS].values.astype('float32')
    traj_mean = traj_vals.mean(axis=0)
    traj_std  = traj_vals.std(axis=0) + 1e-6

    train_dataset = AV_Dataset(train_df, data_dir=args.data_dir,
                                traj_mean=traj_mean, traj_std=traj_std)
    test_dataset  = AV_Dataset(test_df,  data_dir=args.data_dir,
                                traj_mean=traj_mean, traj_std=traj_std)

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True,  num_workers=0)
    testloader  = DataLoader(test_dataset,  batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    print(f"\t Dataset loaded — train: {len(train_dataset)}, test: {len(test_dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = AVmodel(num_classes=args.num_classes,
                    num_latents=args.num_latent,
                    dim=args.adapter_dim,
                    mode=args.mode)
    model.to(args.device)
    print(f"\t Model loaded (mode={args.mode})")
    print('\t Trainable params =',
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn   = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    best_model_state = None
    print("\t Started training\n")
    for epoch in range(args.num_epochs):
        loss,     acc     = train_one_epoch(trainloader, model, optimizer, loss_fn, args.device)
        val_loss, val_acc = val_one_epoch(testloader, model, loss_fn, args.device)

        history['train_loss'].append(loss)
        history['train_acc'].append(acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1:3d}/{args.num_epochs}"
              f"  train loss {loss:.4f}  acc {acc:.2f}%"
              f"  val loss {val_loss:.4f}  val acc {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    print(f"\n\t Training complete — best val acc: {best_val_acc:.2f}%")

    # Save best model checkpoint
    os.makedirs(args.output_dir + '/checkpoints', exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, 'checkpoints', f'{args.mode}_best.pth')
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': best_model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
    }, checkpoint_path)
    print(f"\t Best model checkpoint saved to {checkpoint_path}")

    # ── Compute detailed metrics on final model ────────────────────────────
    print("\n\t Computing detailed classification metrics...")
    val_loss, val_acc, y_pred, y_true = val_one_epoch(
        testloader, model, loss_fn, args.device, return_predictions=True
    )

    detailed_metrics = compute_detailed_metrics(y_true, y_pred, args.num_classes)

    # Print per-class metrics
    print("\n\t Per-class metrics:")
    print(f"\t {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print(f"\t {'-'*40}")
    for i in range(args.num_classes):
        cls_metrics = detailed_metrics['per_class'][f'class_{i}']
        print(f"\t {i:<8} {cls_metrics['precision']:>10.2f} {cls_metrics['recall']:>10.2f} {cls_metrics['f1_score']:>10.2f}")

    print(f"\t {'-'*40}")
    print(f"\t {'Macro':<8} {detailed_metrics['macro_avg']['precision']:>10.2f} {detailed_metrics['macro_avg']['recall']:>10.2f} {detailed_metrics['macro_avg']['f1_score']:>10.2f}")
    print(f"\t {'Weighted':<8} {detailed_metrics['weighted_avg']['precision']:>10.2f} {detailed_metrics['weighted_avg']['recall']:>10.2f} {detailed_metrics['weighted_avg']['f1_score']:>10.2f}")

    # ── Save results ──────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, f'results_{args.mode}.json')
    results = {
        'mode': args.mode,
        'best_val_acc': best_val_acc,
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'metrics': detailed_metrics,
        'history': history,
        'args': {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v
                 for k, v in vars(args).items()}
    }
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\t Results saved to {result_file}")

    return best_val_acc


if __name__ == "__main__":
    opts = parse_options()
    train_test(args=opts)
