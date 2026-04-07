"""
Trajectory prediction visualisation
------------------------------------
1. Train the model (quick, configurable epochs)
2. Run inference on one full video segment
3. Plot ground-truth vs predicted class along the GPS trajectory
"""

import argparse, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')          # headless / WSL-safe
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from dataloader.av_data import AV_Dataset, TRAJ_COLS
from models.visual_model import AVmodel

# ── Class name mapping (adjust if you have official names) ────────────────────
CLASS_NAMES = {
    0: "Cls-0", 1: "Cls-1", 2: "Cls-2",  3: "Cls-3",  4: "Cls-4",
    5: "Cls-5", 6: "Cls-6", 7: "Cls-7",  8: "Cls-8",  9: "Cls-9",
    10: "Cls-10",
}
N_CLASSES = 11

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_loaders(full_df, target_video, traj_mean, traj_std, batch_size, data_dir, seed):
    """Split: target video → test; rest → train."""
    train_df = full_df[full_df['video_file'] != target_video].copy()
    test_df  = full_df[full_df['video_file'] == target_video].copy()

    train_ds = AV_Dataset(train_df, data_dir=data_dir, traj_mean=traj_mean, traj_std=traj_std)
    test_ds  = AV_Dataset(test_df,  data_dir=data_dir, traj_mean=traj_mean, traj_std=traj_std)

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    testloader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)
    return trainloader, testloader, test_df


def train_one_epoch(loader, model, opt, loss_fn, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for traj, imgs, labels in loader:
        traj, imgs, labels = traj.to(device), imgs.to(device), labels.to(device)
        opt.zero_grad()
        preds = model(traj, imgs)
        loss  = loss_fn(preds, labels)
        loss.backward(); opt.step()
        total_loss += loss.item() * len(labels)
        correct    += (preds.argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total * 100


@torch.no_grad()
def predict(loader, model, device):
    model.eval()
    all_preds = []
    for traj, imgs, _ in loader:
        traj, imgs = traj.to(device), imgs.to(device)
        logits = model(traj, imgs)
        all_preds.extend(logits.argmax(1).cpu().numpy())
    return np.array(all_preds)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_trajectory(seg_df: pd.DataFrame, save_path: str):
    """
    Three-panel figure:
      Left  – GPS path coloured by ground-truth class
      Right – GPS path coloured by predicted class
      Bottom – time-series of true vs predicted class over the segment
    """
    all_classes = sorted(range(N_CLASSES))
    palette     = sns.color_palette("tab10", n_colors=N_CLASSES)
    color_map   = {c: palette[c] for c in all_classes}

    fig = plt.figure(figsize=(18, 14))
    gs  = fig.add_gridspec(2, 2, height_ratios=[3, 1.5], hspace=0.35, wspace=0.3)
    ax_gt   = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_ts   = fig.add_subplot(gs[1, :])

    def draw_map(ax, col, title):
        # background path line
        ax.plot(seg_df['Longitude'], seg_df['Latitude'],
                color='lightgrey', linewidth=0.6, zorder=1)
        for cls in all_classes:
            sub = seg_df[seg_df[col] == cls]
            if sub.empty:
                continue
            ax.scatter(sub['Longitude'], sub['Latitude'],
                       c=[color_map[cls]], s=18, label=CLASS_NAMES[cls],
                       zorder=2, alpha=0.85)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude',  fontsize=11)
        ax.tick_params(labelsize=9)
        # start / end markers
        ax.scatter(seg_df['Longitude'].iloc[0],  seg_df['Latitude'].iloc[0],
                   marker='^', s=120, c='black', zorder=5, label='Start')
        ax.scatter(seg_df['Longitude'].iloc[-1], seg_df['Latitude'].iloc[-1],
                   marker='s', s=120, c='black', zorder=5, label='End')
        ax.legend(fontsize=7.5, loc='upper right',
                  ncol=2, framealpha=0.8)

    seg_df['Longitude'] = seg_df['经度']
    seg_df['Latitude']  = seg_df['纬度']
    draw_map(ax_gt,   'GT',   'Ground Truth')
    draw_map(ax_pred, 'Pred', 'Prediction')

    # ── Accuracy patch ───────────────────────────────────────────────────
    acc = (seg_df['GT'] == seg_df['Pred']).mean() * 100
    fig.text(0.5, 0.96, f'Segment Accuracy: {acc:.1f}%',
             ha='center', fontsize=13, color='darkgreen', fontweight='bold')

    # ── Time-series panel ────────────────────────────────────────────────
    idx = np.arange(len(seg_df))
    ax_ts.step(idx, seg_df['GT'],   where='mid', linewidth=1.4,
               color='steelblue',   label='Ground Truth',  alpha=0.9)
    ax_ts.step(idx, seg_df['Pred'], where='mid', linewidth=1.2,
               color='tomato', linestyle='--', label='Prediction', alpha=0.9)

    # shade mismatched regions
    mismatch = seg_df['GT'].values != seg_df['Pred'].values
    for i, m in enumerate(mismatch):
        if m:
            ax_ts.axvspan(i - 0.5, i + 0.5, color='salmon', alpha=0.25, linewidth=0)

    ax_ts.set_yticks(all_classes)
    ax_ts.set_yticklabels([CLASS_NAMES[c] for c in all_classes], fontsize=8)
    ax_ts.set_xlabel('Frame index', fontsize=11)
    ax_ts.set_ylabel('Activity Class', fontsize=11)
    ax_ts.set_title('Time-series: Ground Truth vs Prediction  (pink = mismatch)',
                    fontsize=12, fontweight='bold')
    ax_ts.legend(fontsize=10, loc='upper right')
    ax_ts.set_xlim(0, len(seg_df))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved → {save_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file',      default='../../data/aligned_output/aligned_data.csv')
    parser.add_argument('--data_dir',      default='../../')
    parser.add_argument('--target_video',  default='20241018104130.mp4',
                        help='video held out for visualisation')
    parser.add_argument('--num_epochs',    type=int,   default=10)
    parser.add_argument('--batch_size',    type=int,   default=8)
    parser.add_argument('--lr',            type=float, default=3e-4)
    parser.add_argument('--adapter_dim',   type=int,   default=8)
    parser.add_argument('--num_latent',    type=int,   default=4)
    parser.add_argument('--seed',          type=int,   default=42)
    parser.add_argument('--save_fig',      default='trajectory_comparison.png')
    parser.add_argument('--device',        default='cuda:0')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    full_df = pd.read_csv(args.csv_file)
    train_rows = full_df[full_df['video_file'] != args.target_video]
    traj_vals  = train_rows[TRAJ_COLS].values.astype('float32')
    traj_mean  = traj_vals.mean(axis=0)
    traj_std   = traj_vals.std(axis=0) + 1e-6

    trainloader, testloader, seg_df = make_loaders(
        full_df, args.target_video, traj_mean, traj_std,
        args.batch_size, args.data_dir, args.seed
    )
    print(f"Train: {len(trainloader.dataset)} | Test (viz): {len(testloader.dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = AVmodel(num_classes=N_CLASSES,
                    num_latents=args.num_latent,
                    dim=args.adapter_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn   = nn.CrossEntropyLoss()

    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Training ──────────────────────────────────────────────────────────
    print("\n── Training ──────────────────────────────────────────────────")
    for ep in range(1, args.num_epochs + 1):
        loss, acc = train_one_epoch(trainloader, model, optimizer, loss_fn, device)
        print(f"  Epoch {ep:3d}/{args.num_epochs}  loss {loss:.4f}  acc {acc:.2f}%")

    # ── Inference on target video ──────────────────────────────────────────
    print("\n── Inference ─────────────────────────────────────────────────")
    preds = predict(testloader, model, device)
    seg_df = seg_df.reset_index(drop=True)
    seg_df['GT']   = seg_df['分类'].astype(int)
    seg_df['Pred'] = preds.astype(int)

    acc = (seg_df['GT'] == seg_df['Pred']).mean() * 100
    print(f"  Segment accuracy: {acc:.1f}%  ({int(acc/100*len(seg_df))}/{len(seg_df)} correct)")
    print("\n  Per-class accuracy:")
    for cls in sorted(seg_df['GT'].unique()):
        sub = seg_df[seg_df['GT'] == cls]
        c_acc = (sub['GT'] == sub['Pred']).mean() * 100
        print(f"    Class {cls:2d}: {c_acc:6.1f}%  (n={len(sub)})")

    # ── Plot ──────────────────────────────────────────────────────────────
    print("\n── Plotting ──────────────────────────────────────────────────")
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.save_fig)
    plot_trajectory(seg_df, save_path)


if __name__ == '__main__':
    main()
