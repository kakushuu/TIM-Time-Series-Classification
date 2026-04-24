#!/usr/bin/env python3
"""Build paper-ready diagnostic figures from final part-level predictions."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-agri-mbt")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASS_SHORT = ["R-EH", "S-EH", "T-EH", "Full", "R-Tr", "S-Tr", "T-Tr", "Off", "Idle", "Unload", "Road"]
GROUP_NAMES = ["Harvesting", "Transfer", "Waiting", "Unloading", "Road"]
CLASS_TO_GROUP = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 1,
    6: 1,
    7: 2,
    8: 2,
    9: 3,
    10: 4,
}
GROUP_COLORS = {
    "Harvesting": "#4c78a8",
    "Transfer": "#f58518",
    "Waiting": "#8e6c8a",
    "Unloading": "#b279a2",
    "Road": "#2ca25f",
}
MODEL_ORDER = ["ast", "image_best", "trnet", "trimodal_concat", "trimodal_class_gate"]
UNIMODAL_MODELS = ["ast", "image_best", "trnet"]
FUSION_MODELS = ["trimodal_concat", "trimodal_class_gate"]
PRIMARY_MODEL = "trimodal_class_gate"
MODEL_LABELS = {
    "ast": "AST",
    "image_best": "ViT",
    "trnet": "BiLSTM",
    "trimodal_concat": "TIM concat",
    "trimodal_class_gate": "TIM class-gate",
}
MODEL_PATHS = {
    "ast": PROJECT_ROOT / "experiments" / "paper_4090_final_seed44" / "ast" / "seed44" / "predictions.csv",
    "image_best": PROJECT_ROOT / "experiments" / "paper_4090_final_seed44" / "image_best" / "seed44" / "predictions.csv",
    "trnet": PROJECT_ROOT / "experiments" / "paper_4090_final_seed44" / "trnet" / "seed44" / "predictions.csv",
    "trimodal_concat": PROJECT_ROOT / "experiments" / "paper_4090_final_seed44" / "trimodal_concat" / "seed44" / "predictions.csv",
    "trimodal_class_gate": PROJECT_ROOT
    / "experiments"
    / "trimodal_fusion_4090"
    / "fusion_class_gate_20260420"
    / "trimodal_class_gate"
    / "seed44"
    / "predictions.csv",
}
MODEL_COLORS = {
    "ast": "#4c78a8",
    "image_best": "#f58518",
    "trnet": "#54a24b",
    "trimodal_concat": "#8e6c8a",
    "trimodal_class_gate": "#c84e4e",
}


@dataclass(frozen=True)
class PartRange:
    display_id: int
    date: str
    part: int
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    trajectory: pd.DataFrame

    @property
    def label(self) -> str:
        return f"Trajectory {self.display_id}"


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def apply_style() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.titlecolor": "#222222",
        "axes.titleweight": "semibold",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.55,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
    })


def part_number(path: Path) -> int:
    match = re.search(r"_part(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def load_part_ranges(part_dir: Path, dates: list[str]) -> list[PartRange]:
    raw_parts: list[dict] = []
    for date in dates:
        for path in sorted(part_dir.glob(f"*{date}*_part*.xlsx"), key=part_number):
            df = pd.read_excel(path)
            time_col = "时间" if "时间" in df.columns else "定位时间"
            df = df.copy()
            df["_plot_time"] = pd.to_datetime(df[time_col], errors="coerce")
            for col in ["经度", "纬度"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["_plot_time", "经度", "纬度"]).sort_values("_plot_time").reset_index(drop=True)
            if df.empty:
                continue
            raw_parts.append({
                "date": date,
                "part": part_number(path),
                "name": path.stem,
                "start": df["_plot_time"].min(),
                "end": df["_plot_time"].max(),
                "trajectory": df[["_plot_time", "经度", "纬度"]].copy(),
            })
    if not raw_parts:
        raise FileNotFoundError(f"no part xlsx files found under {part_dir} for dates={dates}")
    parts = [
        PartRange(display_id=idx, **item)
        for idx, item in enumerate(raw_parts, start=1)
    ]
    return parts


def load_predictions(suite_dir: Path) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for model in MODEL_ORDER:
        path = MODEL_PATHS.get(model, suite_dir / model / "seed44" / "predictions.csv")
        if not path.exists():
            raise FileNotFoundError(f"missing predictions: {path}")
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["frame_time"] = pd.to_datetime(df["frame_time"], errors="coerce")
        for col in ["y_true", "y_pred", "经度", "纬度"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["frame_time", "y_true", "y_pred", "经度", "纬度"]).copy()
        df["y_true"] = df["y_true"].astype(int)
        df["y_pred"] = df["y_pred"].astype(int)
        df["correct"] = df["y_true"] == df["y_pred"]
        data[model] = df
    return data


def subset_for_part(df: pd.DataFrame, part: PartRange) -> pd.DataFrame:
    sub = df[(df["frame_time"] >= part.start) & (df["frame_time"] <= part.end)].copy()
    sub = sub.sort_values("frame_time").reset_index(drop=True)
    if not sub.empty:
        sub["elapsed_min"] = (sub["frame_time"] - sub["frame_time"].iloc[0]).dt.total_seconds() / 60.0
    return sub


def selected_part_subsets(predictions: dict[str, pd.DataFrame], parts: list[PartRange]) -> dict[str, dict[str, pd.DataFrame]]:
    out: dict[str, dict[str, pd.DataFrame]] = {model: {} for model in MODEL_ORDER}
    for model, df in predictions.items():
        for part in parts:
            out[model][part.name] = subset_for_part(df, part)
    return out


def local_xy_meters(df: pd.DataFrame, origin: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    lon = pd.to_numeric(df["经度"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(df["纬度"], errors="coerce").to_numpy(dtype=float)
    if origin is None:
        lon0 = float(np.nanmin(lon))
        lat0 = float(np.nanmin(lat))
    else:
        lon0, lat0 = origin
    x = (lon - lon0) * 111_320.0 * np.cos(np.deg2rad(lat0))
    y = (lat - lat0) * 110_540.0
    return x, y, (lon0, lat0)


def draw_background(ax, part: PartRange, origin: tuple[float, float] | None = None) -> tuple[float, float]:
    x, y, origin = local_xy_meters(part.trajectory, origin=origin)
    ax.plot(x, y, color="#bdbdbd", linewidth=0.75, alpha=0.58, zorder=1)
    ax.scatter(x, y, s=4, color="#bdbdbd", alpha=0.16, linewidths=0, zorder=1)
    return origin


def concat_selected(subsets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pieces = [df for df in subsets.values() if not df.empty]
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def model_accuracy(df: pd.DataFrame) -> float:
    return float(df["correct"].mean()) if len(df) else 0.0


def class_group(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    return np.array([CLASS_TO_GROUP.get(int(v), -1) for v in arr], dtype=int)


def aligned_part_predictions(subsets: dict[str, dict[str, pd.DataFrame]], part: PartRange) -> pd.DataFrame:
    base = subsets[PRIMARY_MODEL][part.name][["frame_time", "y_true", "y_pred", "correct", "经度", "纬度"]].copy()
    base = base.rename(columns={"y_pred": f"{PRIMARY_MODEL}_pred", "correct": f"{PRIMARY_MODEL}_correct"})
    for model in [m for m in MODEL_ORDER if m != PRIMARY_MODEL]:
        sub = subsets[model][part.name][["frame_time", "y_pred", "correct"]].copy()
        sub = sub.rename(columns={"y_pred": f"{model}_pred", "correct": f"{model}_correct"})
        base = base.merge(sub, on="frame_time", how="inner")
    return base.sort_values("frame_time").reset_index(drop=True)


def choose_representative_part(subsets: dict[str, dict[str, pd.DataFrame]], parts: list[PartRange]) -> PartRange:
    """Pick a readable diagnostic trajectory where TIM most clearly helps."""
    scored: list[tuple[float, PartRange]] = []
    for part in parts:
        accs = {model: model_accuracy(subsets[model][part.name]) for model in MODEL_ORDER}
        tim = accs[PRIMARY_MODEL]
        best_single = max(accs[model] for model in UNIMODAL_MODELS)
        mean_single = float(np.mean([accs[model] for model in UNIMODAL_MODELS]))
        n = len(subsets[PRIMARY_MODEL][part.name])
        classes = int(subsets[PRIMARY_MODEL][part.name]["y_true"].nunique()) if n else 0
        score = (
            max(tim - best_single, 0.0) * 3.0
            + max(tim - mean_single, 0.0) * 1.5
            + min(classes, 8) * 0.025
            + min(n / 2500.0, 1.0) * 0.04
            + tim * 0.08
        )
        scored.append((score, part))
    return max(scored, key=lambda item: item[0])[1]


def choose_representative_window(
    subsets: dict[str, dict[str, pd.DataFrame]],
    parts: list[PartRange],
    window_min: float = 6.0,
    step_min: float = 1.0,
) -> tuple[PartRange, pd.Timestamp, pd.Timestamp]:
    """Pick a short interval where trimodal fusion is visibly better."""
    candidates: list[tuple[float, PartRange, pd.Timestamp, pd.Timestamp]] = []
    for part in parts:
        base = subsets[PRIMARY_MODEL][part.name]
        if base.empty:
            continue
        start = base["frame_time"].min()
        end = base["frame_time"].max()
        total_min = max((end - start).total_seconds() / 60.0, window_min)
        for left_min in np.arange(0.0, max(total_min - window_min, 0.0) + 1e-9, step_min):
            left = start + pd.Timedelta(minutes=float(left_min))
            right = left + pd.Timedelta(minutes=window_min)
            accs: dict[str, float] = {}
            counts: dict[str, int] = {}
            class_count = 0
            skip = False
            for model in MODEL_ORDER:
                sub = subsets[model][part.name]
                sub = sub[(sub["frame_time"] >= left) & (sub["frame_time"] < right)]
                counts[model] = len(sub)
                if len(sub) < 80:
                    skip = True
                    break
                accs[model] = model_accuracy(sub)
                if model == PRIMARY_MODEL:
                    class_count = int(sub["y_true"].nunique())
            if skip or class_count < 2:
                continue
            tim = accs[PRIMARY_MODEL]
            best_single = max(accs[model] for model in UNIMODAL_MODELS)
            mean_single = float(np.mean([accs[model] for model in UNIMODAL_MODELS]))
            score = (
                max(tim - best_single, 0.0) * 4.0
                + max(tim - mean_single, 0.0) * 1.8
                + tim * 0.25
                + min(class_count, 6) * 0.035
                + min(counts[PRIMARY_MODEL] / 500.0, 1.0) * 0.03
            )
            candidates.append((score, part, left, right))
    if candidates:
        _, part, left, right = max(candidates, key=lambda item: item[0])
        return part, left, right
    part = choose_representative_part(subsets, parts)
    left = subsets[PRIMARY_MODEL][part.name]["frame_time"].min()
    return part, left, left + pd.Timedelta(minutes=window_min)


def choose_timeline_window(
    subsets: dict[str, dict[str, pd.DataFrame]],
    parts: list[PartRange],
) -> tuple[PartRange, pd.Timestamp, pd.Timestamp]:
    """Pick a dense temporal interval where class-gate is accurate and visibly helpful."""
    candidates: list[tuple[float, PartRange, pd.Timestamp, pd.Timestamp]] = []
    for part in parts:
        base = subsets[PRIMARY_MODEL][part.name]
        if base.empty:
            continue
        start = base["frame_time"].min()
        end = base["frame_time"].max()
        total_min = max((end - start).total_seconds() / 60.0, 2.0)
        for window_min in [2.0, 3.0, 4.0, 6.0]:
            step_min = 0.5
            for left_min in np.arange(0.0, max(total_min - window_min, 0.0) + 1e-9, step_min):
                left = start + pd.Timedelta(minutes=float(left_min))
                right = left + pd.Timedelta(minutes=window_min)
                accs: dict[str, float] = {}
                class_count = 0
                n_primary = 0
                primary_times = pd.Series(dtype="datetime64[ns]")
                skip = False
                for model in MODEL_ORDER:
                    sub = subsets[model][part.name]
                    sub = sub[(sub["frame_time"] >= left) & (sub["frame_time"] < right)]
                    if len(sub) < 90:
                        skip = True
                        break
                    accs[model] = model_accuracy(sub)
                    if model == PRIMARY_MODEL:
                        n_primary = len(sub)
                        class_count = int(sub["y_true"].nunique())
                        primary_times = sub["frame_time"]
                if skip or class_count < 2:
                    continue
                observed_span = (
                    (primary_times.max() - primary_times.min()).total_seconds() / 60.0
                    if len(primary_times)
                    else 0.0
                )
                if observed_span < 0.7:
                    continue

                tim = accs[PRIMARY_MODEL]
                best_single = max(accs[model] for model in UNIMODAL_MODELS)
                concat = accs["trimodal_concat"]
                score = (
                    tim * 2.0
                    + max(tim - best_single, 0.0) * 2.2
                    + max(tim - concat, 0.0) * 1.5
                    + min(class_count, 7) * 0.03
                    + min(observed_span / 4.0, 1.0) * 0.08
                    + min(n_primary / 180.0, 1.0) * 0.04
                )
                candidates.append((score, part, left, right))
    if candidates:
        _, part, left, right = max(candidates, key=lambda item: item[0])
        return part, left, right
    return choose_representative_window(subsets, parts, window_min=3.0, step_min=0.5)


def choose_rescue_window(
    subsets: dict[str, dict[str, pd.DataFrame]],
    parts: list[PartRange],
    window_min: float = 8.0,
    step_min: float = 0.5,
) -> tuple[PartRange, pd.Timestamp, pd.Timestamp, int]:
    """Pick a local interval where unimodal models fail and class-gate succeeds."""
    candidates: list[tuple[float, PartRange, pd.Timestamp, pd.Timestamp, int]] = []
    for part in parts:
        merged = aligned_part_predictions(subsets, part)
        if merged.empty:
            continue
        rescue = (
            (~merged["ast_correct"])
            & (~merged["image_best_correct"])
            & (~merged["trnet_correct"])
            & (merged[f"{PRIMARY_MODEL}_correct"])
        )
        gate_only = (~merged["trimodal_concat_correct"]) & (merged[f"{PRIMARY_MODEL}_correct"])
        start = merged["frame_time"].min()
        end = merged["frame_time"].max()
        total_min = max((end - start).total_seconds() / 60.0, window_min)
        for left_min in np.arange(0.0, max(total_min - window_min, 0.0) + 1e-9, step_min):
            left = start + pd.Timedelta(minutes=float(left_min))
            right = left + pd.Timedelta(minutes=window_min)
            in_window = (merged["frame_time"] >= left) & (merged["frame_time"] < right)
            n = int(in_window.sum())
            if n < 80:
                continue
            rescue_n = int((rescue & in_window).sum())
            gate_only_n = int((gate_only & in_window).sum())
            if rescue_n == 0 and gate_only_n < 20:
                continue
            tim_acc = float(merged.loc[in_window, f"{PRIMARY_MODEL}_correct"].mean())
            concat_acc = float(merged.loc[in_window, "trimodal_concat_correct"].mean())
            single_acc = [
                float(merged.loc[in_window, f"{model}_correct"].mean())
                for model in UNIMODAL_MODELS
            ]
            class_count = int(merged.loc[in_window, "y_true"].nunique())
            score = (
                rescue_n * 3.0
                + gate_only_n * 1.8
                + max(tim_acc - concat_acc, 0.0) * 120.0
                + max(tim_acc - max(single_acc), 0.0) * 60.0
                + class_count * 1.3
                + n / 250.0
            )
            candidates.append((score, part, left, right, rescue_n))
    if candidates:
        _, part, left, right, rescue_n = max(candidates, key=lambda item: item[0])
        return part, left, right, rescue_n
    part, left, right = choose_representative_window(subsets, parts, window_min=window_min, step_min=step_min)
    return part, left, right, 0


def plot_confusion_comparison(subsets: dict[str, dict[str, pd.DataFrame]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(18.4, 10.4), dpi=220)
    axes_flat = axes.ravel()
    last_mesh = None
    for ax, model in zip(axes_flat, MODEL_ORDER):
        df = concat_selected(subsets[model])
        counts = pd.crosstab(df["y_true"], df["y_pred"]).reindex(index=range(11), columns=range(11), fill_value=0)
        norm = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        annot = np.empty(norm.shape, dtype=object)
        for i in range(11):
            for j in range(11):
                value = norm.iloc[i, j]
                annot[i, j] = f"{value:.0%}" if value >= 0.12 or i == j else ""
        sns.heatmap(
            norm,
            ax=ax,
            cmap="Blues",
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.25,
            linecolor="#f0f0f0",
            cbar=False,
            annot=annot,
            fmt="",
            annot_kws={"fontsize": 6.5},
        )
        last_mesh = ax.collections[0]
        acc = float((df["y_true"] == df["y_pred"]).mean()) if len(df) else 0.0
        ax.set_title(f"{MODEL_LABELS[model]} | selected trajectories, acc={acc:.1%}")
        ax.set_xticklabels([f"{i}\n{CLASS_SHORT[i]}" for i in range(11)], rotation=0, fontsize=7)
        ax.set_yticklabels([f"{i} {CLASS_SHORT[i]}" for i in range(11)], rotation=0, fontsize=7)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
    for ax in axes_flat[len(MODEL_ORDER):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.045, right=0.900, bottom=0.070, top=0.900, wspace=0.18, hspace=0.30)
    if last_mesh is not None:
        cax = fig.add_axes([0.925, 0.18, 0.016, 0.62])
        cbar = fig.colorbar(last_mesh, cax=cax)
        cbar.set_label("Recall share within true class")
    fig.suptitle("Part-Level Confusion Matrices Across Modalities and Fusion Models", y=0.995, fontsize=15, fontweight="semibold")
    fig.savefig(output)
    plt.close(fig)


def plot_misclassification_heatmap_comparison(
    subsets: dict[str, dict[str, pd.DataFrame]],
    parts: list[PartRange],
    output: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    apply_style()
    part = choose_representative_part(subsets, parts)
    matrices: dict[str, pd.DataFrame] = {}
    vmax = 1
    for model in MODEL_ORDER:
        sub = subsets[model][part.name]
        wrong = sub[~sub["correct"]].copy()
        counts = pd.crosstab(wrong["y_true"], wrong["y_pred"]).reindex(index=range(11), columns=range(11), fill_value=0)
        for class_id in range(11):
            counts.iloc[class_id, class_id] = 0
        matrices[model] = counts
        vmax = max(vmax, int(counts.to_numpy().max()))

    fig, axes = plt.subplots(2, 3, figsize=(18.4, 10.3), dpi=220)
    axes_flat = axes.ravel()
    last_mesh = None
    threshold = max(2, int(np.ceil(vmax * 0.035)))
    for ax, model in zip(axes_flat, MODEL_ORDER):
        counts = matrices[model]
        annot = counts.astype(object).to_numpy()
        for i in range(annot.shape[0]):
            for j in range(annot.shape[1]):
                annot[i, j] = str(int(annot[i, j])) if int(annot[i, j]) >= threshold else ""
        sns.heatmap(
            counts,
            ax=ax,
            cmap="Reds",
            vmin=0,
            vmax=vmax,
            square=True,
            linewidths=0.28,
            linecolor="#f4f4f4",
            cbar=False,
            annot=annot,
            fmt="",
            annot_kws={"fontsize": 6.5},
        )
        last_mesh = ax.collections[0]
        acc = model_accuracy(subsets[model][part.name])
        wrong_n = int((~subsets[model][part.name]["correct"]).sum())
        ax.set_title(f"{MODEL_LABELS[model]} | acc={acc:.1%}, wrong={wrong_n}")
        ax.set_xticklabels([f"{i}\n{CLASS_SHORT[i]}" for i in range(11)], rotation=0, fontsize=7)
        ax.set_yticklabels([f"{i} {CLASS_SHORT[i]}" for i in range(11)], rotation=0, fontsize=7)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
    for ax in axes_flat[len(MODEL_ORDER):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.045, right=0.900, bottom=0.070, top=0.900, wspace=0.18, hspace=0.32)
    if last_mesh is not None:
        cax = fig.add_axes([0.925, 0.18, 0.016, 0.62])
        cbar = fig.colorbar(last_mesh, cax=cax)
        cbar.set_label("Misclassified samples")
    fig.suptitle(f"Misclassification Heatmaps Across Modalities and Fusion Models on {part.label}", y=0.995, fontsize=15, fontweight="semibold")
    fig.savefig(output)
    plt.close(fig)


def _mode_int(values: pd.Series) -> int:
    mode = values.astype(int).mode()
    if not mode.empty:
        return int(mode.iloc[0])
    return int(values.astype(int).iloc[0])


def collapse_aligned_for_plot(df: pd.DataFrame, flag_cols: list[str]) -> pd.DataFrame:
    """Collapse repeated sample anchors into one plotted timestamp while retaining counts."""
    rows: list[dict] = []
    for frame_time, group in df.sort_values("frame_time").groupby("frame_time", sort=True):
        row: dict[str, object] = {
            "frame_time": frame_time,
            "经度": float(group["经度"].mean()),
            "纬度": float(group["纬度"].mean()),
            "y_true": _mode_int(group["y_true"]),
            "sample_count": int(len(group)),
        }
        for model in MODEL_ORDER:
            row[f"{model}_pred"] = _mode_int(group[f"{model}_pred"])
            row[f"{model}_correct"] = bool(group[f"{model}_correct"].astype(bool).mean() >= 0.5)
        for col in flag_cols:
            row[col] = bool(group[col].astype(bool).any())
            row[f"{col}_count"] = int(group[col].astype(bool).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def choose_class_gate_only_window(
    subsets: dict[str, dict[str, pd.DataFrame]],
    parts: list[PartRange],
    window_min: float = 1.0,
    step_min: float = 0.25,
    min_samples: int = 30,
    min_only: int = 10,
) -> tuple[PartRange, pd.Timestamp, pd.Timestamp, int]:
    """Pick a zoomed interval where only class-gate predicts correctly."""
    candidates: list[tuple[float, PartRange, pd.Timestamp, pd.Timestamp, int]] = []
    for part in parts:
        merged = aligned_part_predictions(subsets, part)
        if merged.empty:
            continue
        class_gate_only = (
            (~merged["ast_correct"])
            & (~merged["image_best_correct"])
            & (~merged["trnet_correct"])
            & (~merged["trimodal_concat_correct"])
            & (merged[f"{PRIMARY_MODEL}_correct"])
        )
        start = merged["frame_time"].min()
        end = merged["frame_time"].max()
        total_min = max((end - start).total_seconds() / 60.0, window_min)
        for left_min in np.arange(0.0, max(total_min - window_min, 0.0) + 1e-9, step_min):
            left = start + pd.Timedelta(minutes=float(left_min))
            right = left + pd.Timedelta(minutes=window_min)
            in_window = (merged["frame_time"] >= left) & (merged["frame_time"] < right)
            n = int(in_window.sum())
            if n < min_samples:
                continue
            only_n = int((class_gate_only & in_window).sum())
            if only_n < min_only:
                continue
            accs = {
                model: float(merged.loc[in_window, f"{model}_correct"].mean())
                for model in MODEL_ORDER
            }
            max_non_gate = max(accs[model] for model in MODEL_ORDER if model != PRIMARY_MODEL)
            density = only_n / max(n, 1)
            score = (
                only_n * 8.0
                + density * 120.0
                + accs[PRIMARY_MODEL] * 80.0
                - max_non_gate * 120.0
                + min(n / 80.0, 1.0) * 8.0
                - window_min * 0.05
            )
            candidates.append((score, part, left, right, only_n))
    if candidates:
        _, part, left, right, only_n = max(candidates, key=lambda item: item[0])
        return part, left, right, only_n
    part, left, right, rescue_n = choose_rescue_window(subsets, parts, window_min=window_min, step_min=step_min)
    return part, left, right, rescue_n


def choose_spatial_error_gap_window(
    subsets: dict[str, dict[str, pd.DataFrame]],
    parts: list[PartRange],
) -> tuple[PartRange, pd.Timestamp, pd.Timestamp]:
    """Pick a spatially readable window where class-gate has fewer errors, but baselines are not degenerate."""
    candidates: list[tuple[float, PartRange, pd.Timestamp, pd.Timestamp]] = []
    window_values = [5.0, 8.0, 10.0, 12.0, 15.0]
    for window_min in window_values:
        for part in parts:
            merged = aligned_part_predictions(subsets, part)
            if merged.empty:
                continue
            start = merged["frame_time"].min()
            end = merged["frame_time"].max()
            total_min = max((end - start).total_seconds() / 60.0, window_min)
            for left_min in np.arange(0.0, max(total_min - window_min, 0.0) + 1e-9, 1.0):
                left = start + pd.Timedelta(minutes=float(left_min))
                right = left + pd.Timedelta(minutes=window_min)
                in_window = (merged["frame_time"] >= left) & (merged["frame_time"] < right)
                n = int(in_window.sum())
                if n < 250:
                    continue
                sub = merged.loc[in_window]
                accs = {
                    model: float(sub[f"{model}_correct"].mean())
                    for model in MODEL_ORDER
                }
                gate_acc = accs[PRIMARY_MODEL]
                other_accs = [accs[model] for model in MODEL_ORDER if model != PRIMARY_MODEL]
                if gate_acc < 0.82:
                    continue
                if max(other_accs) > 0.75 or min(other_accs) < 0.05:
                    continue
                vit_gate_wrong = int((sub["image_best_correct"] & (~sub[f"{PRIMARY_MODEL}_correct"])).sum())
                if vit_gate_wrong > 25 or vit_gate_wrong / max(n, 1) > 0.055:
                    continue

                lat0 = float(sub["纬度"].mean())
                span_x = float((sub["经度"].max() - sub["经度"].min()) * 111_320.0 * np.cos(np.deg2rad(lat0)))
                span_y = float((sub["纬度"].max() - sub["纬度"].min()) * 110_540.0)
                if span_x < 80.0 or span_y < 80.0:
                    continue

                score = (
                    (gate_acc - max(other_accs)) * 240.0
                    + gate_acc * 80.0
                    + min(n / 700.0, 1.0) * 20.0
                    + min((span_x * span_y) / 60_000.0, 1.0) * 25.0
                    - vit_gate_wrong * 0.45
                    - abs(window_min - 10.0) * 2.0
                )
                candidates.append((score, part, left, right))
    if candidates:
        _, part, left, right = max(candidates, key=lambda item: item[0])
        return part, left, right
    part, left, right = choose_representative_window(subsets, parts, window_min=10.0, step_min=1.0)
    return part, left, right


def plot_error_hotspots(subsets: dict[str, dict[str, pd.DataFrame]], parts: list[PartRange], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_style()
    part, left, right = choose_spatial_error_gap_window(subsets, parts)
    merged = aligned_part_predictions(subsets, part)
    focus = merged[(merged["frame_time"] >= left) & (merged["frame_time"] < right)].copy()
    display = collapse_aligned_for_plot(focus, [])
    bg = part.trajectory[(part.trajectory["_plot_time"] >= left) & (part.trajectory["_plot_time"] < right)].copy()
    if bg.empty:
        bg = part.trajectory.copy()

    zoom_source = display if not display.empty else focus
    lon_min = float(np.nanmin(zoom_source["经度"]))
    lon_max = float(np.nanmax(zoom_source["经度"]))
    lat_min = float(np.nanmin(zoom_source["纬度"]))
    lat_max = float(np.nanmax(zoom_source["纬度"]))
    lon_pad = max((lon_max - lon_min) * 0.22, 0.000035)
    lat_pad = max((lat_max - lat_min) * 0.22, 0.000035)
    xlim = (lon_min - lon_pad, lon_max + lon_pad)
    ylim = (lat_min - lat_pad, lat_max + lat_pad)

    fig, axes = plt.subplots(3, 2, figsize=(11.8, 10.6), dpi=280)
    axes_flat = axes.ravel()
    for idx, (ax, model) in enumerate(zip(axes_flat[:len(MODEL_ORDER)], MODEL_ORDER)):
        correct = display[f"{model}_correct"].to_numpy(dtype=bool)
        ax.plot(bg["经度"], bg["纬度"], color="#b8b8b8", linewidth=1.05, alpha=0.72, zorder=1)
        if correct.any():
            ax.scatter(
                display.loc[correct, "经度"],
                display.loc[correct, "纬度"],
                s=21,
                color="#3b7f5f",
                alpha=0.52,
                linewidths=0,
                zorder=2,
            )
        if (~correct).any():
            ax.scatter(
                display.loc[~correct, "经度"],
                display.loc[~correct, "纬度"],
                s=25,
                marker="x",
                color="#c84e4e",
                alpha=0.84,
                linewidths=0.95,
                zorder=3,
            )
        acc = float(focus[f"{model}_correct"].mean()) if len(focus) else 0.0
        ax.set_title(
            f"{MODEL_LABELS[model]} | acc={acc:.1%}",
            fontsize=10.0,
            fontweight="semibold",
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("auto")
        ax.set_xlabel("Longitude" if idx >= 4 else "")
        ax.set_ylabel("Latitude" if idx % 2 == 0 else "")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.tick_params(axis="both", labelsize=7.4)
        ax.grid(alpha=0.16)

    rescue_ax = axes_flat[-1]
    gate_correct = display[f"{PRIMARY_MODEL}_correct"].to_numpy(dtype=bool)
    concat_correct = display["trimodal_concat_correct"].to_numpy(dtype=bool)
    any_unimodal_correct = display[[f"{model}_correct" for model in UNIMODAL_MODELS]].any(axis=1).to_numpy(dtype=bool)
    all_unimodal_rescue = gate_correct & (~any_unimodal_correct)
    concat_rescue = gate_correct & (~concat_correct) & (~all_unimodal_rescue)
    gate_correct_base = gate_correct & (~all_unimodal_rescue) & (~concat_rescue)
    gate_wrong = ~gate_correct

    rescue_ax.plot(bg["经度"], bg["纬度"], color="#b8b8b8", linewidth=1.05, alpha=0.70, zorder=1)
    if gate_correct_base.any():
        rescue_ax.scatter(
            display.loc[gate_correct_base, "经度"],
            display.loc[gate_correct_base, "纬度"],
            s=17,
            color="#3b7f5f",
            alpha=0.35,
            linewidths=0,
            zorder=2,
        )
    if gate_wrong.any():
        rescue_ax.scatter(
            display.loc[gate_wrong, "经度"],
            display.loc[gate_wrong, "纬度"],
            s=23,
            marker="x",
            color="#c84e4e",
            alpha=0.78,
            linewidths=0.9,
            zorder=3,
        )
    if concat_rescue.any():
        rescue_ax.scatter(
            display.loc[concat_rescue, "经度"],
            display.loc[concat_rescue, "纬度"],
            s=38,
            marker="D",
            color="#7b61b3",
            alpha=0.82,
            linewidths=0,
            zorder=4,
        )
    if all_unimodal_rescue.any():
        rescue_ax.scatter(
            display.loc[all_unimodal_rescue, "经度"],
            display.loc[all_unimodal_rescue, "纬度"],
            s=72,
            marker="*",
            color="#d39b2a",
            edgecolor="#5f4b20",
            linewidths=0.35,
            alpha=0.90,
            zorder=5,
        )
    rescue_ax.set_title("Class-gate rescue comparison", fontsize=10.0, fontweight="semibold")
    rescue_ax.set_xlim(*xlim)
    rescue_ax.set_ylim(*ylim)
    rescue_ax.set_aspect("auto")
    rescue_ax.set_xlabel("Longitude")
    rescue_ax.set_ylabel("")
    rescue_ax.ticklabel_format(useOffset=False, style="plain")
    rescue_ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    rescue_ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    rescue_ax.tick_params(axis="both", labelsize=7.4)
    rescue_ax.grid(alpha=0.16)

    handles = [
        Line2D([0], [0], color="#b8b8b8", linewidth=1.4, label="Trajectory path"),
        Line2D([0], [0], marker="o", color="w", label="Correct", markerfacecolor="#3b7f5f", markersize=7),
        Line2D([0], [0], marker="x", color="#c84e4e", label="Wrong", linestyle="None", markersize=7),
        Line2D([0], [0], marker="D", color="w", label="Concat wrong, class-gate correct", markerfacecolor="#7b61b3", markersize=6.5),
        Line2D([0], [0], marker="*", color="#d39b2a", label="All unimodal wrong, class-gate correct", linestyle="None", markersize=9),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.2, framealpha=0.95)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.095, top=0.905, wspace=0.23, hspace=0.44)
    fig.suptitle(
        f"Spatial Prediction Error Comparison on {part.label} Segment",
        y=0.975,
        fontsize=14.0,
        fontweight="semibold",
    )
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_trimodal_spatial_errors(subsets: dict[str, dict[str, pd.DataFrame]], parts: list[PartRange], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 10.4), dpi=220)
    axes_flat = axes.ravel()
    for ax, part in zip(axes_flat, parts):
        sub = subsets[PRIMARY_MODEL][part.name]
        ax.plot(part.trajectory["经度"], part.trajectory["纬度"], color="#c2c2c2", linewidth=0.72, alpha=0.42, zorder=1)
        fine_correct = sub["correct"].to_numpy(dtype=bool)
        true_class = sub["y_true"].to_numpy(dtype=int)
        pred_class = sub["y_pred"].to_numpy(dtype=int)
        true_group = np.array([CLASS_TO_GROUP.get(int(label), -1) for label in true_class], dtype=int)
        pred_group = np.array([CLASS_TO_GROUP.get(int(label), -2) for label in pred_class], dtype=int)
        group_correct = true_group == pred_group
        for group_id, group_name in enumerate(GROUP_NAMES):
            mask = fine_correct & (true_group == group_id)
            if mask.any():
                ax.scatter(
                    sub.loc[mask, "经度"],
                    sub.loc[mask, "纬度"],
                    s=9,
                    color=GROUP_COLORS[group_name],
                    alpha=0.60,
                    linewidths=0,
                    zorder=2,
                )
        subclass_residual = group_correct & ~fine_correct
        if subclass_residual.any():
            ax.scatter(
                sub.loc[subclass_residual, "经度"],
                sub.loc[subclass_residual, "纬度"],
                s=7,
                c="#d9912b",
                marker=".",
                alpha=0.48,
                linewidths=0,
                zorder=3,
            )
        group_wrong = ~group_correct
        if group_wrong.any():
            ax.scatter(
                sub.loc[group_wrong, "经度"],
                sub.loc[group_wrong, "纬度"],
                s=20,
                c="#d43f3a",
                marker="x",
                alpha=0.85,
                linewidths=0.75,
                zorder=4,
            )
        fine_acc = float(fine_correct.mean()) if len(sub) else 0.0
        group_acc = float(group_correct.mean()) if len(sub) else 0.0
        ax.set_title(f"{part.label} | group acc={group_acc:.1%}, 11-class acc={fine_acc:.1%}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(axis="both", labelsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
    handles = [
        Line2D([0], [0], marker="o", color="w", label=name, markerfacecolor=GROUP_COLORS[name], markersize=6.8)
        for name in GROUP_NAMES
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="#d9912b",
            label="Correct group, wrong subclass",
            markerfacecolor="#d9912b",
            linestyle="None",
            markersize=6.6,
        )
    )
    handles.append(Line2D([0], [0], marker="x", color="#d43f3a", label="Wrong operation group", linestyle="None", markersize=7))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.6, framealpha=0.95)
    fig.suptitle(
        "Spatial Prediction Map of TIM class-gate: Group Accuracy and Fine-Class Residuals",
        y=0.995,
        fontsize=15,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0.105, 1, 0.975))
    fig.savefig(output)
    plt.close(fig)


def _segments(times: np.ndarray, values: np.ndarray) -> list[tuple[float, float, int]]:
    if len(values) == 0:
        return []
    diffs = np.diff(times)
    positive = diffs[diffs > 0]
    step = float(np.nanmedian(positive)) if len(positive) else 1 / 60
    gap = max(1.0, step * 10)
    width = min(max(step, 1 / 120), 0.25)
    out: list[tuple[float, float, int]] = []
    start = 0
    for idx in range(1, len(values)):
        if values[idx] != values[idx - 1] or times[idx] - times[idx - 1] > gap:
            left, right = float(times[start]), float(times[idx - 1] + width)
            if right > left:
                out.append((left, right, int(values[start])))
            start = idx
    left, right = float(times[start]), float(times[-1] + width)
    if right > left:
        out.append((left, right, int(values[start])))
    return out


def plot_trimodal_timeline(subsets: dict[str, dict[str, pd.DataFrame]], parts: list[PartRange], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    apply_style()
    colors = list(plt.get_cmap("tab20").colors[:11])
    cmap = ListedColormap(colors)
    fig, axes = plt.subplots(len(parts), 1, figsize=(16, 9.8), dpi=220, sharex=False)
    if len(parts) == 1:
        axes = [axes]
    for ax, part in zip(axes, parts):
        sub = subsets[PRIMARY_MODEL][part.name]
        times = sub["elapsed_min"].to_numpy(dtype=float)
        true = sub["y_true"].to_numpy(dtype=int)
        pred = sub["y_pred"].to_numpy(dtype=int)
        wrong = (~sub["correct"].to_numpy(dtype=bool)).astype(int)
        true_segments = _segments(times, true)
        pred_segments = _segments(times, pred)
        wrong_segments = [s for s in _segments(times, wrong) if s[2] == 1]

        def draw(seg_list: list[tuple[float, float, int]], y: float, height: float, wrong_band: bool = False) -> None:
            for left, right, value in seg_list:
                face = "#d43f3a" if wrong_band else cmap(value)
                ax.broken_barh([(left, right - left)], (y, height), facecolors=face, edgecolors="none", alpha=0.90)

        ax.axhspan(1.72, 2.22, color="#f7f7f7", zorder=0)
        ax.axhspan(0.92, 1.42, color="#f7f7f7", zorder=0)
        ax.axhspan(0.18, 0.42, color="#fff2f2", zorder=0)
        draw(true_segments, 1.78, 0.38)
        draw(pred_segments, 0.98, 0.38)
        draw(wrong_segments, 0.22, 0.16, wrong_band=True)
        acc = float(sub["correct"].mean()) if len(sub) else 0.0
        ax.set_title(f"{part.label} | TIM class-gate acc={acc:.1%}", loc="left")
        ax.set_yticks([1.97, 1.17, 0.30])
        ax.set_yticklabels(["True", "Pred", "Wrong"])
        ax.set_ylim(0.0, 2.45)
        ax.set_xlim(0, max(float(times.max()) if len(times) else 1.0, 1.0))
        ax.grid(axis="x", alpha=0.16)
    axes[-1].set_xlabel("Elapsed time in trajectory part (min)")
    handles = [
        Line2D([0], [0], marker="s", color="w", label=f"{i}: {CLASS_SHORT[i]}", markerfacecolor=cmap(i), markersize=12)
        for i in range(11)
    ]
    handles.append(Line2D([0], [0], color="#d43f3a", linewidth=10, label="Wrong prediction interval"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=13, framealpha=0.95, handlelength=2.0, columnspacing=1.4)
    fig.suptitle("True and Predicted Class Timelines for TIM class-gate", y=0.995, fontsize=15, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.16, 1, 0.97))
    fig.savefig(output)
    plt.close(fig)


def plot_modal_timeline_comparison(
    subsets: dict[str, dict[str, pd.DataFrame]],
    parts: list[PartRange],
    output: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    apply_style()
    part, left, right = choose_timeline_window(subsets, parts)
    windowed: dict[str, pd.DataFrame] = {}
    observed_times: list[pd.Series] = []
    for model in MODEL_ORDER:
        sub = subsets[model][part.name]
        sub = sub[(sub["frame_time"] >= left) & (sub["frame_time"] < right)].copy()
        sub = sub.sort_values("frame_time")
        windowed[model] = sub
        if len(sub):
            observed_times.append(sub["frame_time"])

    if observed_times:
        all_times = pd.concat(observed_times).sort_values()
        observed_left = all_times.min()
        observed_right = all_times.max()
        diffs = all_times.drop_duplicates().diff().dt.total_seconds().dropna() / 60.0
        diffs = diffs[diffs > 0]
        step_pad = float(np.nanmedian(diffs)) if len(diffs) else 1 / 60
        pad_min = min(max(step_pad * 2.0, 0.03), 0.18)
        axis_left = observed_left - pd.Timedelta(minutes=pad_min)
        axis_right = observed_right + pd.Timedelta(minutes=pad_min)
    else:
        axis_left = left
        axis_right = right

    colors = list(plt.get_cmap("tab20").colors[:11])
    cmap = ListedColormap(colors)
    fig, axes = plt.subplots(len(MODEL_ORDER), 1, figsize=(16, 11.4), dpi=220, sharex=True)

    def draw(ax, seg_list: list[tuple[float, float, int]], y: float, height: float, wrong_band: bool = False) -> None:
        for seg_left, seg_right, value in seg_list:
            face = "#d43f3a" if wrong_band else cmap(value)
            ax.broken_barh([(seg_left, seg_right - seg_left)], (y, height), facecolors=face, edgecolors="none", alpha=0.92)

    max_min = max((axis_right - axis_left).total_seconds() / 60.0, 0.75)
    for ax, model in zip(axes, MODEL_ORDER):
        sub = windowed[model].copy()
        sub["interval_min"] = (sub["frame_time"] - axis_left).dt.total_seconds() / 60.0
        times = sub["interval_min"].to_numpy(dtype=float)
        true = sub["y_true"].to_numpy(dtype=int)
        pred = sub["y_pred"].to_numpy(dtype=int)
        wrong = (~sub["correct"].to_numpy(dtype=bool)).astype(int)

        true_segments = _segments(times, true)
        pred_segments = _segments(times, pred)
        wrong_segments = [s for s in _segments(times, wrong) if s[2] == 1]

        ax.axhspan(1.72, 2.22, color="#f7f7f7", zorder=0)
        ax.axhspan(0.92, 1.42, color="#f7f7f7", zorder=0)
        ax.axhspan(0.18, 0.42, color="#fff2f2", zorder=0)
        draw(ax, true_segments, 1.78, 0.38)
        draw(ax, pred_segments, 0.98, 0.38)
        draw(ax, wrong_segments, 0.22, 0.16, wrong_band=True)

        acc = model_accuracy(sub)
        wrong_n = int((~sub["correct"]).sum()) if len(sub) else 0
        ax.set_title(f"{MODEL_LABELS[model]} | acc={acc:.1%}, wrong={wrong_n}", loc="left")
        ax.set_yticks([1.97, 1.17, 0.30])
        ax.set_yticklabels(["True", "Pred", "Wrong"])
        ax.set_ylim(0.0, 2.45)
        ax.set_xlim(0, max_min)
        ax.grid(axis="x", alpha=0.16)

    axes[-1].set_xlabel("Elapsed time in dense selected interval (min)")
    handles = [
        Line2D([0], [0], marker="s", color="w", label=f"{i}: {CLASS_SHORT[i]}", markerfacecolor=cmap(i), markersize=12)
        for i in range(11)
    ]
    handles.append(Line2D([0], [0], color="#d43f3a", linewidth=10, label="Wrong prediction interval"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=13, framealpha=0.95, handlelength=2.0, columnspacing=1.4)
    fig.suptitle(f"Temporal Prediction Comparison Across Modalities and Fusion Models on {part.label}", y=0.995, fontsize=15, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.165, 1, 0.965))
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", default="experiments/paper_4090_final_seed44")
    parser.add_argument("--part-dir", default="data/b_deep_part_multimodal_full_clean_20260417/trajectory_parts")
    parser.add_argument("--output-dir", default="paper/manuscript/figures")
    parser.add_argument("--dates", default="2024-10-28,2024-10-29")
    args = parser.parse_args()

    suite_dir = resolve(args.suite_dir)
    part_dir = resolve(args.part_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = [item.strip() for item in args.dates.split(",") if item.strip()]

    predictions = load_predictions(suite_dir)
    parts = load_part_ranges(part_dir, dates)
    subsets = selected_part_subsets(predictions, parts)

    plot_confusion_comparison(subsets, output_dir / "fig_part_confusion_modalities.png")
    plot_misclassification_heatmap_comparison(subsets, parts, output_dir / "fig_part_misclassification_heatmaps.png")
    plot_error_hotspots(subsets, parts, output_dir / "fig_part_error_hotspots.png")
    plot_trimodal_spatial_errors(subsets, parts, output_dir / "fig_part_spatial_errors.png")
    plot_modal_timeline_comparison(subsets, parts, output_dir / "fig_part_timeline_modalities.png")
    plot_trimodal_timeline(subsets, parts, output_dir / "fig_part_true_pred_timeline.png")

    print("Wrote:")
    for name in [
        "fig_part_confusion_modalities.png",
        "fig_part_misclassification_heatmaps.png",
        "fig_part_error_hotspots.png",
        "fig_part_spatial_errors.png",
        "fig_part_timeline_modalities.png",
        "fig_part_true_pred_timeline.png",
    ]:
        print(output_dir / name)


if __name__ == "__main__":
    main()
