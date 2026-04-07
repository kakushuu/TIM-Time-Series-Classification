#!/usr/bin/env python3
"""
Agri-MBT: 农业视频 + 轨迹数据对齐工具（v2 严格版）
====================================================

核心流程：
  1. cv2 每秒提取一帧（保留 1920×1080 原始分辨率）
  2. pytesseract OCR 识别每帧左上角时间戳
  3. 时间戳校验：基于相邻帧一致性检查，剔除 OCR 误读帧
  4. 时间去重：同一秒只保留一张帧图片
  5. 仅对短间隔（≤ MAX_INTERP_GAP 秒）的 OCR 失败帧做插值
     超出间隔的帧直接排除（不在 CSV 中出现）
  6. 按识别时间顺序匹配轨迹数据（容差 ±2 秒）
  7. 生成对齐 CSV

相比 v1 的关键修复：
  旧版 RANSAC 会把正确的 OCR 读数当"异常值"丢弃，并用全局
  线性拟合替代——当拟合直线本身有偏差时（长段失败、变帧率），
  最终时间戳可能偏差数分钟。
  新版策略：
    - OCR 成功的帧永远不覆盖，只用相邻帧做一致性检查剔除误读
    - OCR 失败的帧只在短间隔内插值，长段失败直接排除

ocr_status 取值：
  ok           - OCR 成功且通过一致性检查，时间戳可信
  ocr_error    - OCR 成功但与相邻帧不一致（误读），已排除
  interpolated - OCR 失败，但距两侧 ok 帧各在 MAX_INTERP_GAP 以内，线性插值
  excluded     - OCR 失败且间隔过大，已排除

用法：
  python scripts/align_agri_data.py \\
      --trajectory data/trajectory/B-2024-10-19/trajectory_B_20241019.csv \\
      --video-dir   data/video/B-2024-10-19 \\
      --output      data/aligned_output/B-2024-10-19 \\
      --traj-format csv

  python scripts/align_agri_data.py \\
      --trajectory data/trajectory/B-2024-10-18/12-12-49_23-59-58.xlsx \\
      --video-dir   data/video/B-2024-10-18 \\
      --output      data/aligned_output \\
      --traj-format xlsx
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# OCR 区域（测试验证：h=5%, w=30%, thresh=200, psm7 为最优配置）
OCR_REGIONS = [
    (0.05, 0.30),
]
OCR_THRESHOLDS = [200]
OCR_CONFIGS = [
    r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:- ',
    r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:- ',
]

# 时间戳正则（兼容日期与时间间无空格，如 "2024-10-1907:11:55"）
TS_PATTERNS = [
    r'(\d{4})-(\d{2})-(\d{2})\s*(\d{2}):(\d{2}):(\d{2})',
    r'(\d{4})/(\d{2})/(\d{2})\s*(\d{2}):(\d{2}):(\d{2})',
]

# 无意义列
COLUMNS_TO_DROP = [
    '上点时间', '补点', 'unitid', 'GPS标记',
    '播种播肥 / 油耗 / 压力', '播种播肥/油耗/压力',
    '抛肥量(立方)', '定位间隔',
]

# 时间戳一致性检查：相邻 OCR 帧之间速率应为 ~1 s/s，允许范围
OCR_RATE_MIN = 0.5   # 秒/秒（允许视频轻微慢放）
OCR_RATE_MAX = 1.5   # 秒/秒（允许视频轻微快放）

# 插值最大间隔：OCR 失败帧距最近有效锚点超过此秒数则排除
MAX_INTERP_GAP_S: int = 5

# 轨迹匹配容差（秒）
TRAJ_TOLERANCE_S = 2


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameRecord:
    second_offset: int
    frame_number: int
    video_file: str
    ocr_time: Optional[datetime]
    repaired_time: Optional[datetime] = None
    ocr_status: str = 'unknown'   # ok | ocr_error | interpolated | excluded
    _saved_frame_path: Optional[str] = None  # 临时属性，运行时赋值


# ──────────────────────────────────────────────────────────────────────────────
# OCR helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_ts(text: str) -> Optional[datetime]:
    for pat in TS_PATTERNS:
        m = re.search(pat, text)
        if m:
            try:
                return datetime(*map(int, m.groups()))
            except ValueError:
                continue
    return None


def ocr_frame_timestamp(frame: np.ndarray) -> Optional[datetime]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    for h_ratio, w_ratio in OCR_REGIONS:
        region = gray[0:int(h * h_ratio), 0:int(w * w_ratio)]
        for thresh in OCR_THRESHOLDS:
            _, binary = cv2.threshold(region, thresh, 255, cv2.THRESH_BINARY)
            pil_img = Image.fromarray(binary)
            for cfg in OCR_CONFIGS:
                try:
                    text = pytesseract.image_to_string(pil_img, config=cfg).strip()
                    ts = _parse_ts(text)
                    if ts is not None:
                        return ts
                except Exception:
                    continue
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Timestamp repair  (v2 — strict, no global fit)
# ──────────────────────────────────────────────────────────────────────────────

def repair_timestamps_strict(
    records: List[FrameRecord],
    max_interp_gap: int = MAX_INTERP_GAP_S,
) -> List[FrameRecord]:
    """
    严格时间戳修复（v2）。

    策略：
      1. 收集所有 OCR 成功的帧，用相邻帧速率检查过滤误读
         - 速率在 [OCR_RATE_MIN, OCR_RATE_MAX] 之间且与至少一侧邻帧一致 → 'ok'
         - 否则 → 'ocr_error'，排除，不覆盖为拟合值
      2. 对 OCR 失败的帧（ocr_time=None）：
         - 找最近的左侧和右侧 'ok' 锚点
         - 两侧间隔均 ≤ max_interp_gap → 线性插值，'interpolated'
         - 否则 → 'excluded'
    """
    if not records:
        return records

    # ── Step 1: 找出所有 OCR 成功的帧并做速率一致性检查 ──────────────────────
    # 先收集候选锚点（OCR 成功）
    candidates: List[Tuple[int, float]] = [
        (r.second_offset, r.ocr_time.timestamp())
        for r in records if r.ocr_time is not None
    ]

    # 一致性检查：遍历相邻候选对，标记速率异常的帧
    bad_offsets: set = set()
    for i in range(len(candidates) - 1):
        off_a, ts_a = candidates[i]
        off_b, ts_b = candidates[i + 1]
        delta_off = off_b - off_a
        if delta_off <= 0:
            continue
        rate = (ts_b - ts_a) / delta_off
        if not (OCR_RATE_MIN <= rate <= OCR_RATE_MAX):
            # 这一对速率异常；找哪一端与另一邻居不一致
            # 简单策略：如果左邻也异常则标记 candidates[i+1]，否则标记 candidates[i]
            if i > 0:
                off_prev, ts_prev = candidates[i - 1]
                rate_prev = (ts_a - ts_prev) / max(1, off_a - off_prev)
                if OCR_RATE_MIN <= rate_prev <= OCR_RATE_MAX:
                    # candidates[i] 与左邻一致，问题在 candidates[i+1]
                    bad_offsets.add(off_b)
                else:
                    bad_offsets.add(off_a)
            else:
                # 没有左邻，先标记 candidates[i+1] 为可疑
                bad_offsets.add(off_b)

    # 构建"干净"锚点列表（排除 bad_offsets）
    clean_anchors: List[Tuple[int, float]] = [
        (off, ts) for off, ts in candidates if off not in bad_offsets
    ]

    # 如果经过 bad_offsets 过滤后某端仍然造成速率异常，再做一轮
    for _ in range(3):
        updated_bad: set = set()
        for i in range(len(clean_anchors) - 1):
            off_a, ts_a = clean_anchors[i]
            off_b, ts_b = clean_anchors[i + 1]
            rate = (ts_b - ts_a) / max(1, off_b - off_a)
            if not (OCR_RATE_MIN <= rate <= OCR_RATE_MAX):
                # 标记速率较偏的一端
                if i > 0:
                    rate_l = (ts_a - clean_anchors[i - 1][1]) / max(1, off_a - clean_anchors[i - 1][0])
                    if OCR_RATE_MIN <= rate_l <= OCR_RATE_MAX:
                        updated_bad.add(off_b)
                    else:
                        updated_bad.add(off_a)
                else:
                    updated_bad.add(off_b)
        if not updated_bad:
            break
        bad_offsets |= updated_bad
        clean_anchors = [(off, ts) for off, ts in clean_anchors if off not in bad_offsets]

    # 转换为按 second_offset 排序的列表
    clean_anchors.sort(key=lambda x: x[0])
    clean_set: set = {off for off, _ in clean_anchors}

    # ── Step 2: 对每帧赋予最终 repaired_time 和 ocr_status ──────────────────
    for r in records:
        if r.ocr_time is not None:
            if r.second_offset in clean_set:
                r.repaired_time = r.ocr_time
                r.ocr_status = 'ok'
            else:
                # OCR 误读（速率异常）
                r.repaired_time = None
                r.ocr_status = 'ocr_error'
        else:
            # OCR 失败：尝试短间隔插值
            off = r.second_offset
            # 找左侧最近 ok 锚点
            left = None
            for a_off, a_ts in reversed(clean_anchors):
                if a_off < off:
                    left = (a_off, a_ts)
                    break
            # 找右侧最近 ok 锚点
            right = None
            for a_off, a_ts in clean_anchors:
                if a_off > off:
                    right = (a_off, a_ts)
                    break

            left_gap = (off - left[0]) if left else float('inf')
            right_gap = (right[0] - off) if right else float('inf')

            if left_gap <= max_interp_gap and right_gap <= max_interp_gap:
                # 双侧插值（更精确）
                l_off, l_ts = left
                r_off, r_ts = right
                interp_ts = l_ts + (off - l_off) * (r_ts - l_ts) / (r_off - l_off)
                r.repaired_time = datetime.fromtimestamp(interp_ts)
                r.ocr_status = 'interpolated'
            elif left_gap <= max_interp_gap:
                # 仅左侧锚点够近：向右外推
                l_off, l_ts = left
                r.repaired_time = datetime.fromtimestamp(l_ts + (off - l_off))
                r.ocr_status = 'interpolated'
            elif right_gap <= max_interp_gap:
                # 仅右侧锚点够近：向左外推
                r_off, r_ts = right
                r.repaired_time = datetime.fromtimestamp(r_ts - (r_off - off))
                r.ocr_status = 'interpolated'
            else:
                # 间隔过大，排除
                r.repaired_time = None
                r.ocr_status = 'excluded'

    return records


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory loading
# ──────────────────────────────────────────────────────────────────────────────

def load_trajectory(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == 'xlsx':
        df = pd.read_excel(path)
    elif fmt == 'csv':
        df = pd.read_csv(path)
    else:
        raise ValueError(f"不支持的格式: {fmt}")

    drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if drop:
        df = df.drop(columns=drop)

    df['定位时间'] = pd.to_datetime(df['定位时间'])
    df = df.sort_values('定位时间').drop_duplicates('定位时间').reset_index(drop=True)
    df['_ts'] = df['定位时间'].apply(lambda x: int(pd.Timestamp(x).timestamp()))
    return df


def match_trajectory(
    ts: datetime,
    traj_df: pd.DataFrame,
    ts_index: Dict[int, int],
    tolerance: int = TRAJ_TOLERANCE_S,
) -> Optional[int]:
    target = int(pd.Timestamp(ts).timestamp())
    for delta in range(tolerance + 1):
        for offset in ([0] if delta == 0 else [-delta, delta]):
            idx = ts_index.get(target + offset)
            if idx is not None:
                return idx
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Video processing
# ──────────────────────────────────────────────────────────────────────────────

def extract_frames_from_video(
    video_path: Path,
    frames_dir: Path,
    max_interp_gap: int = MAX_INTERP_GAP_S,
    verbose: bool = True,
) -> List[FrameRecord]:
    """
    从单个视频每秒提取一帧（原始分辨率），OCR 时间戳，
    严格校验后返回 FrameRecord 列表（仅含 ok / interpolated 帧）。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] 无法打开视频: {video_path.name}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = int(total_frames / fps) if fps > 0 else 0

    if fps <= 0 or duration_s <= 0:
        cap.release()
        return []

    if verbose:
        print(f"  处理 {video_path.name}  ({duration_s}s, {fps:.1f}fps) ...", flush=True)

    # 临时目录（帧立即写盘，避免全视频帧占满内存）
    tmp_dir = frames_dir / f'_tmp_{video_path.stem}'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    records: List[FrameRecord] = []
    tmp_paths: Dict[int, Path] = {}

    cur_pos = 0  # cap 当前指向的帧号

    for sec in range(duration_s):
        target_frame = int(sec * fps)
        if target_frame >= total_frames:
            break

        # grab() 跳帧（不解码）+ read() 取目标帧
        skipped_ok = True
        while cur_pos < target_frame:
            if not cap.grab():
                skipped_ok = False
                break
            cur_pos += 1

        if not skipped_ok:
            break

        ret, frame = cap.read()
        cur_pos += 1

        if not ret or frame is None:
            records.append(FrameRecord(
                second_offset=sec, frame_number=target_frame,
                video_file=video_path.name, ocr_time=None, ocr_status='excluded',
            ))
            continue

        # 立即写盘（避免将所有帧放内存）
        tmp_path = tmp_dir / f'{sec:06d}.jpg'
        cv2.imwrite(str(tmp_path), frame)
        tmp_paths[sec] = tmp_path

        ts = ocr_frame_timestamp(frame)
        records.append(FrameRecord(
            second_offset=sec, frame_number=target_frame,
            video_file=video_path.name, ocr_time=ts,
        ))
        del frame

    cap.release()

    if verbose:
        ocr_ok = sum(1 for r in records if r.ocr_time is not None)
        print(f"    OCR 成功: {ocr_ok}/{len(records)} 帧", flush=True)

    # 时间戳严格校验
    records = repair_timestamps_strict(records, max_interp_gap=max_interp_gap)

    # 统计
    if verbose:
        cnt = {s: sum(1 for r in records if r.ocr_status == s)
               for s in ('ok', 'ocr_error', 'interpolated', 'excluded')}
        print(f"    ok={cnt['ok']} interp={cnt['interpolated']} "
              f"ocr_error={cnt['ocr_error']} excluded={cnt['excluded']}", flush=True)

    # 整理帧文件：只保留 ok / interpolated，删除其余临时文件
    frames_dir.mkdir(parents=True, exist_ok=True)
    seen_seconds: set = set()
    deduplicated: List[FrameRecord] = []

    for r in records:
        # 排除 ocr_error 和 excluded
        if r.ocr_status in ('ocr_error', 'excluded') or r.repaired_time is None:
            if r.second_offset in tmp_paths:
                tmp_paths[r.second_offset].unlink(missing_ok=True)
            continue

        ts_floor = r.repaired_time.replace(microsecond=0)
        ts_key = int(ts_floor.timestamp())

        if ts_key in seen_seconds:
            # 重复时间戳：删除临时文件
            if r.second_offset in tmp_paths:
                tmp_paths[r.second_offset].unlink(missing_ok=True)
            continue

        seen_seconds.add(ts_key)

        # 重命名为最终文件名
        final_name = ts_floor.strftime('%Y%m%d_%H%M%S') + '.jpg'
        final_path = frames_dir / final_name
        if r.second_offset in tmp_paths:
            src = tmp_paths[r.second_offset]
            if src.exists():
                src.rename(final_path)
                r._saved_frame_path = str(final_path)
            else:
                r._saved_frame_path = None
        else:
            r._saved_frame_path = None

        deduplicated.append(r)

    # 清理空临时目录
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    if verbose:
        print(f"    最终保留: {len(deduplicated)} 帧", flush=True)

    return deduplicated


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    trajectory_path: Path,
    video_dir: Path,
    output_dir: Path,
    traj_format: str = 'xlsx',
    time_tolerance: int = TRAJ_TOLERANCE_S,
    max_interp_gap: int = MAX_INTERP_GAP_S,
    verbose: bool = True,
) -> pd.DataFrame:

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / 'aligned_frames'

    # ── 1. 加载轨迹 ──────────────────────────────────────────────────────────
    if verbose:
        print('=' * 70)
        print('步骤 1/4  加载轨迹数据')
        print('=' * 70)

    traj_df = load_trajectory(trajectory_path, traj_format)
    ts_index: Dict[int, int] = {row['_ts']: idx for idx, row in traj_df.iterrows()}

    if verbose:
        print(f'  轨迹记录: {len(traj_df):,} 条')
        print(f'  时间范围: {traj_df["定位时间"].min()} → {traj_df["定位时间"].max()}')

    # ── 2. 视频帧提取 + OCR + 严格校验 ──────────────────────────────────────
    if verbose:
        print('\n' + '=' * 70)
        print('步骤 2/4  视频帧提取（cv2, 1fps, 原始分辨率）+ OCR + 严格校验')
        print(f'  插值最大间隔: {max_interp_gap} 秒（超出即排除）')
        print('=' * 70)

    video_files = sorted(video_dir.glob('*.mp4'))
    if not video_files:
        video_files = sorted(video_dir.glob('*.MP4'))

    if verbose:
        print(f'  找到 {len(video_files)} 个视频文件')

    all_records: List[FrameRecord] = []
    for vf in video_files:
        recs = extract_frames_from_video(
            vf, frames_dir, max_interp_gap=max_interp_gap, verbose=verbose,
        )
        all_records.extend(recs)

    if not all_records:
        print('[ERROR] 没有提取到任何帧')
        return pd.DataFrame()

    # ── 3. 全局时间排序 + 二次去重 ──────────────────────────────────────────
    if verbose:
        print('\n' + '=' * 70)
        print('步骤 3/4  全局时间排序 + 二次去重')
        print('=' * 70)

    all_records.sort(key=lambda r: r.repaired_time or datetime.max)

    global_seen: set = set()
    unique_records: List[FrameRecord] = []
    for r in all_records:
        ts_key = int(r.repaired_time.replace(microsecond=0).timestamp())
        if ts_key not in global_seen:
            global_seen.add(ts_key)
            unique_records.append(r)
        else:
            # 二次重复：删除对应帧文件
            if r._saved_frame_path:
                Path(r._saved_frame_path).unlink(missing_ok=True)

    if verbose:
        print(f'  全局唯一帧: {len(unique_records):,}')

    # ── 4. 轨迹匹配 ─────────────────────────────────────────────────────────
    if verbose:
        print('\n' + '=' * 70)
        print(f'步骤 4/4  匹配轨迹数据（容差 ±{time_tolerance} 秒）')
        print('=' * 70)

    rows = []
    unmatched = 0

    for r in unique_records:
        if r._saved_frame_path is None:
            unmatched += 1
            continue

        frame_ts = r.repaired_time.replace(microsecond=0)
        traj_idx = match_trajectory(frame_ts, traj_df, ts_index, time_tolerance)

        if traj_idx is None:
            unmatched += 1
            continue

        traj_row = traj_df.iloc[traj_idx]
        rows.append({
            'frame_path': r._saved_frame_path,
            'frame_time': str(r.repaired_time),
            'video_file': r.video_file,
            'frame_number': r.frame_number,
            'second_in_video': r.second_offset,
            'ocr_status': r.ocr_status,
            **{k: v for k, v in traj_row.items() if k != '_ts'},
        })

    result_df = pd.DataFrame(rows)

    if verbose:
        print(f'  匹配成功: {len(result_df):,} 帧')
        print(f'  未匹配/跳过: {unmatched} 帧')

    return result_df


# ──────────────────────────────────────────────────────────────────────────────
# Save results
# ──────────────────────────────────────────────────────────────────────────────

def save_results(df: pd.DataFrame, output_dir: Path, verbose: bool = True) -> Path:
    out_csv = output_dir / 'aligned_data.csv'

    if len(df) == 0:
        print('[WARN] 没有数据可保存')
        return out_csv

    df.to_csv(out_csv, index=False, encoding='utf-8-sig')

    stats = {
        'total_aligned_frames': len(df),
        'unique_videos': df['video_file'].nunique() if 'video_file' in df.columns else 0,
        'time_range': {
            'start': str(df['frame_time'].min()),
            'end': str(df['frame_time'].max()),
        },
        'ocr_status_counts': df['ocr_status'].value_counts().to_dict() if 'ocr_status' in df.columns else {},
    }
    with open(output_dir / 'alignment_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

    if verbose:
        print('\n' + '=' * 70)
        print('完成')
        print('=' * 70)
        print(f'  CSV 保存至: {out_csv}')
        print(f'  总对齐帧数: {len(df):,}')
        print(f'  涉及视频数: {stats["unique_videos"]}')
        if 'ocr_status' in df.columns:
            for s, c in sorted(stats['ocr_status_counts'].items()):
                print(f'    ocr_status={s}: {c}')

    return out_csv


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='农业视频-轨迹数据对齐工具（v2 严格版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--trajectory', '-t', required=True)
    parser.add_argument('--video-dir', '-v', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--traj-format', choices=['xlsx', 'csv'], default='xlsx')
    parser.add_argument('--time-tolerance', type=int, default=TRAJ_TOLERANCE_S)
    parser.add_argument('--max-interp-gap', type=int, default=MAX_INTERP_GAP_S,
                        help=f'插值最大间隔秒数，超出则排除（默认: {MAX_INTERP_GAP_S}）')
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args()

    traj_path = Path(args.trajectory)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output)

    if not traj_path.exists():
        print(f'[ERROR] 轨迹文件不存在: {traj_path}', file=sys.stderr)
        sys.exit(1)
    if not video_dir.exists():
        print(f'[ERROR] 视频目录不存在: {video_dir}', file=sys.stderr)
        sys.exit(1)

    df = run_pipeline(
        trajectory_path=traj_path,
        video_dir=video_dir,
        output_dir=output_dir,
        traj_format=args.traj_format,
        time_tolerance=args.time_tolerance,
        max_interp_gap=args.max_interp_gap,
        verbose=not args.quiet,
    )
    save_results(df, output_dir, verbose=not args.quiet)


if __name__ == '__main__':
    main()
