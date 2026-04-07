#!/usr/bin/env python3
"""
Align video frames with GPS trajectory data — batch B-2024-10-19
=================================================================
Uses ffmpeg for fast 1-FPS frame extraction (~8 min for 24 videos).
Video filenames encode UTC start time; trajectory timestamps are CST (UTC+8).

Output
------
data/aligned_output/B-2024-10-19/aligned_frames/YYYYMMDD_HHMMSS.jpg
data/aligned_output/B-2024-10-19/aligned_data.csv
"""

import os, re, sys, shutil, subprocess, argparse, tempfile
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

BASE_DIR   = Path('/home/research/Agri-MBT')
TRAJ_CSV   = BASE_DIR / 'data/trajectory/B-2024-10-19/trajectory_B_20241019.csv'
VIDEO_DIR  = BASE_DIR / 'data/video/B-2024-10-19'
OUTPUT_DIR = BASE_DIR / 'data/aligned_output/B-2024-10-19'
FRAME_DIR  = OUTPUT_DIR / 'aligned_frames'
OUT_CSV    = OUTPUT_DIR / 'aligned_data.csv'

UTC_OFFSET = timedelta(hours=8)   # video filename (UTC) → trajectory (CST)
FRAME_SIZE = '224:224'            # ffmpeg scale filter


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_video_start_cst(filename: str) -> datetime | None:
    m = re.match(r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})', Path(filename).stem)
    if not m:
        return None
    return datetime(*[int(x) for x in m.groups()]) + UTC_OFFSET


def ffmpeg_extract(video_path: Path, out_dir: Path, stride: int) -> int:
    """Extract 1 frame every `stride` seconds into out_dir as f%06d.jpg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fps_out = f'1/{stride}'
    cmd = [
        'ffmpeg', '-y', '-i', str(video_path),
        '-vf', f'fps={fps_out},scale={FRAME_SIZE}',
        '-q:v', '3', '-threads', '4',
        str(out_dir / 'f%06d.jpg')
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [ffmpeg error] {r.stderr[-300:]}")
        return 0
    return len(list(out_dir.glob('f*.jpg')))


# ── Core alignment ────────────────────────────────────────────────────────────

def align(stride: int = 1, dry_run: bool = False) -> pd.DataFrame:

    # ── Load trajectory ───────────────────────────────────────────────────
    print("Loading trajectory …")
    traj = pd.read_csv(TRAJ_CSV, parse_dates=['定位时间'])
    traj = traj.sort_values('定位时间').drop_duplicates('定位时间')
    traj_idx = traj.set_index('定位时间')
    print(f"  {len(traj_idx):,} rows  [{traj_idx.index.min()} → {traj_idx.index.max()}]")

    if not dry_run:
        FRAME_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(VIDEO_DIR.glob('*.mp4'))
    print(f"\nFound {len(videos)} videos  (stride={stride}s)\n")

    all_rows   = []
    total_done = 0

    for vi, vf in enumerate(videos):
        vstart_cst = parse_video_start_cst(vf.name)
        if vstart_cst is None:
            print(f"  [{vi+1:2d}/{len(videos)}] SKIP — cannot parse: {vf.name}")
            continue

        print(f"  [{vi+1:2d}/{len(videos)}] {vf.name}  CST={vstart_cst}", end='  ', flush=True)

        if dry_run:
            # Just check trajectory coverage without touching video
            import cv2
            cap = cv2.VideoCapture(str(vf))
            fps   = cap.get(cv2.CAP_PROP_FPS)
            nfrm  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            dur   = int(nfrm / fps) if fps > 0 else 0
            cap.release()
            matched = sum(
                1 for s in range(0, dur, stride)
                if (vstart_cst + timedelta(seconds=s)).replace(microsecond=0) in traj_idx.index
            )
            print(f"→ {matched:,} frames would match  (dur={dur}s)")
            total_done += matched
            continue

        # ── Extract frames with ffmpeg ─────────────────────────────────
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            n_extracted = ffmpeg_extract(vf, tmp_dir, stride)
            print(f"extracted {n_extracted} frames", end=' … ', flush=True)
            if n_extracted == 0:
                print("SKIP")
                continue

            matched = 0
            # ffmpeg names frames f000001, f000002, ... (1-indexed, one per stride-seconds)
            for img_path in sorted(tmp_dir.glob('f*.jpg')):
                # frame number (1-indexed) → second offset in video
                n = int(img_path.stem[1:])    # strip leading 'f'
                sec = (n - 1) * stride

                ts_cst   = vstart_cst + timedelta(seconds=sec)
                ts_floor = ts_cst.replace(microsecond=0)

                if ts_floor not in traj_idx.index:
                    continue
                trow = traj_idx.loc[ts_floor]

                # Move to final destination
                fname    = ts_cst.strftime('%Y%m%d_%H%M%S') + '.jpg'
                dst      = FRAME_DIR / fname
                rel_path = f'data/aligned_output/B-2024-10-19/aligned_frames/{fname}'
                shutil.move(str(img_path), str(dst))

                all_rows.append({
                    'frame_path'     : rel_path,
                    'frame_time'     : str(ts_cst),
                    'video_file'     : vf.name,
                    'frame_number'   : round(sec * 25),   # approx frame index
                    'second_in_video': sec,
                    '定位时间'          : str(ts_floor),
                    '经度'             : float(trow['经度']),
                    '纬度'             : float(trow['纬度']),
                    '间距(米)'          : float(trow['间距(米)']),
                    '深度'             : int(trow['深度']),
                    '速度'             : float(trow['速度']),
                    '类型'             : str(trow['类型']),
                    '方向角'            : float(trow['方向角']),
                    '分类'             : int(trow['分类']),
                    '时间戳'            : int(ts_cst.timestamp()),
                })
                matched += 1

            total_done += matched
            print(f"matched {matched:,}  (total so far: {total_done:,})")

    result = pd.DataFrame(all_rows)
    if not dry_run and len(result):
        result.to_csv(OUT_CSV, index=False)
        print(f"\nSaved {len(result):,} rows → {OUT_CSV}")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Align B-2024-10-19 video + trajectory')
    parser.add_argument('--stride',  type=int,  default=1,
                        help='extract 1 frame every N seconds (default: 1)')
    parser.add_argument('--dry-run', action='store_true',
                        help='count matches without extracting frames')
    args = parser.parse_args()

    result = align(stride=args.stride, dry_run=args.dry_run)

    print(f"\n{'='*55}")
    print(f"Total aligned rows : {len(result) if not args.dry_run else '(dry-run, see above)'}")
    if len(result):
        print(f"Class distribution :")
        print(result['分类'].value_counts().sort_index().to_string())
        print(f"\nTime range : {result['frame_time'].min()} → {result['frame_time'].max()}")


if __name__ == '__main__':
    main()
