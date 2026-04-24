#!/usr/bin/env python3
"""Build 1-second audio clips aligned to B_deep_part video/GNSS rows.

The input CSVs already contain one row per aligned video second. This script
extracts one WAV clip for each selected (video_file, second_in_video) and writes
metadata CSVs that keep the audio, video frame, GNSS trajectory, and label
columns together.
"""

from __future__ import annotations

import argparse
import json
import wave
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "b_deep_part_full_20241018_29"
DEFAULT_VIDEO_ROOT = PROJECT_ROOT / "data" / "video"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "b_deep_part_audio_1s_20241018_29"


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_command(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def build_video_index(video_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in video_root.rglob("*.mp4"):
        index.setdefault(path.name, path)
    for path in video_root.rglob("*.MP4"):
        index.setdefault(path.name, path)
    return index


def has_audio_stream(video_path: Path) -> bool:
    result = run_command([
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(video_path),
    ])
    return result.returncode == 0 and bool(result.stdout.strip())


def clean_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_silence_wav(path: Path, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * sample_rate)


def clip_dir_for(video_group: pd.DataFrame, output_dir: Path) -> Path:
    split = str(video_group["split"].iloc[0]) if "split" in video_group.columns else "all"
    date = pd.to_datetime(video_group["frame_time"].iloc[0]).strftime("%Y-%m-%d")
    stem = Path(str(video_group["video_file"].iloc[0])).stem
    return output_dir / "audio_segments" / split / date / stem


def expected_clip_path(video_group: pd.DataFrame, output_dir: Path, second: int) -> Path:
    return clip_dir_for(video_group, output_dir) / f"{second:06d}.wav"


def extract_video_clips(
    video_group: pd.DataFrame,
    video_path: Path,
    output_dir: Path,
    sample_rate: int,
    overwrite: bool,
    fill_missing_silence: bool,
) -> dict:
    seconds = sorted({int(v) for v in video_group["second_in_video"].dropna().astype(int).tolist()})
    video_name = str(video_group["video_file"].iloc[0])
    clip_dir = clip_dir_for(video_group, output_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)
    existing = {sec for sec in seconds if (clip_dir / f"{sec:06d}.wav").exists()}
    if existing == set(seconds) and not overwrite:
        return {
            "video_file": video_name,
            "source_video_path": str(video_path),
            "requested_seconds": len(seconds),
            "written_clips": len(existing),
            "missing_clips": 0,
            "min_second": int(min(seconds)) if seconds else None,
            "max_second": int(max(seconds)) if seconds else None,
            "skipped_existing": True,
            "ok": True,
        }
    if overwrite:
        for wav in clip_dir.glob("*.wav"):
            wav.unlink()

    if not seconds:
        return {
            "video_file": video_name,
            "source_video_path": str(video_path),
            "requested_seconds": 0,
            "written_clips": 0,
            "missing_clips": 0,
            "min_second": None,
            "max_second": None,
            "skipped_existing": False,
            "ok": True,
        }
    if not has_audio_stream(video_path):
        return {
            "video_file": video_name,
            "source_video_path": str(video_path),
            "requested_seconds": len(seconds),
            "written_clips": 0,
            "missing_clips": len(seconds),
            "min_second": int(min(seconds)),
            "max_second": int(max(seconds)),
            "skipped_existing": False,
            "ok": False,
            "error": "no audio stream",
        }

    start_second = int(min(seconds))
    end_second = int(max(seconds))
    duration = end_second - start_second + 1
    wanted = set(seconds)
    with tempfile.TemporaryDirectory(prefix=f"audio_{Path(video_name).stem}_", dir=str(output_dir / "_tmp")) as tmp:
        tmp_dir = Path(tmp)
        pattern = tmp_dir / "%06d.wav"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-ss", str(start_second),
            "-i", str(video_path),
            "-t", str(duration),
            "-vn",
            "-map", "0:a:0",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            "-f", "segment",
            "-segment_time", "1",
            "-segment_format", "wav",
            "-reset_timestamps", "1",
            str(pattern),
        ]
        result = run_command(cmd)
        if result.returncode != 0:
            return {
                "video_file": video_name,
                "source_video_path": str(video_path),
                "requested_seconds": len(seconds),
                "written_clips": 0,
                "missing_clips": len(seconds),
                "min_second": start_second,
                "max_second": end_second,
                "skipped_existing": False,
                "ok": False,
                "error": result.stderr[-1000:],
            }
        for tmp_wav in tmp_dir.glob("*.wav"):
            source_second = start_second + int(tmp_wav.stem)
            if source_second not in wanted:
                continue
            target = clip_dir / f"{source_second:06d}.wav"
            if target.exists() and not overwrite:
                continue
            tmp_wav.replace(target)

    written = {sec for sec in seconds if (clip_dir / f"{sec:06d}.wav").exists()}
    missing_after = [sec for sec in seconds if sec not in written]
    silence_filled = []
    if missing_after and fill_missing_silence:
        for sec in missing_after:
            target = clip_dir / f"{sec:06d}.wav"
            write_silence_wav(target, sample_rate)
            silence_filled.append(sec)
        written = {sec for sec in seconds if (clip_dir / f"{sec:06d}.wav").exists()}
        missing_after = [sec for sec in seconds if sec not in written]
    return {
        "video_file": video_name,
        "source_video_path": str(video_path),
        "requested_seconds": len(seconds),
        "written_clips": len(written),
        "missing_clips": len(missing_after),
        "silence_filled_clips": len(silence_filled),
        "silence_filled_seconds": silence_filled[:100],
        "min_second": start_second,
        "max_second": end_second,
        "skipped_existing": False,
        "ok": len(missing_after) == 0,
        "missing_second_examples": missing_after[:20],
    }


def attach_audio_columns(
    df: pd.DataFrame,
    video_index: dict[str, Path],
    output_dir: Path,
    silence_filled_paths: set[str] | None = None,
) -> pd.DataFrame:
    rows = df.copy()
    rows["source_video_path"] = rows["video_file"].map(lambda v: str(video_index.get(str(v), "")))
    rows["audio_start_second"] = rows["second_in_video"].astype(int)
    rows["audio_duration_seconds"] = 1.0

    key_to_group: dict[str, pd.DataFrame] = {
        str(video_file): group
        for video_file, group in rows.groupby("video_file", sort=False)
    }

    def audio_path_for(row: pd.Series) -> str:
        group = key_to_group[str(row["video_file"])]
        path = expected_clip_path(group, output_dir, int(row["second_in_video"]))
        return project_relative(path)

    rows["audio_path"] = rows.apply(audio_path_for, axis=1)
    rows["audio_exists"] = rows["audio_path"].map(lambda p: (PROJECT_ROOT / p).exists() if not Path(p).is_absolute() else Path(p).exists())
    silence_filled_paths = silence_filled_paths or set()
    rows["audio_is_silence_fill"] = rows["audio_path"].isin(silence_filled_paths)
    return rows


def write_split_csvs(enriched: pd.DataFrame, output_dir: Path) -> dict:
    summaries = {}
    for split, group in enriched.groupby("split", sort=False):
        group = group.sort_values(["frame_time", "video_file", "second_in_video", "frame_path"]).reset_index(drop=True)
        csv_path = output_dir / f"{split}.csv"
        group.to_csv(csv_path, index=False, encoding="utf-8-sig")
        summaries[str(split)] = {
            "csv": project_relative(csv_path),
            "rows": int(len(group)),
            "audio_exists_rows": int(group["audio_exists"].sum()),
            "videos": int(group["video_file"].nunique()),
            "time_start": str(group["frame_time"].min()),
            "time_end": str(group["frame_time"].max()),
        }
    all_csv = output_dir / "all.csv"
    enriched.to_csv(all_csv, index=False, encoding="utf-8-sig")
    summaries["all"] = {
        "csv": project_relative(all_csv),
        "rows": int(len(enriched)),
        "audio_exists_rows": int(enriched["audio_exists"].sum()),
        "videos": int(enriched["video_file"].nunique()),
        "time_start": str(enriched["frame_time"].min()),
        "time_end": str(enriched["frame_time"].max()),
    }
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--video-root", default=str(DEFAULT_VIDEO_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-fill-missing-silence", action="store_true", help="Leave clips missing if ffmpeg cannot produce a selected second")
    parser.add_argument("--workers", type=int, default=4, help="Number of videos to process in parallel")
    parser.add_argument("--max-videos", type=int, default=0, help="Debug limit; 0 means all videos")
    args = parser.parse_args()

    input_dir = resolve(args.input_dir)
    video_root = resolve(args.video_root)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(output_dir / "_tmp", ignore_errors=True)
    (output_dir / "_tmp").mkdir(parents=True, exist_ok=True)
    silence_registry_path = output_dir / "silence_filled_clips.json"
    if silence_registry_path.exists():
        existing_silence_filled_paths = set(json.loads(silence_registry_path.read_text(encoding="utf-8")))
    else:
        existing_silence_filled_paths = set()

    source_csv = input_dir / "all.csv"
    df = pd.read_csv(source_csv, encoding="utf-8-sig")
    required = {"split", "video_file", "second_in_video", "frame_time", "frame_path", "分类"}
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise SystemExit(f"missing required columns in {source_csv}: {missing_columns}")
    df["second_in_video"] = pd.to_numeric(df["second_in_video"], errors="raise").astype(int)
    df["frame_time"] = pd.to_datetime(df["frame_time"], errors="raise").dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.sort_values(["frame_time", "video_file", "second_in_video", "frame_path"]).reset_index(drop=True)

    video_index = build_video_index(video_root)
    unique_videos = list(df["video_file"].drop_duplicates())
    missing_videos = [str(v) for v in unique_videos if str(v) not in video_index]
    if missing_videos:
        raise SystemExit(f"missing {len(missing_videos)} source videos under {video_root}; first: {missing_videos[:20]}")
    if args.max_videos:
        keep = set(unique_videos[:args.max_videos])
        df = df[df["video_file"].isin(keep)].reset_index(drop=True)
        unique_videos = list(df["video_file"].drop_duplicates())

    video_jobs = [
        (idx, str(video_file), group.copy(), video_index[str(video_file)])
        for idx, (video_file, group) in enumerate(df.groupby("video_file", sort=False), start=1)
    ]

    def run_video_job(job: tuple[int, str, pd.DataFrame, Path]) -> tuple[int, dict]:
        idx, video_file, group, video_path = job
        summary = extract_video_clips(
            group,
            video_path,
            output_dir,
            args.sample_rate,
            args.overwrite,
            not args.no_fill_missing_silence,
        )
        return idx, summary

    video_summaries_by_idx: dict[int, dict] = {}
    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_video_job, job): job for job in video_jobs}
        for future in as_completed(futures):
            idx, video_file, group, _ = futures[future]
            try:
                result_idx, summary_item = future.result()
            except Exception as exc:  # defensive; keep processing other videos
                result_idx = idx
                summary_item = {
                    "video_file": video_file,
                    "requested_seconds": int(group["second_in_video"].nunique()),
                    "written_clips": 0,
                    "missing_clips": int(group["second_in_video"].nunique()),
                    "ok": False,
                    "error": repr(exc),
                }
            video_summaries_by_idx[result_idx] = summary_item
            status = "ok" if summary_item.get("ok") else "FAILED"
            skipped = " skipped" if summary_item.get("skipped_existing") else ""
            print(
                f"[{len(video_summaries_by_idx)}/{len(video_jobs)}] {video_file} "
                f"rows={len(group)} clips={summary_item.get('written_clips', 0)}/"
                f"{summary_item.get('requested_seconds', 0)} {status}{skipped}",
                flush=True,
            )

    video_summaries = [video_summaries_by_idx[idx] for idx in sorted(video_summaries_by_idx)]

    silence_filled_paths = set(existing_silence_filled_paths)
    for item in video_summaries:
        if not item.get("silence_filled_seconds"):
            continue
        video_file = item["video_file"]
        group = df[df["video_file"] == video_file]
        if group.empty:
            continue
        for sec in item["silence_filled_seconds"]:
            silence_filled_paths.add(project_relative(expected_clip_path(group, output_dir, int(sec))))

    silence_registry_path.write_text(
        json.dumps(sorted(silence_filled_paths), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    enriched = attach_audio_columns(df, video_index, output_dir, silence_filled_paths)
    csv_summaries = write_split_csvs(enriched, output_dir)
    total_requested = int(sum(item["requested_seconds"] for item in video_summaries))
    total_written = int(sum(item["written_clips"] for item in video_summaries))
    total_missing = int(sum(item["missing_clips"] for item in video_summaries))
    total_silence_filled = int(sum(item.get("silence_filled_clips", 0) for item in video_summaries))
    summary = {
        "description": "1-second mono WAV audio clips aligned with B_deep_part full video/GNSS rows.",
        "input_dir": str(input_dir),
        "source_csv": str(source_csv),
        "video_root": str(video_root),
        "output_dir": str(output_dir),
        "sample_rate": int(args.sample_rate),
        "audio_codec": "pcm_s16le",
        "duration_seconds": 1.0,
        "rows": int(len(enriched)),
        "videos": int(len(unique_videos)),
        "requested_clips": total_requested,
        "written_clips": total_written,
        "missing_clips": total_missing,
        "silence_filled_clips": total_silence_filled,
        "csvs": csv_summaries,
        "failed_videos": [item for item in video_summaries if not item.get("ok")],
        "video_summaries": video_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(output_dir / "_tmp", ignore_errors=True)
    print(json.dumps({k: summary[k] for k in ["rows", "videos", "requested_clips", "written_clips", "missing_clips", "silence_filled_clips"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
