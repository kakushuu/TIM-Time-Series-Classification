#!/usr/bin/env python3
"""
视频轨迹数据对齐工具 - 完整流程

流程:
1. 视频处理: 每一秒提取一帧,OCR识别时间,重命名文件
2. 去重处理: 删除重复的帧文件
3. 轨迹对齐: 读取轨迹数据,匹配时间戳,生成最终数据

用法:
    python3 scripts/align_complete_pipeline.py
"""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pytesseract
from PIL import Image
import re
import json
import shutil
from tqdm import tqdm
import argparse
from typing import Tuple, Optional, List, Dict


class VideoFrameExtractor:
    """步骤1: 从视频中提取帧"""

    def __init__(self, video_dir: str, output_dir: str, verbose: bool = True):
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "extracted_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    def extract_frames_from_all_videos(self) -> Dict[str, int]:
        """
        从所有视频中提取帧(每秒一帧)

        Returns:
            视频文件名到帧数的映射
        """
        if self.verbose:
            print("=" * 80)
            print("步骤1: 提取视频帧")
            print("=" * 80)

        video_files = sorted(self.video_dir.glob("*.mp4"))
        if self.verbose:
            print(f"✓ 找到 {len(video_files)} 个视频文件\n")

        video_frame_counts = {}

        for video_path in video_files:
            frame_count = self._extract_frames_from_video(video_path)
            video_frame_counts[video_path.name] = frame_count

        if self.verbose:
            print(f"\n✓ 总共提取了 {sum(video_frame_counts.values())} 个帧")

        return video_frame_counts

    def _extract_frames_from_video(self, video_path: Path) -> int:
        """从单个视频提取帧"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            if self.verbose:
                print(f"✗ 无法打开视频: {video_path.name}")
            return 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(total_frames / fps)

        if self.verbose:
            print(f"处理 {video_path.name}:")
            print(f"  时长: {duration} 秒, FPS: {fps:.2f}")

        # 每秒提取一帧
        for second in tqdm(range(duration), desc=f"  提取帧", disable=not self.verbose):
            frame_number = int(second * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if not ret:
                continue

            # 临时保存(使用临时文件名)
            temp_filename = f"temp_{video_path.stem}_{second:06d}.jpg"
            frame_path = self.frames_dir / temp_filename
            cv2.imwrite(str(frame_path), frame)

        cap.release()
        return duration


class FrameTimeRecognizer:
    """步骤2: OCR识别和重命名帧"""

    def __init__(self, frames_dir: str, verbose: bool = True):
        self.frames_dir = Path(frames_dir)
        self.verbose = verbose
        self.rename_log = []
        self.timestamp_source_file = self.frames_dir / "frame_timestamp_sources.json"
        self.ocr_results_file = self.frames_dir / "ocr_results.jsonl"

    def _normalize_ocr_text(self, text: str) -> str:
        text = text.strip().upper()
        text = re.sub(r'\s+', ' ', text)
        replacements = str.maketrans({
            'O': '0',
            'I': '1',
            'L': '1',
            'Z': '2',
            'S': '5',
            'B': '8',
        })
        text = text.translate(replacements)
        text = text.replace('.', ':').replace('_', '-')
        text = re.sub(r'[^0-9:/\-\s]', '', text)
        text = re.sub(r'(\d{4}[-/]\d{2}[-/]\d{2})(\d{2}:\d{2}:\d{2})', r'\1 \2', text)
        return text

    def _parse_timestamp_text(self, text: str) -> Optional[datetime]:
        patterns = [
            r'(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})',
            r'(\d{4})/(\d{2})/(\d{2})\s+(\d{2}):(\d{2}):(\d{2})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                try:
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                    hour, minute, second = int(groups[3]), int(groups[4]), int(groups[5])
                    return datetime(year, month, day, hour, minute, second)
                except (ValueError, IndexError):
                    continue
        return None

    def _parse_temp_frame_name(self, frame_path: Path) -> Tuple[Optional[str], Optional[int]]:
        match = re.match(r'temp_(.+)_(\d{6})$', frame_path.stem)
        if not match:
            return None, None
        try:
            return f"{match.group(1)}.mp4", int(match.group(2))
        except ValueError:
            return None, None

    def _extract_frame_second(self, frame_path: Path) -> Optional[int]:
        return self._parse_temp_frame_name(frame_path)[1]

    def _extract_frame_video(self, frame_path: Path) -> Optional[str]:
        return self._parse_temp_frame_name(frame_path)[0]

    def _load_ocr_results(self) -> Dict[str, Dict[str, object]]:
        if not self.ocr_results_file.exists():
            return {}

        results = {}
        with open(self.ocr_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                frame_name = record.get('frame')
                if isinstance(frame_name, str):
                    results[frame_name] = record
        return results

    def _append_ocr_result(self, record: Dict[str, object]) -> None:
        with open(self.ocr_results_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _record_to_raw_result(self, temp_frame: Path, record: Dict[str, object]) -> Dict[str, object]:
        timestamp = None
        timestamp_text = record.get('ocr_timestamp')
        if isinstance(timestamp_text, str) and timestamp_text:
            try:
                timestamp = datetime.fromisoformat(timestamp_text)
            except ValueError:
                timestamp = None

        second = record.get('second')
        if not isinstance(second, int):
            second = self._extract_frame_second(temp_frame)
        video_file = record.get('video_file')
        if not isinstance(video_file, str):
            video_file = self._extract_frame_video(temp_frame)

        return {
            'path': temp_frame,
            'video_file': video_file,
            'second': second,
            'ocr_timestamp': timestamp,
            'timestamp': timestamp,
            'source': 'ocr' if timestamp is not None else None,
        }

    def recognize_and_rename_all(self) -> Tuple[int, int]:
        """
        OCR识别所有帧并重命名

        Returns:
            (成功数, 失败数)
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("步骤2: OCR识别和重命名帧")
            print("=" * 80)

        temp_frames = sorted(
            self.frames_dir.glob("temp_*.jpg"),
            key=lambda p: (
                self._extract_frame_video(p) or '',
                self._extract_frame_second(p) is None,
                self._extract_frame_second(p) or 0,
                p.name,
            )
        )
        if self.verbose:
            print(f"✓ 找到 {len(temp_frames)} 个临时帧文件\n")

        cached_results = self._load_ocr_results()
        cached_count = 0
        raw_results = []
        for temp_frame in tqdm(temp_frames, desc="OCR识别", disable=not self.verbose):
            cached_record = cached_results.get(temp_frame.name)
            if cached_record is not None:
                raw_results.append(self._record_to_raw_result(temp_frame, cached_record))
                cached_count += 1
                continue

            video_file = self._extract_frame_video(temp_frame)
            second = self._extract_frame_second(temp_frame)
            timestamp = self._extract_timestamp_ocr(temp_frame)
            record = {
                'frame': temp_frame.name,
                'video_file': video_file,
                'second': second,
                'ocr_timestamp': timestamp.isoformat() if timestamp is not None else None,
            }
            self._append_ocr_result(record)
            raw_results.append({
                'path': temp_frame,
                'video_file': video_file,
                'second': second,
                'ocr_timestamp': timestamp,
                'timestamp': timestamp,
                'source': 'ocr' if timestamp is not None else None,
            })

        ocr_count = sum(item['ocr_timestamp'] is not None for item in raw_results)

        # 每个视频独立计算“视频内秒数 -> OCR时间”的时间轴偏移。
        # 相邻视频可能不连续，不能把所有视频混在一条全局秒序列上。
        valid_offsets_by_video: Dict[str, List[int]] = {}
        for item in raw_results:
            video_file = item.get('video_file')
            if not video_file or item['ocr_timestamp'] is None or item['second'] is None:
                continue
            offset = int(item['ocr_timestamp'].timestamp()) - item['second']
            valid_offsets_by_video.setdefault(video_file, []).append(offset)

        dominant_offsets: Dict[str, int] = {}
        for video_file, offsets in valid_offsets_by_video.items():
            if offsets:
                dominant_offsets[video_file] = int(pd.Series(offsets).median())

        # 先接受与本视频时间轴一致的 OCR 结果，再只在该视频内部对中间缺口做连续补值。
        accepted_seconds_by_video: Dict[str, set] = {}
        for item in raw_results:
            video_file = item.get('video_file')
            second = item['second']
            timestamp = item['ocr_timestamp']
            dominant_offset = dominant_offsets.get(video_file) if video_file else None
            if timestamp is None or second is None or dominant_offset is None:
                continue
            offset = int(timestamp.timestamp()) - second
            if abs(offset - dominant_offset) <= 2:
                item['timestamp'] = timestamp
                item['source'] = 'ocr'
                accepted_seconds_by_video.setdefault(video_file, set()).add(second)
            else:
                item['timestamp'] = None
                item['source'] = None

        inferred_fill_count = 0
        accepted_ranges_by_video = {
            video_file: (min(seconds), max(seconds))
            for video_file, seconds in accepted_seconds_by_video.items()
            if seconds and video_file in dominant_offsets
        }
        if accepted_ranges_by_video:
            for item in raw_results:
                video_file = item.get('video_file')
                second = item['second']
                if item['timestamp'] is not None or second is None:
                    continue
                dominant_offset = dominant_offsets.get(video_file) if video_file else None
                accepted_range = accepted_ranges_by_video.get(video_file) if video_file else None
                if dominant_offset is None or accepted_range is None:
                    continue
                min_second, max_second = accepted_range
                if min_second <= second <= max_second:
                    item['timestamp'] = datetime.fromtimestamp(dominant_offset + second)
                    item['source'] = 'inferred'
                    inferred_fill_count += 1

        success_count = 0
        fail_count = 0
        continuity_reject_count = 0
        timestamp_source_map = {}

        for item in raw_results:
            temp_frame = item['path']
            timestamp = item['timestamp']
            video_file = item.get('video_file')
            second = item['second']

            keep = timestamp is not None
            dominant_offset = dominant_offsets.get(video_file) if video_file else None
            if keep and dominant_offset is not None and second is not None and item['ocr_timestamp'] is not None:
                offset = int(item['ocr_timestamp'].timestamp()) - second
                if abs(offset - dominant_offset) > 2:
                    keep = False
                    continuity_reject_count += 1

            if keep:
                new_filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                new_path = self.frames_dir / new_filename

                if new_path.exists() and new_path != temp_frame:
                    counter = 1
                    while True:
                        new_filename_with_suffix = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{counter:03d}.jpg"
                        new_path = self.frames_dir / new_filename_with_suffix
                        if not new_path.exists():
                            break
                        counter += 1

                temp_frame.rename(new_path)
                self.rename_log.append({
                    'original': temp_frame.name,
                    'renamed': new_path.name,
                    'video_file': video_file,
                    'second_in_video': second,
                    'timestamp': timestamp.isoformat(),
                    'timestamp_source': item['source'] or 'ocr',
                })
                timestamp_source_map[new_path.name] = {
                    'timestamp': timestamp.isoformat(),
                    'source': item['source'] or 'ocr',
                    'inferred': bool(item['source'] == 'inferred'),
                    'video_file': video_file,
                    'second_in_video': second,
                    'video_time_offset': dominant_offset,
                }
                success_count += 1
            else:
                temp_frame.unlink()
                fail_count += 1

        with open(self.timestamp_source_file, 'w', encoding='utf-8') as f:
            json.dump(timestamp_source_map, f, ensure_ascii=False, indent=2)

        if self.verbose:
            print(f"\n✓ OCR成功: {success_count}")
            print(f"✓ OCR失败: {fail_count}")
            print(f"✓ OCR识别成功: {ocr_count}")
            print(f"✓ 复用OCR缓存: {cached_count}")
            print(f"✓ 单视频时间轴: {len(dominant_offsets)} 个")
            for video_file, offset in sorted(dominant_offsets.items())[:5]:
                print(f"  - {video_file}: {offset} 秒")
            if len(dominant_offsets) > 5:
                print(f"  - ... 另 {len(dominant_offsets) - 5} 个视频")
            print(f"✓ 连续性剔除: {continuity_reject_count}")
            print(f"✓ 连续补值: {inferred_fill_count}")

        return success_count, fail_count

    def _extract_timestamp_ocr(self, frame_path: Path) -> Optional[datetime]:
        """使用OCR从帧中提取时间戳"""
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # 时间戳固定在左上角，只裁剪该区域。每帧仍先 OCR；失败帧后续由连续时间轴补值。
        timestamp_region = gray[0:int(height * 0.12), 0:int(width * 0.50)]
        candidates = [
            cv2.threshold(timestamp_region, 180, 255, cv2.THRESH_BINARY)[1],
            timestamp_region,
        ]
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789-: /'

        for candidate in candidates:
            pil_image = Image.fromarray(candidate)
            try:
                text = pytesseract.image_to_string(pil_image, config=config)
                normalized = self._normalize_ocr_text(text)
                timestamp = self._parse_timestamp_text(normalized)
                if timestamp is not None:
                    return timestamp
            except Exception:
                continue

        return None


class FrameDeduplicator:
    """步骤3: 删除重复的帧"""

    def __init__(self, frames_dir: str, verbose: bool = True):
        self.frames_dir = Path(frames_dir)
        self.verbose = verbose

    def remove_duplicates(self) -> int:
        """
        删除重复的帧(保留第一个)

        Returns:
            删除的帧数
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("步骤3: 删除重复帧")
            print("=" * 80)

        # 按时间戳分组
        frame_files = list(self.frames_dir.glob("*.jpg"))

        # 提取时间戳(不含序号后缀)
        timestamp_map = {}
        for frame_file in frame_files:
            # 提取基础时间戳(去除 _001 等后缀)
            name = frame_file.stem
            base_name = name.split('_')[0] if '_' in name and name.count('_') >= 3 else name

            if base_name not in timestamp_map:
                timestamp_map[base_name] = []
            timestamp_map[base_name].append(frame_file)

        # 删除重复的帧
        removed_count = 0
        for base_name, files in timestamp_map.items():
            if len(files) > 1:
                # 按文件名排序,保留第一个
                files.sort()
                for duplicate_file in files[1:]:
                    duplicate_file.unlink()
                    removed_count += 1

        if self.verbose:
            total_frames = len(list(self.frames_dir.glob("*.jpg")))
            print(f"✓ 删除了 {removed_count} 个重复帧")
            print(f"✓ 剩余帧数: {total_frames}")

        return removed_count


class TrajectoryAligner:
    """步骤4: 轨迹数据对齐"""

    def __init__(self, trajectory_path: str, frames_dir: str, output_dir: str,
                 video_dir: Optional[str] = None,
                 time_tolerance: int = 2, verbose: bool = True):
        self.trajectory_path = Path(trajectory_path)
        self.frames_dir = Path(frames_dir)
        self.output_dir = Path(output_dir)
        self.video_dir = Path(video_dir) if video_dir is not None else None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.time_tolerance = time_tolerance
        self.verbose = verbose
        self.timestamp_source_map = self._load_timestamp_sources()

        # 最终帧目录
        self.final_frames_dir = self.output_dir / "aligned_frames"
        self.final_frames_dir.mkdir(exist_ok=True)

        # 加载轨迹数据并统一列名
        self.trajectory_df = pd.read_excel(trajectory_path)
        rename_map = {}
        if '时间' in self.trajectory_df.columns:
            rename_map['时间'] = '定位时间'
        if '方向' in self.trajectory_df.columns:
            rename_map['方向'] = '方向角'
        if '标记' in self.trajectory_df.columns:
            rename_map['标记'] = '分类'
        self.trajectory_df = self.trajectory_df.rename(columns=rename_map)
        self.trajectory_df['定位时间'] = pd.to_datetime(self.trajectory_df['定位时间'])

        # 创建时间索引
        self.trajectory_df['时间戳'] = self.trajectory_df['定位时间'].astype(np.int64) // 10**9
        self.time_index = {row['时间戳']: idx for idx, row in self.trajectory_df.iterrows()}
        self.traj_start = self.trajectory_df['定位时间'].min()
        self.traj_end = self.trajectory_df['定位时间'].max()

        if self.verbose:
            print("=" * 80)
            print("步骤4: 轨迹数据对齐")
            print("=" * 80)
            print(f"✓ 加载轨迹数据: {len(self.trajectory_df)} 条记录")
            print(f"✓ 轨迹时间范围: {self.traj_start} 到 {self.traj_end}")

    def _load_timestamp_sources(self) -> Dict[str, Dict[str, object]]:
        source_file = self.frames_dir / "frame_timestamp_sources.json"
        if not source_file.exists():
            return {}
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _parse_frame_time(self, frame_file: Path) -> Optional[datetime]:
        """从帧文件名解析时间，兼容 YYYYMMDD_HHMMSS 和带后缀版本。"""
        match = re.search(r'(\d{8})_(\d{6})', frame_file.stem)
        if not match:
            return None
        return datetime.strptime(f"{match.group(1)}_{match.group(2)}", '%Y%m%d_%H%M%S')

    def _compute_frame_time_range(self, frame_files: List[Path]) -> Tuple[Optional[datetime], Optional[datetime]]:
        frame_times = []
        for frame_file in frame_files:
            frame_time = self._parse_frame_time(frame_file)
            if frame_time is not None:
                frame_times.append(frame_time)
        if not frame_times:
            return None, None
        return min(frame_times), max(frame_times)

    def align_frames_with_trajectory(self) -> pd.DataFrame:
        """
        将帧与轨迹数据对齐

        Returns:
            对齐后的DataFrame
        """
        if self.verbose:
            print(f"✓ 开始对齐...")

        frame_files = sorted(self.frames_dir.glob("*.jpg"))
        frame_start, frame_end = self._compute_frame_time_range(frame_files)
        if frame_start is None or frame_end is None:
            if self.verbose:
                print("⚠ 未找到可解析时间的帧文件")
            return pd.DataFrame()

        overlap_start = max(frame_start, self.traj_start)
        overlap_end = min(frame_end, self.traj_end)
        if overlap_start > overlap_end:
            if self.verbose:
                print("⚠ 视频帧时间与轨迹时间无交集")
                print(f"  帧时间范围: {frame_start} 到 {frame_end}")
                print(f"  轨迹时间范围: {self.traj_start} 到 {self.traj_end}")
            return pd.DataFrame()

        if self.verbose:
            print(f"✓ 取时间交集: {overlap_start} 到 {overlap_end}")

        aligned_data = []
        kept_frame_names = set()

        for frame_file in tqdm(frame_files, desc="对齐数据", disable=not self.verbose):
            # 从文件名提取时间
            try:
                frame_time = self._parse_frame_time(frame_file)
                if frame_time is None:
                    continue
                if frame_time < overlap_start or frame_time > overlap_end:
                    continue

                frame_timestamp = pd.Timestamp(frame_time).value // 10**9

                # 在时间容差范围内查找匹配的轨迹数据
                matched_idx = None
                for tolerance in range(self.time_tolerance + 1):
                    offsets = [0] if tolerance == 0 else [-tolerance, tolerance]
                    for offset in offsets:
                        check_timestamp = frame_timestamp + offset
                        if check_timestamp in self.time_index:
                            matched_idx = self.time_index[check_timestamp]
                            break
                    if matched_idx is not None:
                        break

                if matched_idx is not None:
                    # 复制帧到最终目录
                    final_frame_path = self.final_frames_dir / frame_file.name
                    if frame_file != final_frame_path:
                        shutil.copy2(str(frame_file), str(final_frame_path))
                    kept_frame_names.add(frame_file.name)

                    # 获取对应的轨迹数据
                    trajectory_row = self.trajectory_df.iloc[matched_idx]
                    frame_source = self.timestamp_source_map.get(frame_file.name, {})

                    aligned_data.append({
                        'frame_path': str(final_frame_path),
                        'frame_time': frame_time,
                        'timestamp_source': frame_source.get('source', 'unknown'),
                        'timestamp_inferred': bool(frame_source.get('inferred', False)),
                        'video_file': frame_source.get('video_file', ''),
                        'frame_number': frame_source.get('second_in_video', 0),
                        'second_in_video': frame_source.get('second_in_video', 0),
                        **trajectory_row.to_dict()
                    })
            except Exception as e:
                if self.verbose:
                    print(f"⚠ 处理失败 {frame_file.name}: {e}")
                continue

        result_df = pd.DataFrame(aligned_data)
        if len(result_df) > 0:
            sort_cols = [col for col in ['frame_time', 'video_file', 'second_in_video'] if col in result_df.columns]
            result_df = result_df.sort_values(sort_cols).reset_index(drop=True)

        # 删除交集外或未成功匹配的帧，确保最终目录与 CSV 尽量一致
        for frame_file in self.final_frames_dir.glob("*.jpg"):
            if frame_file.name not in kept_frame_names:
                frame_file.unlink()

        if self.verbose:
            print(f"\n✓ 成功对齐 {len(result_df)} 个帧")
            print(f"✓ 最终保留图片数: {len(list(self.final_frames_dir.glob('*.jpg')))}")

        return result_df

    def save_results(self, aligned_df: pd.DataFrame):
        """保存对齐结果"""
        if self.verbose:
            print("\n保存结果...")

        # 保存 CSV
        output_csv = self.output_dir / "aligned_data.csv"
        aligned_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

        # 保存 JSON
        output_json = self.output_dir / "aligned_data.json"
        aligned_df.to_json(output_json, orient='records', force_ascii=False, indent=2)

        # 保存统计信息
        stats = {
            'total_aligned_frames': len(aligned_df),
            'time_range': {
                'start': str(aligned_df['frame_time'].min()),
                'end': str(aligned_df['frame_time'].max())
            },
            'output_directory': str(self.output_dir)
        }

        stats_file = self.output_dir / "alignment_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"✓ 已保存:")
            print(f"  - {output_csv}")
            print(f"  - {output_json}")
            print(f"  - {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='视频轨迹数据对齐工具 - 完整流程')
    parser.add_argument('--video-dir', '-v', default='data/video/B-2024-10-18',
                       help='视频文件夹路径')
    parser.add_argument('--trajectory', '-t',
                       default='data/trajectory/B-2024-10-18/12-12-49_23-59-58.xlsx',
                       help='轨迹数据Excel文件路径')
    parser.add_argument('--output', '-o', default='data/aligned_output',
                       help='输出目录')
    parser.add_argument('--time-tolerance', type=int, default=2,
                       help='时间容差(秒)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='减少输出')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 步骤1: 提取帧
    extractor = VideoFrameExtractor(
        video_dir=args.video_dir,
        output_dir=args.output,
        verbose=not args.quiet
    )
    video_counts = extractor.extract_frames_from_all_videos()

    # 步骤2: OCR识别和重命名
    frames_dir = output_dir / "extracted_frames"
    recognizer = FrameTimeRecognizer(
        frames_dir=str(frames_dir),
        verbose=not args.quiet
    )
    success, fail = recognizer.recognize_and_rename_all()

    # 步骤3: 删除重复帧
    deduplicator = FrameDeduplicator(
        frames_dir=str(frames_dir),
        verbose=not args.quiet
    )
    removed = deduplicator.remove_duplicates()

    # 步骤4: 轨迹对齐
    aligner = TrajectoryAligner(
        trajectory_path=args.trajectory,
        frames_dir=str(frames_dir),
        output_dir=args.output,
        video_dir=args.video_dir,
        time_tolerance=args.time_tolerance,
        verbose=not args.quiet
    )
    aligned_df = aligner.align_frames_with_trajectory()

    # 保存结果
    if len(aligned_df) > 0:
        aligner.save_results(aligned_df)
        print("\n" + "=" * 80)
        print("✓ 处理完成!")
        print("=" * 80)
        print(f"总帧数: {len(aligned_df)}")
        print(f"输出目录: {args.output}")
    else:
        print("\n✗ 没有找到可对齐的数据")


if __name__ == '__main__':
    main()
