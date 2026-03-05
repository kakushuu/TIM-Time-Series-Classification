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

        temp_frames = list(self.frames_dir.glob("temp_*.jpg"))
        if self.verbose:
            print(f"✓ 找到 {len(temp_frames)} 个临时帧文件\n")

        success_count = 0
        fail_count = 0

        for temp_frame in tqdm(temp_frames, desc="OCR识别", disable=not self.verbose):
            # OCR识别时间
            timestamp = self._extract_timestamp_ocr(temp_frame)

            if timestamp:
                # 重命名文件
                new_filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                new_path = self.frames_dir / new_filename

                # 处理重名
                if new_path.exists() and new_path != temp_frame:
                    # 添加序号后缀
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
                    'timestamp': timestamp.isoformat()
                })
                success_count += 1
            else:
                # OCR失败,删除临时文件
                temp_frame.unlink()
                fail_count += 1

        if self.verbose:
            print(f"\n✓ OCR成功: {success_count}")
            print(f"✓ OCR失败: {fail_count}")

        return success_count, fail_count

    def _extract_timestamp_ocr(self, frame_path: Path) -> Optional[datetime]:
        """使用OCR从帧中提取时间戳"""
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # 尝试不同的区域大小
        for h_ratio in [0.08, 0.10, 0.12]:
            for w_ratio in [0.4, 0.45, 0.5]:
                timestamp_region = gray[0:int(height * h_ratio), 0:int(width * w_ratio)]

                _, binary = cv2.threshold(timestamp_region, 150, 255, cv2.THRESH_BINARY)
                pil_image = Image.fromarray(binary)

                configs = [
                    r'--oem 3 --psm 7',
                    r'--oem 3 --psm 6',
                    r'--oem 3 --psm 11',
                ]

                for config in configs:
                    try:
                        text = pytesseract.image_to_string(pil_image, config=config)
                        text = text.strip()

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
                 time_tolerance: int = 2, verbose: bool = True):
        self.trajectory_path = Path(trajectory_path)
        self.frames_dir = Path(frames_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.time_tolerance = time_tolerance
        self.verbose = verbose

        # 最终帧目录
        self.final_frames_dir = self.output_dir / "aligned_frames"
        self.final_frames_dir.mkdir(exist_ok=True)

        # 加载轨迹数据
        self.trajectory_df = pd.read_excel(trajectory_path)
        self.trajectory_df['定位时间'] = pd.to_datetime(self.trajectory_df['定位时间'])

        # 创建时间索引
        self.trajectory_df['时间戳'] = self.trajectory_df['定位时间'].apply(
            lambda x: int(pd.Timestamp(x).timestamp()))
        self.time_index = {row['时间戳']: idx for idx, row in self.trajectory_df.iterrows()}

        if self.verbose:
            print("=" * 80)
            print("步骤4: 轨迹数据对齐")
            print("=" * 80)
            print(f"✓ 加载轨迹数据: {len(self.trajectory_df)} 条记录")

    def align_frames_with_trajectory(self) -> pd.DataFrame:
        """
        将帧与轨迹数据对齐

        Returns:
            对齐后的DataFrame
        """
        if self.verbose:
            print(f"✓ 开始对齐...")

        frame_files = sorted(self.frames_dir.glob("*.jpg"))
        aligned_data = []

        for frame_file in tqdm(frame_files, desc="对齐数据", disable=not self.verbose):
            # 从文件名提取时间
            try:
                time_str = frame_file.stem
                if '_' in time_str and time_str.count('_') >= 2:
                    # 格式: YYYYMMDD_HHMMSS 或 YYYYMMDD_HHMMSS_001
                    parts = time_str.split('_')
                    datetime_str = f"{parts[0]}_{parts[1]}_{parts[2]}"
                    frame_time = datetime.strptime(datetime_str, '%Y%m%d_%H%M%S')
                else:
                    frame_time = datetime.strptime(time_str, '%Y%m%d_%H%M%S')

                frame_timestamp = int(pd.Timestamp(frame_time).timestamp())

                # 在时间容差范围内查找匹配的轨迹数据
                matched_idx = None
                for tolerance in range(self.time_tolerance + 1):
                    for offset in [-tolerance, tolerance]:
                        check_timestamp = frame_timestamp + offset
                        if check_timestamp in self.time_index:
                            matched_idx = self.time_index[check_timestamp]
                            break
                    if matched_idx is not None:
                        break

                if matched_idx is not None:
                    # 复制帧到最终目录
                    import shutil
                    final_frame_path = self.final_frames_dir / frame_file.name
                    if frame_file != final_frame_path:
                        shutil.copy2(str(frame_file), str(final_frame_path))

                    # 获取对应的轨迹数据
                    trajectory_row = self.trajectory_df.iloc[matched_idx]

                    aligned_data.append({
                        'frame_path': str(final_frame_path),
                        'frame_time': frame_time,
                        'video_file': self._get_video_file(frame_file.name),
                        'frame_number': 0,  # 简化处理
                        'second_in_video': 0,
                        **trajectory_row.to_dict()
                    })
            except Exception as e:
                if self.verbose:
                    print(f"⚠ 处理失败 {frame_file.name}: {e}")
                continue

        result_df = pd.DataFrame(aligned_data)

        if self.verbose:
            print(f"\n✓ 成功对齐 {len(result_df)} 个帧")

        return result_df

    def _get_video_file(self, frame_filename: str) -> str:
        """从帧文件名推断视频文件(简化版本)"""
        # 这里可以根据实际需求实现更复杂的逻辑
        return "unknown.mp4"

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
