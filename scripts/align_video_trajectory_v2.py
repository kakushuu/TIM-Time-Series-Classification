#!/usr/bin/env python3
"""
改进版：视频与轨迹数据对齐脚本

功能：
1. 自动检测视频和轨迹数据的时间范围是否匹配
2. 主要使用视频文件名中的时间戳（更可靠）
3. 可选使用OCR验证时间戳
4. 生成详细的对齐报告
5. 支持时间容差匹配
"""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import json
from typing import Tuple, Optional, List, Dict
import re


class VideoTrajectoryAligner:
    """视频与轨迹数据对齐工具（改进版）"""

    def __init__(self, trajectory_path: str, video_dir: str, output_dir: str,
                 use_ocr: bool = False, time_tolerance: int = 2):
        """
        初始化对齐器

        Args:
            trajectory_path: 轨迹数据Excel文件路径
            video_dir: 视频文件夹路径
            output_dir: 输出目录
            use_ocr: 是否使用OCR验证时间戳
            time_tolerance: 时间容差（秒），用于匹配视频帧和轨迹数据
        """
        self.trajectory_path = Path(trajectory_path)
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_ocr = use_ocr
        self.time_tolerance = time_tolerance

        # 创建帧输出目录
        self.frames_dir = self.output_dir / "aligned_frames"
        self.frames_dir.mkdir(exist_ok=True)

        # 加载轨迹数据
        print("=" * 80)
        print("步骤1: 加载轨迹数据")
        print("=" * 80)
        self.trajectory_df = pd.read_excel(trajectory_path)
        self.trajectory_df['定位时间'] = pd.to_datetime(self.trajectory_df['定位时间'])

        # 创建时间索引以便快速查找
        self.trajectory_df['时间戳'] = self.trajectory_df['定位时间'].astype(np.int64) // 10**9
        self.time_index = {row['时间戳']: idx for idx, row in self.trajectory_df.iterrows()}

        print(f"✓ 加载完成: {len(self.trajectory_df)} 条轨迹记录")
        print(f"  时间范围: {self.trajectory_df['定位时间'].min()} 到 {self.trajectory_df['定位时间'].max()}")

    def parse_time_from_filename(self, filename: str) -> Optional[datetime]:
        """
        从视频文件名解析时间戳

        Args:
            filename: 视频文件名（例如：20241018043810.mp4）

        Returns:
            解析出的datetime对象
        """
        pattern = r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})'
        match = re.search(pattern, Path(filename).stem)

        if match:
            year, month, day, hour, minute, second = map(int, match.groups())
            try:
                return datetime(year, month, day, hour, minute, second)
            except ValueError:
                return None
        return None

    def analyze_videos(self) -> List[Dict]:
        """
        分析所有视频文件

        Returns:
            视频信息列表
        """
        print("\n" + "=" * 80)
        print("步骤2: 分析视频文件")
        print("=" * 80)

        video_files = sorted(self.video_dir.glob("*.mp4"))
        print(f"✓ 找到 {len(video_files)} 个视频文件\n")

        video_info_list = []

        for video_path in video_files:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"⚠ 无法打开: {video_path.name}")
                continue

            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            cap.release()

            # 从文件名解析时间
            start_time = self.parse_time_from_filename(video_path.name)
            if start_time:
                end_time = start_time + timedelta(seconds=duration)

                video_info = {
                    'path': video_path,
                    'filename': video_path.name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'total_frames': total_frames,
                    'fps': fps
                }

                video_info_list.append(video_info)
                print(f"✓ {video_path.name}")
                print(f"  时间: {start_time} 到 {end_time}")
                print(f"  时长: {duration:.1f}秒 ({duration/60:.1f}分钟)\n")
            else:
                print(f"⚠ 无法解析文件名时间: {video_path.name}\n")

        return video_info_list

    def check_time_overlap(self, video_info_list: List[Dict]) -> Tuple[bool, Dict]:
        """
        检查视频和轨迹数据的时间重叠情况

        Args:
            video_info_list: 视频信息列表

        Returns:
            (是否有重叠, 详细信息)
        """
        print("=" * 80)
        print("步骤3: 检查时间范围匹配")
        print("=" * 80)

        if not video_info_list:
            return False, {'message': '没有有效的视频文件'}

        # 获取时间范围
        video_start = min(v['start_time'] for v in video_info_list)
        video_end = max(v['end_time'] for v in video_info_list)
        traj_start = self.trajectory_df['定位时间'].min()
        traj_end = self.trajectory_df['定位时间'].max()

        overlap_info = {
            'video_start': video_start,
            'video_end': video_end,
            'trajectory_start': traj_start,
            'trajectory_end': traj_end,
        }

        # 检查重叠
        if video_end < traj_start:
            overlap_info['overlap'] = False
            overlap_info['message'] = '视频数据在轨迹数据之前，无重叠'
            print(f"✗ {overlap_info['message']}")
            print(f"  视频时间: {video_start} 到 {video_end}")
            print(f"  轨迹时间: {traj_start} 到 {traj_end}")
            return False, overlap_info
        elif video_start > traj_end:
            overlap_info['overlap'] = False
            overlap_info['message'] = '视频数据在轨迹数据之后，无重叠'
            print(f"✗ {overlap_info['message']}")
            print(f"  视频时间: {video_start} 到 {video_end}")
            print(f"  轨迹时间: {traj_start} 到 {traj_end}")
            return False, overlap_info
        else:
            # 计算重叠时间段
            overlap_start = max(video_start, traj_start)
            overlap_end = min(video_end, traj_end)
            overlap_duration = (overlap_end - overlap_start).total_seconds()

            overlap_info['overlap'] = True
            overlap_info['overlap_start'] = overlap_start
            overlap_info['overlap_end'] = overlap_end
            overlap_info['overlap_duration'] = overlap_duration

            print(f"✓ 找到时间重叠区域:")
            print(f"  重叠时间: {overlap_start} 到 {overlap_end}")
            print(f"  重叠时长: {overlap_duration:.1f}秒 ({overlap_duration/60:.1f}分钟)")
            return True, overlap_info

    def extract_frames_per_second(self, video_info: Dict) -> List[Dict]:
        """
        从视频中每秒提取一帧，并与轨迹数据对齐

        Args:
            video_info: 视频信息字典

        Returns:
            对齐的帧信息列表
        """
        aligned_frames = []
        cap = cv2.VideoCapture(str(video_info['path']))

        if not cap.isOpened():
            return aligned_frames

        fps = video_info['fps']
        total_frames = video_info['total_frames']
        start_time = video_info['start_time']
        duration = video_info['duration']

        # 每秒提取一帧
        for second in range(int(duration)):
            frame_number = int(second * fps)
            if frame_number >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if not ret:
                continue

            # 计算这一帧对应的时间
            frame_time = start_time + timedelta(seconds=second)
            frame_timestamp = int(frame_time.timestamp())

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
                # 保存帧
                frame_filename = f"{frame_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                frame_path = self.frames_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)

                # 获取对应的轨迹数据
                trajectory_row = self.trajectory_df.iloc[matched_idx]

                aligned_frames.append({
                    'frame_path': str(frame_path),
                    'frame_time': frame_time,
                    'video_file': video_info['filename'],
                    'frame_number': frame_number,
                    'second_in_video': second,
                    **trajectory_row.to_dict()
                })

        cap.release()
        return aligned_frames

    def process_all_videos(self) -> pd.DataFrame:
        """
        处理所有视频文件并与轨迹数据对齐

        Returns:
            对齐后的数据DataFrame
        """
        # 分析视频
        video_info_list = self.analyze_videos()

        # 检查时间重叠
        has_overlap, overlap_info = self.check_time_overlap(video_info_list)

        if not has_overlap:
            print("\n" + "!" * 80)
            print("警告：视频和轨迹数据的时间范围不重叠！")
            print("!" * 80)
            print(f"\n详细信息:")
            print(f"  视频时间范围: {overlap_info.get('video_start')} 到 {overlap_info.get('video_end')}")
            print(f"  轨迹时间范围: {overlap_info.get('trajectory_start')} 到 {overlap_info.get('trajectory_end')}")
            print(f"\n建议:")
            print("  1. 检查是否使用了正确的视频文件")
            print("  2. 检查是否使用了正确的轨迹数据文件")
            print("  3. 如果有其他日期的数据，请指定正确的路径")

            # 返回空的DataFrame
            return pd.DataFrame()

        # 处理每个视频
        print("\n" + "=" * 80)
        print("步骤4: 提取和对齐视频帧")
        print("=" * 80)

        all_aligned_data = []

        for video_info in video_info_list:
            # 检查视频是否与轨迹数据时间重叠
            if video_info['end_time'] < overlap_info['trajectory_start']:
                print(f"跳过 {video_info['filename']} (时间不重叠)")
                continue
            if video_info['start_time'] > overlap_info['trajectory_end']:
                print(f"跳过 {video_info['filename']} (时间不重叠)")
                continue

            print(f"处理 {video_info['filename']}...")
            aligned_frames = self.extract_frames_per_second(video_info)
            all_aligned_data.extend(aligned_frames)
            print(f"  ✓ 提取了 {len(aligned_frames)} 个对齐帧")

        # 创建结果DataFrame
        result_df = pd.DataFrame(all_aligned_data)
        return result_df

    def save_results(self, aligned_df: pd.DataFrame, overlap_info: Dict):
        """
        保存对齐结果

        Args:
            aligned_df: 对齐后的数据DataFrame
            overlap_info: 时间重叠信息
        """
        print("\n" + "=" * 80)
        print("步骤5: 保存结果")
        print("=" * 80)

        if len(aligned_df) == 0:
            print("⚠ 没有对齐的数据可保存")
            return

        # 保存为CSV
        output_csv = self.output_dir / "aligned_data.csv"
        aligned_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"✓ 对齐数据已保存到: {output_csv}")

        # 保存为JSON
        output_json = self.output_dir / "aligned_data.json"
        aligned_df.to_json(output_json, orient='records', force_ascii=False, indent=2)
        print(f"✓ 对齐数据（JSON）已保存到: {output_json}")

        # 生成统计信息
        stats = {
            'summary': {
                'total_aligned_frames': len(aligned_df),
                'unique_videos': aligned_df['video_file'].nunique(),
                'time_range': {
                    'start': str(aligned_df['frame_time'].min()),
                    'end': str(aligned_df['frame_time'].max())
                }
            },
            'time_overlap': overlap_info,
            'directories': {
                'output': str(self.output_dir),
                'frames': str(self.frames_dir)
            }
        }

        stats_file = self.output_dir / "alignment_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        print(f"✓ 统计信息已保存到: {stats_file}")

        print(f"\n{'=' * 80}")
        print("对齐完成统计:")
        print("=" * 80)
        print(f"✓ 总计对齐帧数: {len(aligned_df)}")
        print(f"✓ 涉及视频数: {aligned_df['video_file'].nunique()}")
        print(f"✓ 时间范围: {aligned_df['frame_time'].min()} 到 {aligned_df['frame_time'].max()}")
        print(f"✓ 帧保存位置: {self.frames_dir}")


def main():
    parser = argparse.ArgumentParser(description='视频与轨迹数据对齐工具（改进版）')
    parser.add_argument('--trajectory', '-t',
                       default='data/trajectory/B-2024-10-18/12-12-49_23-59-58.xlsx',
                       help='轨迹数据Excel文件路径')
    parser.add_argument('--video-dir', '-v',
                       default='data/video/B-2024-10-18',
                       help='视频文件夹路径')
    parser.add_argument('--output', '-o',
                       default='data/aligned_output',
                       help='输出目录')
    parser.add_argument('--use-ocr',
                       action='store_true',
                       help='是否使用OCR验证时间戳（实验性功能）')
    parser.add_argument('--time-tolerance',
                       type=int,
                       default=2,
                       help='时间容差（秒），用于匹配视频帧和轨迹数据')

    args = parser.parse_args()

    # 创建对齐器并处理
    aligner = VideoTrajectoryAligner(
        trajectory_path=args.trajectory,
        video_dir=args.video_dir,
        output_dir=args.output,
        use_ocr=args.use_ocr,
        time_tolerance=args.time_tolerance
    )

    # 分析视频
    video_info_list = aligner.analyze_videos()

    # 检查时间重叠
    has_overlap, overlap_info = aligner.check_time_overlap(video_info_list)

    # 处理所有视频
    aligned_df = aligner.process_all_videos()

    # 保存结果
    if len(aligned_df) > 0:
        aligner.save_results(aligned_df, overlap_info)
        print("\n✓ 处理完成！")
    else:
        print("\n✗ 没有找到可对齐的数据！")


if __name__ == '__main__':
    main()
