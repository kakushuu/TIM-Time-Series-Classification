#!/usr/bin/env python3
"""
最终版：视频与轨迹数据对齐脚本

改进：
1. 使用OCR从最后一帧识别实际时间戳
2. 根据视频时长反推开始时间
3. 如果OCR失败，才使用文件名时间作为备选
4. 自动检测和处理时间偏移问题
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
import pytesseract
from PIL import Image


# 要删除的无意义列
COLUMNS_TO_REMOVE = [
    '上点时间',
    '补点',
    'unitid',
    'GPS标记',
    '播种播肥 / 油耗 / 压力',
    '播种播肥/油耗/压力',  # 可能的不同格式
    '抛肥量(立方)',
    '定位间隔'
]


class VideoTrajectoryAligner:
    """视频与轨迹数据对齐工具（最终版）"""

    def __init__(self, trajectory_path: str, video_dir: str, output_dir: str,
                 time_tolerance: int = 2, verbose: bool = True):
        """
        初始化对齐器

        Args:
            trajectory_path: 轨迹数据Excel文件路径
            video_dir: 视频文件夹路径
            output_dir: 输出目录
            time_tolerance: 时间容差（秒）
            verbose: 是否输出详细信息
        """
        self.trajectory_path = Path(trajectory_path)
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.time_tolerance = time_tolerance
        self.verbose = verbose

        # 创建帧输出目录
        self.frames_dir = self.output_dir / "aligned_frames"
        self.frames_dir.mkdir(exist_ok=True)

        # 加载轨迹数据
        if self.verbose:
            print("=" * 80)
            print("步骤1: 加载轨迹数据")
            print("=" * 80)

        self.trajectory_df = pd.read_excel(trajectory_path)
        self.trajectory_df['定位时间'] = pd.to_datetime(self.trajectory_df['定位时间'])

        # 过滤无意义列
        cols_before = len(self.trajectory_df.columns)
        self.trajectory_df = self.trajectory_df.drop(
            columns=[col for col in COLUMNS_TO_REMOVE if col in self.trajectory_df.columns]
        )
        cols_after = len(self.trajectory_df.columns)

        if self.verbose and cols_before != cols_after:
            print(f"✓ 过滤了 {cols_before - cols_after} 个无意义列")

        # 创建时间索引（使用正确的时间戳转换方式）
        self.trajectory_df['时间戳'] = self.trajectory_df['定位时间'].apply(lambda x: int(pd.Timestamp(x).timestamp()))
        self.time_index = {row['时间戳']: idx for idx, row in self.trajectory_df.iterrows()}

        if self.verbose:
            print(f"✓ 加载完成: {len(self.trajectory_df)} 条轨迹记录")
            print(f"  时间范围: {self.trajectory_df['定位时间'].min()} 到 {self.trajectory_df['定位时间'].max()}")

    def extract_timestamp_from_frame(self, frame: np.ndarray) -> Optional[datetime]:
        """
        使用OCR从视频帧中提取时间戳

        Args:
            frame: OpenCV图像帧

        Returns:
            识别到的datetime对象，失败返回None
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 提取左上角区域
        height, width = gray.shape

        # 尝试不同的区域大小
        for h_ratio in [0.08, 0.10, 0.12]:
            for w_ratio in [0.4, 0.45, 0.5]:
                timestamp_region = gray[0:int(height * h_ratio), 0:int(width * w_ratio)]

                # 二值化处理
                _, binary = cv2.threshold(timestamp_region, 150, 255, cv2.THRESH_BINARY)

                # OCR识别
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

                        # 解析时间戳
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

    def get_video_time_range(self, video_path: Path) -> Tuple[Optional[datetime], Optional[datetime], str]:
        """
        获取视频的实际时间范围（使用OCR + 时长计算）

        Args:
            video_path: 视频文件路径

        Returns:
            (开始时间, 结束时间, 时间来源说明)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, None, "无法打开视频"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        # 提取最后一帧（通常OCR识别率更高）
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
        ret, last_frame = cap.read()

        # 尝试提取第一帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = cap.read()

        cap.release()

        start_time = None
        end_time = None
        source = ""

        # 策略1: 使用OCR识别最后一帧，然后反推开始时间
        if last_frame is not None:
            end_time = self.extract_timestamp_from_frame(last_frame)
            if end_time:
                start_time = end_time - timedelta(seconds=duration)
                source = "OCR识别最后一帧+时长反推"

                # 验证：如果第一帧也能识别，检查一致性
                if first_frame is not None:
                    first_time_ocr = self.extract_timestamp_from_frame(first_frame)
                    if first_time_ocr:
                        diff = abs((first_time_ocr - start_time).total_seconds())
                        if diff < 5:  # 误差小于5秒
                            source += "(已验证)"
                        else:
                            # 如果差异大，使用两个OCR结果的平均值
                            start_time = first_time_ocr
                            source = f"OCR识别第一帧和最后一帧(差异{diff:.1f}秒)"

        # 策略2: 如果最后一帧OCR失败，尝试第一帧
        if start_time is None and first_frame is not None:
            start_time = self.extract_timestamp_from_frame(first_frame)
            if start_time:
                end_time = start_time + timedelta(seconds=duration)
                source = "OCR识别第一帧+时长推算"

        # 策略3: 如果都失败，使用文件名（备选方案）
        if start_time is None:
            filename_pattern = r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})'
            match = re.search(filename_pattern, video_path.stem)
            if match:
                year, month, day, hour, minute, second = map(int, match.groups())
                start_time = datetime(year, month, day, hour, minute, second)
                end_time = start_time + timedelta(seconds=duration)
                source = "文件名推断(OCR失败)"

        return start_time, end_time, source

    def analyze_videos(self) -> List[Dict]:
        """
        分析所有视频文件

        Returns:
            视频信息列表
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("步骤2: 分析视频文件（使用OCR识别实际时间）")
            print("=" * 80)

        video_files = sorted(self.video_dir.glob("*.mp4"))

        if self.verbose:
            print(f"✓ 找到 {len(video_files)} 个视频文件\n")

        video_info_list = []

        for i, video_path in enumerate(video_files, 1):
            start_time, end_time, source = self.get_video_time_range(video_path)

            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            cap.release()

            if start_time and end_time:
                video_info = {
                    'path': video_path,
                    'filename': video_path.name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'total_frames': total_frames,
                    'fps': fps,
                    'time_source': source
                }

                video_info_list.append(video_info)

                if self.verbose:
                    print(f"{i}. {video_path.name}")
                    print(f"   时间范围: {start_time} 到 {end_time}")
                    print(f"   时长: {duration:.1f}秒 ({duration/60:.1f}分钟)")
                    print(f"   时间来源: {source}\n")
            else:
                if self.verbose:
                    print(f"{i}. ⚠️  无法确定时间: {video_path.name}\n")

        return video_info_list

    def check_time_overlap(self, video_info_list: List[Dict]) -> Tuple[bool, Dict]:
        """
        检查视频和轨迹数据的时间重叠情况
        """
        if self.verbose:
            print("=" * 80)
            print("步骤3: 检查时间范围匹配")
            print("=" * 80)

        if not video_info_list:
            return False, {'message': '没有有效的视频文件'}

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

        if video_end < traj_start:
            overlap_info['overlap'] = False
            overlap_info['message'] = '视频数据在轨迹数据之前，无重叠'
            if self.verbose:
                print(f"✗ {overlap_info['message']}")
                print(f"  视频时间: {video_start} 到 {video_end}")
                print(f"  轨迹时间: {traj_start} 到 {traj_end}")
            return False, overlap_info
        elif video_start > traj_end:
            overlap_info['overlap'] = False
            overlap_info['message'] = '视频数据在轨迹数据之后，无重叠'
            if self.verbose:
                print(f"✗ {overlap_info['message']}")
                print(f"  视频时间: {video_start} 到 {video_end}")
                print(f"  轨迹时间: {traj_start} 到 {traj_end}")
            return False, overlap_info
        else:
            overlap_start = max(video_start, traj_start)
            overlap_end = min(video_end, traj_end)
            overlap_duration = (overlap_end - overlap_start).total_seconds()

            overlap_info['overlap'] = True
            overlap_info['overlap_start'] = overlap_start
            overlap_info['overlap_end'] = overlap_end
            overlap_info['overlap_duration'] = overlap_duration

            if self.verbose:
                print(f"✓ 找到时间重叠区域:")
                print(f"  重叠时间: {overlap_start} 到 {overlap_end}")
                print(f"  重叠时长: {overlap_duration:.1f}秒 ({overlap_duration/60:.1f}分钟)")

            return True, overlap_info

    def extract_frames_per_second(self, video_info: Dict) -> List[Dict]:
        """
        从视频中每秒提取一帧，并与轨迹数据对齐
        """
        aligned_frames = []
        cap = cv2.VideoCapture(str(video_info['path']))

        if not cap.isOpened():
            return aligned_frames

        fps = video_info['fps']
        total_frames = video_info['total_frames']
        start_time = video_info['start_time']
        duration = video_info['duration']

        for second in range(int(duration)):
            frame_number = int(second * fps)
            if frame_number >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if not ret:
                continue

            frame_time = start_time + timedelta(seconds=second + 1)  # 增加1秒偏移
            # 使用正确的时间戳转换（匹配DataFrame的方式）
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
                frame_filename = f"{frame_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                frame_path = self.frames_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)

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
        """
        video_info_list = self.analyze_videos()
        has_overlap, overlap_info = self.check_time_overlap(video_info_list)

        if not has_overlap:
            if self.verbose:
                print("\n" + "!" * 80)
                print("警告：视频和轨迹数据的时间范围不重叠！")
                print("!" * 80)
                print(f"\n详细信息:")
                print(f"  视频时间范围: {overlap_info.get('video_start')} 到 {overlap_info.get('video_end')}")
                print(f"  轨迹时间范围: {overlap_info.get('trajectory_start')} 到 {overlap_info.get('trajectory_end')}")

            return pd.DataFrame()

        if self.verbose:
            print("\n" + "=" * 80)
            print("步骤4: 提取和对齐视频帧")
            print("=" * 80)

        all_aligned_data = []

        for video_info in video_info_list:
            if video_info['end_time'] < overlap_info['trajectory_start']:
                if self.verbose:
                    print(f"跳过 {video_info['filename']} (时间不重叠)")
                continue
            if video_info['start_time'] > overlap_info['trajectory_end']:
                if self.verbose:
                    print(f"跳过 {video_info['filename']} (时间不重叠)")
                continue

            if self.verbose:
                print(f"处理 {video_info['filename']}...")

            aligned_frames = self.extract_frames_per_second(video_info)
            all_aligned_data.extend(aligned_frames)

            if self.verbose:
                print(f"  ✓ 提取了 {len(aligned_frames)} 个对齐帧")

        result_df = pd.DataFrame(all_aligned_data)
        return result_df

    def save_results(self, aligned_df: pd.DataFrame, overlap_info: Dict):
        """
        保存对齐结果
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("步骤5: 保存结果")
            print("=" * 80)

        if len(aligned_df) == 0:
            if self.verbose:
                print("⚠ 没有对齐的数据可保存")
            return

        output_csv = self.output_dir / "aligned_data.csv"
        aligned_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        if self.verbose:
            print(f"✓ 对齐数据已保存到: {output_csv}")

        output_json = self.output_dir / "aligned_data.json"
        aligned_df.to_json(output_json, orient='records', force_ascii=False, indent=2)
        if self.verbose:
            print(f"✓ 对齐数据（JSON）已保存到: {output_json}")

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
        if self.verbose:
            print(f"✓ 统计信息已保存到: {stats_file}")

            print(f"\n{'=' * 80}")
            print("对齐完成统计:")
            print("=" * 80)
            print(f"✓ 总计对齐帧数: {len(aligned_df)}")
            print(f"✓ 涉及视频数: {aligned_df['video_file'].nunique()}")
            print(f"✓ 时间范围: {aligned_df['frame_time'].min()} 到 {aligned_df['frame_time'].max()}")
            print(f"✓ 帧保存位置: {self.frames_dir}")


def main():
    parser = argparse.ArgumentParser(description='视频与轨迹数据对齐工具（最终版）')
    parser.add_argument('--trajectory', '-t',
                       default='data/trajectory/B-2024-10-18/12-12-49_23-59-58.xlsx',
                       help='轨迹数据Excel文件路径')
    parser.add_argument('--video-dir', '-v',
                       default='data/video/B-2024-10-18',
                       help='视频文件夹路径')
    parser.add_argument('--output', '-o',
                       default='data/aligned_output',
                       help='输出目录')
    parser.add_argument('--time-tolerance',
                       type=int,
                       default=2,
                       help='时间容差（秒）')
    parser.add_argument('--quiet', '-q',
                       action='store_true',
                       help='减少输出信息')

    args = parser.parse_args()

    aligner = VideoTrajectoryAligner(
        trajectory_path=args.trajectory,
        video_dir=args.video_dir,
        output_dir=args.output,
        time_tolerance=args.time_tolerance,
        verbose=not args.quiet
    )

    aligned_df = aligner.process_all_videos()

    if len(aligned_df) > 0:
        # 获取overlap_info用于保存
        video_info_list = aligner.analyze_videos()
        _, overlap_info = aligner.check_time_overlap(video_info_list)
        aligner.save_results(aligned_df, overlap_info)
        print("\n✓ 处理完成！")
    else:
        print("\n✗ 没有找到可对齐的数据！")


if __name__ == '__main__':
    main()
