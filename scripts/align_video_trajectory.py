#!/usr/bin/env python3
"""
视频与轨迹数据对齐脚本
功能：
1. 从视频中每秒提取一帧
2. 使用OCR识别视频帧上的时间戳
3. 将视频帧与轨迹数据按时间对齐
"""

import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pytesseract
from PIL import Image
import argparse
import json
from typing import Tuple, Optional, List, Dict
import re


class VideoTrajectoryAligner:
    """视频与轨迹数据对齐工具"""

    def __init__(self, trajectory_path: str, video_dir: str, output_dir: str):
        """
        初始化对齐器

        Args:
            trajectory_path: 轨迹数据Excel文件路径
            video_dir: 视频文件夹路径
            output_dir: 输出目录
        """
        self.trajectory_path = Path(trajectory_path)
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建帧输出目录
        self.frames_dir = self.output_dir / "aligned_frames"
        self.frames_dir.mkdir(exist_ok=True)

        # 加载轨迹数据
        print("加载轨迹数据...")
        self.trajectory_df = pd.read_excel(trajectory_path)
        self.trajectory_df['定位时间'] = pd.to_datetime(self.trajectory_df['定位时间'])
        print(f"轨迹数据: {len(self.trajectory_df)} 条记录")
        print(f"时间范围: {self.trajectory_df['定位时间'].min()} 到 {self.trajectory_df['定位时间'].max()}")

    def extract_timestamp_from_frame(self, frame: np.ndarray) -> Optional[datetime]:
        """
        从视频帧中提取时间戳

        Args:
            frame: OpenCV图像帧

        Returns:
            识别到的datetime对象，失败返回None
        """
        # 转换为灰度图以提高OCR准确率
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 提取左上角区域（时间戳区域）
        height, width = gray.shape
        # 时间戳通常在前40%的宽度和8%的高度
        timestamp_region = gray[0:int(height * 0.08), 0:int(width * 0.4)]

        # 二值化处理，增强文字对比度
        _, binary = cv2.threshold(timestamp_region, 150, 255, cv2.THRESH_BINARY)

        # 转换为PIL图像
        pil_image = Image.fromarray(binary)

        # OCR识别 - 使用数字模式提高准确率
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789-: '
        text = pytesseract.image_to_string(pil_image, config=custom_config)

        # 清理文本
        text = text.strip()

        # 解析时间戳 - 针对格式 "2024-10-18 04:38:10"
        pattern = r'(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})'
        match = re.search(pattern, text)

        if match:
            try:
                year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                hour, minute, second = int(match.group(4)), int(match.group(5)), int(match.group(6))
                return datetime(year, month, day, hour, minute, second)
            except ValueError:
                pass

        # 如果OCR失败，返回None
        return None

    def get_video_time_range(self, video_path: Path) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        获取视频的时间范围（通过OCR识别第一帧和最后一帧）

        Args:
            video_path: 视频文件路径

        Returns:
            (开始时间, 结束时间)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return None, None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        print(f"\n处理视频: {video_path.name}")
        print(f"  总帧数: {total_frames}, FPS: {fps:.2f}, 时长: {duration:.2f}秒")

        # 提取第一帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = cap.read()

        # 提取最后一帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
        ret, last_frame = cap.read()

        cap.release()

        # OCR识别时间戳
        start_time = self.extract_timestamp_from_frame(first_frame) if first_frame is not None else None
        end_time = self.extract_timestamp_from_frame(last_frame) if last_frame is not None else None

        if start_time and end_time:
            print(f"  识别到时间范围: {start_time} 到 {end_time}")
        else:
            # 尝试从文件名解析时间
            filename_pattern = r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})'
            match = re.search(filename_pattern, video_path.stem)
            if match:
                year, month, day, hour, minute, second = map(int, match.groups())
                start_time = datetime(year, month, day, hour, minute, second)
                # 根据视频时长计算结束时间
                end_time = start_time + timedelta(seconds=duration)
                print(f"  从文件名推断时间范围: {start_time} 到 {end_time}")

        return start_time, end_time

    def extract_frames_per_second(self, video_path: Path, start_time: datetime,
                                   trajectory_times: List[datetime]) -> List[Dict]:
        """
        从视频中每秒提取一帧，并与轨迹时间对齐

        Args:
            video_path: 视频文件路径
            start_time: 视频开始时间
            trajectory_times: 轨迹时间列表

        Returns:
            对齐的帧信息列表
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        aligned_frames = []
        trajectory_time_set = set(t for t in trajectory_times)

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

            # 检查是否与轨迹时间匹配
            if frame_time in trajectory_time_set:
                # 保存帧
                frame_filename = f"{frame_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                frame_path = self.frames_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)

                # 获取对应的轨迹数据
                trajectory_row = self.trajectory_df[
                    self.trajectory_df['定位时间'] == frame_time
                ].iloc[0]

                aligned_frames.append({
                    'frame_path': str(frame_path),
                    'frame_time': frame_time,
                    'video_file': video_path.name,
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
        # 获取所有视频文件
        video_files = sorted(self.video_dir.glob("*.mp4"))
        print(f"\n找到 {len(video_files)} 个视频文件")

        # 获取轨迹时间列表
        trajectory_times = self.trajectory_df['定位时间'].tolist()

        all_aligned_data = []

        # 处理每个视频
        for video_path in video_files:
            start_time, end_time = self.get_video_time_range(video_path)

            if start_time is None or end_time is None:
                print(f"  跳过视频（无法确定时间范围）: {video_path.name}")
                continue

            # 检查视频时间范围是否与轨迹数据重叠
            traj_start = self.trajectory_df['定位时间'].min()
            traj_end = self.trajectory_df['定位时间'].max()

            if end_time < traj_start or start_time > traj_end:
                print(f"  跳过视频（时间不重叠）: {video_path.name}")
                continue

            # 提取并对齐帧
            aligned_frames = self.extract_frames_per_second(
                video_path, start_time, trajectory_times
            )
            all_aligned_data.extend(aligned_frames)
            print(f"  提取了 {len(aligned_frames)} 个对齐帧")

        # 创建结果DataFrame
        result_df = pd.DataFrame(all_aligned_data)
        return result_df

    def save_results(self, aligned_df: pd.DataFrame):
        """
        保存对齐结果

        Args:
            aligned_df: 对齐后的数据DataFrame
        """
        if len(aligned_df) == 0:
            print("\n警告：没有找到匹配的视频帧和轨迹数据！")
            return

        # 保存为CSV
        output_csv = self.output_dir / "aligned_data.csv"
        aligned_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n对齐数据已保存到: {output_csv}")

        # 保存为JSON（更详细的信息）
        output_json = self.output_dir / "aligned_data.json"
        aligned_df.to_json(output_json, orient='records', force_ascii=False, indent=2)
        print(f"对齐数据（JSON）已保存到: {output_json}")

        # 生成统计信息
        stats = {
            'total_aligned_frames': len(aligned_df),
            'unique_videos': aligned_df['video_file'].nunique(),
            'time_range': {
                'start': str(aligned_df['frame_time'].min()),
                'end': str(aligned_df['frame_time'].max())
            },
            'output_directory': str(self.output_dir),
            'frames_directory': str(self.frames_dir)
        }

        stats_file = self.output_dir / "alignment_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"统计信息已保存到: {stats_file}")

        print(f"\n总计对齐帧数: {len(aligned_df)}")
        print(f"涉及视频数: {aligned_df['video_file'].nunique()}")
        print(f"时间范围: {aligned_df['frame_time'].min()} 到 {aligned_df['frame_time'].max()}")


def main():
    parser = argparse.ArgumentParser(description='视频与轨迹数据对齐工具')
    parser.add_argument('--trajectory', '-t',
                       default='data/trajectory/B-2024-10-18/12-12-49_23-59-58.xlsx',
                       help='轨迹数据Excel文件路径')
    parser.add_argument('--video-dir', '-v',
                       default='data/video/B-2024-10-18',
                       help='视频文件夹路径')
    parser.add_argument('--output', '-o',
                       default='data/aligned_output',
                       help='输出目录')

    args = parser.parse_args()

    # 创建对齐器并处理
    aligner = VideoTrajectoryAligner(
        trajectory_path=args.trajectory,
        video_dir=args.video_dir,
        output_dir=args.output
    )

    # 处理所有视频
    aligned_df = aligner.process_all_videos()

    # 保存结果
    aligner.save_results(aligned_df)

    print("\n处理完成！")


if __name__ == '__main__':
    main()
