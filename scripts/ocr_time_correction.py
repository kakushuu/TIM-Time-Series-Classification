#!/usr/bin/env python3
"""
OCR时间识别和矫正工具

功能：
1. 对18:41:34之后的每一帧进行OCR识别
2. 比较OCR识别时间与CSV时间
3. 自动矫正不一致的帧
4. 生成详细的矫正报告

用法：
    python3 scripts/ocr_time_correction.py
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


class OCRTimeCorrector:
    """OCR时间识别和矫正器"""

    def __init__(self, csv_file, frames_dir, output_dir):
        self.csv_file = Path(csv_file)
        self.frames_dir = Path(frames_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载数据
        self.df = pd.read_csv(csv_file)
        self.corrections = []
        self.failed_ocr = []

    def extract_timestamp_ocr(self, frame_path):
        """
        使用OCR从帧中提取时间戳

        Returns:
            (datetime对象, OCR文本) 或 (None, "")
        """
        # 读取图像
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return None, ""

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
                        pattern = r'(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})'
                        match = re.search(pattern, text)

                        if match:
                            groups = match.groups()
                            try:
                                year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                                hour, minute, second = int(groups[3]), int(groups[4]), int(groups[5])
                                dt = datetime(year, month, day, hour, minute, second)
                                return dt, text
                            except (ValueError, IndexError):
                                continue
                    except Exception:
                        continue

        return None, ""

    def correct_times_after_timestamp(self, timestamp_str, auto_correct=True, threshold_seconds=0.5):
        """
        矫正指定时间戳之后的所有帧

        Args:
            timestamp_str: 起始时间戳字符串
            auto_correct: 是否自动应用矫正
            threshold_seconds: 时间差异阈值（秒）
        """
        print("=" * 80)
        print(f"OCR时间识别和矫正: {timestamp_str} 之后")
        print("=" * 80)

        # 筛选目标帧
        target_df = self.df[self.df['frame_time'] >= timestamp_str].copy()
        print(f"\n目标帧数: {len(target_df)}")

        if len(target_df) == 0:
            print("没有找到目标帧")
            return

        # 按视频分组处理
        video_files = target_df['video_file'].unique()
        print(f"涉及视频: {len(video_files)} 个\n")

        # 处理每个视频
        for video_file in video_files:
            video_df = target_df[target_df['video_file'] == video_file].copy()
            print(f"\n处理视频: {video_file}")
            print(f"  帧数: {len(video_df)}")

            # OCR识别每一帧
            print(f"  正在进行OCR识别...")

            corrections_for_video = []

            for idx in tqdm(video_df.index, desc=f"  处理 {video_file}"):
                row = self.df.loc[idx]
                frame_path = Path(row['frame_path'])

                # OCR识别
                ocr_time, ocr_text = self.extract_timestamp_ocr(frame_path)

                if ocr_time is None:
                    self.failed_ocr.append({
                        'index': idx,
                        'frame_path': str(frame_path),
                        'video_file': row['video_file'],
                        'csv_time': row['frame_time']
                    })
                    continue

                # 解析CSV时间
                csv_time_str = str(row['frame_time'])
                if '.' in csv_time_str:
                    csv_time = datetime.strptime(csv_time_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
                else:
                    csv_time = datetime.strptime(csv_time_str, '%Y-%m-%d %H:%M:%S')

                # 计算差异
                diff_seconds = (ocr_time - csv_time).total_seconds()

                # 如果差异超过阈值，记录
                if abs(diff_seconds) > threshold_seconds:
                    corrections_for_video.append({
                        'index': idx,
                        'frame_path': str(frame_path),
                        'video_file': row['video_file'],
                        'csv_time': csv_time,
                        'ocr_time': ocr_time,
                        'diff_seconds': diff_seconds,
                        'second_in_video': row['second_in_video']
                    })

            # 显示该视频的矫正统计
            if len(corrections_for_video) > 0:
                print(f"\n  发现 {len(corrections_for_video)} 处需要矫正:")

                # 统计差异分布
                diff_counts = {}
                for corr in corrections_for_video:
                    diff = corr['diff_seconds']
                    if diff not in diff_counts:
                        diff_counts[diff] = 0
                    diff_counts[diff] += 1

                print(f"  差异分布:")
                for diff, count in sorted(diff_counts.items()):
                    print(f"    {diff:+.0f}秒: {count} 帧")

                # 显示前10个矫正
                print(f"\n  示例（前10个）:")
                for i, corr in enumerate(corrections_for_video[:10]):
                    print(f"    {i+1}. CSV时间: {corr['csv_time']}")
                    print(f"       OCR时间: {corr['ocr_time']}")
                    print(f"       差异: {corr['diff_seconds']:+.1f}秒")

                self.corrections.extend(corrections_for_video)
            else:
                print(f"  ✓ 所有帧时间一致")

        # 总结
        print("\n" + "=" * 80)
        print("OCR识别总结:")
        print("=" * 80)
        print(f"总帧数: {len(target_df)}")
        print(f"OCR成功: {len(target_df) - len(self.failed_ocr)}")
        print(f"OCR失败: {len(self.failed_ocr)}")
        print(f"需要矫正: {len(self.corrections)}")

        if len(self.failed_ocr) > 0:
            print(f"\nOCR失败的帧:")
            for fail in self.failed_ocr[:10]:
                print(f"  - {fail['frame_path']}")

        # 自动矫正
        if auto_correct and len(self.corrections) > 0:
            self.apply_corrections()
        elif len(self.corrections) > 0:
            print(f"\n发现 {len(self.corrections)} 处需要矫正")
            print("使用 --apply 参数应用矫正")

    def apply_corrections(self):
        """应用时间矫正"""
        print("\n" + "=" * 80)
        print("应用时间矫正")
        print("=" * 80)

        # 创建备份
        backup_file = self.output_dir / "aligned_data_backup.csv"
        self.df.to_csv(backup_file, index=False, encoding='utf-8-sig')
        print(f"✓ 已创建备份: {backup_file}")

        # 应用矫正
        for corr in tqdm(self.corrections, desc="应用矫正"):
            idx = corr['index']
            ocr_time = corr['ocr_time']
            diff = corr['diff_seconds']

            # 更新时间
            self.df.loc[idx, 'frame_time'] = ocr_time

            # 更新帧文件名（处理文件名冲突）
            old_path = Path(self.df.loc[idx, 'frame_path'])
            new_filename = f"{ocr_time.strftime('%Y%m%d_%H%M%S')}.jpg"
            new_path = old_path.parent / new_filename

            # 如果目标文件已存在，添加序号后缀避免冲突
            if new_path.exists() and new_path != old_path:
                # 添加毫秒级后缀
                suffix_counter = 1
                while True:
                    new_filename_with_suffix = f"{ocr_time.strftime('%Y%m%d_%H%M%S')}_{suffix_counter:03d}.jpg"
                    new_path_with_suffix = old_path.parent / new_filename_with_suffix
                    if not new_path_with_suffix.exists():
                        new_path = new_path_with_suffix
                        break
                    suffix_counter += 1

            # 重命名文件
            if old_path.exists():
                old_path.rename(new_path)
                self.df.loc[idx, 'frame_path'] = str(new_path)

        # 保存结果
        output_csv = self.output_dir / "aligned_data.csv"
        output_json = self.output_dir / "aligned_data.json"

        self.df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        self.df.to_json(output_json, orient='records', force_ascii=False, indent=2)

        print(f"\n✓ 已保存矫正后的数据:")
        print(f"  - {output_csv}")
        print(f"  - {output_json}")

        # 生成矫正报告
        self.generate_report()

        # 更新统计信息
        self.update_stats()

    def generate_report(self):
        """生成矫正报告"""
        report_file = self.output_dir / "ocr_correction_report.json"

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_corrections': len(self.corrections),
            'total_failed_ocr': len(self.failed_ocr),
            'corrections': self.corrections,
            'failed_ocr': self.failed_ocr
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"✓ 已生成矫正报告: {report_file}")

        # 也生成一个可读的文本报告
        text_report_file = self.output_dir / "ocr_correction_report.txt"

        with open(text_report_file, 'w', encoding='utf-8') as f:
            f.write("OCR时间矫正报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"时间: {datetime.now()}\n")
            f.write(f"总矫正数: {len(self.corrections)}\n")
            f.write(f"OCR失败数: {len(self.failed_ocr)}\n\n")

            if len(self.corrections) > 0:
                f.write("矫正详情:\n")
                f.write("-" * 80 + "\n")
                for i, corr in enumerate(self.corrections, 1):
                    f.write(f"{i}. {corr['frame_path']}\n")
                    f.write(f"   CSV时间: {corr['csv_time']}\n")
                    f.write(f"   OCR时间: {corr['ocr_time']}\n")
                    f.write(f"   差异: {corr['diff_seconds']:+.1f}秒\n\n")

            if len(self.failed_ocr) > 0:
                f.write("\nOCR失败:\n")
                f.write("-" * 80 + "\n")
                for fail in self.failed_ocr:
                    f.write(f"  - {fail['frame_path']}\n")

        print(f"✓ 已生成文本报告: {text_report_file}")

    def update_stats(self):
        """更新统计信息"""
        stats_file = self.output_dir / "alignment_stats.json"

        stats = {
            'summary': {
                'total_aligned_frames': len(self.df),
                'unique_videos': int(self.df['video_file'].nunique()),
                'time_range': {
                    'start': str(self.df['frame_time'].min()),
                    'end': str(self.df['frame_time'].max())
                },
                'ocr_corrections': len(self.corrections),
                'ocr_failures': len(self.failed_ocr)
            },
            'time_overlap': {
                'video_start': '2024-10-18 05:08:11',
                'video_end': '2024-10-18 19:41:31.779922',
                'trajectory_start': '2024-10-18 12:12:49',
                'trajectory_end': '2024-10-18 23:59:58',
                'overlap': True,
                'overlap_start': '2024-10-18 12:12:49',
                'overlap_end': '2024-10-18 19:41:31.779922'
            },
            'directories': {
                'output': str(self.output_dir),
                'frames': str(self.output_dir / 'aligned_frames')
            }
        }

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"✓ 已更新统计信息: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='OCR时间识别和矫正工具')
    parser.add_argument('--csv', default='data/aligned_output/aligned_data.csv',
                       help='输入CSV文件')
    parser.add_argument('--frames-dir', default='data/aligned_output/aligned_frames',
                       help='帧图像目录')
    parser.add_argument('--output-dir', default='data/aligned_output',
                       help='输出目录')
    parser.add_argument('--timestamp', default='2024-10-18 18:41:34',
                       help='起始时间戳')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='时间差异阈值（秒）')
    parser.add_argument('--apply', action='store_true',
                       help='自动应用矫正')
    parser.add_argument('--check-only', action='store_true',
                       help='仅检查不矫正')

    args = parser.parse_args()

    # 创建矫正器
    corrector = OCRTimeCorrector(
        csv_file=args.csv,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir
    )

    # 执行矫正
    auto_correct = args.apply and not args.check_only

    corrector.correct_times_after_timestamp(
        timestamp_str=args.timestamp,
        auto_correct=auto_correct,
        threshold_seconds=args.threshold
    )

    if args.check_only and len(corrector.corrections) > 0:
        print(f"\n发现 {len(corrector.corrections)} 处需要矫正")
        print("运行以下命令应用矫正:")
        print(f"  python3 {__file__} --apply")


if __name__ == '__main__':
    main()
