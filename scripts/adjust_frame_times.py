#!/usr/bin/env python3
"""
时间校正工具 - 检查和校正18:41:34之后的帧时间

用法:
1. 首先运行检查模式，查看问题帧
2. 然后手动指定需要调整的帧
"""

import pandas as pd
from datetime import timedelta
import argparse


def check_time_issues(csv_file):
    """检查时间问题"""
    df = pd.read_csv(csv_file)

    # 18:41:34之后的数据
    target_df = df[df['frame_time'] >= '2024-10-18 18:41:34'].copy()

    print("=" * 70)
    print("检查 18:41:34 之后的时间问题")
    print("=" * 70)

    # 按视频分组检查
    for video_file in target_df['video_file'].unique():
        video_df = target_df[target_df['video_file'] == video_file].copy()
        video_df = video_df.reset_index()

        print(f"\n视频: {video_file}")
        print(f"  帧数: {len(video_df)}")
        print(f"  时间范围: {video_df['frame_time'].min()} 到 {video_df['frame_time'].max()}")

        # 显示前10帧和后10帧
        print(f"\n  前10帧:")
        for idx in range(min(10, len(video_df))):
            row = video_df.iloc[idx]
            print(f"    {idx}. {row['frame_time']} (second: {row['second_in_video']})")

        print(f"\n  后10帧:")
        for idx in range(max(0, len(video_df) - 10), len(video_df)):
            row = video_df.iloc[idx]
            print(f"    {idx}. {row['frame_time']} (second: {row['second_in_video']})")


def show_sample_frames(csv_file, video_file=None, start_idx=None, count=10):
    """显示样本帧"""
    df = pd.read_csv(csv_file)
    target_df = df[df['frame_time'] >= '2024-10-18 18:41:34'].copy()

    if video_file:
        target_df = target_df[target_df['video_file'] == video_file]

    if start_idx is not None:
        target_df = target_df.iloc[start_idx:start_idx + count]
    else:
        target_df = target_df.head(count)

    print("\n样本帧:")
    print("=" * 70)
    for idx, row in target_df.iterrows():
        print(f"索引 {idx}:")
        print(f"  时间: {row['frame_time']}")
        print(f"  文件: {row['frame_path'].split('/')[-1]}")
        print(f"  视频: {row['video_file']}")
        print(f"  second_in_video: {row['second_in_video']}")
        print()


def adjust_frame_time(csv_file, output_file, adjustments):
    """
    调整帧时间

    Args:
        csv_file: 输入CSV文件
        output_file: 输出CSV文件
        adjustments: 调整列表，格式: [(video_file, start_second, end_second, offset_seconds), ...]
    """
    df = pd.read_csv(csv_file)

    for video_file, start_sec, end_sec, offset in adjustments:
        # 找到需要调整的帧
        mask = (
            (df['video_file'] == video_file) &
            (df['second_in_video'] >= start_sec) &
            (df['second_in_video'] <= end_sec)
        )

        # 调整时间
        df.loc[mask, 'frame_time'] = pd.to_datetime(df.loc[mask, 'frame_time']) + timedelta(seconds=offset)

        print(f"✓ 调整 {video_file} 的 {start_sec}-{end_sec} 秒，偏移 {offset}秒 ({len(df[mask])} 帧)")

    # 保存
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='帧时间校正工具')
    parser.add_argument('--csv', default='data/aligned_output/aligned_data.csv', help='CSV文件')
    parser.add_argument('--check', action='store_true', help='检查时间问题')
    parser.add_argument('--show-samples', action='store_true', help='显示样本帧')
    parser.add_argument('--video', help='指定视频文件')
    parser.add_argument('--adjust', action='store_true', help='执行时间调整')

    args = parser.parse_args()

    if args.check:
        check_time_issues(args.csv)
    elif args.show_samples:
        show_sample_frames(args.csv, args.video)
    elif args.adjust:
        print("请手动编辑此脚本，在 adjustments 列表中指定调整参数")
        print("\n示例:")
        print("adjustments = [")
        print("    ('20241018104130.mp4', 100, 200, 1),   # 第100-200秒，加1秒")
        print("    ('20241018111131.mp4', 0, 100, -2),    # 第0-100秒，减2秒")
        print("]")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
