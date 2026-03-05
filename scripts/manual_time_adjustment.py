#!/usr/bin/env python3
"""
手动时间校正工具 - 查看帧图像并指定时间调整

用法:
1. 运行查看模式，显示帧图像
2. 根据实际视频内容确定需要调整的秒数
3. 创建调整配置文件
4. 运行应用调整
"""

import pandas as pd
from datetime import timedelta
from pathlib import Path
import json
import argparse


def create_adjustment_template():
    """创建调整配置模板"""
    template = {
        "description": "指定需要调整时间偏移的帧范围",
        "adjustments": [
            {
                "video_file": "20241018104130.mp4",
                "start_second": 0,
                "end_second": 100,
                "offset_seconds": 1,
                "reason": "前100秒需要加1秒"
            },
            {
                "video_file": "20241018111131.mp4",
                "start_second": 0,
                "end_second": 100,
                "offset_seconds": -2,
                "reason": "前100秒需要减2秒"
            }
        ]
    }

    with open('time_adjustments_template.json', 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    print("✓ 已创建调整配置模板: time_adjustments_template.json")
    print("\n请编辑此文件，指定需要调整的帧范围和偏移量")


def apply_adjustments(csv_file, adjustment_file, output_file):
    """应用时间调整"""
    # 加载CSV
    df = pd.read_csv(csv_file)

    # 加载调整配置
    with open(adjustment_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    adjustments = config['adjustments']

    print(f"加载了 {len(adjustments)} 个调整规则")
    print("=" * 70)

    total_adjusted = 0

    for adj in adjustments:
        video_file = adj['video_file']
        start_sec = adj['start_second']
        end_sec = adj['end_second']
        offset = adj['offset_seconds']
        reason = adj.get('reason', '')

        print(f"\n调整: {video_file}")
        print(f"  范围: 第{start_sec}-{end_sec}秒")
        print(f"  偏移: {offset:+d}秒")
        print(f"  原因: {reason}")

        # 找到需要调整的帧
        mask = (
            (df['video_file'] == video_file) &
            (df['second_in_video'] >= start_sec) &
            (df['second_in_video'] <= end_sec)
        )

        frames_to_adjust = df[mask]
        print(f"  受影响帧数: {len(frames_to_adjust)}")

        if len(frames_to_adjust) > 0:
            # 调整时间
            df.loc[mask, 'frame_time'] = pd.to_datetime(
                df.loc[mask, 'frame_time']) + timedelta(seconds=offset)

            total_adjusted += len(frames_to_adjust)
            print(f"  ✓ 已调整")

    # 保存结果
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    df.to_json(output_file.replace('.csv', '.json'),
               orient='records', force_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"✓ 总共调整了 {total_adjusted} 帧")
    print(f"✓ 已保存到: {output_file}")
    print(f"✓ JSON已保存到: {output_file.replace('.csv', '.json')}")


def show_frames_for_review(csv_file, video_file=None, start_second=None, count=10):
    """显示需要检查的帧"""
    df = pd.read_csv(csv_file)

    # 18:41:34 之后的帧
    target_df = df[df['frame_time'] >= '2024-10-18 18:41:34'].copy()

    if video_file:
        target_df = target_df[target_df['video_file'] == video_file]

    if start_second is not None:
        target_df = target_df[target_df['second_in_video'] >= start_second]
        target_df = target_df.head(count)

    print(f"\n显示 {len(target_df)} 个帧供检查:")
    print("=" * 70)

    for idx, row in target_df.iterrows():
        print(f"\n索引 {idx}:")
        print(f"  帧文件: {row['frame_path']}")
        print(f"  CSV时间: {row['frame_time']}")
        print(f"  视频文件: {row['video_file']}")
        print(f"  second_in_video: {row['second_in_video']}")
        print(f"\n  请查看此帧图像，确认实际时间戳与CSV时间的差异")


def interactive_adjustment(csv_file, output_file):
    """交互式调整工具"""
    df = pd.read_csv(csv_file)

    print("\n交互式时间调整工具")
    print("=" * 70)
    print("此工具将引导您逐个检查和调整帧时间")
    print("\n请先手动检查帧图像，然后告诉我要调整哪些帧\n")

    adjustments = []

    while True:
        print("\n" + "-" * 70)
        video_file = input("输入视频文件名 (或输入 'q' 退出): ").strip()

        if video_file.lower() == 'q':
            break

        if video_file not in df['video_file'].values:
            print(f"⚠️  视频文件不存在: {video_file}")
            continue

        try:
            start_sec = int(input("输入开始秒数 (second_in_video): "))
            end_sec = int(input("输入结束秒数: "))
            offset = int(input("输入时间偏移 (秒，正数=加，负数=减): "))
            reason = input("输入调整原因 (可选): ").strip()
        except ValueError:
            print("⚠️  输入无效，请重试")
            continue

        adjustments.append({
            'video_file': video_file,
            'start_second': start_sec,
            'end_second': end_sec,
            'offset_seconds': offset,
            'reason': reason
        })

        print(f"✓ 已添加调整规则")

    if len(adjustments) > 0:
        # 应用调整
        config = {'adjustments': adjustments}

        with open('time_adjustments.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 已保存调整配置到: time_adjustments.json")

        # 应用
        apply_adjustments(csv_file, 'time_adjustments.json', output_file)
    else:
        print("\n未进行任何调整")


def main():
    parser = argparse.ArgumentParser(description='帧时间手动校正工具')
    parser.add_argument('--csv', default='data/aligned_output/aligned_data.csv',
                       help='输入CSV文件')
    parser.add_argument('--output', default='data/aligned_output/aligned_data_adjusted.csv',
                       help='输出CSV文件')
    parser.add_argument('--create-template', action='store_true',
                       help='创建调整配置模板')
    parser.add_argument('--apply', metavar='ADJUSTMENT_FILE',
                       help='应用调整配置文件')
    parser.add_argument('--show-frames', action='store_true',
                       help='显示需要检查的帧')
    parser.add_argument('--video', help='指定视频文件')
    parser.add_argument('--start-second', type=int, help='起始秒数')
    parser.add_argument('--interactive', action='store_true',
                       help='交互式调整模式')

    args = parser.parse_args()

    if args.create_template:
        create_adjustment_template()
    elif args.apply:
        apply_adjustments(args.csv, args.apply, args.output)
    elif args.show_frames:
        show_frames_for_review(args.csv, args.video, args.start_second)
    elif args.interactive:
        interactive_adjustment(args.csv, args.output)
    else:
        parser.print_help()
        print("\n使用示例:")
        print("1. 创建调整模板:")
        print("   python3 scripts/manual_time_adjustment.py --create-template")
        print("\n2. 显示需要检查的帧:")
        print("   python3 scripts/manual_time_adjustment.py --show-frames --video 20241018104130.mp4")
        print("\n3. 应用调整:")
        print("   python3 scripts/manual_time_adjustment.py --apply time_adjustments.json")
        print("\n4. 交互式调整:")
        print("   python3 scripts/manual_time_adjustment.py --interactive")


if __name__ == '__main__':
    main()
