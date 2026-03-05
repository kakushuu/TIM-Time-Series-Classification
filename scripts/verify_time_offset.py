#!/usr/bin/env python3
"""
验证1秒时间偏移是否正确应用
"""

import pandas as pd
from datetime import datetime

# 加载对齐数据
df = pd.read_csv('data/aligned_output/aligned_data.csv')

print("=" * 80)
print("验证1秒时间偏移")
print("=" * 80)

print(f"\n总帧数: {len(df)}")
print(f"时间范围: {df['frame_time'].min()} 到 {df['frame_time'].max()}")

# 检查每个视频的第一帧
print("\n" + "=" * 80)
print("每个视频的第一帧时间（应该比开始时间+1秒）")
print("=" * 80)

for video in df['video_file'].unique():
    video_df = df[df['video_file'] == video]
    first_frame = video_df.iloc[0]

    print(f"\n{video}:")
    print(f"  第一帧时间: {first_frame['frame_time']}")
    print(f"  帧序号: {first_frame['frame_number']}")
    print(f"  秒数: {first_frame['second_in_video']}")

    # 根据second_in_video=0和+1秒偏移，第一帧应该是start_time + 1秒
    # 例如：视频12:38:12开始，第一帧应该是12:38:13

    # 从文件名推断的原始开始时间（无偏移）
    import re
    match = re.search(r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})', video)
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        filename_time = datetime(year, month, day, hour, minute, second)
        print(f"  文件名时间: {filename_time}")
        print(f"  第一帧应该是: {filename_time} (如果OCR正确) 或 {filename_time.replace(hour=hour+8)} (如果8小时偏移)")

# 检查时间连续性
print("\n" + "=" * 80)
print("时间连续性检查")
print("=" * 80)

for video in df['video_file'].unique():
    video_df = df[df['video_file'] == video].sort_values('second_in_video')

    # 检查时间差
    time_diffs = video_df['frame_time'].diff()
    print(f"\n{video}:")
    print(f"  帧数: {len(video_df)}")
    print(f"  平均时间差: {time_diffs.mean()}")
    print(f"  最小时间差: {time_diffs.min()}")
    print(f"  最大时间差: {time_diffs.max()}")

    # 检查是否有缺失的秒
    expected_seconds = set(range(int(video_df['second_in_video'].min()),
                                  int(video_df['second_in_video'].max()) + 1))
    actual_seconds = set(video_df['second_in_video'].astype(int))
    missing_seconds = expected_seconds - actual_seconds

    if missing_seconds:
        print(f"  ⚠️  缺失的秒数: {sorted(missing_seconds)}")
    else:
        print(f"  ✅ 无缺失秒数")

print("\n" + "=" * 80)
print("验证完成")
print("=" * 80)
