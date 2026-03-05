#!/usr/bin/env python3
"""
调试脚本：检查为什么没有匹配到轨迹数据
"""

import pandas as pd
import numpy as np
from datetime import datetime

# 加载轨迹数据
traj_df = pd.read_excel('data/trajectory/B-2024-10-18/12-12-49_23-59-58.xlsx')
traj_df['定位时间'] = pd.to_datetime(traj_df['定位时间'])
traj_df['时间戳'] = traj_df['定位时间'].astype(np.int64) // 10**9

print(f"轨迹数据总数: {len(traj_df)}")
print(f"时间范围: {traj_df['定位时间'].min()} 到 {traj_df['定位时间'].max()}")

# 定义视频时间（从OCR识别的结果）
video_times = [
    ("视频1", datetime(2024, 10, 18, 12, 38, 12), datetime(2024, 10, 18, 13, 8, 10)),
    ("视频14", datetime(2024, 10, 18, 18, 11, 31), datetime(2024, 10, 18, 18, 41, 29)),
    ("视频15", datetime(2024, 10, 18, 18, 41, 33), datetime(2024, 10, 18, 19, 11, 30)),
    ("视频16", datetime(2024, 10, 18, 19, 11, 34), datetime(2024, 10, 18, 19, 41, 31)),
]

print("\n" + "=" * 80)
print("检查每个视频时间范围内的轨迹数据")
print("=" * 80)

for video_name, start_time, end_time in video_times:
    print(f"\n{video_name}: {start_time} 到 {end_time}")

    # 查找时间范围内的轨迹数据
    mask = (traj_df['定位时间'] >= start_time) & (traj_df['定位时间'] <= end_time)
    matching_rows = traj_df[mask]

    print(f"  匹配的轨迹记录数: {len(matching_rows)}")

    if len(matching_rows) > 0:
        print(f"  第一条: {matching_rows.iloc[0]['定位时间']}")
        print(f"  最后一条: {matching_rows.iloc[-1]['定位时间']}")

        # 检查时间间隔
        time_diffs = matching_rows['定位时间'].diff().dt.total_seconds()
        print(f"  平均时间间隔: {time_diffs.mean():.2f}秒")
        print(f"  最小间隔: {time_diffs.min():.2f}秒")
        print(f"  最大间隔: {time_diffs.max():.2f}秒")
    else:
        # 检查是否有接近的数据
        print(f"  没有精确匹配，检查附近的数据...")

        # 查找最近的轨迹数据
        time_diffs = abs((traj_df['定位时间'] - start_time).dt.total_seconds())
        min_idx = time_diffs.idxmin()
        closest_time = traj_df.loc[min_idx, '定位时间']
        print(f"  最接近的时间: {closest_time} (相差: {time_diffs[min_idx]:.0f}秒)")

    # 检查时间戳匹配
    start_ts = int(start_time.timestamp())
    end_ts = int(end_time.timestamp())

    print(f"\n  开始时间戳: {start_ts}")
    print(f"  结束时间戳: {end_ts}")

    # 检查是否有精确匹配
    exact_matches = traj_df[traj_df['时间戳'] == start_ts]
    print(f"  精确匹配数: {len(exact_matches)}")

    # 检查时间范围内的数据
    time_range_data = traj_df[(traj_df['时间戳'] >= start_ts) & (traj_df['时间戳'] <= end_ts)]
    print(f"  时间范围内数据: {len(time_range_data)}")

    if len(time_range_data) > 0:
        print(f"  范围内第一条: {time_range_data.iloc[0]['定位时间']}")
        print(f"  范围内最后一条: {time_range_data.iloc[-1]['定位时间']}")

        # 检查时间戳分布
        print(f"\n  时间戳示例:")
        for idx in range(min(5, len(time_range_data))):
            row = time_range_data.iloc[idx]
            print(f"    {row['定位时间']} -> {row['时间戳']}")
