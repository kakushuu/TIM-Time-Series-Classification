#!/usr/bin/env python3
"""
测试时区问题
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone

# 加载轨迹数据
traj_df = pd.read_excel('data/trajectory/B-2024-10-18/12-12-49_23-59-58.xlsx')
traj_df['定位时间'] = pd.to_datetime(traj_df['定位时间'])

print("检查时间戳转换问题:\n")

# 检查几个样本
sample_times = [
    datetime(2024, 10, 18, 12, 38, 12),
    datetime(2024, 10, 18, 18, 11, 31),
]

for dt in sample_times:
    print(f"Datetime: {dt}")

    # 方法1: 直接timestamp()（本地时间）
    ts1 = int(dt.timestamp())
    print(f"  方法1 (timestamp()): {ts1}")

    # 方法2: pandas方式
    ts2 = int(pd.Timestamp(dt).timestamp())
    print(f"  方法2 (pd.Timestamp): {ts2}")

    # 方法3: 从DataFrame计算
    ts3 = int(pd.Timestamp(dt).value // 10**9)
    print(f"  方法3 (value//10^9): {ts3}")

    # 在DataFrame中查找
    matching = traj_df[traj_df['定位时间'] == dt]
    print(f"  在DataFrame中找到: {len(matching)} 条")

    if len(matching) > 0:
        # 计算DataFrame中的时间戳
        df_ts = int(matching.iloc[0]['定位时间'].timestamp())
        print(f"  DataFrame时间戳: {df_ts}")
        print(f"  时间戳差异: {ts1 - df_ts}")

    print()

# 检查DataFrame中的时间戳生成方式
print("\n检查DataFrame时间戳生成:")
traj_df['时间戳_方法1'] = traj_df['定位时间'].astype(np.int64) // 10**9
traj_df['时间戳_方法2'] = traj_df['定位时间'].apply(lambda x: int(x.timestamp()))

sample_row = traj_df[traj_df['定位时间'] == datetime(2024, 10, 18, 12, 38, 12)]
if len(sample_row) > 0:
    print(f"定位时间: {sample_row.iloc[0]['定位时间']}")
    print(f"方法1 (astype): {sample_row.iloc[0]['时间戳_方法1']}")
    print(f"方法2 (timestamp): {sample_row.iloc[0]['时间戳_方法2']}")

    # 用方法2的时间戳查找
    ts = int(sample_row.iloc[0]['定位时间'].timestamp())
    found = traj_df[traj_df['时间戳_方法2'] == ts]
    print(f"使用方法2时间戳查找: 找到 {len(found)} 条")
