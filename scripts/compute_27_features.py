#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算 27 维轨迹特征（仿照 GAN-BiLSTM 论文）

特征组成：
  - 2 维经纬度: lon, lat
  - 25 维运动特征: 5 个基础特征 × 5 个统计量

5 个基础特征：
  1. speed (速度)
  2. acceleration (加速度)
  3. angular_speed (角速度)
  4. angular_acceleration (角加速度)
  5. angle_diff (角度差)

5 个统计量：
  1. 原始值
  2. median(window=5)
  3. median(window=50)
  4. SD(window=5)
  5. SD(window=50)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 窗口大小（仿照 GAN-BiLSTM）
WINDOW_SMALL = 5
WINDOW_LARGE = 50


def compute_motion_features(df):
    """
    计算 5 个运动特征（对应论文公式 1-4）

    输入: df 包含 [经度, 纬度, 速度, 方向, 时间]
    输出: DataFrame 添加 4 列运动特征
    """
    n = len(df)

    # 初始化（适配 Agri-MBT 列名）
    speed = df['速度'].values if '速度' in df.columns else df['speed'].values
    direction = df['方向角'].values if '方向角' in df.columns else (df['方向'].values if '方向' in df.columns else df['direction'].values)

    time_diff = np.zeros(n)
    acceleration = np.zeros(n)
    angle_diff = np.zeros(n)
    angular_speed = np.zeros(n)
    angular_acceleration = np.zeros(n)

    # 解析时间（Agri-MBT 使用 frame_time）
    if 'frame_time' in df.columns:
        time_col = df['frame_time'].values
    else:
        time_col = df['时间'].values if '时间' in df.columns else df['time'].values
    timestamps = []
    for t in time_col:
        try:
            dt = datetime.strptime(str(t), '%Y/%m/%d %H:%M:%S')
        except:
            try:
                dt = datetime.strptime(str(t), '%Y-%m-%d %H:%M:%S')
            except:
                dt = datetime.strptime(str(t).split('.')[0], '%Y-%m-%d %H:%M:%S')
        timestamps.append(dt)

    # 计算时间差（公式 1, 3, 4 分母）
    for i in range(1, n):
        time_diff[i] = (timestamps[i] - timestamps[i-1]).total_seconds()

    # 公式 1: acceleration = Δspeed / Δtime
    for i in range(1, n):
        if time_diff[i] > 0:
            acceleration[i] = (speed[i] - speed[i-1]) / time_diff[i]

    # 公式 2: angle_diff = Δdirection
    for i in range(1, n):
        angle_diff[i] = direction[i] - direction[i-1]

    # 公式 3: angular_speed = angle_diff / Δtime
    for i in range(1, n):
        if time_diff[i] > 0:
            angular_speed[i] = angle_diff[i] / time_diff[i]

    # 公式 4: angular_acceleration = Δangular_speed / Δtime
    for i in range(2, n):
        if time_diff[i] > 0:
            angular_acceleration[i] = (angular_speed[i] - angular_speed[i-1]) / time_diff[i]

    df['acceleration'] = acceleration
    df['angular_speed'] = angular_speed
    df['angular_acceleration'] = angular_acceleration
    df['angle_diff'] = angle_diff

    # 保存速度列（统一命名为 speed）
    if '速度' in df.columns:
        df['speed'] = df['速度']

    return df


def compute_window_features(df):
    """
    计算时间窗口特征（对应论文公式 5-6）

    对 5 个运动特征分别计算:
      - median(window=5)
      - median(window=50)
      - SD(window=5)
      - SD(window=50)
    """
    motion_features = ['speed', 'acceleration', 'angular_speed', 'angular_acceleration', 'angle_diff']

    # 速度列名兼容
    speed_col = '速度' if '速度' in df.columns else 'speed'

    for feat in motion_features:
        # 处理列名
        if feat == 'speed':
            col = speed_col
        else:
            col = feat

        values = df[col].values
        n = len(values)

        # 初始化窗口统计量
        med_5 = np.zeros(n)
        med_50 = np.zeros(n)
        sd_5 = np.zeros(n)
        sd_50 = np.zeros(n)

        for i in range(n):
            # 小窗口 (5)
            if i < WINDOW_SMALL:
                window_small = values[:i+1]
            else:
                window_small = values[i-WINDOW_SMALL+1:i+1]

            # 大窗口 (50)
            if i < WINDOW_LARGE:
                window_large = values[:i+1]
            else:
                window_large = values[i-WINDOW_LARGE+1:i+1]

            # 公式 5: Median
            med_5[i] = np.median(window_small)
            med_50[i] = np.median(window_large)

            # 公式 6: SD (标准差)
            sd_5[i] = np.std(window_small) if len(window_small) > 1 else 0
            sd_50[i] = np.std(window_large) if len(window_large) > 1 else 0

        # 添加到 DataFrame
        df[f'{feat}_med_5'] = med_5
        df[f'{feat}_med_50'] = med_50
        df[f'{feat}_SD_5'] = sd_5
        df[f'{feat}_SD_50'] = sd_50

    return df


def main(input_csv, output_csv):
    """
    主函数：计算 27 维特征

    输入: aligned_data.csv (包含经度、纬度、速度、方向、时间、分类、frame_path)
    输出: aligned_data_27features.csv (27 维特征 + 标签 + frame_path)
    """
    print(f"读取原始数据: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"原始数据形状: {df.shape}")
    print(f"原始列名: {df.columns.tolist()}")

    # 保留必要列
    keep_cols = []
    for col in ['经度', '纬度', '速度', '方向角', '方向', 'frame_time', '时间', '分类', 'frame_path']:
        if col in df.columns:
            keep_cols.append(col)
        elif col == '经度' and 'longitude' in df.columns:
            keep_cols.append('longitude')
        elif col == '纬度' and 'latitude' in df.columns:
            keep_cols.append('latitude')
        elif col == '速度' and 'speed' in df.columns:
            keep_cols.append('speed')
        elif col == '方向角' and 'direction' in df.columns:
            keep_cols.append('direction')
        elif col == '时间' and 'time' in df.columns:
            keep_cols.append('time')
        elif col == '分类' and 'label' in df.columns:
            keep_cols.append('label')

    df = df[keep_cols].copy()
    print(f"保留列: {df.columns.tolist()}")

    # 统一列名
    col_mapping = {
        'longitude': '经度',
        'latitude': '纬度',
        'speed': '速度',
        'direction': '方向角',
        'time': 'frame_time',
        'label': '分类'
    }
    df = df.rename(columns=col_mapping)

    # 步骤 1: 计算运动特征
    print("\n步骤 1: 计算 5 个运动特征...")
    df = compute_motion_features(df)

    # 步骤 2: 计算时间窗口特征
    print("步骤 2: 计算时间窗口特征 (20 维)...")
    df = compute_window_features(df)

    # 构建最终 27 维特征
    print("\n构建 27 维特征向量...")
    feature_27 = [
        '经度', '纬度',  # 2 维

        # 5 个运动特征 × 5 个统计量 = 25 维
        '速度', 'speed_med_5', 'speed_med_50', 'speed_SD_5', 'speed_SD_50',
        'acceleration', 'acceleration_med_5', 'acceleration_med_50', 'acceleration_SD_5', 'acceleration_SD_50',
        'angular_speed', 'angular_speed_med_5', 'angular_speed_med_50', 'angular_speed_SD_5', 'angular_speed_SD_50',
        'angular_acceleration', 'angular_acceleration_med_5', 'angular_acceleration_med_50', 'angular_acceleration_SD_5', 'angular_acceleration_SD_50',
        'angle_diff', 'angle_diff_med_5', 'angle_diff_med_50', 'angle_diff_SD_5', 'angle_diff_SD_50'
    ]

    output_df = df[feature_27 + ['分类', 'frame_path']].copy()

    # 保存
    print(f"\n保存 27 维特征数据: {output_csv}")
    output_parent = os.path.dirname(output_csv)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    print(f"输出数据形状: {output_df.shape}")
    print(f"输出列名:\n{output_df.columns.tolist()}")

    # 统计信息
    print("\n特征统计:")
    print(output_df[feature_27].describe())

    return output_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算 27 维轨迹特征")
    parser.add_argument('--input', type=str,
                        default='/home/research/Agri-MBT/data/aligned_output/aligned_data.csv',
                        help='输入 CSV 文件路径')
    parser.add_argument('--output', type=str,
                        default='/home/research/Agri-MBT/data/aligned_output/aligned_data_27features.csv',
                        help='输出 CSV 文件路径')

    args = parser.parse_args()

    main(args.input, args.output)
