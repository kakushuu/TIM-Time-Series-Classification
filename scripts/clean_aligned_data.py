#!/usr/bin/env python3
"""
清理 aligned_data.csv 中无意义的列

移除以下列：
- 上点时间
- 补点
- unitid
- GPS标记
- 播种播肥 / 油耗 / 压力
- 抛肥量(立方)
- 定位间隔
"""

import pandas as pd
from pathlib import Path

# 定义要删除的列
COLUMNS_TO_REMOVE = [
    '上点时间',
    '补点',
    'unitid',
    'GPS标记',
    '播种播肥 / 油耗 / 压力',
    '抛肥量(立方)',
    '定位间隔'
]

def clean_aligned_data(input_path: str, output_path: str = None):
    """
    清理对齐数据，删除无意义的列

    Args:
        input_path: 输入CSV文件路径
        output_path: 输出CSV文件路径（如果为None，覆盖原文件）
    """
    input_file = Path(input_path)

    if not input_file.exists():
        print(f"❌ 文件不存在: {input_path}")
        return

    # 读取数据
    print(f"📖 读取文件: {input_path}")
    df = pd.read_csv(input_path, encoding='utf-8-sig')

    print(f"\n原始数据:")
    print(f"  - 总行数: {len(df)}")
    print(f"  - 总列数: {len(df.columns)}")
    print(f"\n原始列名:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

    # 检查哪些列存在
    existing_cols_to_remove = [col for col in COLUMNS_TO_REMOVE if col in df.columns]
    missing_cols = [col for col in COLUMNS_TO_REMOVE if col not in df.columns]

    if missing_cols:
        print(f"\n⚠️  以下列不存在（可能已被删除）:")
        for col in missing_cols:
            print(f"  - {col}")

    if not existing_cols_to_remove:
        print("\n✓ 所有需要删除的列都已经不存在了，无需处理")
        return

    print(f"\n将要删除的列:")
    for col in existing_cols_to_remove:
        print(f"  - {col}")

    # 删除列
    df_cleaned = df.drop(columns=existing_cols_to_remove)

    print(f"\n清理后数据:")
    print(f"  - 总行数: {len(df_cleaned)}")
    print(f"  - 总列数: {len(df_cleaned.columns)}")
    print(f"\n保留的列名:")
    for i, col in enumerate(df_cleaned.columns, 1):
        print(f"  {i}. {col}")

    # 确定输出路径
    if output_path is None:
        output_path = input_path

    # 保存
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df_cleaned.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 清理后的数据已保存到: {output_path}")

    # 统计
    print(f"\n📊 统计信息:")
    print(f"  - 删除列数: {len(existing_cols_to_remove)}")
    print(f"  - 保留列数: {len(df_cleaned.columns)}")
    print(f"  - 数据行数: {len(df_cleaned)}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='清理对齐数据中的无意义列')
    parser.add_argument('--input', '-i',
                       default='data/aligned_output/aligned_data.csv',
                       help='输入CSV文件路径')
    parser.add_argument('--output', '-o',
                       help='输出CSV文件路径（默认覆盖原文件）')

    args = parser.parse_args()

    clean_aligned_data(args.input, args.output)
