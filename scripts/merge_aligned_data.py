#!/usr/bin/env python3
"""
合并新对齐数据到主数据集（自动备份）。

用法：
  python scripts/merge_aligned_data.py \
      --new  data/aligned_output/B-2024-10-19/aligned_data.csv \
      --master data/aligned_output/aligned_data.csv
"""
import argparse
import shutil
from datetime import datetime
from pathlib import Path
import pandas as pd


def merge(new_path: Path, master_path: Path) -> None:
    # 备份
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = master_path.with_name(f'aligned_data_backup_{ts}.csv')
    shutil.copy2(master_path, backup)
    print(f'备份: {backup}')

    master = pd.read_csv(master_path)
    new = pd.read_csv(new_path)
    print(f'主数据集: {len(master):,} 行  |  新数据: {len(new):,} 行')

    # 统一 ocr_status 列（主数据集可能没有此列）
    if 'ocr_status' not in master.columns and 'ocr_status' in new.columns:
        new = new.drop(columns=['ocr_status'])

    merged = pd.concat([master, new], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=['frame_time'])
    merged = merged.sort_values('frame_time').reset_index(drop=True)
    after = len(merged)

    merged.to_csv(master_path, index=False, encoding='utf-8-sig')
    print(f'合并后: {after:,} 行  (去重 {before - after} 条)')
    print(f'已保存: {master_path}')

    # 分类分布
    if '分类' in merged.columns:
        print('\n分类分布（合并后）:')
        print(merged['分类'].value_counts().sort_index().to_string())


def main():
    parser = argparse.ArgumentParser(description='合并新对齐数据到主数据集')
    parser.add_argument('--new', required=True, help='新生成的 aligned_data.csv 路径')
    parser.add_argument('--master', default='data/aligned_output/aligned_data.csv',
                        help='主数据集路径（默认: data/aligned_output/aligned_data.csv）')
    args = parser.parse_args()

    new_path = Path(args.new)
    master_path = Path(args.master)

    if not new_path.exists():
        print(f'[ERROR] 新数据文件不存在: {new_path}')
        return
    if not master_path.exists():
        print(f'[ERROR] 主数据集不存在: {master_path}')
        return

    merge(new_path, master_path)


if __name__ == '__main__':
    main()
