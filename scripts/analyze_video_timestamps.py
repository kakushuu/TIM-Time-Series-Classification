#!/usr/bin/env python3
"""
测试OCR时间戳识别并检查所有视频的时间范围
"""

import cv2
import sys
from pathlib import Path
import pytesseract
from PIL import Image
import re
from datetime import datetime
import numpy as np

def extract_timestamp(frame):
    """从帧中提取时间戳"""
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 提取左上角区域
    height, width = gray.shape
    timestamp_region = gray[0:int(height * 0.08), 0:int(width * 0.4)]

    # 二值化处理
    _, binary = cv2.threshold(timestamp_region, 150, 255, cv2.THRESH_BINARY)

    # OCR识别
    pil_image = Image.fromarray(binary)
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789-: '
    text = pytesseract.image_to_string(pil_image, config=custom_config)

    # 解析时间戳
    pattern = r'(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})'
    match = re.search(pattern, text.strip())

    if match:
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        hour, minute, second = int(match.group(4)), int(match.group(5)), int(match.group(6))
        return datetime(year, month, day, hour, minute, second), text.strip()

    return None, text.strip()

def analyze_all_videos(video_dir):
    """分析所有视频的时间范围"""
    video_dir = Path(video_dir)
    video_files = sorted(video_dir.glob("*.mp4"))

    print(f"找到 {len(video_files)} 个视频文件\n")
    print("=" * 80)

    results = []

    for i, video_path in enumerate(video_files, 1):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"{i}. {video_path.name}: 无法打开")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        # 提取第一帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = cap.read()

        # 提取最后一帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
        ret, last_frame = cap.read()

        cap.release()

        # OCR识别
        start_time, ocr_text_start = extract_timestamp(first_frame) if first_frame is not None else (None, "")
        end_time, ocr_text_end = extract_timestamp(last_frame) if last_frame is not None else (None, "")

        # 如果OCR失败，尝试从文件名解析
        if start_time is None:
            filename_pattern = r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})'
            match = re.search(filename_pattern, video_path.stem)
            if match:
                year, month, day, hour, minute, second = map(int, match.groups())
                start_time = datetime(year, month, day, hour, minute, second)
                end_time = start_time + __import__('datetime').timedelta(seconds=duration)

        result = {
            'index': i,
            'filename': video_path.name,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'ocr_start': ocr_text_start,
            'ocr_end': ocr_text_end
        }
        results.append(result)

        print(f"{i}. {video_path.name}")
        print(f"   时长: {duration:.1f}秒 ({duration/60:.1f}分钟)")
        print(f"   OCR识别 - 开始: '{ocr_text_start}' -> {start_time}")
        print(f"           结束: '{ocr_text_end}' -> {end_time}")
        print()

    return results

if __name__ == '__main__':
    video_dir = "data/video/B-2024-10-18"
    results = analyze_all_videos(video_dir)

    print("=" * 80)
    print("\n视频时间范围汇总：")
    print(f"最早开始时间: {min(r['start_time'] for r in results if r['start_time'])}")
    print(f"最晚结束时间: {max(r['end_time'] for r in results if r['end_time'])}")
