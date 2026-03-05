#!/usr/bin/env python3
"""
验证脚本：对比文件名时间和OCR识别的实际视频时间
"""

import cv2
import pytesseract
from PIL import Image
import re
from datetime import datetime
from pathlib import Path

def extract_timestamp_ocr(frame):
    """使用OCR从帧中提取时间戳"""
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

            # OCR识别 - 尝试不同配置
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

                    # 尝试解析时间 - 支持多种格式
                    patterns = [
                        (r'(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})', 'YYYY-MM-DD HH:MM:SS'),
                        (r'(\d{4})/(\d{2})/(\d{2})\s+(\d{2}):(\d{2}):(\d{2})', 'YYYY/MM/DD HH:MM:SS'),
                        (r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})', 'YYYYMMDDHHMMSS'),
                    ]

                    for pattern, fmt in patterns:
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

def verify_video_time(video_path, expected_start=None, expected_end=None):
    """验证视频时间范围"""
    video_path = Path(video_path)

    print(f"\n{'='*80}")
    print(f"验证视频: {video_path.name}")
    print(f"{'='*80}\n")

    # 从文件名解析时间
    filename_pattern = r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})'
    match = re.search(filename_pattern, video_path.stem)

    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        filename_time = datetime(year, month, day, hour, minute, second)
        print(f"📁 文件名时间: {filename_time}")
    else:
        print(f"⚠️  无法从文件名解析时间")
        filename_time = None

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 无法打开视频")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    print(f"\n📊 视频信息:")
    print(f"   总帧数: {total_frames}")
    print(f"   FPS: {fps:.2f}")
    print(f"   时长: {duration:.1f}秒 ({duration/60:.1f}分钟)")

    # 提取第一帧
    print(f"\n🖼️  提取第一帧...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()

    if ret:
        first_time, first_ocr_text = extract_timestamp_ocr(first_frame)
        print(f"   OCR识别文本: '{first_ocr_text}'")
        print(f"   ✅ 第一帧时间: {first_time if first_time else '识别失败'}")

        # 保存第一帧用于调试
        debug_path = Path("test_frames") / f"{video_path.stem}_verify_first.jpg"
        cv2.imwrite(str(debug_path), first_frame)
        print(f"   已保存: {debug_path}")

    # 提取最后一帧
    print(f"\n🖼️  提取最后一帧...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
    ret, last_frame = cap.read()

    if ret:
        last_time, last_ocr_text = extract_timestamp_ocr(last_frame)
        print(f"   OCR识别文本: '{last_ocr_text}'")
        print(f"   ✅ 最后一帧时间: {last_time if last_time else '识别失败'}")

        # 保存最后一帧用于调试
        debug_path = Path("test_frames") / f"{video_path.stem}_verify_last.jpg"
        cv2.imwrite(str(debug_path), last_frame)
        print(f"   已保存: {debug_path}")

    cap.release()

    # 对比分析
    print(f"\n{'='*80}")
    print(f"📊 时间对比分析")
    print(f"{'='*80}\n")

    if first_time:
        print(f"✅ OCR识别的开始时间: {first_time}")
    else:
        print(f"❌ OCR未能识别开始时间")

    if last_time:
        print(f"✅ OCR识别的结束时间: {last_time}")
    else:
        print(f"❌ OCR未能识别结束时间")

    if expected_start:
        print(f"\n🎯 预期开始时间: {expected_start}")
        if first_time:
            diff = (first_time - expected_start).total_seconds()
            if abs(diff) < 5:
                print(f"   ✅ 时间匹配！（误差: {diff:.1f}秒）")
            else:
                print(f"   ⚠️  时间不匹配（差异: {diff:.1f}秒）")

    if expected_end:
        print(f"\n🎯 预期结束时间: {expected_end}")
        if last_time:
            diff = (last_time - expected_end).total_seconds()
            if abs(diff) < 5:
                print(f"   ✅ 时间匹配！（误差: {diff:.1f}秒）")
            else:
                print(f"   ⚠️  时间不匹配（差异: {diff:.1f}秒）")

    # 文件名时间 vs 实际时间
    if filename_time and first_time:
        time_diff = (first_time - filename_time).total_seconds()
        print(f"\n📝 文件名时间 vs 实际时间:")
        print(f"   文件名: {filename_time}")
        print(f"   实际:   {first_time}")
        print(f"   差异:   {time_diff:.0f}秒 ({time_diff/3600:.1f}小时)")

        if abs(time_diff) > 300:  # 超过5分钟差异
            print(f"\n   ⚠️  警告：文件名时间与实际视频时间差异较大！")
            print(f"   建议使用OCR识别的实际时间戳")

    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    video_path = "data/video/B-2024-10-18/20241018043810.mp4"

    # 预期时间（用户提供的）
    expected_start = datetime(2024, 10, 18, 12, 38, 13)
    expected_end = datetime(2024, 10, 18, 13, 8, 10)

    verify_video_time(video_path, expected_start, expected_end)
