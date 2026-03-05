#!/usr/bin/env python3
"""
测试脚本：提取视频的第一帧和最后一帧，用于调试时间戳识别
"""

import cv2
import sys
from pathlib import Path

def extract_first_last_frames(video_path: str, output_dir: str = "test_frames"):
    """提取视频的第一帧和最后一帧"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"视频: {video_path}")
    print(f"总帧数: {total_frames}, FPS: {fps:.2f}")

    # 提取第一帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if ret:
        first_path = output_path / f"{Path(video_path).stem}_first.jpg"
        cv2.imwrite(str(first_path), first_frame)
        print(f"第一帧已保存: {first_path}")

    # 提取最后一帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
    ret, last_frame = cap.read()
    if ret:
        last_path = output_path / f"{Path(video_path).stem}_last.jpg"
        cv2.imwrite(str(last_path), last_frame)
        print(f"最后一帧已保存: {last_path}")

    cap.release()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # 使用第一个视频文件
        video_path = "data/video/B-2024-10-18/20241018043810.mp4"

    extract_first_last_frames(video_path)
