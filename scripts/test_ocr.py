#!/usr/bin/env python3
"""
测试OCR在已提取帧上的识别效果
"""

import cv2
import pytesseract
from PIL import Image
import re
from datetime import datetime

def test_ocr_on_image(image_path):
    """测试OCR在指定图像上的识别"""
    print(f"\n测试图像: {image_path}")
    print("-" * 60)

    # 读取图像
    frame = cv2.imread(image_path)
    if frame is None:
        print("无法读取图像")
        return

    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 提取左上角区域
    height, width = gray.shape
    print(f"图像尺寸: {width}x{height}")

    # 尝试不同的区域大小
    regions = [
        ("小区域 (20%x5%)", gray[0:int(height * 0.05), 0:int(width * 0.2)]),
        ("中区域 (40%x8%)", gray[0:int(height * 0.08), 0:int(width * 0.4)]),
        ("大区域 (50%x10%)", gray[0:int(height * 0.10), 0:int(width * 0.5)]),
    ]

    for region_name, timestamp_region in regions:
        print(f"\n{region_name}:")

        # 二值化处理
        _, binary = cv2.threshold(timestamp_region, 150, 255, cv2.THRESH_BINARY)

        # 保存处理后的区域以便查看
        debug_path = f"test_frames/debug_{region_name.replace(' ', '_').replace('(', '').replace(')', '')}.jpg"
        cv2.imwrite(debug_path, binary)
        print(f"  调试图像已保存: {debug_path}")

        # OCR识别
        pil_image = Image.fromarray(binary)

        # 尝试不同的配置
        configs = [
            ('默认', ''),
            ('数字模式', r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789-: '),
            ('单行文本', r'--oem 3 --psm 7'),
            ('稀疏文本', r'--oem 3 --psm 11'),
        ]

        for config_name, config in configs:
            try:
                text = pytesseract.image_to_string(pil_image, config=config)
                text = text.strip()
                if text:
                    print(f"  {config_name}: '{text}'")

                    # 尝试解析时间
                    pattern = r'(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})'
                    match = re.search(pattern, text)
                    if match:
                        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                        hour, minute, second = int(match.group(4)), int(match.group(5)), int(match.group(6))
                        dt = datetime(year, month, day, hour, minute, second)
                        print(f"    ✓ 解析成功: {dt}")
            except Exception as e:
                print(f"  {config_name}: 错误 - {e}")

if __name__ == '__main__':
    # 测试第一帧和最后一帧
    test_ocr_on_image("test_frames/20241018043810_first.jpg")
    test_ocr_on_image("test_frames/20241018043810_last.jpg")
