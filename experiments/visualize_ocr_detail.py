#!/usr/bin/env python3
"""
OCR Timestamp Extraction Detailed Pipeline
Shows each step from raw frame to OCR text extraction
"""
import matplotlib
matplotlib.use('Agg')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import pytesseract

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_ocr_detail_pipeline():
    """Create detailed OCR extraction pipeline visualization"""

    # Load sample frame
    sample_frame_path = '/home/research/Agri-MBT/data/aligned_output/aligned_frames/20241018_123813.jpg'

    if not Path(sample_frame_path).exists():
        frames_dir = Path('/home/research/Agri-MBT/data/aligned_output/aligned_frames')
        if frames_dir.exists():
            sample_frames = list(frames_dir.glob('*.jpg'))
            if sample_frames:
                sample_frame_path = str(sample_frames[0])

    # Read frame
    rgb_frame = cv2.imread(sample_frame_path)
    rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

    # Step 1: Original frame
    h, w = rgb_frame.shape[:2]

    # Step 2: Convert to grayscale
    gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)

    # Step 3: Extract ROI (upper-left 5% height, 30% width)
    roi_h = int(h * 0.05)
    roi_w = int(w * 0.30)
    timestamp_roi = gray_frame[0:roi_h, 0:roi_w]

    # Step 4: Compare different binarization thresholds
    thresholds = [150, 180, 200, 220]
    binary_results = []

    for thresh in thresholds:
        _, binary = cv2.threshold(timestamp_roi, thresh, 255, cv2.THRESH_BINARY)
        binary_results.append((thresh, binary))

    # Step 5: Best binarization result (thresh=200)
    _, binary_best = cv2.threshold(timestamp_roi, 200, 255, cv2.THRESH_BINARY)

    # Step 6: OCR extraction
    ocr_config = '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:- '
    ocr_text = pytesseract.image_to_string(binary_best, config=ocr_config).strip()

    # ============ Create detailed pipeline figure ============
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

    # Title
    ax = fig.add_subplot(gs[0, :])
    ax.axis('off')
    ax.text(0.5, 0.5, 'OCR Timestamp Extraction Pipeline',
            ha='center', va='center', fontsize=26, fontweight='bold',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow',
                     edgecolor='orange', linewidth=3))

    # Row 1: Step 1-2
    # Step 1: Original RGB Frame
    ax = fig.add_subplot(gs[1, 0:2])
    ax.imshow(rgb_frame)
    ax.set_title('Step 1: Original RGB Frame (1920×1080)', fontsize=16, fontweight='bold')
    ax.axis('off')

    # Mark ROI region
    rect = patches.Rectangle((0, 0), roi_w, roi_h,
                             linewidth=4, edgecolor='red', facecolor='none',
                             linestyle='--')
    ax.add_patch(rect)
    ax.text(roi_w/2, roi_h + 30, 'Timestamp Region',
           ha='center', fontsize=13, color='red', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))

    # Step 2: Grayscale Conversion
    ax = fig.add_subplot(gs[1, 2:4])
    ax.imshow(gray_frame, cmap='gray')
    ax.set_title('Step 2: Grayscale Conversion', fontsize=16, fontweight='bold')
    ax.axis('off')

    # Mark ROI
    rect = patches.Rectangle((0, 0), roi_w, roi_h,
                             linewidth=4, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    # Row 2: Step 3-4
    # Step 3: Extract ROI
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(timestamp_roi, cmap='gray')
    ax.set_title(f'Step 3: Extract ROI\nSize: {timestamp_roi.shape}',
                fontsize=14, fontweight='bold')
    ax.axis('off')

    # Step 4: Binarization comparison (show first 3 thresholds)
    for idx, (thresh, binary_img) in enumerate(binary_results[:3]):
        ax = fig.add_subplot(gs[2, idx+1])
        ax.imshow(binary_img, cmap='gray')
        ax.set_title(f'Threshold = {thresh}', fontsize=13, fontweight='bold')

        # OCR test
        try:
            text = pytesseract.image_to_string(binary_img, config=ocr_config).strip()
            color = 'green' if text and len(text) > 10 else 'red'
            ax.text(0.5, -0.1, f'OCR: {text[:20] if text else "Failed"}',
                   ha='center', transform=ax.transAxes, fontsize=9,
                   color=color, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except:
            ax.text(0.5, -0.1, 'OCR Failed',
                   ha='center', transform=ax.transAxes, fontsize=9, color='red')

        ax.axis('off')

        if idx == 2:  # thresh=200
            ax.text(0.5, 1.05, '✓ Best Threshold',
                   ha='center', transform=ax.transAxes, fontsize=11,
                   color='green', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

    # Row 3: Step 5-8
    # Step 5: Preprocessing
    ax = fig.add_subplot(gs[3, 0])
    ax.axis('off')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.text(0.5, 0.95, 'Step 5: Preprocessing', ha='center', fontsize=15, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightblue'))

    steps = [
        ('1. Grayscale', 'Convert RGB to grayscale'),
        ('2. ROI Extract', f'Region: (0,0) to ({roi_w},{roi_h})'),
        ('3. Binarize', 'Threshold = 200'),
        ('4. OCR Config', '--oem 3 --psm 7'),
        ('5. Whitelist', '0123456789:- ')
    ]

    y_pos = 0.8
    for step, desc in steps:
        ax.text(0.05, y_pos, step, ha='left', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue', pad=0.3))
        ax.text(0.5, y_pos, desc, ha='left', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray', pad=0.3))
        y_pos -= 0.15

    # Step 6: OCR Result
    ax = fig.add_subplot(gs[3, 1])
    ax.axis('off')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.text(0.5, 0.95, 'Step 6: OCR Result', ha='center', fontsize=15, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen'))

    ax.imshow(binary_best, cmap='gray', extent=[0.1, 0.9, 0.5, 0.85])

    ax.text(0.5, 0.35, 'Raw OCR Output:', ha='center', fontsize=11, fontweight='bold')
    ax.text(0.5, 0.25, f'"{ocr_text}"', ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', linewidth=2),
           family='monospace')

    if ocr_text:
        ax.text(0.5, 0.1, '✓ Success', ha='center', fontsize=13, color='green', fontweight='bold')
    else:
        ax.text(0.5, 0.1, '✗ Failed', ha='center', fontsize=13, color='red', fontweight='bold')

    # Step 7: Regex Parsing
    ax = fig.add_subplot(gs[3, 2])
    ax.axis('off')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.text(0.5, 0.95, 'Step 7: Regex Parsing', ha='center', fontsize=15, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax.text(0.5, 0.8, 'Pattern:', ha='center', fontsize=11, fontweight='bold')
    ax.text(0.5, 0.7, r'(\d{4}-\d{2}-\d{2}\s*\d{2}:\d{2}:\d{2})',
           ha='center', fontsize=10, family='monospace',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue', pad=0.3))

    ax.text(0.5, 0.55, 'Match Groups:', ha='center', fontsize=11, fontweight='bold')

    # Simulate match result
    if ocr_text and len(ocr_text) > 10:
        import re
        match = re.search(r'(\d{4}-\d{2}-\d{2})\s*(\d{2}:\d{2}:\d{2})', ocr_text)
        if match:
            date_part = match.group(1)
            time_part = match.group(2)

            ax.text(0.3, 0.4, 'Date:', ha='right', fontsize=10, fontweight='bold')
            ax.text(0.35, 0.4, date_part, ha='left', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='gray'))

            ax.text(0.3, 0.25, 'Time:', ha='right', fontsize=10, fontweight='bold')
            ax.text(0.35, 0.25, time_part, ha='left', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', edgecolor='gray'))

            ax.text(0.5, 0.1, '✓ Parsed Successfully', ha='center', fontsize=12,
                   color='green', fontweight='bold')

    # Step 8: Final Output
    ax = fig.add_subplot(gs[3, 3])
    ax.axis('off')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.text(0.5, 0.95, 'Step 8: Final Output', ha='center', fontsize=15, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen'))

    ax.text(0.5, 0.8, 'Datetime Object:', ha='center', fontsize=11, fontweight='bold')

    if ocr_text:
        import re
        match = re.search(r'(\d{4}-\d{2}-\d{2})\s*(\d{2}:\d{2}:\d{2})', ocr_text)
        if match:
            datetime_str = f"{match.group(1)} {match.group(2)}"
            ax.text(0.5, 0.65, datetime_str, ha='center', fontsize=14,
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', linewidth=2),
                   family='monospace', fontweight='bold')

            ax.text(0.5, 0.45, 'Format:', ha='center', fontsize=10, fontweight='bold')
            ax.text(0.5, 0.35, 'YYYY-MM-DD HH:MM:SS', ha='center', fontsize=11,
                   family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='gray'))

            ax.text(0.5, 0.2, 'Ready for:', ha='center', fontsize=10, fontweight='bold')
            ax.text(0.5, 0.1, '• Trajectory Matching\n• Data Alignment', ha='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

    plt.suptitle('Detailed OCR Timestamp Extraction Process',
                 fontsize=22, fontweight='bold', y=0.995)

    output_path = '/home/research/Agri-MBT/experiments/visualizations/ocr_extraction_detail.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    print("\n" + "="*70)
    print("OCR Extraction Detail Visualization")
    print("="*70)

    print("\n→ Creating OCR extraction detail visualization...")
    create_ocr_detail_pipeline()

    print("\n" + "="*70)
    print("✓ OCR detail visualization complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
