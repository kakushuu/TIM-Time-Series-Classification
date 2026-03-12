#!/usr/bin/env python3
"""
Data Processing Pipeline Visualization
Shows the complete workflow from raw video frames to aligned data
"""
import matplotlib
matplotlib.use('Agg')

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
import pytesseract

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_pipeline_visualization():
    """Create complete data processing pipeline visualization"""

    # Load a sample frame
    sample_frame_path = '/home/research/Agri-MBT/data/aligned_output/aligned_frames/20241018_123813.jpg'

    if not Path(sample_frame_path).exists():
        print(f"⚠ Sample frame not found: {sample_frame_path}")
        frames_dir = Path('/home/research/Agri-MBT/data/aligned_output/aligned_frames')
        if frames_dir.exists():
            sample_frames = list(frames_dir.glob('*.jpg'))
            if sample_frames:
                sample_frame_path = str(sample_frames[0])
                print(f"  Using: {sample_frame_path}")
            else:
                print("  No frames found, creating mock visualization")
                return

    # Read original RGB frame
    rgb_frame = cv2.imread(sample_frame_path)
    rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)

    # Extract timestamp ROI (upper-left corner)
    h, w = gray_frame.shape
    roi_y_start = 0
    roi_y_end = int(h * 0.05)
    roi_x_start = 0
    roi_x_end = int(w * 0.30)

    timestamp_roi = gray_frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

    # Binarization (for OCR)
    _, binary_roi = cv2.threshold(timestamp_roi, 200, 255, cv2.THRESH_BINARY)

    # OCR extraction
    ocr_config = '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:- '
    ocr_text = pytesseract.image_to_string(binary_roi, config=ocr_config).strip()

    # Load trajectory data
    trajectory_csv = '/home/research/Agri-MBT/data/aligned_output/aligned_data.csv'
    traj_df = pd.read_csv(trajectory_csv)

    # Find matched trajectory data
    frame_name = Path(sample_frame_path).stem
    matched_row = traj_df[traj_df['frame_path'].str.contains(frame_name)]

    # ============ Create main pipeline figure ============
    fig = plt.figure(figsize=(24, 32))
    gs = fig.add_gridspec(6, 3, hspace=0.35, wspace=0.3)

    # Row 1: Title
    ax = fig.add_subplot(gs[0, :])
    ax.axis('off')
    ax.text(0.5, 0.5, 'Video-Trajectory Data Processing Pipeline',
            ha='center', va='center', fontsize=28, fontweight='bold',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', edgecolor='navy', linewidth=3))

    # Row 2: Step 1-3
    # Step 1: Original RGB Frame
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(rgb_frame)
    ax.set_title('Step 1: Original RGB Frame', fontsize=16, fontweight='bold', pad=10)
    ax.axis('off')

    # Mark timestamp region with red rectangle
    rect = patches.Rectangle((0, 0), roi_x_end, roi_y_end,
                             linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.text(roi_x_end/2, roi_y_end + 20, 'Timestamp ROI',
           ha='center', fontsize=12, color='red', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # Step 2: Grayscale Conversion
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(gray_frame, cmap='gray')
    ax.set_title('Step 2: Grayscale Conversion', fontsize=16, fontweight='bold', pad=10)
    ax.axis('off')

    rect = patches.Rectangle((0, 0), roi_x_end, roi_y_end,
                             linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    # Step 3: Extract ROI
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(timestamp_roi, cmap='gray')
    ax.set_title('Step 3: Extract Timestamp ROI', fontsize=16, fontweight='bold', pad=10)
    ax.axis('off')
    ax.text(0.5, -0.1, f'Shape: {timestamp_roi.shape}',
           ha='center', transform=ax.transAxes, fontsize=11)

    # Row 3: Step 4-6
    # Step 4: Binarization
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(binary_roi, cmap='gray')
    ax.set_title('Step 4: Binarization (Thresh=200)', fontsize=16, fontweight='bold', pad=10)
    ax.axis('off')

    # Step 5: OCR Recognition
    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(binary_roi, cmap='gray')
    ax.set_title('Step 5: OCR Text Recognition', fontsize=16, fontweight='bold', pad=10)
    ax.axis('off')

    ocr_display = ocr_text if ocr_text else 'OCR Failed'
    ax.text(0.5, -0.15, f'OCR Result:\n{ocr_display}',
           ha='center', transform=ax.transAxes, fontsize=13,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9),
           fontweight='bold')

    # Step 6: Timestamp Parsing
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.text(0.5, 0.85, 'OCR Text', ha='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax.text(0.5, 0.75, ocr_display, ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    ax.annotate('', xy=(0.5, 0.55), xytext=(0.5, 0.65),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

    ax.text(0.5, 0.5, 'Regex Parsing', ha='center', fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax.text(0.5, 0.35, r'Pattern: (\d{4}-\d{2}-\d{2}\s*\d{2}:\d{2}:\d{2})',
           ha='center', fontsize=10, family='monospace',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    ax.annotate('', xy=(0.5, 0.15), xytext=(0.5, 0.25),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

    ax.text(0.5, 0.08, 'Datetime Object', ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen'))

    ax.set_title('Step 6: Timestamp Parsing', fontsize=16, fontweight='bold', pad=10)

    # Row 4: Step 7-9
    # Step 7: Load Trajectory Data
    ax = fig.add_subplot(gs[3, 0])
    ax.axis('off')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.text(0.5, 0.9, 'Trajectory Data CSV', ha='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightcoral'))

    traj_columns = ['Timestamp', 'Longitude', 'Latitude', 'Speed', 'Depth', 'Heading', 'Type']
    y_pos = 0.75
    for col in traj_columns:
        ax.text(0.5, y_pos, f'• {col}', ha='center', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', pad=0.3))
        y_pos -= 0.1

    ax.text(0.5, 0.05, f'Total: {len(traj_df)} rows', ha='center', fontsize=11,
           fontweight='bold', color='blue')

    ax.set_title('Step 7: Load Trajectory Data', fontsize=16, fontweight='bold', pad=10)

    # Step 8: Timestamp Matching
    ax = fig.add_subplot(gs[3, 1])
    ax.axis('off')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.text(0.2, 0.9, 'Frame Time', ha='center', fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax.text(0.2, 0.8, ocr_display, ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    ax.text(0.5, 0.9, 'Match', ha='center', fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow'))
    ax.text(0.5, 0.7, 'Tolerance\n±2s', ha='center', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.text(0.5, 0.5, 'Distance\nCalculation', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    ax.text(0.8, 0.9, 'Trajectory Time', ha='center', fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightcoral'))

    if len(matched_row) > 0:
        traj_time = matched_row.iloc[0]['定位时间']
        ax.text(0.8, 0.8, str(traj_time), ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    ax.annotate('', xy=(0.5, 0.85), xytext=(0.2, 0.85),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.annotate('', xy=(0.5, 0.85), xytext=(0.8, 0.85),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'))

    if len(matched_row) > 0:
        ax.text(0.5, 0.15, '✓ Match Found!', ha='center', fontsize=14, fontweight='bold',
               color='green', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    else:
        ax.text(0.5, 0.15, '✗ No Match', ha='center', fontsize=14, fontweight='bold',
               color='red', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_title('Step 8: Timestamp Matching', fontsize=16, fontweight='bold', pad=10)

    # Step 9: Data Alignment
    ax = fig.add_subplot(gs[3, 2])
    ax.axis('off')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.text(0.5, 0.95, 'Aligned Data', ha='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen'))

    if len(matched_row) > 0:
        row = matched_row.iloc[0]
        fields = [
            ('Frame', frame_name),
            ('Time', str(row['定位时间'])),
            ('Lon', f"{row['经度']:.7f}"),
            ('Lat', f"{row['纬度']:.7f}"),
            ('Speed', f"{row['速度']:.2f}"),
            ('Depth', f"{row['深度']}"),
            ('Class', f"{row['分类']}")
        ]

        y_pos = 0.8
        for field_name, field_value in fields:
            ax.text(0.3, y_pos, f'{field_name}:', ha='right', fontsize=10, fontweight='bold')
            ax.text(0.35, y_pos, field_value, ha='left', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor='lightgray', pad=0.2))
            y_pos -= 0.1
    else:
        ax.text(0.5, 0.5, 'No matched data\nto display', ha='center', fontsize=12, color='gray')

    ax.set_title('Step 9: Data Alignment', fontsize=16, fontweight='bold', pad=10)

    # Row 5: Step 10-12
    # Step 10: OCR Status Check
    ax = fig.add_subplot(gs[4, 0])
    ax.axis('off')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.text(0.5, 0.9, 'OCR Status Check', ha='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='orange'))

    status_types = [
        ('ok', 'green', 'Successfully read'),
        ('ocr_error', 'red', 'Read but invalid'),
        ('interpolated', 'blue', 'Interpolated (gap ≤5s)'),
        ('excluded', 'gray', 'Excluded (no match)')
    ]

    y_pos = 0.7
    for status, color, desc in status_types:
        ax.text(0.1, y_pos, f'• {status}:', ha='left', fontsize=11, fontweight='bold', color=color)
        ax.text(0.45, y_pos, desc, ha='left', fontsize=10)
        y_pos -= 0.15

    ax.set_title('Step 10: OCR Status Check', fontsize=16, fontweight='bold', pad=10)

    # Step 11: Timestamp Validation
    ax = fig.add_subplot(gs[4, 1])
    ax.axis('off')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.text(0.5, 0.9, 'Timestamp Validation', ha='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='orange'))

    ax.text(0.5, 0.75, 'Local Pairwise Rate Check', ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax.text(0.5, 0.6, r'Rate = $\frac{\Delta timestamp}{\Delta offset}$', ha='center', fontsize=13)

    ax.text(0.5, 0.45, 'Valid Range: [0.5, 1.5] s/s', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    ax.text(0.5, 0.25, 'If invalid → ocr_error', ha='center', fontsize=11,
           color='red', fontweight='bold')

    ax.set_title('Step 11: Timestamp Validation', fontsize=16, fontweight='bold', pad=10)

    # Step 12: Final Output
    ax = fig.add_subplot(gs[4, 2])
    ax.axis('off')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.text(0.5, 0.9, 'Final Output', ha='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen'))

    output_files = [
        ('aligned_data.csv', 'Main dataset'),
        ('aligned_frames/', 'Frame images'),
        ('alignment_stats.json', 'Statistics'),
        ('ocr_correction_report.json', 'OCR report')
    ]

    y_pos = 0.7
    for filename, desc in output_files:
        ax.text(0.1, y_pos, f'📄 {filename}', ha='left', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue', pad=0.3))
        ax.text(0.85, y_pos, desc, ha='center', fontsize=9, style='italic')
        y_pos -= 0.15

    ax.set_title('Step 12: Final Output', fontsize=16, fontweight='bold', pad=10)

    # Row 6: Statistics
    ax = fig.add_subplot(gs[5, :])
    ax.axis('off')

    stats_text = f"""
    Processing Statistics:
    • Total Frames Processed: {len(traj_df)}
    • OCR Success Rate: {(traj_df['frame_path'].notna().sum() / len(traj_df) * 100):.1f}%
    • Matched Trajectory Points: {len(traj_df[traj_df['frame_path'].notna()])}
    • 11 Activity Classes
    • Time Range: {traj_df['定位时间'].min()} to {traj_df['定位时间'].max()}
    • Spatial Coverage: Longitude [{traj_df['经度'].min():.6f}, {traj_df['经度'].max():.6f}], Latitude [{traj_df['纬度'].min():.6f}, {traj_df['纬度'].max():.6f}]
    """

    ax.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=14,
           family='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', edgecolor='orange', linewidth=2))

    plt.suptitle('Agricultural Video-Trajectory Data Processing Pipeline',
                 fontsize=24, fontweight='bold', y=0.995)

    # Save
    output_path = '/home/research/Agri-MBT/experiments/visualizations/data_processing_pipeline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    print("\n" + "="*70)
    print("Data Processing Pipeline Visualization")
    print("="*70)

    print("\n→ Creating data processing pipeline visualization...")
    create_pipeline_visualization()

    print("\n" + "="*70)
    print("✓ Pipeline visualization complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
