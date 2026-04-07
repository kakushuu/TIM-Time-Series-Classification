# Story 2: Multimodal Dataset Loader Implementation

## User Story
As a model trainer, I want a PyTorch Dataset class that loads aligned video-GNSS data with 5-second sliding windows, so that I can efficiently train the TC-AdaptFormer model with proper batching.

## Context
- **Data Source**: `/home/research/Agri-MBT/data/aligned_output/aligned_data.csv`
- **Total Samples**: 6272 aligned frame-GNSS pairs at 1Hz
- **Window Size**: T=5 consecutive frames (5 seconds)
- **GNSS Features**: 7 numerical columns (经度, 纬度, 速度, 深度, 方向角, 间距, 类型)
- **Video Frames**: JPG images at paths specified in `frame_path` column
- **Labels**: `分类` column (0-10, 11 classes)

## Acceptance Criteria

### AC1: Dataset Class Structure
- [ ] Class `AgriMultimodalDataset(torch.utils.data.Dataset)` created in `src/dataset.py`
- [ ] `__init__` accepts: csv_path, window_size=5, transform=None, normalize_gnss=True
- [ ] `__len__` returns number of valid windows (not raw samples)
- [ ] `__getitem__` returns tuple: (video_tensor, gnss_tensor, label)

### AC2: Sliding Window Logic
- [ ] Extract consecutive T=5 frame sequences from CSV
- [ ] Handle boundary cases: skip windows with <5 consecutive frames
- [ ] Ensure temporal continuity: check `second_in_video` or `时间戳` gaps
- [ ] Example: samples [0,1,2,3,4] → window 0, samples [1,2,3,4,5] → window 1

### AC3: Video Frame Loading
- [ ] Load T=5 JPG images from `frame_path` column
- [ ] Resize to 224×224 if needed
- [ ] Apply torchvision transforms (ToTensor, Normalize with ImageNet stats)
- [ ] Output shape: `(T, 3, 224, 224)` = `(5, 3, 224, 224)`
- [ ] Handle missing files gracefully (log warning, skip window)

### AC4: GNSS Feature Extraction
- [ ] Extract 7 columns: `['经度', '纬度', '速度', '深度', '方向角', '间距(米)', '类型']`
- [ ] Convert to float32 numpy array
- [ ] Shape for T=5 window: `(5, 7)` → flatten to `(35,)` OR keep `(5, 7)`
- [ ] Decide: use only first frame GNSS `(7,)` or all T frames `(5, 7)`

### AC5: GNSS Normalization
- [ ] Compute mean/std from training set for each of 7 features
- [ ] Save normalization stats to `data/gnss_normalization.json`
- [ ] Apply z-score normalization: `(x - mean) / std`
- [ ] Handle special cases: 类型 (categorical) → one-hot or keep as-is

### AC6: Label Handling
- [ ] Extract `分类` column as integer label (0-10)
- [ ] Verify label range: assert all labels in [0, 10]
- [ ] Return as torch.long tensor

### AC7: Train/Val/Test Split
- [ ] Implement `split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)`
- [ ] Use stratified split to preserve class distribution
- [ ] Return 3 Subset objects
- [ ] Save split indices to `data/split_indices.json` for reproducibility

### AC8: DataLoader Integration Test
- [ ] Create test script `tests/test_dataset.py`
- [ ] Load 1 batch with batch_size=8
- [ ] Print shapes: video `(8, 5, 3, 224, 224)`, gnss `(8, 7)` or `(8, 5, 7)`, labels `(8,)`
- [ ] Verify no NaN values in tensors
- [ ] Measure loading time for 100 samples

### AC9: Class Imbalance Handling
- [ ] Compute class weights: `weight[i] = 1 / count[i]` normalized
- [ ] Save to `data/class_weights.json`
- [ ] Document usage: `criterion = nn.CrossEntropyLoss(weight=class_weights)`

## Definition of Done
- File `src/dataset.py` exists with `AgriMultimodalDataset` class
- File `tests/test_dataset.py` runs without errors
- Test output shows correct tensor shapes
- Normalization stats and class weights saved to JSON files
- All 9 acceptance criteria verified with unit tests
- Code includes Chinese comments for key logic

## Technical Notes
- Use `pandas.read_csv` for CSV loading
- Use `PIL.Image.open` or `cv2.imread` for JPG loading
- Consider caching loaded images if memory allows
- Handle class imbalance: Class 3 (2304 samples) vs Class 1 (98 samples)
- GNSS decision: recommend using only first frame GNSS `(7,)` to match architecture design (single query vector)

## Example Usage
```python
from src.dataset import AgriMultimodalDataset
from torch.utils.data import DataLoader

dataset = AgriMultimodalDataset(
    csv_path='data/aligned_output/aligned_data.csv',
    window_size=5,
    normalize_gnss=True
)

train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

for video, gnss, labels in train_loader:
    # video: (8, 5, 3, 224, 224)
    # gnss: (8, 7)
    # labels: (8,)
    break
```
