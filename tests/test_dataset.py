"""
数据集加载器测试
pytest tests/test_dataset.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import pytest
from torch.utils.data import DataLoader
from dataset import AgriMultimodalDataset, split_dataset

CSV_PATH = 'data/aligned_output/aligned_data.csv'


@pytest.fixture(scope='module')
def dataset():
    return AgriMultimodalDataset(csv_path=CSV_PATH, window_size=5, normalize_gnss=True)


def test_dataset_size(dataset):
    assert len(dataset) > 0, "数据集为空"
    print(f"\n数据集大小: {len(dataset)}")


def test_single_sample_shapes(dataset):
    video, gnss, label = dataset[0]
    assert video.shape == (5, 3, 224, 224), f"video 形状错误: {video.shape}"
    assert gnss.shape == (5, 7), f"gnss 形状错误: {gnss.shape}"
    assert isinstance(label, int), f"label 类型错误: {type(label)}"


def test_label_range(dataset):
    for i in range(min(100, len(dataset))):
        _, _, label = dataset[i]
        assert 0 <= label <= 10, f"label 越界: {label} at index {i}"


def test_no_nan_values(dataset):
    video, gnss, _ = dataset[0]
    assert not torch.isnan(video).any(), "video 中存在 NaN"
    assert not torch.isnan(gnss).any(), "gnss 中存在 NaN"


def test_dataloader_batch_shapes(dataset):
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    v_batch, g_batch, l_batch = next(iter(loader))
    assert v_batch.shape == (8, 5, 3, 224, 224), f"批次 video 形状错误: {v_batch.shape}"
    assert g_batch.shape == (8, 5, 7), f"批次 gnss 形状错误: {g_batch.shape}"
    assert l_batch.shape == (8,), f"批次 label 形状错误: {l_batch.shape}"
    assert l_batch.min() >= 0 and l_batch.max() <= 10


def test_gnss_normalization_files():
    dataset = AgriMultimodalDataset(csv_path=CSV_PATH, window_size=5, normalize_gnss=True)
    split_dataset(dataset, save_path='data/test_split_indices.json')
    assert Path('data/gnss_normalization.json').exists(), "GNSS 归一化文件未生成"
    assert Path('data/class_weights.json').exists(), "类别权重文件未生成"
