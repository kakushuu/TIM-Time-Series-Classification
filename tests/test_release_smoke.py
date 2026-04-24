from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import AgriMultimodalDataset
from train_ablation import LongWindowDataset


ROOT = Path(__file__).parent.parent
DEMO = ROOT / "sample_data" / "masked_demo"


def test_masked_demo_dataset_loader():
    dataset = AgriMultimodalDataset(
        csv_path=str(DEMO / "train.csv"),
        window_size=4,
        normalize_gnss=True,
        gnss_stats_path="sample_data/masked_demo/gnss_normalization.json",
        img_size=64,
    )
    video, gnss, label = dataset[0]
    assert video.shape == (4, 3, 64, 64)
    assert gnss.shape == (4, 5)
    assert 0 <= label <= 10


def test_masked_demo_long_window_trimodal():
    dataset = LongWindowDataset(
        csv_path=str(DEMO / "train.csv"),
        seq_len=4,
        stride=1,
        eval_stride=1,
        context_mode="causal",
        sampling_strategy="fixed",
        duration_stats="",
        adaptive_min_window=4,
        adaptive_max_window=8,
        adaptive_context_scale=2.0,
        adaptive_min_stride=1,
        adaptive_max_stride=2,
        adaptive_stride_ratio=0.25,
        image_window_size=3,
        image_sampling="nearest_causal",
        image_radius=2,
        image_radius_mode="fixed",
        image_radius_duration_scale=0.5,
        image_radius_classes="",
        image_frame_dropout=0.0,
        image_jpeg_draft_size=0,
        feature_mode="engineered",
        mode="trimodal",
        is_train=False,
        max_time_gap=1.0,
        audio_sample_rate=16000,
        img_size=64,
    )
    traj, images, audio, label = dataset[0]
    assert traj.shape == (4, 36)
    assert images.shape == (3, 3, 64, 64)
    assert audio.numel() == 16000
    assert 0 <= label <= 10
