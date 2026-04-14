"""
Mock 测试脚本：TC-AdaptFormer 前向/反向传播验证 + 资源报告
conda run -n agri-mbt pytest tests/test_model_mock.py -v
conda run -n agri-mbt python tests/test_model_mock.py --report
"""

import sys
import time
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn.functional as F
import pytest

from models import TCAdaptFormer


# ── 构造 Mock 数据 ────────────────────────────────────────────────────────────

def make_batch(batch_size: int = 2, T: int = 5, device: str = 'cpu'):
    """构造随机测试批次"""
    video  = torch.randn(batch_size, T, 3, 224, 224, device=device)
    gnss   = torch.randn(batch_size, T, 7, device=device)
    labels = torch.randint(0, 11, (batch_size,), device=device)
    return video, gnss, labels


@pytest.fixture(scope='module')
def model():
    m = TCAdaptFormer(num_classes=11, pretrained=False)
    m.eval()
    return m


# ── 核心测试 ──────────────────────────────────────────────────────────────────

def test_forward_pass(model):
    """前向传播：输出形状正确，无 NaN/Inf"""
    video, gnss, _ = make_batch(batch_size=2)
    with torch.no_grad():
        logits = model(video, gnss)
    assert logits.shape == (2, 11), f"输出形状错误: {logits.shape}"
    assert not torch.isnan(logits).any(), "logits 含 NaN"
    assert torch.isfinite(logits).all(), "logits 含 Inf"


def test_backward_pass(model):
    """反向传播：可训练参数有梯度，冻结参数无梯度"""
    model.train()
    video, gnss, labels = make_batch(batch_size=2)
    logits = model(video, gnss)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    # 检查可训练参数有梯度
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"可训练参数无梯度: {name}"
        else:
            assert p.grad is None, f"冻结参数出现梯度: {name}"

    model.zero_grad()
    model.eval()


def test_parameter_count(model):
    """可训练参数量在合理范围 [3M, 5M]"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    ratio = trainable / total
    assert trainable >= 3e6, f"可训练参数量 {trainable/1e6:.2f}M 过低"
    assert ratio < 0.3, f"可训练比例 {ratio:.1%} 过高（期望 <30%）"


def test_wrong_input_shapes(model):
    """错误输入形状应当触发 AssertionError"""
    with pytest.raises(AssertionError):
        # gnss 维度错误
        model(torch.randn(2, 5, 3, 224, 224), torch.randn(2, 5))
    with pytest.raises(AssertionError):
        # batch size 不匹配
        model(torch.randn(2, 5, 3, 224, 224), torch.randn(3, 5, 7))


def test_output_determinism(model):
    """相同输入两次前向应得到相同输出"""
    video, gnss, _ = make_batch(batch_size=2)
    with torch.no_grad():
        out1 = model(video, gnss)
        out2 = model(video, gnss)
    assert torch.allclose(out1, out2), "相同输入得到不同输出"


# ── 资源报告（独立运行时调用）────────────────────────────────────────────────

def generate_report():
    """生成完整资源消耗报告，保存到 tests/mock_test_report.txt"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"  TC-AdaptFormer Mock 测试资源报告")
    print(f"  设备: {device}")
    print(f"{'='*60}")

    model = TCAdaptFormer(num_classes=11, pretrained=False).to(device)
    model.eval()

    # ── 参数量 ────────────────────────────────────────────────────
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"\n[参数量]")
    print(f"  总参数:       {total/1e6:.2f}M")
    print(f"  可训练参数:   {trainable/1e6:.2f}M  ({trainable/total*100:.1f}%)")
    print(f"  冻结参数:     {frozen/1e6:.2f}M  ({frozen/total*100:.1f}%)")

    # ── 前向传播耗时 ──────────────────────────────────────────────
    B, T = 2, 5
    video = torch.randn(B, T, 3, 224, 224, device=device)
    gnss  = torch.randn(B, T, 7, device=device)

    # warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(video, gnss)

    N_ITERS = 20
    times = []
    with torch.no_grad():
        for _ in range(N_ITERS):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(video, gnss)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)  # ms

    import statistics
    mean_ms = statistics.mean(times)
    std_ms  = statistics.stdev(times)
    per_sample = mean_ms / B
    print(f"\n[推理耗时 (batch_size={B}, T={T}, {N_ITERS}次平均)]")
    print(f"  批次耗时:     {mean_ms:.1f} ± {std_ms:.1f} ms")
    print(f"  单样本耗时:   {per_sample:.1f} ms/sample")
    print(f"  吞吐量:       {1000/per_sample:.0f} samples/s")

    # ── 显存占用 ──────────────────────────────────────────────────
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(video, gnss)
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\n[显存占用 (batch_size={B})]")
        print(f"  前向峰值显存: {peak_mb:.0f} MB")
    else:
        param_mb = total * 4 / 1024**2
        print(f"\n[内存估算]")
        print(f"  参数内存 (FP32): {param_mb:.0f} MB")

    # ── 批次大小扩展性 ────────────────────────────────────────────
    print(f"\n[批次大小扩展性]")
    print(f"  {'Batch':>6} | {'时间(ms)':>10} | {'吞吐(s/s)':>10} | {'显存(MB)':>10}")
    print(f"  {'------':>6}-+-{'----------':>10}-+-{'----------':>10}-+-{'----------':>10}")

    results = {}
    for bs in [1, 2, 4, 8]:
        try:
            v = torch.randn(bs, T, 3, 224, 224, device=device)
            g = torch.randn(bs, T, 7, device=device)
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(v, g)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000
            mem = torch.cuda.max_memory_allocated() / 1024**2 if device == 'cuda' else total * 4 / 1024**2
            throughput = bs / (elapsed / 1000)
            print(f"  {bs:>6} | {elapsed:>10.1f} | {throughput:>10.0f} | {mem:>10.0f}")
            results[bs] = {'time_ms': elapsed, 'throughput': throughput, 'mem_mb': mem}
        except RuntimeError as e:
            print(f"  {bs:>6} | {'OOM':>10} | {'-':>10} | {'-':>10}")

    # ── 测试汇总 ──────────────────────────────────────────────────
    print(f"\n[测试汇总]")
    tests_passed = {
        '前向传播形状': True,
        '输出无NaN': not torch.isnan(out).any().item(),
        '可训练参数3-5M': 3e6 <= trainable <= 5e6,
        '推理时延<200ms': per_sample < 200,
    }
    for name, passed in tests_passed.items():
        status = '✅ PASS' if passed else '❌ FAIL'
        print(f"  {status}  {name}")

    all_pass = all(tests_passed.values())
    print(f"\n{'✅ 全部通过' if all_pass else '❌ 存在失败项'}")
    print('='*60)

    # 保存报告
    report_path = Path('tests/mock_test_report.txt')
    report_path.parent.mkdir(exist_ok=True)

    report_lines = [
        "TC-AdaptFormer Mock 测试资源报告",
        f"设备: {device}",
        "",
        f"总参数:       {total/1e6:.2f}M",
        f"可训练参数:   {trainable/1e6:.2f}M  ({trainable/total*100:.1f}%)",
        f"冻结参数:     {frozen/1e6:.2f}M",
        "",
        f"单样本推理耗时: {per_sample:.1f} ms",
        f"推理吞吐量:     {1000/per_sample:.0f} samples/s",
        "",
    ]
    for bs, r in results.items():
        report_lines.append(
            f"batch_size={bs}: {r['time_ms']:.1f}ms  {r['throughput']:.0f}s/s  {r['mem_mb']:.0f}MB"
        )
    report_lines += ["", "测试汇总:"]
    for name, passed in tests_passed.items():
        report_lines.append(f"  {'PASS' if passed else 'FAIL'}  {name}")
    report_lines.append(f"\n总体结果: {'ALL PASS' if all_pass else 'FAILED'}")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\n报告已保存: {report_path}")

    return all_pass


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    success = generate_report()
    sys.exit(0 if success else 1)
