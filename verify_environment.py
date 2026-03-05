#!/usr/bin/env python3
"""
验证 Agri-MBT 环境配置

检查所有必要的依赖是否正确安装
"""

import sys
from typing import Dict, List, Tuple

def check_python_version() -> Tuple[bool, str]:
    """检查 Python 版本"""
    version = sys.version_info
    required = (3, 8)
    if version >= required:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor}.{version.micro} (需要 >= 3.8)"

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """检查单个包是否安装"""
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, "未安装"

def check_cuda() -> Tuple[bool, str]:
    """检查 CUDA 是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
            cuda_version = torch.version.cuda
            return True, f"CUDA {cuda_version}, {device_count} GPU(s), {device_name}"
        else:
            return False, "CUDA 不可用"
    except Exception as e:
        return False, f"检查失败: {str(e)}"

def main():
    """主检查流程"""
    print("=" * 80)
    print("Agri-MBT 环境验证")
    print("=" * 80)
    print()

    # 定义要检查的包
    packages: Dict[str, List[Tuple[str, str]]] = {
        "核心依赖": [
            ("numpy", "numpy"),
            ("pandas", "pandas"),
            ("scipy", "scipy"),
            ("scikit-learn", "sklearn"),
        ],
        "深度学习": [
            ("torch", "torch"),
            ("torchvision", "torchvision"),
            ("torchaudio", "torchaudio"),
            ("timm", "timm"),
            ("transformers", "transformers"),
            ("einops", "einops"),
        ],
        "计算机视觉": [
            ("opencv-python", "cv2"),
            ("pillow", "PIL"),
            ("pytesseract", "pytesseract"),
        ],
        "数据处理": [
            ("openpyxl", "openpyxl"),
            ("tqdm", "tqdm"),
        ],
        "可视化": [
            ("matplotlib", "matplotlib"),
            ("seaborn", "seaborn"),
        ],
    }

    results = []

    # 检查 Python 版本
    print("检查 Python 版本...")
    success, info = check_python_version()
    status = "✓" if success else "✗"
    print(f"  {status} {info}")
    results.append(success)
    print()

    # 检查各个包
    for category, package_list in packages.items():
        print(f"检查 {category}...")
        for package_name, import_name in package_list:
            success, version = check_package(package_name, import_name)
            status = "✓" if success else "✗"
            print(f"  {status} {package_name:20s} {version}")
            results.append(success)
        print()

    # 检查 CUDA
    print("检查 CUDA...")
    success, info = check_cuda()
    status = "✓" if success else "✗"
    print(f"  {status} {info}")
    results.append(success)
    print()

    # 总结
    print("=" * 80)
    if all(results):
        print("✓ 环境配置完成! 所有依赖都已正确安装。")
        print()
        print("下一步:")
        print("  1. 检查数据: ls data/aligned_output/")
        print("  2. 运行训练: cd Multimodal-Fusion-with-Attention-Bottlenecks-main/MBT/")
        print("              python train_test.py --batch_size 8 --num_epochs 15")
    else:
        print("✗ 部分依赖缺失或配置有问题。")
        print()
        print("请运行以下命令安装缺失的依赖:")
        print("  conda activate agri-mbt")
        print("  pip install -r requirements.txt")
    print("=" * 80)

    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())
