# Erase-and-Restore vs PatchGuard 对比项目

复现并对比两篇顶会对抗性防御论文：**Erase-and-Restore** (AsiaCCS'21) 和 **PatchGuard** (USENIX Security'21)，在 ImageNette 数据集上评估各自在 L2 对抗样本和对抗性 patch 场景下的表现。

## 目录结构

```
Erase-and-Restore/          # AsiaCCS'21 原始仓库
  └── erase_restore/
      ├── erase_restore_lib.py          # 原始实现 (Keras, ImageNet)
      └── ern_imagenette.py             # 适配版 (PyTorch GPU, ImageNette)

PatchGuard/                 # USENIX Security'21 原始仓库
  ├── mask_bn.py            # robust masking 防御
  ├── mask_ds.py            # derandomized smoothing 防御
  ├── det_bn.py             # PatchGuard++ 攻击检测
  └── ...                   # 其余原仓库文件

attacks/                    # 对抗样本生成脚本
  ├── generate_l2_adv.py    # L2 PGD 对抗样本 (ε=0.5/1.0/2.0/4.0)
  └── generate_patch_adv.py # 随机方块 patch 对抗样本 (16/32/48)

shared_data/                # 共享数据目录
  ├── benign_imagenette.pkl # 干净样本
  ├── adv_l2_eps*.pkl       # L2 对抗样本
  ├── adv_patch*.pkl        # Patch 对抗样本
  └── results_ern/          # E&R 实验结果

analysis/                   # 交叉测试与报告
  ├── cross_test_ern_on_patch.py  # E&R 检测 patch 攻击
  ├── cross_test_pg_on_l2.py      # PatchGuard 防御 L2 攻击
  └── report.md                   # 最终对比报告

docs/superpowers/
  ├── specs/                # 设计文档
  └── plans/                # 实现计划

CHANGELOG.md                # 详细改动记录
```

## 环境

```bash
conda create -n adv-defense python=3.11 -y
conda activate adv-defense

# PyTorch CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 其余依赖
pip install opencv-python scikit-learn scikit-image pandas joblib tqdm scipy
```

硬件: NVIDIA RTX 4060 8GB, 16GB RAM

## 快速开始

### 1. 准备数据和模型

- ImageNette (full size) 放至 `PatchGuard/data/imagenette/` (含 train/val)
- PatchGuard 预训练 checkpoint 放至 `PatchGuard/checkpoints/`
  - `bagnet17_nette.pth`, `bagnet33_nette.pth`, `ds_nette.pth`
  - 从 [Google Drive](https://drive.google.com/drive/folders/1u5RsCuZNf7ddWW0utI4OrgWGmJCUDCuT) 下载

### 2. 生成对抗样本

```bash
# L2 对抗样本 (3000 张, 4 种强度)
python attacks/generate_l2_adv.py --max_images 3000

# Patch 对抗样本 (3000 张, 3 种尺寸)
python attacks/generate_patch_adv.py
```

### 3. 运行 Erase-and-Restore

```bash
python Erase-and-Restore/erase_restore/ern_imagenette.py --n_samples 1000
```

### 4. 运行 PatchGuard

```bash
cd PatchGuard

# Mask-BN
python mask_bn.py --model bagnet17 --dataset imagenette --patch_size 32 --m
python mask_bn.py --model bagnet17 --dataset imagenette --patch_size 32 --cbn
python mask_bn.py --model bagnet17 --dataset imagenette --patch_size 16 --m
python mask_bn.py --model bagnet17 --dataset imagenette --patch_size 48 --m

# Mask-DS / DS
python mask_ds.py --dataset imagenette --patch_size 42 --ds
python mask_ds.py --dataset imagenette --patch_size 42 --m
```

### 5. 交叉测试

```bash
python analysis/cross_test_ern_on_patch.py
python analysis/cross_test_pg_on_l2.py
```

## 实验结果摘要

### Erase-and-Restore: L2 对抗样本检测

| L2 ε | AdaBoost acc | AUC |
|:---:|:---:|:---:|
| 0.5 | 91.4% | 0.973 |
| 1.0 | 91.2% | 0.972 |
| 2.0 | 91.6% | 0.971 |
| 4.0 | 91.0% | 0.972 |

### PatchGuard: 可证明 Patch 防御 (Mask-BN, BagNet17)

| Patch Size | 可证明鲁棒 | 干净精度 |
|:---:|:---:|:---:|
| 16×16 | 90.7% | 95.2% |
| 32×32 | 85.8% | 95.0% |
| 48×48 | 78.3% | 94.5% |

### 交叉测试

| | L2 对抗样本 | Patch 攻击 |
|:---|:---:|:---:|
| Erase-and-Restore | **91%** (专长) | 5-11% (失效) |
| PatchGuard | ~0% (失效) | **86%** (专长) |

## 核心结论

**两个方案互补而非竞争**。Erase-and-Restore 擅长检测全局 L2 扰动（经验性），PatchGuard 擅长防御局部 patch 攻击（可证明）。具体选择取决于实际威胁模型。

## 参考文献

- [Exploiting the Sensitivity of L2 Adversarial Examples to Erase-and-Restore](https://doi.org/10.1145/3433210.3453094) (AsiaCCS 2021)
- [PatchGuard: A Provably Robust Defense against Adversarial Patches via Small Receptive Fields and Masking](https://www.usenix.org/conference/usenixsecurity21/presentation/xiang) (USENIX Security 2021)
