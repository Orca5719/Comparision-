# 项目改动记录

## Erase-and-Restore (原仓库)

### 新增文件
| 文件 | 说明 |
|------|------|
| `erase_restore/ern_imagenette.py` | 核心适配脚本。将原论文的 ImageNet 1000类/Keras 实现改为 ImageNette 10类/PyTorch GPU 实现。保留原论文的随机掩码生成、Telea inpaint 算法、AdaBoost/SVM 检测器训练逻辑 |

### 修改内容
| 方面 | 原实现 | 适配后 |
|------|--------|--------|
| 模型框架 | Keras ResNet50 | PyTorch ResNet50 (torchvision) |
| 类别数 | 1000 (完整 ImageNet) | 10 (ImageNette, 通过提取 ImageNet 中对应索引实现) |
| 推理方式 | 单张循环 `model.predict()` | 批量 GPU 推理 (batch_size=32) |
| 特征维度 | 12×1000→PCA→120D | 12×10=120D 直接使用 (无需 PCA) |
| 数据集 | ImageNet ILSVRC2012 | ImageNette (full size) |
| 路径 | 硬编码绝对路径 `/adv/...` | 命令行参数指定 |

### 未修改
- 随机掩码生成算法 (`get_single_mask`) — 完全保留原文 5 种形状
- 掩码数量 (11) 和像素密度 (300)
- cv2.inpaint 参数 (INPAINT_TELEA, radius=3)
- 检测器: AdaBoost (DecisionTree max_depth=1, 200 estimators) / SVM (RBF kernel, gamma=0.01, C=5)

---

## PatchGuard (原仓库)

### 未修改原有代码
- 所有 `.py` 文件、模型定义、防御算法**未做任何修改**
- 仅将 `misc/patch_attack.py` 和 `misc/PatchAttacker.py` 复制到项目根目录以便运行 (README 要求)

### 新增文件 (项目根目录)
| 文件 | 说明 |
|------|------|
| `attacks/generate_l2_adv.py` | L2 对抗样本生成 (PGD-L2, 4种ε强度) |
| `attacks/generate_patch_adv.py` | Patch 对抗样本生成 (随机有色方块, 3种尺寸) |
| `analysis/cross_test_ern_on_patch.py` | 交叉测试: E&R 检测 patch 攻击 |
| `analysis/cross_test_pg_on_l2.py` | 交叉测试: PatchGuard 防御 L2 攻击 |
| `analysis/compare_results.py` | 对比汇总脚本 |
| `shared_data/` | 共享数据目录 (对抗样本 + 实验结果) |
| `docs/superpowers/specs/...-design.md` | 实验设计文档 |
| `docs/superpowers/plans/...-plan.md` | 实现计划 |

### 下载的外部资源
- 预训练模型: `checkpoints/bagnet17_nette.pth`, `bagnet33_nette.pth`, `ds_nette.pth` (Google Drive)
- 数据集: ImageNette (full size, ~1.5GB) 放至 `data/imagenette/`

---

## 对比维度

| 维度 | Erase-and-Restore | PatchGuard |
|------|-------------------|------------|
| 论文 | AsiaCCS 2021 | USENIX Security 2021 |
| 防御类型 | 检测 (后处理, 二分类) | 防御 (鲁棒分类 + 可证明) |
| 威胁模型 | L2 约束对抗扰动 (CW/PGD) | 空间局部对抗 patch |
| 核心机制 | 擦除修复 + 预测向量变化 | 小感受野 BagNet + 鲁棒掩码 |
| 可证明性 | 无 (经验性评估) | 有 (provable robustness) |
| 推理开销 | 12× 前向传播/图 | 1× 前向传播 + masking |
| 对干净精度影响 | 无 (后处理, 不影响分类器) | 有 (BagNet 精度 < 标准 ResNet) |
