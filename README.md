# YOLO11s 迁移训练学习日志

基于 Ultralytics 官方在 COCO（80 类）上预训练的 **YOLO11s** 权重进行两阶段迁移微调，最终得到可识别 **9 种食材**的检测模型。

- **预训练权重**：[`yolo11s.pt`](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt)（[官方 Releases](https://github.com/ultralytics/assets/releases)）

## 学习思路

> 看优秀开源库（ultralytics）是怎么写的、对比着拆解并仿写，是入门工程代码最快的方式。

`ultralytics` 是封装在 PyTorch 之上的中间层。本仓库采用两步对照学习：
1. 先用 `train_ultralytics.py` 跑通高层封装流程；
2. 再拆解 `ultralytics` 底层的 PyTorch 实现，并在 `train.py` 中亲手仿写（数据加载、权重迁移、冻结/解冻、训练循环、mAP 评估与推理）。

## 文件说明

- `train_ultralytics.py`：基于 `ultralytics` 的两阶段迁移训练脚本（冻结前 11 层 → 全网微调）
- `dataset.yaml`：9 类食材数据集配置
- `train.py`：个人仿写的 PyTorch 底层训练代码
