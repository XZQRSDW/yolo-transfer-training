# YOLO11s 迁移训练学习日志

基于 Ultralytics 官方在 COCO（80 类）上预训练的 **YOLO11s** 权重进行两阶段迁移微调，最终得到可识别 **9 种食材**的检测模型。

- **预训练权重**：[`yolo11s.pt`](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt)（[官方 Releases](https://github.com/ultralytics/assets/releases)）

## 学习思路

> 看优秀开源库（ultralytics）是怎么写的、对比着拆解并仿写，是入门工程代码最快的方式。

`ultralytics` 是封装在 PyTorch 之上的中间层。本仓库采用**两步对照 + 需求倒推**的学习方式：
1. **高层跑通**：先用 `train_ultralytics.py` 跑通两阶段迁移训练流程，并通过跳转源码观察 `ultralytics` 的调用链；
2. **底层仿写（需求倒推法）**：在 `train.py` 中亲手用 PyTorch 仿写底层实现（数据加载、权重迁移、冻结/解冻、训练循环、mAP 评估与推理）。编写时不从头盲目堆砌代码与依赖，而是**自顶向下从 `main()` 的训练主线出发**：
   - 按“环境初始化 → 数据管道 → 模型与权重迁移 → 阶段一冻结练头 → 阶段二全网微调 → 推理验货”的顺序拆分为小步；
   - 每一步根据 `main()` 的实际需要，**倒推**去前面补写对应的常量、工具函数或类；
   - **按需导入依赖**，明确每一条 `import` 是因为哪一行代码而引入，每完成一小步即刻验证，确保理解每一行代码的作用。

## 文件说明

- `train_ultralytics.py`：基于 `ultralytics` 的两阶段迁移训练脚本（冻结前 11 层 → 全网微调）
- `dataset.yaml`：9 类食材数据集配置
- `train.py`：个人仿写的 PyTorch 底层训练代码
