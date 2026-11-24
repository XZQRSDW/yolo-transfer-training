# ==============================================================================
# 基于 Ultralytics 的 YOLO11s 两阶段迁移学习训练脚本（COCO 80 类 -> 9 种食材）
#
# 学习提示（配合本脚本拆解 ultralytics 底层实现）：
#   1. 建议先读底部 main() 掌握两阶段迁移训练的整体流程，再逐个阅读上方函数。
#   2. 在编辑器中按住 Ctrl + 左键点击 YOLO / model.train / model.predict /
#      model.export，可追踪到底层 PyTorch 训练循环、数据加载、冻结与评估实现，
#      作为编写 train.py 的对照参考。
# ==============================================================================

from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


# ==============================================================================
# 1. 路径与超参数配置
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_YAML = SCRIPT_DIR / "dataset.yaml"
OUTPUT_DIR = SCRIPT_DIR / "runs" / "ultralytics"  # runs/ 已在 .gitignore 中忽略

# 预训练权重：若当前目录已有 yolo11s.pt 则直接使用，否则由 ultralytics 自动下载
LOCAL_PRETRAINED = SCRIPT_DIR / "yolo11s.pt"
PRETRAINED_WEIGHTS = str(LOCAL_PRETRAINED) if LOCAL_PRETRAINED.exists() else "yolo11s.pt"

# 基础训练配置
SEED = 42
IMG_SIZE = 640
BATCH_SIZE = 8
NUM_WORKERS = 0  # Windows 调试阶段建议为 0；GPU 正式训练可按需调整为 2 或 4
DEVICE = 0 if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available()

# 阶段一：冻结 Backbone（前 11 层：0~10 层），仅训练 Neck（11~22）与新 9 类检测头（23）
FREEZE_LAYERS = 11
FREEZE_EPOCHS = 10
FREEZE_LR = 1e-3

# 阶段二：解冻全网（0~23 层全部参与微调），使用较小学习率精调
FINETUNE_EPOCHS = 40
FINETUNE_LR = 1e-4

# 优化器、学习率衰减与早停配置
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 2.0
PATIENCE = 15

# 推理验货与部署导出配置
CONF_THRES = 0.25
IOU_THRES = 0.45
EXPORT_ONNX = False  # 设为 True 可在训练结束后自动导出 best.onnx

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ==============================================================================
# 2. 启动前自检：解析 dataset.yaml 并确认数据集就绪
# ==============================================================================
def find_images(image_dir: Path) -> list[Path]:
    """扫描指定目录下支持格式的图片文件并按路径排序。"""
    if not image_dir.exists():
        return []
    return sorted(
        p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def resolve_dataset_config(dataset_yaml: Path) -> tuple[Path, Path, int]:
    """解析 dataset.yaml 中的训练集目录、验证集目录和类别数。"""
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"未找到数据集配置文件: {dataset_yaml}")

    with dataset_yaml.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    for key in ("train", "val"):
        if not config.get(key):
            raise ValueError(f"{dataset_yaml} 缺少必需字段: '{key}'")

    names = config.get("names")
    nc = config.get("nc")
    if names is None and nc is None:
        raise ValueError(f"{dataset_yaml} 必须包含 'names' 或 'nc' 字段")
    if names is not None and nc is not None and len(names) != nc:
        raise ValueError(f"{dataset_yaml} 中 nc={nc} 与 names 数量 ({len(names)}) 不一致")
    num_classes = len(names) if names is not None else int(nc)

    raw_path = config.get("path")
    if raw_path:
        base_dir = Path(raw_path)
        if not base_dir.is_absolute():
            base_dir = (dataset_yaml.parent / base_dir).resolve()
    else:
        base_dir = dataset_yaml.parent.resolve()

    train_dir = (base_dir / str(config["train"])).resolve()
    val_dir = (base_dir / str(config["val"])).resolve()
    return train_dir, val_dir, num_classes


def validate_ready(dataset_yaml: Path) -> tuple[list[Path], list[Path]]:
    """在训练前校验配置文件与训练/验证集图片是否准备完成。"""
    train_dir, val_dir, num_classes = resolve_dataset_config(dataset_yaml)
    train_images = find_images(train_dir)
    val_images = find_images(val_dir)

    if not train_images or not val_images:
        raise RuntimeError(
            "数据集尚未准备好，请按 dataset.yaml 配置放入图片和对应 YOLO 标签后再运行：\n"
            f"  训练集图片目录: {train_dir}（当前 {len(train_images)} 张）\n"
            f"  验证集图片目录: {val_dir}（当前 {len(val_images)} 张）"
        )

    print(
        f"[检查通过] 设备={DEVICE} | AMP={USE_AMP} | 类别数={num_classes} | "
        f"训练集={len(train_images)} 张 | 验证集={len(val_images)} 张"
    )
    return train_images, val_images


# ==============================================================================
# 3. 阶段一：冻结前 11 层 Backbone，快速把新初始化的 9 类检测头练稳
# ==============================================================================
def train_stage1_freeze(
    pretrained_weights: str,
    dataset_yaml: Path,
    output_dir: Path,
    epochs: int = FREEZE_EPOCHS,
) -> Path:
    """加载 COCO 80 类预训练权重，冻结前 FREEZE_LAYERS 层并训练 Neck 与 9 类检测头。"""
    print("\n" + "=" * 60)
    print(f"[阶段一] 冻结前 {FREEZE_LAYERS} 层训练检测头 | epochs={epochs} | lr0={FREEZE_LR}")
    print("=" * 60)

    # 源码追踪入口 1：YOLO(...) -> ultralytics/engine/model.py (Model.__init__)
    model = YOLO(pretrained_weights)

    # 源码追踪入口 2：model.train(...) ->
    #   engine/model.py (Model.train)
    #   -> models/yolo/detect/train.py (DetectionTrainer)
    #   -> engine/trainer.py (BaseTrainer._setup_train & _do_train)
    # Ultralytics 会根据 dataset.yaml 自动将检测头重建为 9 类，并复用其余预训练权重
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=NUM_WORKERS,
        seed=SEED,
        amp=USE_AMP,
        optimizer=OPTIMIZER,
        lr0=FREEZE_LR,
        lrf=0.01,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        cos_lr=True,
        freeze=FREEZE_LAYERS,  # 底层在 BaseTrainer._setup_train 中冻结 model.0~model.10
        patience=PATIENCE,
        project=str(output_dir),
        name="stage1_freeze",
        exist_ok=True,
    )

    best_path = Path(model.trainer.best)
    last_path = Path(model.trainer.last)
    stage1_ckpt = best_path if best_path.exists() else last_path
    print(f"[阶段一完成] 产出权重: {stage1_ckpt}")
    return stage1_ckpt


# ==============================================================================
# 4. 阶段二：加载阶段一最优权重，全网解冻并小学习率端到端微调
# ==============================================================================
def train_stage2_finetune(
    stage1_ckpt: Path,
    dataset_yaml: Path,
    output_dir: Path,
    epochs: int = FINETUNE_EPOCHS,
) -> Path:
    """基于阶段一权重解冻全部网络层，以较小学习率端到端微调。"""
    print("\n" + "=" * 60)
    print(f"[阶段二] 全网解冻微调 | epochs={epochs} | lr0={FINETUNE_LR}")
    print("=" * 60)

    model = YOLO(str(stage1_ckpt))

    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=NUM_WORKERS,
        seed=SEED,
        amp=USE_AMP,
        optimizer=OPTIMIZER,
        lr0=FINETUNE_LR,
        lrf=0.01,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        cos_lr=True,
        freeze=None,  # 全网解冻，0~23 层全部参与梯度更新（.dfl 常量投影层除外）
        patience=PATIENCE,
        project=str(output_dir),
        name="stage2_finetune",
        exist_ok=True,
    )

    best_path = Path(model.trainer.best)
    last_path = Path(model.trainer.last)
    stage2_ckpt = best_path if best_path.exists() else last_path
    print(f"[阶段二完成] 最终最优权重: {stage2_ckpt}")
    return stage2_ckpt


# ==============================================================================
# 5. 推理验货与可选部署导出
# ==============================================================================
def infer_and_export(
    best_ckpt: Path,
    sample_image: Path,
    output_dir: Path,
    export_onnx: bool = EXPORT_ONNX,
) -> None:
    """使用最终权重对单张验证集图片进行推理并保存可视化结果，按需导出 ONNX。"""
    print("\n" + "=" * 60)
    print(f"[推理验货] 权重={best_ckpt.name} | 样本图={sample_image.name}")
    print("=" * 60)

    best_model = YOLO(str(best_ckpt))

    # 源码追踪入口 3：model.predict(...) -> engine/predictor.py（预处理 -> 前向 -> NMS -> 画框）
    results = best_model.predict(
        source=str(sample_image),
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device=DEVICE,
        save=True,
        project=str(output_dir),
        name="infer_check",
        exist_ok=True,
    )

    boxes = results[0].boxes
    print(f"[推理验货完成] 检测到 {len(boxes)} 个目标，可视化结果已保存至: {output_dir / 'infer_check'}")

    if export_onnx:
        # 源码追踪入口 4：model.export(...) -> engine/exporter.py
        onnx_path = best_model.export(format="onnx", imgsz=IMG_SIZE, simplify=True)
        print(f"[部署导出完成] ONNX 模型路径: {onnx_path}")


# ==============================================================================
# 6. 主流程入口
# ==============================================================================
def main() -> None:
    """串联：启动前自检 -> 阶段一冻结练头 -> 阶段二全网微调 -> 推理验货。"""
    _train_images, val_images = validate_ready(DATASET_YAML)

    stage1_ckpt = train_stage1_freeze(
        pretrained_weights=PRETRAINED_WEIGHTS,
        dataset_yaml=DATASET_YAML,
        output_dir=OUTPUT_DIR,
        epochs=FREEZE_EPOCHS,
    )

    final_ckpt = train_stage2_finetune(
        stage1_ckpt=stage1_ckpt,
        dataset_yaml=DATASET_YAML,
        output_dir=OUTPUT_DIR,
        epochs=FINETUNE_EPOCHS,
    )

    infer_and_export(
        best_ckpt=final_ckpt,
        sample_image=val_images[0],
        output_dir=OUTPUT_DIR,
        export_onnx=EXPORT_ONNX,
    )


if __name__ == "__main__":
    main()
