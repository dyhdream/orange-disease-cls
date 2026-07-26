"""
脐橙病虫害分类 - 训练脚本 (适配 ModelEngine 平台 Ascend 910B)
用法:
  1. 解压数据集: !unzip orange_disease_cls_dataset.zip -d .
  2. 安装依赖:   !pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
  3. 运行训练:   !python train.py
"""
import os, sys, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from argparse import Namespace

# ==================== 设备检测 ====================
def get_device():
    """自动检测可用设备: NPU > CUDA > CPU"""
    try:
        import torch_npu
        if torch.npu.is_available():
            n = torch.npu.device_count()
            print(f"Ascend NPU 可用: {n} 张卡")
            for i in range(n):
                props = torch.npu.get_device_properties(i)
                print(f"  NPU {i}: {props.name}, {props.total_mem/1024**3:.0f}GB")
            return "npu:0"
    except ImportError:
        pass

    if torch.cuda.is_available():
        print(f"CUDA 可用: {torch.cuda.get_device_name(0)}")
        return "cuda:0"

    print("使用 CPU")
    return "cpu"


# ==================== 配置 ====================
DEVICE = get_device()
BATCH = 64 if "npu" in DEVICE else 32    # NPU 64GB 显存可以开大
EPOCHS = 100
LR = 0.005
IMSZ = 224
N_CLASSES = 14
WEIGHT_PATH = "yolo26s-cls.pt"
BEST_PATH = "best.pt"
CKPT_PATH = "checkpoint.pt"

CLASS_NAMES = [
    "S10_Anthracnose", "S11_Citrus_greasy_spot", "S12_Algal_spot_and_moss",
    "S13_Sooty_mould", "S14_Canker", "S1_Healthy_fruit", "S2_Healthy_leaf",
    "S3_Blotchy_mottling", "S4_Red_nose_fruit", "S5_Zinc_deficiency",
    "S6_Vein_yellowing", "S7_Uniform_yellowing", "S8_Magnesium_deficiency",
    "S9_Boron_deficiency",
]


def main():
    # ==================== 加载模型 ====================
    print("\n" + "=" * 50)
    print("加载模型")
    print("=" * 50)

    if not os.path.exists(WEIGHT_PATH):
        print(f"[!] 权重文件不存在: {WEIGHT_PATH}")
        print("    请上传 yolo26s-cls.pt 到当前目录")
        sys.exit(1)

    m = YOLO(WEIGHT_PATH, task="classify")
    model = m.model

    # 替换分类头
    for _, mod in model.named_modules():
        if hasattr(mod, "linear") and mod.linear.out_features == 1000:
            mod.linear = nn.Linear(mod.linear.in_features, N_CLASSES)
            print(f"分类头: {mod.linear.in_features} -> {N_CLASSES}")
            break

    for p in model.parameters():
        p.requires_grad = True
    model.train()
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数: {total_params:,}")
    print(f"设备: {DEVICE}")

    # ==================== 加载数据 ====================
    print("\n" + "=" * 50)
    print("加载数据集")
    print("=" * 50)

    for d in ["dataset/train", "dataset/val"]:
        if not os.path.exists(d):
            print(f"[!] 数据集目录不存在: {d}")
            print("    请先解压: !unzip orange_disease_cls_dataset.zip -d .")
            sys.exit(1)

    args = Namespace(
        fraction=1.0, cache=False, imgsz=IMSZ,
        fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
        perspective=0.0, erasing=0.4, auto_augment="randaugment",
    )

    train_ds = ClassificationDataset("dataset/train", args, augment=True)
    val_ds = ClassificationDataset("dataset/val", args, augment=False)
    train_loader = DataLoader(train_ds, BATCH, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, BATCH, shuffle=False, num_workers=0, pin_memory=True)

    print(f"训练集: {len(train_ds)} 张")
    print(f"验证集: {len(val_ds)} 张")
    print(f"类别: {len(train_ds.base.classes)} 个")
    print(f"batch: {BATCH}, imgsz: {IMSZ}, lr: {LR}")

    # ==================== 训练 ====================
    print("\n" + "=" * 50)
    print(f"开始训练: {EPOCHS} epochs")
    print("=" * 50)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.937, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    t_start = time.time()

    for epoch in range(EPOCHS):
        # ---- 训练 ----
        model.train()
        train_loss = 0
        for bi, batch in enumerate(train_loader):
            imgs = batch["img"].to(DEVICE)
            labels = batch["cls"].to(DEVICE).long().view(-1)

            optimizer.zero_grad()
            preds = model(imgs)
            if isinstance(preds, tuple):
                preds = preds[0]
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)

        # NaN 检测
        if avg_loss != avg_loss:
            print(f"  [NaN] epoch {epoch+1}")
            if os.path.exists(BEST_PATH):
                model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE, weights_only=True))
                for g in optimizer.param_groups:
                    g["lr"] *= 0.5
            continue

        scheduler.step()

        # ---- 验证 ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["img"].to(DEVICE)
                labels = batch["cls"].to(DEVICE).long().view(-1)
                preds = model(imgs)
                if isinstance(preds, tuple):
                    preds = preds[0]
                correct += preds.argmax(1).eq(labels).sum().item()
                total += labels.size(0)

        acc = 100.0 * correct / total
        elapsed = (time.time() - t_start) / 60
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:3d}/{EPOCHS}  loss={avg_loss:.4f}  val={acc:5.2f}%  lr={lr_now:.6f}  {elapsed:.1f}min")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), BEST_PATH)
            print(f"  [Best] {best_acc:.2f}%")

        # 断点
        torch.save({
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        }, CKPT_PATH)

    # ==================== 测试集评估 ====================
    if os.path.exists("dataset/test"):
        print("\n" + "=" * 50)
        print("测试集评估")
        print("=" * 50)

        model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE, weights_only=True))
        model.eval()

        test_ds = ClassificationDataset("dataset/test", args, augment=False)
        test_loader = DataLoader(test_ds, BATCH, shuffle=False, num_workers=0)

        correct, total = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                imgs = batch["img"].to(DEVICE)
                labels = batch["cls"].to(DEVICE).long().view(-1)
                preds = model(imgs)
                if isinstance(preds, tuple):
                    preds = preds[0]
                correct += preds.argmax(1).eq(labels).sum().item()
                total += labels.size(0)

        test_acc = 100.0 * correct / total
        print(f"测试集: {correct}/{total} = {test_acc:.2f}%")

    print(f"\n完成! 验证集最佳: {best_acc:.2f}%  耗时: {(time.time()-t_start)/60:.1f}min")
    print(f"模型已保存: {BEST_PATH}")

    # 清理 checkpoint
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)


if __name__ == "__main__":
    main()
