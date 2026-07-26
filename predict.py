"""
脐橙病虫害分类 - 推理脚本
用法: python predict.py <图片路径>
"""
import os, sys, json
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.data.augment import classify_transforms
from PIL import Image

N_CLASSES = 14
IMG_SIZE = 224

CLASS_NAMES = [
    "S10_Anthracnose", "S11_Citrus_greasy_spot", "S12_Algal_spot_and_moss",
    "S13_Sooty_mould", "S14_Canker", "S1_Healthy_fruit", "S2_Healthy_leaf",
    "S3_Blotchy_mottling", "S4_Red_nose_fruit", "S5_Zinc_deficiency",
    "S6_Vein_yellowing", "S7_Uniform_yellowing", "S8_Magnesium_deficiency",
    "S9_Boron_deficiency",
]

CN_NAMES = {
    "S1_Healthy_fruit": "健康果实", "S2_Healthy_leaf": "健康叶片",
    "S3_Blotchy_mottling": "斑驳花纹(黄龙病早期)", "S4_Red_nose_fruit": "红鼻果(黄龙病特征)",
    "S5_Zinc_deficiency": "缺锌", "S6_Vein_yellowing": "叶脉黄化",
    "S7_Uniform_yellowing": "均匀黄化", "S8_Magnesium_deficiency": "缺镁",
    "S9_Boron_deficiency": "缺硼", "S10_Anthracnose": "炭疽病",
    "S11_Citrus_greasy_spot": "柑橘脂点黄斑病", "S12_Algal_spot_and_moss": "藻斑与苔藓",
    "S13_Sooty_mould": "煤污病", "S14_Canker": "溃疡病",
}

CATEGORY = {
    "S1_Healthy_fruit": "健康", "S2_Healthy_leaf": "健康",
    "S3_Blotchy_mottling": "黄龙病", "S4_Red_nose_fruit": "黄龙病",
    "S5_Zinc_deficiency": "缺素症", "S6_Vein_yellowing": "缺素症",
    "S7_Uniform_yellowing": "缺素症", "S8_Magnesium_deficiency": "缺素症",
    "S9_Boron_deficiency": "缺素症", "S10_Anthracnose": "真菌病害",
    "S11_Citrus_greasy_spot": "真菌病害", "S12_Algal_spot_and_moss": "藻类/苔藓",
    "S13_Sooty_mould": "真菌病害", "S14_Canker": "细菌病害",
}


def get_device():
    try:
        import torch_npu
        if torch.npu.is_available():
            return "npu:0"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def load_model(weight_path="best.pt", base_path="yolo26s-cls.pt", device="cpu"):
    m = YOLO(base_path, task="classify")
    model = m.model
    for _, mod in model.named_modules():
        if hasattr(mod, "linear") and mod.linear.out_features == 1000:
            mod.linear = nn.Linear(mod.linear.in_features, N_CLASSES)
            break
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def predict(model, image_path, device="cpu"):
    transform = classify_transforms(size=IMG_SIZE)
    im = Image.open(image_path).convert("RGB")
    x = transform(im).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(x)
        if isinstance(preds, tuple):
            preds = preds[0]
        probs = preds[0]

    results = []
    for i, name in enumerate(CLASS_NAMES):
        p = float(probs[i])
        results.append({
            "class_id": i, "name_en": name,
            "name_cn": CN_NAMES[name], "category": CATEGORY[name],
            "probability": round(p, 6),
        })
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results


def print_results(image_path, results):
    top1 = results[0]
    print(f"\n{'='*60}")
    print(f"  图片: {os.path.basename(image_path)}")
    print(f"  预测: {top1['name_cn']} ({top1['probability']*100:.1f}%)")
    print(f"  分类: {top1['category']}")
    print(f"{'='*60}")
    print(f"  {'类别':<20} {'概率':>8}  概率分布")
    print(f"  {'-'*20} {'-'*8}   {'-'*30}")
    for r in results:
        p = r["probability"]
        bar = "#" * int(p * 40)
        marker = " <--" if r is results[0] else ""
        print(f"  {r['name_cn']:<18} {p*100:>7.1f}%  {bar}{marker}")

    rag_json = {
        "image": os.path.basename(image_path),
        "top_prediction": {
            "name_cn": top1["name_cn"], "name_en": top1["name_en"],
            "category": top1["category"], "probability": top1["probability"],
        },
        "all_probabilities": {r["name_en"]: r["probability"] for r in results},
    }
    print(f"\n  JSON (for RAG):")
    print(f"  {json.dumps(rag_json, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    device = get_device()
    print(f"设备: {device}")

    model = load_model(device=device)
    print("模型加载完成")

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("请输入图片路径: ").strip().strip('"')

    if not image_path or not os.path.exists(image_path):
        print(f"图片不存在: {image_path}")
        sys.exit(1)

    results = predict(model, image_path, device)
    print_results(image_path, results)
