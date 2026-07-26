# 脐橙病虫害分类模型

基于 YOLO26s-cls 的脐橙 14 类病虫害图像分类模型。

## 病害类别

| ID | 英文名 | 中文名 | 分类 |
|----|--------|--------|------|
| S1 | Healthy fruit | 健康果实 | 健康 |
| S2 | Healthy leaf | 健康叶片 | 健康 |
| S3 | Blotchy mottling | 斑驳花纹 | 黄龙病(早期) |
| S4 | Red nose fruit | 红鼻果 | 黄龙病(特征) |
| S5 | Zinc deficiency | 缺锌 | 缺素症 |
| S6 | Vein yellowing | 叶脉黄化 | 缺素症 |
| S7 | Uniform yellowing | 均匀黄化 | 缺素症 |
| S8 | Magnesium deficiency | 缺镁 | 缺素症 |
| S9 | Boron deficiency | 缺硼 | 缺素症 |
| S10 | Anthracnose | 炭疽病 | 真菌病害 |
| S11 | Citrus greasy spot | 柑橘脂点黄斑病 | 真菌病害 |
| S12 | Algal spot and moss | 藻斑与苔藓 | 藻类/苔藓 |
| S13 | Sooty mould | 煤污病 | 真菌病害 |
| S14 | Canker | 溃疡病 | 细菌病害 |

## 训练结果

| 指标 | 数值 |
|------|------|
| 验证集精度 | 98.44% |
| 测试集精度 | 81.49% |
| 模型大小 | 22MB |
| 训练设备 | RTX 3060 Laptop (6GB) |

## 使用方法

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- ultralytics 8.4+

### 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 数据准备

1. 下载数据集: `orange_disease_cls_dataset.zip`
2. 解压到项目目录: `unzip orange_disease_cls_dataset.zip -d .`
3. 确保目录结构:
   ```
   dataset/
   ├── train/
   │   ├── S1_Healthy_fruit/
   │   ├── S2_Healthy_leaf/
   │   └── ... (14 个类别)
   ├── val/
   │   └── ...
   └── test/
       └── ...
   ```

### 权重文件

1. 下载预训练权重: `yolo26s-cls.pt` (13MB)
2. 放到项目根目录

### 训练

```bash
python train.py
```

训练完成后会生成 `best.pt` 模型文件。

### 推理

```bash
python predict.py <图片路径>
```

输出 JSON 格式的病害概率，可直接接入 RAG 系统。

## 模型架构

- 基础模型: YOLO26s-cls (Ultralytics)
- 分类头: Linear(1280, 14)
- 训练策略: SGD + CosineAnnealing + 梯度裁剪
- 数据增强: RandomHorizontalFlip + HSV抖动 + 随机擦除

## 文件说明

| 文件 | 说明 |
|------|------|
| `train.py` | 训练脚本 (支持 NPU/CUDA/CPU) |
| `predict.py` | 推理脚本 |
| `requirements.txt` | Python 依赖 |
| `yolo26s-cls.pt` | 预训练权重 (需下载) |
| `best.pt` | 训练后的最佳模型 |

## License

仅用于学术研究和内部使用。
