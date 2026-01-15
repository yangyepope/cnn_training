# CNN 颜值预测模型

基于 **SCUT-FBP5500** 数据集，使用 **CNN (卷积神经网络)** 训练颜值预测模型。

## 📁 项目结构

```
cnn_training/
├── README.md           # 项目说明文档（你正在看的这个）
├── requirements.txt    # Python依赖包列表
├── .gitignore          # Git忽略文件配置
├── config.py           # 配置文件（路径、超参数）
├── dataset.py          # 数据集加载模块
├── model.py            # CNN模型定义
├── train.py            # 训练脚本
├── predict.py          # 预测脚本
├── data/               # 数据目录（需要自己创建并放入数据）
│   └── SCUT-FBP5500/
│       ├── Images/           # 5500张人脸图片
│       └── All_Ratings.xlsx  # 颜值评分文件
└── models/             # 模型保存目录
    └── beauty_cnn.pth  # 训练好的模型（训练后生成）
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

或者手动安装：
```bash
pip install torch torchvision pandas openpyxl pillow numpy
```

### 2. 准备数据

下载 SCUT-FBP5500 数据集，将数据放到正确位置：

```
data/SCUT-FBP5500/
├── Images/              # 放入5500张图片
│   ├── AF1.jpg
│   ├── AF2.jpg
│   └── ...
└── All_Ratings.xlsx     # 放入评分文件
```

### 3. 训练模型

```bash
# 使用默认参数训练（MobileNet，20轮）
python train.py

# 自定义参数
python train.py --epochs 30 --batch_size 8 --model resnet18
```

### 4. 预测新图片

```bash
# 预测单张图片
python predict.py test.jpg

# 预测多张图片
python predict.py image1.jpg image2.jpg

# 预测整个文件夹
python predict.py --dir /path/to/images
```

## ⚙️ 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 20 | 训练轮数 |
| `--batch_size` | 16 | 每批处理的图片数量 |
| `--lr` | 0.001 | 学习率 |
| `--model` | mobilenet | 模型选择：mobilenet / resnet18 / resnet34 |

## 🖥️ 模型选择建议

| 模型 | 参数量 | CPU训练时间(每轮) | 适用场景 |
|------|--------|------------------|---------|
| **MobileNet** | 3.4M | 5-10分钟 | CPU训练首选，轻量快速 |
| **ResNet18** | 11M | 10-15分钟 | 效果更好，需要更多时间 |
| **ResNet34** | 21M | 15-25分钟 | 效果最好，训练最慢 |

**建议**：如果用 CPU 训练，推荐使用 MobileNet。

## 📊 评估指标说明

训练过程中会显示以下指标：

| 指标 | 含义 | 目标值 |
|------|------|--------|
| **MAE** | 平均绝对误差 | 越小越好（< 0.3 较好） |
| **RMSE** | 均方根误差 | 越小越好 |
| **相关系数** | 预测与真实值的相关性 | 越接近 1 越好（> 0.85 较好） |

## 📝 代码文件说明

| 文件 | 功能 | 主要内容 |
|------|------|---------|
| `config.py` | 配置文件 | 数据路径、训练参数 |
| `dataset.py` | 数据加载 | BeautyDataset类、数据增强 |
| `model.py` | 模型定义 | BeautyModel类（支持多种骨干网络） |
| `train.py` | 训练脚本 | 训练循环、验证、模型保存 |
| `predict.py` | 预测脚本 | 加载模型、预测新图片 |

## ❓ 常见问题

### Q: 训练时内存不足怎么办？
A: 减小 `batch_size`，例如 `--batch_size 8` 或 `--batch_size 4`

### Q: 训练太慢怎么办？
A: 
- 使用 MobileNet（最轻量）
- 减少训练轮数 `--epochs 10`
- 如果有 NVIDIA GPU，安装 CUDA 版 PyTorch

### Q: 如何继续训练？
A: 目前不支持断点续训，需要从头开始。可以修改代码加载已保存的模型继续训练。

### Q: 预测结果不准怎么办？
A:
- 确保输入图片是正面人脸照片
- 尝试使用更大的模型（ResNet18/34）
- 增加训练轮数

## 📚 参考资料

- [SCUT-FBP5500 数据集](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [迁移学习教程](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## 📄 License

MIT License
