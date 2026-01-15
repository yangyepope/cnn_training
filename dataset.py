"""
数据集模块
==========
定义了 BeautyDataset 类，用于加载 SCUT-FBP5500 数据集

主要功能：
1. 读取评分文件（Excel/CSV）
2. 加载图片并进行预处理
3. 提供给 PyTorch DataLoader 使用

核心概念：
---------
PyTorch 的数据加载流程：
    Dataset（定义如何获取单个样本） 
        ↓
    DataLoader（将多个样本组成batch，打乱顺序，多线程加载）
        ↓
    训练循环（for batch in dataloader）

我们需要继承 torch.utils.data.Dataset 类，并实现三个方法：
    - __init__: 初始化，读取数据文件
    - __len__: 返回数据集大小
    - __getitem__: 根据索引返回一个样本
"""

# ==============================================================================
# 导入必要的库
# ==============================================================================

import os                          # 操作系统接口，用于文件路径操作
import pandas as pd                # 数据处理库，用于读取Excel文件
import torch                       # PyTorch核心库
from torch.utils.data import Dataset  # 数据集基类
from PIL import Image              # Python图像库，用于读取图片
from torchvision import transforms # 图像变换库，用于数据增强和预处理

# 导入配置
from config import IMAGE_SIZE


# ==============================================================================
# 数据集类定义
# ==============================================================================

class BeautyDataset(Dataset):
    """
    颜值预测数据集类
    
    这个类负责：
    1. 读取评分文件，获取每张图片的颜值评分
    2. 根据索引加载对应的图片
    3. 对图片进行预处理（缩放、归一化等）
    4. 返回（图片张量，评分）的元组供训练使用
    
    参数说明：
    ---------
    images_dir : str
        图片文件夹的路径，里面存放着所有人脸图片
        
    ratings_file : str
        评分文件的路径（Excel或CSV格式）
        文件应包含两列：图片文件名、颜值评分
        
    transform : torchvision.transforms.Compose, 可选
        图像预处理/数据增强的变换操作
        如果为None，则不做任何变换（不推荐）
    
    使用示例：
    ---------
    >>> dataset = BeautyDataset(
    ...     images_dir="data/Images",
    ...     ratings_file="data/All_Ratings.xlsx",
    ...     transform=train_transform
    ... )
    >>> image, rating = dataset[0]  # 获取第一个样本
    >>> len(dataset)  # 获取数据集大小
    5500
    """
    
    def __init__(self, images_dir: str, ratings_file: str, transform=None):
        """
        初始化数据集
        
        这个方法在创建数据集对象时自动调用，负责：
        1. 保存传入的参数
        2. 读取评分文件
        3. 匹配图片和评分，构建数据列表
        """
        
        # ----------------------------------------------------------------------
        # 保存参数到实例变量
        # ----------------------------------------------------------------------
        # self.xxx 表示这是实例变量，在类的其他方法中也可以访问
        
        self.images_dir = images_dir    # 保存图片目录路径
        self.transform = transform       # 保存变换操作
        
        # ----------------------------------------------------------------------
        # 读取评分文件
        # ----------------------------------------------------------------------
        # pandas 可以读取多种格式的表格文件
        # 根据文件扩展名选择不同的读取方式
        
        if ratings_file.endswith('.xlsx'):
            # Excel 2007及以后的格式（.xlsx）
            # 需要安装 openpyxl 库：pip install openpyxl
            df = pd.read_excel(ratings_file)
        else:
            # 其他格式（如.csv）
            df = pd.read_csv(ratings_file)
        
        # 打印列名，方便调试和确认数据格式
        # df.columns 返回所有列名的列表
        print(f"[数据集] 评分文件列名: {list(df.columns)}")
        
        # ----------------------------------------------------------------------
        # 自动识别列名
        # ----------------------------------------------------------------------
        # 不同数据集的列名可能不同，这里自动查找包含特定关键词的列
        # 这样代码就不用写死列名，更通用
        
        filename_col = None   # 用于存储文件名所在的列名
        rating_col = None     # 用于存储评分所在的列名
        
        # 遍历所有列名
        for col in df.columns:
            # 转为小写，方便不区分大小写地匹配
            col_lower = col.lower()
            
            # 查找文件名列：包含'file'、'name'或'image'关键词的列
            if 'file' in col_lower or 'name' in col_lower or 'image' in col_lower:
                filename_col = col
            
            # 查找评分列：包含'rating'、'score'或'beauty'关键词的列
            if 'rating' in col_lower or 'score' in col_lower or 'beauty' in col_lower:
                rating_col = col
        
        # 如果自动识别失败，就使用默认的第一列和第二列
        if filename_col is None or rating_col is None:
            filename_col = df.columns[0]   # 第一列作为文件名
            rating_col = df.columns[1]      # 第二列作为评分
        
        print(f"[数据集] 使用列: 文件名列='{filename_col}', 评分列='{rating_col}'")
        
        # ----------------------------------------------------------------------
        # 构建数据列表
        # ----------------------------------------------------------------------
        # 遍历评分文件的每一行，检查对应的图片是否存在
        # 只保留存在的图片，跳过缺失的
        
        self.data = []          # 存储 (图片路径, 评分) 元组的列表
        missing_count = 0       # 统计缺失的图片数量
        
        # df.iterrows() 返回 (索引, 行数据) 的迭代器
        # 这里用 _ 表示我们不需要索引
        for _, row in df.iterrows():
            # 获取当前行的文件名和评分
            filename = str(row[filename_col])    # 转为字符串，防止数字类型
            rating = float(row[rating_col])      # 转为浮点数
            
            # 拼接完整的图片路径
            # os.path.join 会根据操作系统自动处理路径分隔符（Windows用\，Linux用/）
            img_path = os.path.join(images_dir, filename)
            
            # 检查图片文件是否真的存在
            if os.path.exists(img_path):
                # 存在就添加到数据列表
                self.data.append((img_path, rating))
            else:
                # 不存在就计数
                missing_count += 1
        
        # 打印加载结果
        print(f"[数据集] 成功加载: {len(self.data)} 张图片")
        if missing_count > 0:
            print(f"[数据集] ⚠️ 缺失图片: {missing_count} 张")
    
    def __len__(self):
        """
        返回数据集的大小（样本总数）
        
        这个方法是 Dataset 基类要求实现的
        DataLoader 需要知道数据集有多少样本，才能正确地分batch
        
        返回：
        ------
        int : 数据集中的样本数量
        
        示例：
        ------
        >>> len(dataset)
        5500
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        根据索引获取一个样本
        
        这个方法是 Dataset 基类要求实现的
        当你用 dataset[i] 访问数据集时，Python会自动调用这个方法
        DataLoader 也会调用这个方法来获取数据
        
        参数：
        ------
        idx : int
            样本的索引，范围是 0 到 len(dataset)-1
        
        返回：
        ------
        image : torch.Tensor
            预处理后的图片张量
            形状为 (3, 224, 224)，表示 (通道数, 高度, 宽度)
            3个通道分别是 R（红）、G（绿）、B（蓝）
            
        rating : torch.Tensor
            归一化后的评分张量
            形状为 ()，即标量（0维张量）
            值在 0 到 1 之间（原始1-5分被归一化）
        
        示例：
        ------
        >>> image, rating = dataset[0]
        >>> image.shape
        torch.Size([3, 224, 224])
        >>> rating
        tensor(0.5625)  # 对应原始3.25分
        """
        
        # 从数据列表中获取该索引对应的图片路径和评分
        img_path, rating = self.data[idx]
        
        # ----------------------------------------------------------------------
        # 读取图片
        # ----------------------------------------------------------------------
        # PIL.Image.open() 打开图片文件
        # .convert('RGB') 确保图片是RGB格式（3通道）
        # 有些图片可能是灰度图（1通道）或RGBA（4通道，带透明度）
        # 统一转换为RGB可以保证输入格式一致
        
        image = Image.open(img_path).convert('RGB')
        
        # ----------------------------------------------------------------------
        # 应用图像变换
        # ----------------------------------------------------------------------
        # 如果定义了transform，就对图片进行变换
        # 变换可能包括：缩放、翻转、旋转、归一化等
        
        if self.transform:
            image = self.transform(image)
        
        # ----------------------------------------------------------------------
        # 评分归一化
        # ----------------------------------------------------------------------
        # 原始评分范围是 1-5 分
        # 归一化到 0-1 范围，公式：(rating - 1) / 4
        # 
        # 为什么要归一化？
        # 1. 神经网络输出层用了Sigmoid激活函数，输出范围就是0-1
        # 2. 归一化后的值更容易训练（梯度更稳定）
        # 
        # 对应关系：
        #   1分 -> 0.00
        #   2分 -> 0.25
        #   3分 -> 0.50
        #   4分 -> 0.75
        #   5分 -> 1.00
        
        rating_normalized = (rating - 1.0) / 4.0
        
        # ----------------------------------------------------------------------
        # 返回结果
        # ----------------------------------------------------------------------
        # 将评分转换为PyTorch张量
        # dtype=torch.float32 指定数据类型为32位浮点数
        # 这与神经网络的默认计算类型一致
        
        return image, torch.tensor(rating_normalized, dtype=torch.float32)


# ==============================================================================
# 数据变换函数
# ==============================================================================

def get_train_transform():
    """
    获取训练集的数据变换
    
    训练时使用数据增强（Data Augmentation）技术：
    - 随机翻转、旋转、颜色变化等
    - 增加数据多样性，防止过拟合
    - 让模型学到更鲁棒的特征
    
    返回：
    ------
    transform : torchvision.transforms.Compose
        组合多个变换操作的对象
    """
    
    return transforms.Compose([
        # 1. 调整图片大小到 224x224
        # 预训练模型（ImageNet）使用的标准尺寸
        # Resize 会保持图片比例，可能会有变形
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        
        # 2. 随机水平翻转
        # 50%的概率左右翻转图片
        # 人脸左右翻转后颜值应该不变，这是合理的增强
        transforms.RandomHorizontalFlip(),
        
        # 3. 随机旋转
        # 在 -10度 到 +10度 之间随机旋转
        # 模拟拍照时头部轻微倾斜的情况
        transforms.RandomRotation(10),
        
        # 4. 颜色抖动
        # 随机调整亮度和对比度
        # brightness=0.2 表示亮度可变化 ±20%
        # contrast=0.2 表示对比度可变化 ±20%
        # 模拟不同光照条件
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        
        # 5. 转换为张量
        # PIL图片格式：(H, W, C)，值范围 0-255
        # PyTorch张量格式：(C, H, W)，值范围 0-1
        # ToTensor 会自动完成这两个转换
        transforms.ToTensor(),
        
        # 6. 标准化（归一化）
        # 使用 ImageNet 数据集的均值和标准差
        # 公式：output = (input - mean) / std
        # 这是因为预训练模型是在ImageNet上训练的，使用相同的标准化
        # mean: RGB三个通道的均值 [0.485, 0.456, 0.406]
        # std: RGB三个通道的标准差 [0.229, 0.224, 0.225]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transform():
    """
    获取验证集的数据变换
    
    验证时不使用数据增强，只做必要的预处理：
    - 缩放到固定尺寸
    - 转为张量
    - 标准化
    
    返回：
    ------
    transform : torchvision.transforms.Compose
        组合多个变换操作的对象
    """
    
    return transforms.Compose([
        # 只做必要的预处理，不做随机增强
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ==============================================================================
# 测试代码
# ==============================================================================
# 当直接运行这个文件时（python dataset.py），会执行下面的测试代码
# 当被其他文件导入时（from dataset import ...），不会执行

if __name__ == "__main__":
    """
    测试数据集加载是否正常
    """
    from config import IMAGES_DIR, RATINGS_FILE
    
    print("=" * 60)
    print("测试数据集加载")
    print("=" * 60)
    
    # 创建数据集（使用训练变换）
    transform = get_train_transform()
    dataset = BeautyDataset(IMAGES_DIR, RATINGS_FILE, transform=transform)
    
    # 测试获取样本
    if len(dataset) > 0:
        image, rating = dataset[0]
        print(f"\n第一个样本:")
        print(f"  图片形状: {image.shape}")  # 应该是 torch.Size([3, 224, 224])
        print(f"  评分(归一化): {rating.item():.4f}")
        print(f"  评分(原始): {rating.item() * 4 + 1:.2f}")
    
    print("\n✅ 数据集加载测试完成!")
