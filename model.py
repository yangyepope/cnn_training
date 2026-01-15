"""
模型定义模块
============
定义了颜值预测的神经网络模型

核心概念：
---------
1. 迁移学习（Transfer Learning）
   - 使用在 ImageNet（120万张图片）上预训练好的模型
   - 这些模型已经学会了提取图像特征（边缘、纹理、形状等）
   - 我们只需要修改最后的分类层，让它适应我们的任务
   
2. 为什么用迁移学习？
   - 我们只有5500张图片，从零训练CNN容易过拟合
   - 预训练模型已经有很强的特征提取能力
   - 训练更快，效果更好

3. 模型结构
   - 骨干网络（Backbone）：预训练的CNN，用于提取特征
   - 回归头（Head）：新的全连接层，输出颜值分数

支持的模型：
-----------
- MobileNetV2：轻量级，3.4M参数，CPU友好
- ResNet18：中等大小，11M参数，效果较好
- ResNet34：较大，21M参数，效果可能更好
"""

# ==============================================================================
# 导入必要的库
# ==============================================================================

import torch                       # PyTorch核心库
import torch.nn as nn             # 神经网络模块，包含各种层
from torchvision import models    # 预训练模型库


# ==============================================================================
# 模型类定义
# ==============================================================================

class BeautyModel(nn.Module):
    """
    颜值预测模型
    
    继承自 nn.Module，这是所有 PyTorch 模型的基类
    
    模型结构：
    ---------
    输入图片 (batch, 3, 224, 224)
         ↓
    预训练骨干网络（卷积层，提取特征）
         ↓
    特征向量 (batch, num_features)
         ↓
    Dropout（随机丢弃，防止过拟合）
         ↓
    全连接层（num_features -> 1）
         ↓
    Sigmoid激活（压缩到0-1范围）
         ↓
    输出 (batch,)
    
    参数说明：
    ---------
    model_name : str
        选择哪个预训练模型
        - 'mobilenet': MobileNetV2，最轻量
        - 'resnet18': ResNet18，中等
        - 'resnet34': ResNet34，较大
    
    使用示例：
    ---------
    >>> model = BeautyModel(model_name='mobilenet')
    >>> images = torch.randn(16, 3, 224, 224)  # 16张随机图片
    >>> outputs = model(images)  # 前向传播
    >>> outputs.shape
    torch.Size([16])  # 16个预测分数
    """
    
    def __init__(self, model_name='mobilenet'):
        """
        初始化模型
        
        步骤：
        1. 调用父类初始化
        2. 加载预训练模型
        3. 修改最后的分类层为回归层
        """
        
        # ----------------------------------------------------------------------
        # 调用父类的初始化方法
        # ----------------------------------------------------------------------
        # 这是 Python 继承的标准写法，必须调用
        super(BeautyModel, self).__init__()
        
        # 保存模型名称，方便后续查看
        self.model_name = model_name
        
        # ----------------------------------------------------------------------
        # 根据选择加载不同的预训练模型
        # ----------------------------------------------------------------------
        
        if model_name == 'mobilenet':
            # ==================================================================
            # MobileNetV2
            # ==================================================================
            # 特点：
            # - 轻量级，专为移动设备设计
            # - 使用深度可分离卷积，参数量小
            # - 约 3.4M 参数
            # - CPU 上也能较快运行
            
            # 加载预训练的 MobileNetV2 模型
            # weights 参数指定使用 ImageNet 预训练权重
            # IMAGENET1K_V1 表示在 ImageNet-1K（1000类）数据集上训练的第1版权重
            self.backbone = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
            )
            
            # 获取原始分类器的输入特征数
            # MobileNetV2 的 classifier 结构是：
            #   classifier = Sequential(
            #       Dropout(0.2),
            #       Linear(1280, 1000)  # 1000是ImageNet的类别数
            #   )
            # 我们需要获取 Linear 层的输入特征数（1280）
            num_features = self.backbone.classifier[1].in_features
            
            # 替换分类器为回归头
            # 原来是输出1000个类别的概率，现在改为输出1个颜值分数
            self.backbone.classifier = nn.Sequential(
                # Dropout 层：训练时随机将20%的神经元输出置为0
                # 作用：防止过拟合，让模型不依赖特定的神经元
                nn.Dropout(0.2),
                
                # 全连接层（Linear）：将1280维特征映射到1维输出
                # in_features=1280, out_features=1
                nn.Linear(num_features, 1),
                
                # Sigmoid 激活函数：将输出压缩到 (0, 1) 范围
                # 公式：sigmoid(x) = 1 / (1 + e^(-x))
                # 无论输入是多少，输出都在0到1之间
                nn.Sigmoid()
            )
        
        elif model_name == 'resnet18':
            # ==================================================================
            # ResNet18
            # ==================================================================
            # 特点：
            # - 残差网络（Residual Network），有跳跃连接
            # - 18层深，约 11M 参数
            # - 效果比 MobileNet 好，但更慢
            
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1
            )
            
            # ResNet 的最后一层叫 fc（fully connected，全连接层）
            # 原始结构：fc = Linear(512, 1000)
            num_features = self.backbone.fc.in_features  # 512
            
            # 替换全连接层
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 1),
                nn.Sigmoid()
            )
        
        elif model_name == 'resnet34':
            # ==================================================================
            # ResNet34
            # ==================================================================
            # 特点：
            # - 34层深，比ResNet18更深
            # - 约 21M 参数
            # - 理论上效果更好，但训练更慢
            
            self.backbone = models.resnet34(
                weights=models.ResNet34_Weights.IMAGENET1K_V1
            )
            
            num_features = self.backbone.fc.in_features  # 512
            
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 1),
                nn.Sigmoid()
            )
        
        else:
            # 不支持的模型，抛出错误
            raise ValueError(f"不支持的模型: {model_name}，可选: mobilenet, resnet18, resnet34")
    
    def forward(self, x):
        """
        前向传播
        
        这个方法定义了数据如何流经网络
        当你调用 model(input) 时，PyTorch 会自动调用这个方法
        
        参数：
        ------
        x : torch.Tensor
            输入图片张量
            形状：(batch_size, 3, 224, 224)
            - batch_size: 一批有多少张图片
            - 3: RGB三个颜色通道
            - 224, 224: 图片的高度和宽度
        
        返回：
        ------
        output : torch.Tensor
            预测的颜值分数
            形状：(batch_size,)
            值在 0 到 1 之间
        
        计算过程：
        ---------
        1. x 通过骨干网络的卷积层，提取特征
        2. 特征通过 Dropout，随机丢弃一部分
        3. 特征通过全连接层，得到1维输出
        4. 输出通过 Sigmoid，压缩到0-1范围
        5. squeeze() 去掉多余的维度
        """
        
        # 通过骨干网络得到输出
        # 输出形状是 (batch_size, 1)
        output = self.backbone(x)
        
        # squeeze() 去掉值为1的维度
        # (batch_size, 1) -> (batch_size,)
        # 这样输出形状就和标签形状一致了
        return output.squeeze()
    
    def count_parameters(self):
        """
        统计模型的参数数量
        
        返回：
        ------
        total : int
            总参数数量
        trainable : int
            可训练的参数数量（requires_grad=True）
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ==============================================================================
# 测试代码
# ==============================================================================

if __name__ == "__main__":
    """
    测试模型是否能正常运行
    """
    print("=" * 60)
    print("测试模型")
    print("=" * 60)
    
    # 测试每个支持的模型
    for model_name in ['mobilenet', 'resnet18', 'resnet34']:
        print(f"\n--- {model_name} ---")
        
        # 创建模型
        model = BeautyModel(model_name=model_name)
        
        # 统计参数
        total, trainable = model.count_parameters()
        print(f"总参数: {total:,}")
        print(f"可训练参数: {trainable:,}")
        
        # 测试前向传播
        # 创建一个假的输入：4张 224x224 的RGB图片
        dummy_input = torch.randn(4, 3, 224, 224)
        
        # 前向传播
        output = model(dummy_input)
        
        print(f"输入形状: {dummy_input.shape}")
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    print("\n✅ 模型测试完成!")
