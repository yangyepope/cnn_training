"""
训练脚本
========
用于训练颜值预测CNN模型的主脚本

使用方法：
---------
# 使用默认参数训练
python train.py

# 自定义参数
python train.py --epochs 30 --batch_size 8 --model resnet18

训练流程：
---------
1. 解析命令行参数
2. 加载数据集，划分训练集/验证集
3. 创建模型
4. 训练循环：
   - 遍历训练集，更新模型参数
   - 在验证集上评估模型
   - 保存最佳模型
5. 输出最终结果
"""

# ==============================================================================
# 导入必要的库
# ==============================================================================

import os
import argparse
import time
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    IMAGES_DIR, RATINGS_FILE, MODEL_SAVE_PATH,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR,
    TRAIN_RATIO, RANDOM_SEED, DEFAULT_MODEL
)
from dataset import BeautyDataset, get_train_transform, get_val_transform
from model import BeautyModel


# ==============================================================================
# 训练一个Epoch的函数
# ==============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0
    valid_batches = 0  # 有效batch计数
    batch_count = len(dataloader)
    
    start_time = time.time()
    
    for batch_idx, (images, ratings) in enumerate(dataloader):
        images = images.to(device)
        ratings = ratings.to(device)
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        loss = criterion(outputs, ratings)
        
        # 检查loss是否为nan或inf，如果是则跳过这个batch
        if math.isnan(loss.item()) or math.isinf(loss.item()):
            print(f"  ⚠️ Batch {batch_idx}: Loss is nan/inf, skipping...")
            continue
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪：防止梯度爆炸导致nan
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        valid_batches += 1
        
        # 打印进度
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == batch_count:
            elapsed = time.time() - start_time
            eta = elapsed / (batch_idx + 1) * (batch_count - batch_idx - 1)
            current_avg_loss = total_loss / valid_batches if valid_batches > 0 else 0
            print(f"  Epoch [{epoch}/{total_epochs}] "
                  f"Batch [{batch_idx + 1}/{batch_count}] "
                  f"Loss: {loss.item():.4f} "
                  f"Avg: {current_avg_loss:.4f} "
                  f"ETA: {eta/60:.1f}min")
    
    return total_loss / valid_batches if valid_batches > 0 else float('inf')


# ==============================================================================
# 评估函数
# ==============================================================================

def evaluate(model, dataloader, criterion, device):
    """
    在验证集上评估模型性能
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, ratings in dataloader:
            images = images.to(device)
            ratings = ratings.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, ratings)
            
            # 跳过无效的loss
            if not (math.isnan(loss.item()) or math.isinf(loss.item())):
                total_loss += loss.item()
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(ratings.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 过滤掉nan值
    valid_mask = ~(np.isnan(all_preds) | np.isnan(all_labels))
    all_preds = all_preds[valid_mask]
    all_labels = all_labels[valid_mask]
    
    if len(all_preds) == 0:
        return float('inf'), float('inf'), float('inf'), 0
    
    # 转回1-5分
    preds_original = all_preds * 4.0 + 1.0
    labels_original = all_labels * 4.0 + 1.0
    
    mae = np.mean(np.abs(preds_original - labels_original))
    mse = np.mean((preds_original - labels_original) ** 2)
    rmse = np.sqrt(mse)
    
    if len(preds_original) > 1:
        correlation = np.corrcoef(preds_original, labels_original)[0, 1]
    else:
        correlation = 0
    
    return total_loss / len(dataloader), mae, rmse, correlation


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    """
    主函数：程序的入口点
    """
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="CNN颜值预测模型训练")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="批大小")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="学习率")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       choices=["mobilenet", "resnet18", "resnet34"], help="模型选择")
    parser.add_argument("--images_dir", type=str, default=IMAGES_DIR, help="图片目录")
    parser.add_argument("--ratings_file", type=str, default=RATINGS_FILE, help="评分文件")
    parser.add_argument("--save_path", type=str, default=MODEL_SAVE_PATH, help="模型保存路径")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CNN 颜值预测模型训练")
    print("=" * 60)
    
    # 选择计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📱 计算设备: {device}")
    
    if device.type == 'cpu':
        print("⚠️  使用CPU训练，速度较慢，请耐心等待...")
    
    # 加载数据集
    print("\n📂 加载数据集...")
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    full_dataset = BeautyDataset(args.images_dir, args.ratings_file, transform=train_transform)
    
    if len(full_dataset) == 0:
        print("❌ 数据集为空，请检查路径配置！")
        return
    
    # 划分训练集和验证集
    train_size = int(TRAIN_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    print(f"   训练集: {train_size} 张图片")
    print(f"   验证集: {val_size} 张图片")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # 创建模型
    print(f"\n🔧 创建模型: {args.model}")
    model = BeautyModel(model_name=args.model)
    model = model.to(device)
    
    total_params, trainable_params = model.count_parameters()
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    
    # 使用较小的学习率，并且使用权重衰减防止过拟合
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # 打印训练配置
    print(f"\n🚀 开始训练...")
    print(f"   训练轮数: {args.epochs}")
    print(f"   批大小: {args.batch_size}")
    print(f"   学习率: {args.lr}")
    print()
    
    # 训练循环
    best_mae = float('inf')
    best_correlation = 0
    train_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        
        # 验证
        val_loss, mae, rmse, correlation = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        # 更新学习率（根据验证损失）
        scheduler.step(val_loss)
        
        # 打印结果
        print(f"\n📊 Epoch {epoch}/{args.epochs} 完成 (耗时: {epoch_time/60:.1f}分钟)")
        print(f"   训练损失: {train_loss:.4f}")
        print(f"   验证损失: {val_loss:.4f}")
        print(f"   MAE: {mae:.4f} (越小越好)")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   相关系数: {correlation:.4f} (越接近1越好)")
        
        # 保存最佳模型
        if mae < best_mae and not math.isnan(mae):
            best_mae = mae
            best_correlation = correlation
            
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mae': mae,
                'correlation': correlation,
                'model_name': args.model
            }, args.save_path)
            print(f"   ✅ 保存最佳模型！(MAE: {mae:.4f})")
        
        print()
    
    # 训练完成
    total_time = time.time() - train_start
    print("=" * 60)
    print("🎉 训练完成！")
    print("=" * 60)
    print(f"   总耗时: {total_time / 60:.1f} 分钟")
    print(f"   最佳 MAE: {best_mae:.4f}")
    print(f"   最佳相关系数: {best_correlation:.4f}")
    print(f"   模型保存位置: {args.save_path}")


if __name__ == "__main__":
    main()
