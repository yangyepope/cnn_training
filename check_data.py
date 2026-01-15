
"""
测试脚本：检查数据是否正常
"""
import pandas as pd
import os
from PIL import Image

# 配置
IMAGES_DIR = r"D:\8-python-project\cnn_training\data\SCUT-FBP5500\Images"
RATINGS_FILE = r"D:\8-python-project\cnn_training\data\SCUT-FBP5500\All_Ratings.xlsx"

print("=" * 60)
print("数据检查")
print("=" * 60)

# 1. 检查评分文件
print("\n📊 检查评分文件...")
try:
    df = pd.read_excel(RATINGS_FILE)
    print(f"   文件读取成功!")
    print(f"   行数: {len(df)}")
    print(f"   列名: {list(df.columns)}")
    print(f"\n   前5行数据:")
    print(df.head())
    print(f"\n   数据类型:")
    print(df.dtypes)
    print(f"\n   评分统计:")
    # 假设第二列是评分
    rating_col = df.columns[1]
    print(f"   最小值: {df[rating_col].min()}")
    print(f"   最大值: {df[rating_col].max()}")
    print(f"   平均值: {df[rating_col].mean():.2f}")
    print(f"   是否有nan: {df[rating_col].isna().sum()}")
except Exception as e:
    print(f"   ❌ 读取失败: {e}")

# 2. 检查图片文件夹
print("\n📷 检查图片文件夹...")
if os.path.exists(IMAGES_DIR):
    files = os.listdir(IMAGES_DIR)
    print(f"   图片数量: {len(files)}")
    if len(files) > 0:
        print(f"   前5个文件: {files[:5]}")
        
        # 检查第一张图片
        first_img = os.path.join(IMAGES_DIR, files[0])
        try:
            img = Image.open(first_img)
            print(f"   第一张图片: {files[0]}")
            print(f"   图片尺寸: {img.size}")
            print(f"   图片模式: {img.mode}")
        except Exception as e:
            print(f"   ❌ 读取图片失败: {e}")
else:
    print(f"   ❌ 文件夹不存在!")

# 3. 检查文件名匹配
print("\n🔗 检查文件名匹配...")
try:
    df = pd.read_excel(RATINGS_FILE)
    filename_col = df.columns[0]
    
    # 获取评分文件中的文件名
    rating_filenames = set(df[filename_col].astype(str).tolist())
    
    # 获取实际图片文件名
    actual_filenames = set(os.listdir(IMAGES_DIR))
    
    # 检查匹配
    matched = rating_filenames & actual_filenames
    only_in_rating = rating_filenames - actual_filenames
    only_in_folder = actual_filenames - rating_filenames
    
    print(f"   评分文件中的文件数: {len(rating_filenames)}")
    print(f"   实际图片文件数: {len(actual_filenames)}")
    print(f"   匹配的文件数: {len(matched)}")
    
    if only_in_rating:
        print(f"   ⚠️ 只在评分文件中存在: {len(only_in_rating)} 个")
        print(f"      例如: {list(only_in_rating)[:3]}")
    
    if only_in_folder:
        print(f"   ⚠️ 只在图片文件夹中存在: {len(only_in_folder)} 个")
        print(f"      例如: {list(only_in_folder)[:3]}")
        
except Exception as e:
    print(f"   ❌ 检查失败: {e}")

print("\n" + "=" * 60)
print("检查完成!")
print("=" * 60)
