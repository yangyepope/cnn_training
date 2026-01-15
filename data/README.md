# 数据目录

请将 SCUT-FBP5500 数据集放到这个目录下：

```
data/
└── SCUT-FBP5500/
    ├── Images/              ← 放入5500张人脸图片
    │   ├── AF1.jpg
    │   ├── AF2.jpg
    │   ├── ...
    │   └── CM500.jpg
    │
    └── All_Ratings.xlsx     ← 放入评分文件
```

## 数据集下载

SCUT-FBP5500 数据集可以从以下地址获取：
- GitHub: https://github.com/HCIILAB/SCUT-FBP5500-Database-Release

## 数据说明

- **Images/**: 包含5500张人脸图片
  - AF: 亚洲女性 (Asian Female)
  - AM: 亚洲男性 (Asian Male)
  - CF: 白人女性 (Caucasian Female)
  - CM: 白人男性 (Caucasian Male)

- **All_Ratings.xlsx**: 颜值评分文件
  - 60名志愿者对每张图片打分（1-5分）
  - 文件包含每张图片的平均评分
