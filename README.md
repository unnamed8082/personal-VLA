# VLA - Visual-Language-Action for Grinding SystemVLA——研磨系统的视觉语言动作

🤖 **深度优化的 VLA 模型用于机器人磨削系统**

[![License: Apache 2.0   许可证：Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)[![许可证：Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)]（https://opensource.org/licenses/Apache-2.0）
[![Python 3.12+   Python 3.12Python 3.12](https://img.shields.io/badge/python-3.12+-blue.svghttps://img.shields.io/badge/python-3.12 -blue.svg)(https://img.shields.io/badge/python-3.12 -blue.svghttps://img.shields.io/badge/python-3.12 -blue.svg)](https://www.python.org/downloads/)[![Python 3.12](https://img.shields.io/badge/python-3.12 -blue.svg)]（https://www.python.org/downloads/）[![Python 3.12 Python 3.12   Python 3.12](https://img.shields.io/badge/python-3.12 -blue.svghttps://img.shields.io/badge/ Python -3.12 -blue.svg)(https://img.shields.io/badge/python-3.12 -blue.svghttps://img.shields.io/badge/ Python -3.12 -blue.svg)(https://img.shields.io/badge/python - 3.12 -blue.svghttps: / / img.shields。io/badge/ Python -3.12 -blue.svg)（https://img.shields.io/badge/python-3.12 -blue.svghttps://img.shields. svg）io/badge/ Python -3.12 -blue.svg)](https://www.python.org/downloads/)[！[Python 3.12](https://img.shields.io/badge/python-3.12 -blue.svg)]（https://www.python.org/downloads/）
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svghttps://img.shields.io/badge/PyTorch-2.2 -red.svg)(https://img.shields.io/badge/PyTorch-2.2 -red.svghttps://img.shields.io/badge/PyTorch-2.2 -red.svg)](https://pytorch.org/)[![PyTorch](https://img.shields.io/badge/PyTorch-2.2 -red.svghttps://img.shields.io/badge/PyTorch-2.2 -red.svg)(https://img.shields.io/badge/PyTorch-2.2 -red.svghttps://img.shields.io/badge/PyTorch-2.2 -red.svg)](https://pytorch.org/)

## 📋 项目简介

本项目实现了一个深度优化的 VLA（Visual-Language-Action）模型，专门用于机器人磨削任务。基于 LeRobot 框架和 PushT 数据集，实现了像素级动作预测。
由  https://github.com/datawhalechina/every-embodied   这个项目而来

### ✨ 主要特性

- **ResNet-34 Backbone   ResNet-34骨干**: 更强的视觉特征提取能力
- **SE Block 注意力机制**: Squeeze-and-Excitation 模块提升关键特征
- **Dropout 正则化**: 防止过拟合，提升泛化能力
- **Cosine Annealing + Warmup**: 智能学习率调度
- **梯度裁剪**: 稳定训练过程
- **EMA (指数移动平均)**: 平滑模型权重
- **数据增强**: 随机裁剪、颜色抖动、水平翻转

### 🎯 性能指标

| 指标 | 目标 | 实际 |
|------|------|------|
| 像素误差 | < 10px | **4.68px** ✅ |
| MSE Loss | < 5.0 | **4.12** ✅ || MSE Loss | < 5.0 | **4.12** ✅ || MSE Loss | < 5.0 | **4.12** ✅ || MSE Loss | < 5.0 | **4.12** ✅ |
| 训练时间 | - | ~12 小时 |

## 📁 项目结构
VLA\
├── projects/
│   └── grinding_system/│├──grinding_system/│├──grinding_system/│├──grinding_system/
│       ├── train_lerobot_vla_enhanced.py          # 主训练脚本
│       ├── EnhancedVLAModel.py                  # 模型定义
│       ├── test_enhanced_inference.py           # 测试脚本
│       └── visualize_predictions.py             # 可视化脚本
├── lerobot/
│   └── datasets/│├──datasets/
│       └── lerobot_dataset.py                   # 数据集处理
├── outputs/
│   └── checkpoints_enhanced/
│       ├── best_model_enhanced.pt               # 最佳模型检查点（这个是二进制文件，太大了我没上传，可以自己跑）
│       ├── normalization_stats.json             # 归一化参数
│       └── training_log.json                    # 训练日志
├── README.md                                  # 项目说明
├── requirements.txt                           # 依赖包
└── config.yaml                                # 配置文件
<img width="400" height="456" alt="image" src="https://github.com/user-attachments/assets/fec074a8-83ff-4ed8-a17f-a19e419a359c" />   /比;


## 🚀 快速开始

### 1. 环境安装

```bash   ”“bash
# 克隆仓库
git clone https://github.com/你的用户名/VLA.git
cd VLA

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txtPIP install -r requirements.txt
2. 使用预训练模型推理
# 测试增强版模型
python projects/grinding_system/test_enhanced_inference.py \Python projects/grinding_system/test_enhanced_inference.py \python projects/grinding_system/test_enhanced_inference.py \ python projects/grinding_system/test_enhanced_inference.py \python projects/grinding_system/test_enhanced_inference.py \python projects/grinding_system/test_enhanced_inference.py \python projects/grinding_system/test_enhanced_inference.py \python projects/grinding_system/test_enhanced_inference.py \python projects/grinding_system/test_enhanced_inference.py \python projects/grinding_system/test_enhanced_inference.py \python projects/grinding_system/test_enhanced_inference.py \python projects/grinding_system/test_enhanced_inference.py \
    --checkpoint outputs/checkpoints_enhanced/best_model_enhanced.pt——输出/ checkpoints_enhanced / best_model_enhanced.pt检查站——检查点输出/ checkpoints_enhanced/best_model_enhanced.pt检查站。输出/ checkpoints_enhanced/best_model_enhanced.pt检查站——检查点输出/ checkpoints_enhanced/best_model_enhanced.pt检查站。输出/ checkpoints_enhanced/best_model_enhanced.pt检查站
3. 训练自己的模型
# 使用默认配置训练
python projects/grinding_system/train_lerobot_vla_enhanced.pypython项目/ grinding_system / train_lerobot_vla_enhanced.py

# 自定义配置训练
python projects/grinding_system/train_lerobot_vla_enhanced.py \Python projects/grinding_system/ train_lerobot_vla_enhancedpy \
    --epochs 100 \   ——epoch 100 \
    --batch-size 8 \   ——批处理大小8 \
    --learning-rate 0.0002 \   ——学习率0.0002 \
    --save-dir outputs/my_checkpoints——/ my_checkpoints save-dir输出——save-dir outputs/my_checkpoints—&mdash；/ my_checkpoint save-dir输出
4. 可视化预测结果
python projects/grinding_system/visualize_predictions.py \Python projects/grinding_system/visualize_predictions.py \python projects/grinding_system/visualize_predictions.py \ python projects/grinding_system/visualize_predictions.py \
    --checkpoint outputs/checkpoints_enhanced/best_model_enhanced.pt \——checkpoint输出/checkpoints_enhanced/best_model_enhanced.pt \checkpoint输出/checkpoints_enhanced/best_model_enhanced.pt \
    --save-dir outputs/visualizations——save-dir输出/可视化
📊 模型架构
输入图像 (3×H×W)
    ↓
ResNet-34 Backbone   ResNet-34骨干
    ↓
SE Block 注意力
    ↓
视觉特征 (512-dim)
    ↓
状态特征 (2-dim) ──→ 融合层 ──→ 动作解码器 ──→ 输出动作 (2-dim)
                                                      ↓
                                              (Δx, Δy)

🔧 配置说明
编辑 config.yaml 文件来自定义训练参数：
training:   培训:
  epochs: 100   时代:100
  batch_size: 8
  learning_rate: 0.0002
  weight_decay: 0.05
  
augmentation:
  random_crop: true
  color_jitter: true
  horizontal_flip: true
📈 训练日志
查看训练进度和性能指标：
# 查看训练日志
cat outputs/checkpoints_enhanced/training_log.json

# 使用 TensorBoard
tensorboard --logdir outputs/logs
📝 许可证
本项目采用 Apache 2.0 许可证。详见 LICENSE 文件。

🙏 致谢
LeRobot - Hugging Face 的机器人学习框架
PushT Dataset - 斯坦福 PushT 数据集
SmolVLA - 基础 VLA 模型架构
Happy Robotics! 🤖✨
