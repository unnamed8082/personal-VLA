"""
模型推理测试
============
测试训练好的模型在实际任务中的表现
"""

import sys
import torch
from pathlib import Path
import numpy as np

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_inference():
    """测试模型推理"""
    print("="*70)
    print("模型推理测试")
    print("="*70)
    
    # 加载模型
    checkpoint_path = Path('outputs/checkpoints/best_model.pt')
    
    if not checkpoint_path.exists():
        print(f"❌ 模型文件不存在：{checkpoint_path}")
        print(f"💡 请先运行训练脚本：python projects/grinding_system/train_lerobot_vla.py")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"\n✅ 加载模型成功")
    print(f"   路径：{checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    # 安全地获取 loss（如果存在）
    if 'loss' in checkpoint:
        print(f"   训练损失：{checkpoint['loss']:.4f}")
    else:
        print(f"   训练损失：N/A（未保存）")
    
    # 显示配置信息
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"\n📋 训练配置:")
        print(f"   数据集：{config.get('dataset_name', 'N/A')}")
        print(f"   Batch size: {config.get('batch_size', 'N/A')}")
        print(f"   学习率：{config.get('learning_rate', 'N/A')}")
    
    # 加载数据集
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset('lerobot/pusht')
    
    # 创建模型 - 直接在本文件中定义，避免导入问题
    import torch.nn as nn
    
    class SimpleVLAModel(nn.Module):
        """简化版 VLA 模型（用于快速验证）"""
        
        def __init__(self, image_size: tuple = (3, 96, 96), state_dim: int = 5, action_dim: int = 2):
            super().__init__()
            
            # 视觉编码器（简化的 ResNet-18）
            from torchvision.models import resnet18, ResNet18_Weights
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
            self.visual_fc = nn.Linear(512, 256)
            
            # 状态编码器
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256)
            )
            
            # 融合层
            self.fusion = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            # 动作解码器
            self.action_decoder = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim)
            )
            
            print("✅ 简化版 VLA 模型创建完成")
            print(f"   参数量：{sum(p.numel() for p in self.parameters()):,}")
        
        def forward(self, batch):
            """前向传播"""
            # 提取视觉特征 - 支持多种图像键名
            visual_feat = None
            
            image_keys = ['observation.images.top', 'observation.image', 'observation.images.rgb']
            for img_key in image_keys:
                if img_key in batch:
                    images = batch[img_key]
                    visual_feat = self.visual_encoder(images).squeeze(-1).squeeze(-1)
                    visual_feat = self.visual_fc(visual_feat)
                    break
            
            if visual_feat is None:
                visual_feat = torch.zeros(batch['observation.state'].shape[0], 256).to(batch['observation.state'].device)
            
            # 提取状态特征
            state = batch['observation.state']
            state_feat = self.state_encoder(state)
            
            # 融合
            combined = torch.cat([visual_feat, state_feat], dim=-1)
            fused = self.fusion(combined)
            
            # 预测动作
            actions = self.action_decoder(fused)
            
            return {'actions': actions}
    
    sample = dataset[0]
    state_dim = sample['observation.state'].shape[0]
    action_dim = sample['action'].shape[0]
    
    model = SimpleVLAModel(state_dim=state_dim, action_dim=action_dim)
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 尝试直接加载（兼容旧格式）
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print(f"\n📊 模型信息:")
    print(f"   State 维度：{state_dim}")
    print(f"   Action 维度：{action_dim}")
    print(f"   参数量：{sum(p.numel() for p in model.parameters()):,}")
    
        # 测试多个样本
    print(f"\n🧪 推理测试...")
    
    # 先查看数据集的第一个样本，确定正确的键名
    sample = dataset[0]
    print(f"\n🔍 数据集键名检查:")
    print(f"   所有键：{list(sample.keys())}")
    
    # 找到图像键
    image_key = None
    possible_keys = ['observation.images.top', 'observation.image', 'observation.images.rgb', 
                    'observation.image.top', 'image', 'rgb']
    
    for key in possible_keys:
        if key in sample:
            image_key = key
            print(f"   ✅ 找到图像键：{key}")
            break
    
    if image_key is None:
        # 尝试找到任何包含 'image' 的键
        for key in sample.keys():
            if 'image' in key.lower():
                image_key = key
                print(f"   ✅ 找到图像键（模糊匹配）：{key}")
                break
    
    if image_key is None:
        print(f"   ❌ 未找到图像键！")
        print(f"   💡 可用键：{list(sample.keys())}")
        return None
    
    # 找到状态键
    state_key = None
    possible_state_keys = ['observation.state', 'state', 'robot_state', 'observation.robot_state']
    
    for key in possible_state_keys:
        if key in sample:
            state_key = key
            print(f"   ✅ 找到状态键：{key}")
            break
    
    if state_key is None:
        # 尝试找到任何包含 'state' 的键
        for key in sample.keys():
            if 'state' in key.lower() and 'action' not in key.lower():
                state_key = key
                print(f"   ✅ 找到状态键（模糊匹配）：{key}")
                break
    
    if state_key is None:
        print(f"   ❌ 未找到状态键！")
        return None
    
    # 找到动作键
    action_key = 'action'  # 默认
    if action_key not in sample:
        for key in sample.keys():
            if 'action' in key.lower():
                action_key = key
                print(f"   ✅ 找到动作键：{key}")
                break
    
    errors = []
    predictions_list = []
    targets_list = []
    
    for i in range(min(20, len(dataset))):
        sample = dataset[i]
        
        # 准备 batch
        batch = {
            image_key: sample[image_key].unsqueeze(0),
            state_key: sample[state_key].unsqueeze(0)
        }
        
        # 推理
        with torch.no_grad():
            prediction = model(batch)
        
        predicted = prediction['actions'].squeeze().numpy()
        true_action = sample[action_key].numpy()
        
        # 计算误差
        error = np.mean((predicted - true_action) ** 2)
        errors.append(error)
        predictions_list.append(predicted)
        targets_list.append(true_action)
        
        if i < 5:  # 只显示前 5 个
            print(f"\n样本 {i+1}:")
            print(f"   预测：{predicted}")
            print(f"   真实：{true_action}")
            print(f"   绝对误差：{np.abs(predicted - true_action)}")
            print(f"   MSE: {error:.4f}")
    
    avg_mse = np.mean(errors)
    std_mse = np.std(errors)
    
    # 计算整体统计
    all_predictions = np.array(predictions_list)
    all_targets = np.array(targets_list)
    
    overall_mae = np.mean(np.abs(all_predictions - all_targets))
    overall_rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
    
    print(f"\n📊 总体评估:")
    print(f"   平均 MSE: {avg_mse:.4f} ± {std_mse:.4f}")
    print(f"   平均 MAE: {overall_mae:.4f}")
    print(f"   平均 RMSE: {overall_rmse:.4f}")
    print(f"   测试样本数：{len(errors)}")
    
    # 分析误差分布
    print(f"\n📊 误差分布:")
    print(f"   最小 MSE: {np.min(errors):.4f}")
    print(f"   最大 MSE: {np.max(errors):.4f}")
    print(f"   中位数 MSE: {np.median(errors):.4f}")
    
    # 评价
    print(f"\n💡 性能评价:")
    if avg_mse < 0.5:
        print(f"   ✅ 优秀！MSE < 0.5，可以用于实际任务")
    elif avg_mse < 5.0:
        print(f"   ⚠️  可用，MSE 在合理范围，但建议继续训练")
    elif avg_mse < 50:
        print(f"   ⚠️  一般，MSE 偏高，需要更多训练")
    else:
        print(f"   ❌ 需要改进")
        print(f"   可能原因:")
        print(f"     1. 训练轮数不足（建议 50-100 epochs）")
        print(f"     2. 学习率需要调整")
        print(f"     3. Action 数据可能未归一化")
        print(f"     4. 模型架构需要优化")
    
    # 保存评估结果
    results = {
        'checkpoint_path': str(checkpoint_path),
        'epoch': checkpoint.get('epoch', 'N/A'),
        'dataset': 'lerobot/pusht',
        'n_samples': len(errors),
        'avg_mse': float(avg_mse),
        'std_mse': float(std_mse),
        'mae': float(overall_mae),
        'rmse': float(overall_rmse),
        'min_mse': float(np.min(errors)),
        'max_mse': float(np.max(errors)),
        'median_mse': float(np.median(errors))
    }
    
    results_path = Path('outputs/checkpoints/inference_results.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 评估结果已保存：{results_path}")
    
    return avg_mse


if __name__ == "__main__":
    test_inference()