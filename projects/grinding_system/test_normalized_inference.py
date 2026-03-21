
"""
归一化模型推理测试
====================
使用训练好的归一化模型进行推理
"""

import torch
from pathlib import Path
import numpy as np
import json


def test_normalized_inference():
    """测试归一化模型的推理"""
    print("="*70)
    print("归一化模型推理测试")
    print("="*70)
    
    # 加载模型
    checkpoint_path = Path('outputs/checkpoints_normalized_v2/best_model_normalized.pt')
    
    if not checkpoint_path.exists():
        print(f"❌ 模型文件不存在：{checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"\n✅ 加载模型成功")
    print(f"   路径：{checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   验证 MSE: {checkpoint['val_loss']:.4f}")
    print(f"   像素误差：{checkpoint.get('pixel_error', 'N/A'):.2f} pixels")
    
    # 加载归一化统计
    norm_stats = checkpoint['normalization_stats']
    print(f"\n📊 归一化统计:")
    print(f"   Action 均值：{norm_stats['action_mean']}")
    print(f"   Action 标准差：{norm_stats['action_std']}")
    
    # 加载数据集
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset('lerobot/pusht')
    
    # 创建模型
    import torch.nn as nn
    
    class NormalizedVLAModel(nn.Module):
        def __init__(self, state_dim: int = 2, action_dim: int = 2, hidden_dim: int = 256):
            super().__init__()
            from torchvision.models import resnet18, ResNet18_Weights
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
            self.visual_fc = nn.Linear(512, hidden_dim)
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim)
            )
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.action_decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, action_dim)
            )
        
        def forward(self, batch):
            visual_feat = None
            image_keys = ['observation.images.top', 'observation.image', 'observation.images.rgb']
            for img_key in image_keys:
                if img_key in batch:
                    images = batch[img_key]
                    visual_feat = self.visual_encoder(images).squeeze(-1).squeeze(-1)
                    visual_feat = self.visual_fc(visual_feat)
                    break
            if visual_feat is None:
                batch_size = batch['observation.state'].shape[0]
                visual_feat = torch.zeros(batch_size, 256).to(batch['observation.state'].device)
            state = batch['observation.state']
            state_feat = self.state_encoder(state)
            combined = torch.cat([visual_feat, state_feat], dim=-1)
            fused = self.fusion(combined)
            actions = self.action_decoder(fused)
            return {'actions': actions}
    
    dataset_info = checkpoint['dataset_info']
    model = NormalizedVLAModel(
        state_dim=dataset_info['state_dim'],
        action_dim=dataset_info['action_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\n📊 模型信息:")
    print(f"   参数量：{sum(p.numel() for p in model.parameters()):,}")
    
    # 归一化参数
    device = next(model.parameters()).device
    action_mean = torch.tensor(norm_stats['action_mean']).to(device)
    action_std = torch.tensor(norm_stats['action_std']).to(device) + 1e-6
    
       # 测试推理
    print(f"\n🧪 推理测试...")
    
    # 先检测数据集的键名
    sample = dataset[0]
    print(f"\n🔍 检测数据集键名:")
    print(f"   所有键：{list(sample.keys())}")
    
    # 找到图像键
    image_key = None
    possible_image_keys = ['observation.images.top', 'observation.image', 'observation.images.rgb', 
                          'observation.image.top', 'image', 'rgb', 'images.top']
    
    for key in possible_image_keys:
        if key in sample:
            image_key = key
            print(f"   ✅ 找到图像键：{key}")
            break
    
    if image_key is None:
        # 模糊匹配
        for key in sample.keys():
            if 'image' in key.lower():
                image_key = key
                print(f"   ✅ 找到图像键（模糊匹配）：{key}")
                break
    
    if image_key is None:
        print(f"   ❌ 未找到图像键！")
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
        for key in sample.keys():
            if 'state' in key.lower() and 'action' not in key.lower():
                state_key = key
                print(f"   ✅ 找到状态键（模糊匹配）：{key}")
                break
    
    if state_key is None:
        print(f"   ❌ 未找到状态键！")
        return None
    
    # 找到动作键
    action_key = 'action'
    if action_key not in sample:
        for key in sample.keys():
            if 'action' in key.lower():
                action_key = key
                print(f"   ✅ 找到动作键：{key}")
                break
    
    errors = []
    pixel_errors = []
    
    for i in range(min(20, len(dataset))):
        sample = dataset[i]
        
        # 准备 batch
        batch = {
            image_key: sample[image_key].unsqueeze(0),
            state_key: sample[state_key].unsqueeze(0)
        }
        
        # 推理
        with torch.no_grad():
            outputs = model(batch)
            # 反归一化
            pred_denorm = outputs['actions'] * action_std + action_mean
        
        predicted = pred_denorm.squeeze().cpu().numpy()
        true_action = sample[action_key].numpy()
        
        # 计算像素误差
        pixel_error = np.sqrt(np.mean((predicted - true_action) ** 2))
        errors.append(np.mean((predicted - true_action) ** 2))
        pixel_errors.append(pixel_error)
        
        if i < 5:
            print(f"\n样本 {i+1}:")
            print(f"   预测像素坐标：{predicted}")
            print(f"   真实像素坐标：{true_action}")
            print(f"   绝对误差：{np.abs(predicted - true_action)}")
            print(f"   像素误差：{pixel_error:.2f} px")
    
    avg_mse = np.mean(errors)
    avg_pixel_error = np.mean(pixel_errors)
    
    print(f"\n📊 总体评估:")
    print(f"   平均 MSE: {avg_mse:.4f}")
    print(f"   平均 RMSE: {np.sqrt(avg_mse):.4f}")
    print(f"   平均像素误差：{avg_pixel_error:.2f} pixels")
    print(f"   测试样本数：{len(errors)}")
    
    # 评价
    print(f"\n💡 性能评价:")
    if avg_pixel_error < 10:
        print(f"   ✅ 优秀！像素误差 < 10px，可以用于实际任务")
    elif avg_pixel_error < 20:
        print(f"   ⚠️  可用，像素误差在合理范围")
    else:
        print(f"   ❌ 需要改进")
    
    # 保存结果
    results = {
        'checkpoint_path': str(checkpoint_path),
        'epoch': checkpoint['epoch'],
        'val_mse': checkpoint['val_loss'],
        'val_pixel_error': checkpoint.get('pixel_error', 0),
        'test_mse': float(avg_mse),
        'test_rmse': float(np.sqrt(avg_mse)),
        'test_pixel_error': float(avg_pixel_error),
        'n_samples': len(errors)
    }
    
    results_path = Path('outputs/checkpoints_normalized_v2/inference_results.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 评估结果已保存：{results_path}")
    
    return avg_pixel_error


if __name__ == "__main__":
    test_normalized_inference()