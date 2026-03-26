
"""
多模态 VLA 推理测试（视觉 + 力觉 + 状态）
=========================================
测试改进后的三模态融合模型在仿真环境中的表现

功能：
1. 对比有无力量输入的模型输出差异
2. 仿真打磨任务全流程测试
3. 显存占用和推理速度评估

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 注意力模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean(dim=(2, 3))
        y = F.relu(self.fc1(y))
        y = F.sigmoid(self.fc2(y))
        return x * y.view(b, c, 1, 1)


class EnhancedVLAModel(nn.Module):
    """增强版 VLA 模型（支持力觉输入）"""
    
    def __init__(self, state_dim: int = 2, action_dim: int = 2, hidden_dim: int = 512, force_dim: int = 6):
        super().__init__()
        
        from torchvision.models import resnet34, ResNet34_Weights
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        
        self.visual_encoder = nn.Sequential(
            *list(resnet.children())[:-1],
            SEBlock(512)
        )
        self.visual_fc = nn.Linear(512, hidden_dim)
        
        self.force_encoder = nn.Sequential(
            nn.Linear(force_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.force_dim = force_dim
    
    def forward(self, batch):
        batch_size = batch['observation.state'].shape[0]
        device = batch['observation.state'].device
        
        visual_feat = None
        image_keys = ['observation.image', 'observation.images.top', 'observation.images.rgb']
        
        for img_key in image_keys:
            if img_key in batch:
                images = batch[img_key]
                visual_feat = self.visual_encoder(images).squeeze(-1).squeeze(-1)
                visual_feat = self.visual_fc(visual_feat)
                break
        
        if visual_feat is None:
            visual_feat = torch.zeros(batch_size, 512).to(device)
        
        force_feat = None
        if 'observation.force' in batch:
            force_data = batch['observation.force']
            force_feat = self.force_encoder(force_data)
        else:
            force_feat = torch.zeros(batch_size, 512).to(device)
        
        state = batch['observation.state']
        state_feat = self.state_encoder(state)
        
        combined = torch.cat([visual_feat, force_feat, state_feat], dim=-1)
        fused = self.fusion(combined)
        
        actions = self.action_decoder(fused)
        
        return {'actions': actions}

def create_test_batch(batch_size=4, include_force=True, device='cpu'):
    """创建测试批次数据"""
    batch = {
        'observation.state': torch.randn(batch_size, 2).to(device),
        'observation.image': torch.randn(batch_size, 3, 224, 224).to(device),
    }
    
    if include_force:
        batch['observation.force'] = torch.randn(batch_size, 6).to(device)
    
    return batch


def compare_models():
    """比较有无力量输入的模型输出差异"""
    print("="*70)
    print("多模态 VLA 对比测试")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 使用设备：{device}")
    
    checkpoint_path = Path('outputs/checkpoints_enhanced/best_model_enhanced.pt')
    
    if checkpoint_path.exists():
        print(f"\n📥 加载模型：{checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = EnhancedVLAModel(
            state_dim=checkpoint['dataset_info']['state_dim'],
            action_dim=checkpoint['dataset_info']['action_dim'],
            force_dim=checkpoint['dataset_info']['force_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print("✅ 模型加载成功")
    else:
        print("\n⚠️ 未找到训练好的模型，使用随机初始化模型")
        model = EnhancedVLAModel(state_dim=2, action_dim=2, force_dim=6)
        model.to(device)
    
    model.eval()
    
    # 测试 1: 有力量输入
    print("\n" + "-"*70)
    print("测试 1: 有力量输入")
    print("-"*70)
    
    batch_with_force = create_test_batch(batch_size=8, include_force=True, device=device)
    
    with torch.no_grad():
        output_with_force = model(batch_with_force)
    
    pred_actions_with = output_with_force['actions'].cpu().numpy()
    print(f"预测动作（有力量）:")
    print(f"  均值：[{pred_actions_with[:, 0].mean():.4f}, {pred_actions_with[:, 1].mean():.4f}]")
    print(f"  标准差：[{pred_actions_with[:, 0].std():.4f}, {pred_actions_with[:, 1].std():.4f}]")
    print(f"  范围：[{pred_actions_with.min():.4f}, {pred_actions_with.max():.4f}]")
    
    # 测试 2: 无力量输入
    print("\n" + "-"*70)
    print("测试 2: 无力量输入（零填充）")
    print("-"*70)
    
    batch_without_force = create_test_batch(batch_size=8, include_force=False, device=device)
    
    with torch.no_grad():
        output_without_force = model(batch_without_force)
    
    pred_actions_without = output_without_force['actions'].cpu().numpy()
    print(f"预测动作（无力量）:")
    print(f"  均值：[{pred_actions_without[:, 0].mean():.4f}, {pred_actions_without[:, 1].mean():.4f}]")
    print(f"  标准差：[{pred_actions_without[:, 0].std():.4f}, {pred_actions_without[:, 1].std():.4f}]")
    print(f"  范围：[{pred_actions_without.min():.4f}, {pred_actions_without.max():.4f}]")
    
    # 对比分析
    print("\n" + "="*70)
    print("对比分析")
    print("="*70)
    
    diff = np.abs(pred_actions_with - pred_actions_without)
    print(f"\n💡 力量输入的影响:")
    print(f"  平均差异：{diff.mean():.4f}")
    print(f"  最大差异：{diff.max():.4f}")
    print(f"  差异标准差：{diff.std():.4f}")
    
    if diff.mean() > 0.01:
        print("\n✅ 力量输入对模型输出有显著影响，多模态融合有效！")
    else:
        print("\n⚠️ 力量输入影响较小，可能需要重新训练模型")
    
    # 推理速度测试
    print("\n" + "-"*70)
    print("推理速度测试")
    print("-"*70)
    
    n_iterations = 100
    
    start = time.time()
    for _ in range(n_iterations):
        batch = create_test_batch(batch_size=1, include_force=True, device=device)
        with torch.no_grad():
            _ = model(batch)
    elapsed_with_force = time.time() - start
    fps_with_force = n_iterations / elapsed_with_force
    
    start = time.time()
    for _ in range(n_iterations):
        batch = create_test_batch(batch_size=1, include_force=False, device=device)
        with torch.no_grad():
            _ = model(batch)
    elapsed_without_force = time.time() - start
    fps_without_force = n_iterations / elapsed_without_force
    
    print(f"  有力量输入：{fps_with_force:.1f} FPS")
    print(f"  无力量输入：{fps_without_force:.1f} FPS")
    print(f"  额外开销：{(elapsed_with_force - elapsed_without_force)/n_iterations*1000:.2f} ms/样本")
    
    # 显存占用
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\n💾 GPU 显存占用：{allocated:.2f} GB / {reserved:.2f} GB")
    
    print("\n🎉 对比测试完成！")
    print("="*70)
    
    return {
        'diff_mean': diff.mean(),
        'diff_max': diff.max(),
        'fps_with_force': fps_with_force,
        'fps_without_force': fps_without_force
    }


def simulate_grinding_task():
    """仿真打磨任务全流程测试"""
    print("\n" + "="*70)
    print("仿真打磨任务测试")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_path = Path('outputs/checkpoints_enhanced/best_model_enhanced.pt')
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = EnhancedVLAModel(
            state_dim=checkpoint['dataset_info']['state_dim'],
            action_dim=checkpoint['dataset_info']['action_dim'],
            force_dim=checkpoint['dataset_info']['force_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("✅ 模型加载成功")
    else:
        print("⚠️ 使用随机模型进行演示")
        model = EnhancedVLAModel(state_dim=2, action_dim=2, force_dim=6)
        model.to(device)
        model.eval()
    
    n_steps = 50
    print(f"\n🔧 仿真打磨过程：{n_steps} 步")
    
    history = {
        'position': [],
        'force': [],
        'action': []
    }
    
    start_time = time.time()
    
    for step in range(n_steps):
        t = step / n_steps
        
        position = np.array([0.5 * t, 0.3, 0.95])
        velocity = np.array([0.05, 0.0, 0.0])
        
        fz = 20.0 + 2.0 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.5)
        fx = 3.0 * np.sin(2 * np.pi * t * 0.5) + np.random.normal(0, 0.3)
        fy = 3.0 * np.cos(2 * np.pi * t * 0.5) + np.random.normal(0, 0.3)
        force = np.array([fx, fy, fz, 0.1, 0.1, 0.05])
        
        batch = {
            'observation.state': torch.FloatTensor(position[:2]).unsqueeze(0).to(device),
            'observation.image': torch.randn(1, 3, 224, 224).to(device),
            'observation.force': torch.FloatTensor(force).unsqueeze(0).to(device)
        }
        
        with torch.no_grad():
            output = model(batch)
        
        action = output['actions'].cpu().numpy()[0]
        
        history['position'].append(position.copy())
        history['force'].append(force.copy())
        history['action'].append(action.copy())
        
        if step % 10 == 0:
            current_time = time.time() - start_time
            print(f"\r  步数：{step}/{n_steps}, "
                  f"Fz={fz:.2f}N, "
                  f"Action=[{action[0]:.4f}, {action[1]:.4f}], "
                  f"耗时：{current_time:.1f}s", end='')
    
    total_time = time.time() - start_time
    print(f"\n\n✅ 仿真完成，总耗时：{total_time:.1f}s")
    
    forces = np.array(history['force'])
    actions = np.array(history['action'])
    
    print(f"\n📊 统计:")
    print(f"   法向力均值：{forces[:, 2].mean():.2f} N")
    print(f"   法向力波动：±{forces[:, 2].std():.2f} N")
    print(f"   动作平滑度：{np.diff(actions, axis=0).std():.4f}")
    print(f"   推理频率：{n_steps/total_time:.1f} Hz")
    
    return history


def test_batch_inference():
    """测试批量推理性能"""
    print("\n" + "="*70)
    print("批量推理性能测试")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = EnhancedVLAModel(state_dim=2, action_dim=2, force_dim=6)
    model.to(device)
    model.eval()
    
    batch_sizes = [1, 4, 8, 16, 32]
    
    print(f"\n🔧 测试不同 batch size 的推理性能...")
    print(f"{'Batch Size':<12} {'Time (ms)':<12} {'FPS':<12}")
    print("-"*36)
    
    results = {}
    
    for bs in batch_sizes:
        batch = create_test_batch(batch_size=bs, include_force=True, device=device)
        
        start = time.time()
        with torch.no_grad():
            _ = model(batch)
        elapsed = time.time() - start
        
        time_per_sample = elapsed / bs * 1000
        fps = bs / elapsed
        
        results[bs] = {'time': time_per_sample, 'fps': fps}
        print(f"{bs:<12} {time_per_sample:<12.2f} {fps:<12.1f}")
    
    best_bs = max(results.keys(), key=lambda k: results[k]['fps'])
    print(f"\n✅ 最佳 batch size: {best_bs} ({results[best_bs]['fps']:.1f} FPS)")
    
    return results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🤖 多模态 VLA 推理测试套件")
    print("="*70)
    
    results = {}
    
    try:
        result1 = compare_models()
        results['comparison'] = result1
    except Exception as e:
        print(f"\n❌ 对比测试失败：{e}")
        import traceback
        traceback.print_exc()
    
    try:
        result2 = simulate_grinding_task()
        results['simulation'] = result2
    except Exception as e:
        print(f"\n❌ 仿真测试失败：{e}")
        import traceback
        traceback.print_exc()
    
    try:
        result3 = test_batch_inference()
        results['batch_test'] = result3
    except Exception as e:
        print(f"\n❌ 批量测试失败：{e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("🎉 所有测试完成！")
    print("="*70)
    
    if len(results) > 0:
        import json
        report_path = Path('outputs/multimodal_test_report.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {
                    k: (float(v) if isinstance(v, (np.floating, float)) else v)
                    for k, v in value.items()
                }
        
        with open(report_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"\n💾 测试报告已保存：{report_path}")
