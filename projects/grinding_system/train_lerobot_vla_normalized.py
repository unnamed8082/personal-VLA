
"""
归一化版 VLA 训练（关键改进）
==============================
问题诊断：
- 当前 MSE: 83.1 (太高)
- RMSE: 9.12 (平均误差 9 个单位)
- 原因：Action 数据范围可能很大，未归一化

解决方案：
1. 对 Action 进行归一化（减均值除标准差）
2. 训练时使用归一化后的 Action
3. 推理时反归一化输出
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from accelerate import Accelerator
from pathlib import Path
import time
import json
import numpy as np


class NormalizedVLAModel(nn.Module):
    """带归一化的 VLA 模型"""
    
    def __init__(self, state_dim: int = 2, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        
        # 视觉编码器
        from torchvision.models import resnet18, ResNet18_Weights
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.visual_fc = nn.Linear(512, hidden_dim)
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 动作解码器
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        print("✅ 归一化版 VLA 模型创建完成")
        print(f"   参数量：{sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, batch):
        """前向传播"""
        # 视觉特征
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
        
        # 状态特征
        state = batch['observation.state']
        state_feat = self.state_encoder(state)
        
        # 融合
        combined = torch.cat([visual_feat, state_feat], dim=-1)
        fused = self.fusion(combined)
        
        # 预测动作
        actions = self.action_decoder(fused)
        
        return {'actions': actions}


def compute_dataset_stats(dataset):
    """计算数据集的归一化统计量"""
    print("\n📊 计算数据集统计量...")
    
    all_actions = []
    all_states = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        all_actions.append(sample['action'].numpy())
        all_states.append(sample['observation.state'].numpy())
    
    actions = np.array(all_actions)
    states = np.array(all_states)
    
    stats = {
        'action_mean': actions.mean(axis=0).tolist(),
        'action_std': actions.std(axis=0).tolist(),
        'action_min': actions.min(axis=0).tolist(),
        'action_max': actions.max(axis=0).tolist(),
        'state_mean': states.mean(axis=0).tolist(),
        'state_std': states.std(axis=0).tolist()
    }
    
    print(f"   Action 范围：[{actions.min():.2f}, {actions.max():.2f}]")
    print(f"   Action 均值：{actions.mean(axis=0)}")
    print(f"   Action 标准差：{actions.std(axis=0)}")
    
    return stats


def train_normalized():
    """主训练函数"""
    print("="*70)
    print("归一化版 VLA 训练（PushT 数据集）- 修复版")
    print("="*70)
    
    # 配置
    config = {
        'dataset_name': 'lerobot/pusht',
        'batch_size': 4,
        'gradient_accumulation_steps': 4,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_epochs': 50,
        'use_amp': True,
        'num_workers': 2,
        'save_dir': 'outputs/checkpoints_normalized_v2'
    }
    
    print("\n📋 训练配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # 1. 加载数据集
    print("\n1️⃣ 加载 LeRobot 数据集...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    dataset = LeRobotDataset(config['dataset_name'])
    print(f"✅ 数据集加载成功")
    print(f"   大小：{len(dataset)} episodes")
    
    # 2. 计算归一化统计量（使用整个数据集）
    norm_stats = compute_dataset_stats(dataset)
    
    print(f"\n💡 Action 数据分析:")
    print(f"   范围：[{norm_stats['action_min']}, {norm_stats['action_max']}]")
    print(f"   这是 2D 像素坐标（图像中的目标点）")
    print(f"   建议归一化到 [-1, 1] 或 [0, 1] 范围")
    
    # 保存统计量
    stats_path = Path(config['save_dir']) / 'normalization_stats.json'
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    print(f"💾 归一化统计已保存：{stats_path}")
    
    # 3. 获取数据集信息
    sample = dataset[0]
    dataset_info = {
        'state_dim': sample['observation.state'].shape[0],
        'action_dim': sample['action'].shape[0]
    }
    
    # 4. 划分数据集（使用固定随机种子保证可复现性）
    from torch.utils.data import random_split
    
    torch.manual_seed(42)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\n📊 数据集划分:")
    print(f"   训练集：{len(train_dataset)} samples")
    print(f"   验证集：{len(val_dataset)} samples")
    
    # 5. 创建训练器
    print("\n2️⃣ 创建训练器...")
    
    model = NormalizedVLAModel(
        state_dim=dataset_info['state_dim'],
        action_dim=dataset_info['action_dim']
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        mixed_precision='fp16'
    )
    
    device = accelerator.device
    model = model.to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scaler = GradScaler()
    loss_fn = nn.MSELoss()
    
    # 归一化参数（转换为 tensor）
    action_mean = torch.tensor(norm_stats['action_mean']).to(device)
    action_std = torch.tensor(norm_stats['action_std']).to(device) + 1e-6
    
    print(f"\n✅ 训练器初始化完成")
    print(f"   设备：{device}")
    print(f"   Action 归一化：mean={norm_stats['action_mean']}, std={norm_stats['action_std']}")
    
    # 6. 准备 DataLoader
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    # 7. 开始训练
    print("\n3️⃣ 开始训练...")
    start_time = time.time()
    
    best_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"{'='*70}")
        
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for step, batch in enumerate(train_loader):
            # 准备数据
            batch = {k: v.to(device) if hasattr(v, 'to') else v 
                    for k, v in batch.items()}
            
            # 归一化 Action
            action_raw = batch['action']
            action_norm = (action_raw - action_mean) / action_std
            
            # 前向传播
            with torch.amp.autocast('cuda'):
                outputs = model(batch)
                # 使用归一化后的 action 计算损失
                loss = loss_fn(outputs['actions'], action_norm)
            
            # 梯度累积
            loss = loss / config['gradient_accumulation_steps']
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 更新
            if (step + 1) % config['gradient_accumulation_steps'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # 统计
            total_loss += loss.item() * config['gradient_accumulation_steps']
            n_batches += 1
            
            # 进度
            if step % 10 == 0:
                avg_loss = total_loss / max(n_batches, 1)
                print(f"\rEpoch {epoch}, Step {step}, Loss: {avg_loss:.4f}", end='')
        
        avg_train_loss = total_loss / max(n_batches, 1)
        print(f"\n✅ 训练完成，平均损失：{avg_train_loss:.4f}")
        
        # 验证（每 5 个 epoch）
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0.0
            val_n = 0
            
            all_pred_denorm = []
            all_true = []
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                pin_memory=True
            )
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) if hasattr(v, 'to') else v 
                            for k, v in batch.items()}
                    
                    action_raw = batch['action']
                    action_norm = (action_raw - action_mean) / action_std
                    
                    outputs = model(batch)
                    
                    # 反归一化预测
                    pred_denorm = outputs['actions'] * action_std + action_mean
                    
                    # 计算反归一化后的损失（真实世界的损失）
                    loss = loss_fn(pred_denorm, action_raw)
                    
                    val_loss += loss.item()
                    val_n += 1
                    
                    all_pred_denorm.append(pred_denorm.cpu())
                    all_true.append(action_raw.cpu())
            
            avg_val_loss = val_loss / val_n
            
            # 计算像素误差（更有意义的指标）
            all_pred_denorm = torch.cat(all_pred_denorm, dim=0)
            all_true = torch.cat(all_true, dim=0)
            pixel_error = torch.sqrt(torch.mean((all_pred_denorm - all_true) ** 2, dim=1)).mean().item()
            
            print(f"📊 验证集 MSE Loss: {avg_val_loss:.4f}")
            print(f"   验证集 RMSE: {np.sqrt(avg_val_loss):.4f}")
            print(f"   平均像素误差：{pixel_error:.2f} pixels")
            
            # 保存最佳模型
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                checkpoint_path = Path(config['save_dir']) / 'best_model_normalized.pt'
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'dataset_info': dataset_info,
                    'normalization_stats': norm_stats,
                    'val_loss': avg_val_loss,
                    'pixel_error': pixel_error
                }
                
                torch.save(checkpoint, str(checkpoint_path))
                print(f"🏆 新的最佳模型！验证 MSE: {avg_val_loss:.4f}")
                print(f"   像素误差：{pixel_error:.2f} pixels")
                print(f"   💾 已保存：{checkpoint_path}")
        
        # 显存监控
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"💾 GPU 显存：{allocated:.2f}GB / {reserved:.2f}GB")
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  训练完成，总耗时：{elapsed_time/60:.1f} 分钟")
    print(f"🏆 最佳验证 MSE: {best_loss:.4f}")
    print(f"🏆 最佳验证 RMSE: {np.sqrt(best_loss):.4f}")
    
    # 保存训练日志
    log_data = {
        'config': config,
        'best_val_loss': best_loss,
        'training_time_minutes': elapsed_time / 60,
        'normalization_stats': norm_stats
    }
    
    log_path = Path(config['save_dir']) / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n💾 训练日志已保存：{log_path}")
    print("\n🎉 归一化训练完成！")
    print("="*70)


if __name__ == "__main__":
    train_normalized()