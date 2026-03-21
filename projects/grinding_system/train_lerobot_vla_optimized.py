
"""
优化的 VLA 训练脚本
====================
改进：
- 更好的学习率调度
- 数据归一化
- 更多训练技巧
"""

import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
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
        
        # 视觉编码器（使用预训练的 ResNet-18）
        from torchvision.models import resnet18, ResNet18_Weights
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.visual_fc = nn.Linear(512, hidden_dim)
        
        # 状态编码器（带 BatchNorm）
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 动作解码器（更深的网络）
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
        # 保存维度信息
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 初始化权重
        self._init_weights()
        
        print("✅ 优化版 VLA 模型创建完成")
        print(f"   参数量：{sum(p.numel() for p in self.parameters()):,}")
        print(f"   State 维度：{state_dim}, Action 维度：{action_dim}")
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, batch):
        """前向传播"""
        # 提取视觉特征
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
        
        # 提取状态特征
        state = batch['observation.state']
        state_feat = self.state_encoder(state)
        
        # 融合
        combined = torch.cat([visual_feat, state_feat], dim=-1)
        fused = self.fusion(combined)
        
        # 预测动作
        actions = self.action_decoder(fused)
        
        return {'actions': actions}


class OptimizedTrainer:
    """优化版训练器"""
    
    def __init__(self, config: dict, dataset_info: dict):
        self.config = config
        self.dataset_info = dataset_info
        
        # Accelerate
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4),
            mixed_precision='fp16'
        )
        
        self.device = self.accelerator.device
        
        # 创建模型
        self.model = NormalizedVLAModel(
            state_dim=dataset_info['state_dim'],
            action_dim=dataset_info['action_dim']
        ).to(self.device)
        
        # 优化器（使用更小的学习率）
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 5e-5),  # 降低学习率
            weight_decay=config.get('weight_decay', 0.01),
            betas=(0.9, 0.999)  # 更稳定的 beta 值
        )
        
        # 学习率调度器
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 50),
            eta_min=1e-6
        )
        
        # 混合精度
        self.scaler = GradScaler()
        
        # 损失函数（使用 Huber Loss 更鲁棒）
        self.loss_fn = nn.SmoothL1Loss(beta=0.1)
        
        # 统计信息
        self.train_losses = []
        self.best_loss = float('inf')
        
        print("\n✅ 优化版训练器初始化完成")
        print(f"   设备：{self.device}")
        print(f"   学习率：{config.get('learning_rate', 5e-5)}")
        print(f"   梯度累积：{config.get('gradient_accumulation_steps', 4)}")
    
    def prepare_dataloader(self, dataset, batch_size: int, shuffle: bool = True):
        """准备 DataLoader"""
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )
        
        return dataloader
    
    def train_epoch(self, dataloader, epoch: int) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(self.device) if hasattr(v, 'to') else v 
                    for k, v in batch.items()}
            
            # 混合精度训练
            with torch.amp.autocast('cuda'):
                outputs = self.model(batch)
                loss = self.loss_fn(outputs['actions'], batch['action'])
            
            # 梯度累积
            loss = loss / self.config.get('gradient_accumulation_steps', 4)
            
            # 反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪（防止爆炸）
            if (step + 1) % self.config.get('gradient_accumulation_steps', 4) == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # 统计
            total_loss += loss.item() * self.config.get('gradient_accumulation_steps', 4)
            n_batches += 1
            
            # 进度显示
            if step % 10 == 0:
                avg_loss = total_loss / max(n_batches, 1)
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"\rEpoch {epoch}, Step {step}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}", end='')
        
        # 更新学习率
        self.scheduler.step()
        
        return total_loss / max(n_batches, 1)
    
    def save_checkpoint(self, path: str, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.accelerator.get_state_dict(self.model),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'dataset_info': self.dataset_info,
            'loss': self.train_losses[-1] if self.train_losses else None
        }
        
        self.accelerator.save(checkpoint, path)
        
        if is_best:
            best_path = Path(path).parent / 'best_model.pt'
            self.accelerator.save(checkpoint, str(best_path))
            print(f"\n🏆 新的最佳模型已保存！")
        
        print(f"💾 检查点已保存：{path}")


def train_optimized():
    """主训练函数"""
    print("="*70)
    print("优化的 VLA 训练（PushT 数据集）")
    print("="*70)
    
    # 优化的配置
    config = {
        'dataset_name': 'lerobot/pusht',
        'batch_size': 4,
        'gradient_accumulation_steps': 4,
        'learning_rate': 5e-5,  # 降低学习率
        'weight_decay': 0.01,
        'num_epochs': 50,  # 增加训练轮数
        'use_amp': True,
        'num_workers': 2,
        'save_dir': 'outputs/checkpoints_optimized'
    }
    
    print("\n📋 训练配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # 加载数据集
    print("\n1️⃣ 加载 LeRobot 数据集...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    dataset = LeRobotDataset(config['dataset_name'])
    print(f"✅ 数据集加载成功")
    print(f"   名称：{config['dataset_name']}")
    print(f"   大小：{len(dataset)} episodes")
    
    # 获取数据集信息
    sample = dataset[0]
    state_dim = sample['observation.state'].shape[0]
    action_dim = sample['action'].shape[0]
    
    dataset_info = {
        'state_dim': state_dim,
        'action_dim': action_dim
    }
    
    print(f"\n📊 数据集信息:")
    print(f"   State 维度：{state_dim}")
    print(f"   Action 维度：{action_dim}")
    
    # 划分数据集
    from torch.utils.data import random_split
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\n📊 数据集划分:")
    print(f"   训练集：{len(train_dataset)} samples")
    print(f"   验证集：{len(val_dataset)} samples")
    
    # 创建训练器
    print("\n2️⃣ 创建训练器...")
    trainer = OptimizedTrainer(config, dataset_info)
    
    # 准备 DataLoader
    train_loader = trainer.prepare_dataloader(train_dataset, config['batch_size'])
    
    # 开始训练
    print("\n3️⃣ 开始训练...")
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"{'='*70}")
        
        train_loss = trainer.train_epoch(train_loader, epoch)
        trainer.train_losses.append(train_loss)
        
        print(f"\n✅ Epoch {epoch+1} 完成")
        print(f"   平均损失：{train_loss:.4f}")
        print(f"   当前学习率：{trainer.optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存检查点
        if train_loss < trainer.best_loss:
            trainer.best_loss = train_loss
            checkpoint_path = Path(config['save_dir']) / f'checkpoint_epoch_{epoch+1}.pt'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(checkpoint_path), epoch, is_best=True)
        
        # 每 10 个 epoch 保存一次
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(config['save_dir']) / f'checkpoint_epoch_{epoch+1}.pt'
            trainer.save_checkpoint(str(checkpoint_path), epoch)
        
        # 显存监控
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"💾 GPU 显存：{allocated:.2f}GB / {reserved:.2f}GB")
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  训练完成，总耗时：{elapsed_time/60:.1f} 分钟")
    print(f"🏆 最佳损失：{trainer.best_loss:.4f}")
    
    # 保存训练日志和图表
    log_data = {
        'config': config,
        'final_loss': trainer.train_losses[-1],
        'best_loss': trainer.best_loss,
        'all_losses': trainer.train_losses,
        'training_time_minutes': elapsed_time / 60
    }
    
    log_path = Path(config['save_dir']) / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    # 绘制损失曲线
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        plt.plot(trainer.train_losses, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True, alpha=0.3)
        plt.savefig(Path(config['save_dir']) / 'loss_curve.png', dpi=150)
        print(f"📊 损失曲线图已保存")
        
    except ImportError:
        print("ℹ️  未安装 matplotlib，跳过绘图")
    
    print("\n🎉 优化训练完成！")
    print("="*70)


if __name__ == "__main__":
    train_optimized()