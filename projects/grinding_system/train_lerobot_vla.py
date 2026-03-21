
"""
基于 LeRobot 数据集的 VLA 训练脚本
===================================
特性:
- 直接使用 LeRobotDataset
- 支持 RTX 4060 8GB 显存优化
- 混合精度训练 + 梯度累积
- Accelerate 分布式支持
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from accelerate import Accelerator
from pathlib import Path
import time
import json


class SimpleVLAModel(nn.Module):
    """简化版 VLA 模型（用于快速验证）"""
    
    def __init__(self, image_size: tuple = (3, 96, 96), state_dim: int = 5, action_dim: int = 2):
        super().__init__()
        
        # 视觉编码器（简化的 ResNet-18）
        from torchvision.models import resnet18, ResNet18_Weights
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.visual_fc = nn.Linear(512, 256)
        
        # 状态编码器 - 使用实际的 state_dim
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
        print(f"   State 维度：{state_dim}, Action 维度：{action_dim}")
    
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


class LeRobotVLATrainer:
    """LeRobot VLA 训练器"""
    
    def __init__(self, config: dict, dataset_info: dict = None):
        """
        Args:
            config: 训练配置
            dataset_info: 数据集信息（包含 state_dim 和 action_dim）
        """
        self.config = config
        
        # 初始化 Accelerate
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4),
            mixed_precision='fp16' if config.get('use_amp', True) else 'no'
        )
        
        # 设备
        self.device = self.accelerator.device
        
        # 根据数据集信息创建模型
        if dataset_info:
            state_dim = dataset_info.get('state_dim', 5)
            action_dim = dataset_info.get('action_dim', 2)
            self.model = SimpleVLAModel(state_dim=state_dim, action_dim=action_dim).to(self.device)
        else:
            self.model = SimpleVLAModel().to(self.device)
        
        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # 混合精度 scaler
        self.scaler = GradScaler() if config.get('use_amp', True) and self.device.type == 'cuda' else None
        
        # 损失函数
        self.loss_fn = nn.MSELoss()
        
        print("\n✅ 训练器初始化完成")
        print(f"   设备：{self.device}")
        print(f"   混合精度：{config.get('use_amp', True)}")
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
            drop_last=True
        )
        
        return dataloader
    
    def train_epoch(self, dataloader, epoch: int) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for step, batch in enumerate(dataloader):
            # 准备数据
            batch = {k: v.to(self.device) if hasattr(v, 'to') else v 
                    for k, v in batch.items()}
            
            # 前向传播（混合精度）- 使用正确的 API
            if self.device.type == 'cuda' and self.config.get('use_amp', True):
                with torch.amp.autocast('cuda'):
                    outputs = self.model(batch)
                    loss = self.loss_fn(outputs['actions'], batch['action'])
            else:
                outputs = self.model(batch)
                loss = self.loss_fn(outputs['actions'], batch['action'])
            
            # 归一化损失（考虑梯度累积）
            loss = loss / self.config.get('gradient_accumulation_steps', 4)
            
            # 反向传播
            if self.scaler and self.device.type == 'cuda':
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 更新权重（梯度累积后）
            if (step + 1) % self.config.get('gradient_accumulation_steps', 4) == 0:
                if self.scaler and self.device.type == 'cuda':
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # 统计
            total_loss += loss.item() * self.config.get('gradient_accumulation_steps', 4)
            n_batches += 1
            
            # 进度显示
            if step % 10 == 0:
                avg_loss = total_loss / n_batches
                print(f"\rEpoch {epoch}, Step {step}, Loss: {avg_loss:.4f}", end='')
        
        return total_loss / n_batches
    
    def save_checkpoint(self, path: str, epoch: int):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.accelerator.get_state_dict(self.model),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        self.accelerator.save(checkpoint, path)
        print(f"💾 检查点已保存：{path}")


def train_with_lerobot():
    """主训练函数"""
    print("="*70)
    print("基于 LeRobot 数据集的 VLA 训练")
    print("="*70)
    
    # 训练配置（针对 RTX 4060 8GB 优化）
    # 使用 PushT 数据集（无需认证，快速测试）
    config = {
        'dataset_name': 'lerobot/pusht',  # 改为 PushT
        'batch_size': 4,
        'gradient_accumulation_steps': 4,  # 等效 batch_size=16
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_epochs': 10,
        'use_amp': True,  # 启用混合精度
        'num_workers': 2,
        'save_dir': 'outputs/checkpoints'
    }
    
    print("\n📋 训练配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # 1. 加载 LeRobot 数据集
    print("\n1️⃣ 加载 LeRobot 数据集...")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        
        dataset = LeRobotDataset(config['dataset_name'])
        print(f"✅ 数据集加载成功")
        print(f"   名称：{config['dataset_name']}")
        print(f"   大小：{len(dataset)} episodes")
        
        # 显示数据集特征
        if hasattr(dataset, 'features'):
            print(f"\n📊 数据集特征:")
            for key in list(dataset.features.keys())[:10]:  # 只显示前 10 个
                print(f"   - {key}")
        
        # 查看第一个样本的形状，获取实际的维度
        print(f"\n🔍 检查第一个样本...")
        sample = dataset[0]
        
        state_dim = 5  # 默认值
        action_dim = 2  # 默认值
        
        print(f"   样本键:")
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"     - {key}: {value.shape}")
                
                # 提取实际的 state 和 action 维度
                if key == 'observation.state':
                    state_dim = value.shape[0]
                elif key == 'action':
                    action_dim = value.shape[0]
        
        dataset_info = {
            'state_dim': state_dim,
            'action_dim': action_dim
        }
        
        print(f"\n📊 检测到:")
        print(f"   State 维度：{state_dim}")
        print(f"   Action 维度：{action_dim}")
        
    except Exception as e:
        print(f"❌ 数据集加载失败：{e}")
        print("\n💡 提示:")
        print("  1. 确保已安装 LeRobot: pip install lerobot")
        print("  2. 检查网络连接")
        print("  3. 如果使用 Bridge 数据集，需要先设置 HuggingFace Token")
        print("     python projects/grinding_system/data/test_lerobot_dataset.py")
        return
    
    # 2. 划分训练集和验证集
    from torch.utils.data import random_split
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\n📊 数据集划分:")
    print(f"   训练集：{len(train_dataset)} samples")
    print(f"   验证集：{len(val_dataset)} samples")
    
    # 3. 创建训练器（传入数据集信息）
    print("\n2️⃣ 创建训练器...")
    trainer = LeRobotVLATrainer(config, dataset_info=dataset_info)
    
    # 4. 准备 DataLoader
    print("\n3️⃣ 准备 DataLoader...")
    train_loader = trainer.prepare_dataloader(train_dataset, config['batch_size'])
    val_loader = trainer.prepare_dataloader(val_dataset, config['batch_size'], shuffle=False)
    
    # 5. 开始训练
    print("\n4️⃣ 开始训练...")
    start_time = time.time()
    
    best_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"{'='*70}")
        
        # 训练
        train_loss = trainer.train_epoch(train_loader, epoch)
        print(f"\n✅ 训练完成，平均损失：{train_loss:.4f}")
        
        # 验证（简单版本，可以省略）
        # val_loss = trainer.train_epoch(val_loader, epoch)
        # print(f"✅ 验证完成，平均损失：{val_loss:.4f}")
        
        # 保存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path = Path(config['save_dir']) / 'best_model.pt'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(checkpoint_path), epoch)
            print(f"🏆 新的最佳模型！损失：{best_loss:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = Path(config['save_dir']) / f'checkpoint_epoch_{epoch+1}.pt'
            trainer.save_checkpoint(str(checkpoint_path), epoch)
        
        # 显存监控
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"\n💾 GPU 显存使用：{allocated:.2f}GB / {reserved:.2f}GB")
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  训练完成，总耗时：{elapsed_time/60:.1f} 分钟")
    
    # 保存训练日志
    log_data = {
        'config': config,
        'final_loss': train_loss,
        'best_loss': best_loss,
        'training_time_minutes': elapsed_time / 60,
        'dataset_info': dataset_info
    }
    
    log_path = Path(config['save_dir']) / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n💾 训练日志已保存：{log_path}")
    print("\n🎉 VLA 模型训练完成！")
    print("="*70)


if __name__ == "__main__":
    train_with_lerobot()