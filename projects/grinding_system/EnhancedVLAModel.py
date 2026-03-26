"""
增强版 VLA 模型
================
基于视觉-语言-动作（VLA）的深度学习模型，用于高精度打磨任务控制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class EnhancedVLAModel(nn.Module):
    """
    增强版 VLA 模型
    
    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        hidden_dim: 隐藏层维度
        force_dim: 力觉维度
    """
    
    def __init__(self, state_dim: int = 2, action_dim: int = 2, hidden_dim: int = 512, force_dim: int = 6):
        super(EnhancedVLAModel, self).__init__()
        
        # 视觉编码器
        from torchvision.models import resnet34, ResNet34_Weights
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        
        self.visual_encoder = nn.Sequential(
            *list(resnet.children())[:-1],
            SEBlock(512)
        )
        self.visual_fc = nn.Linear(512, hidden_dim)
        
        # 力觉编码器
        self.force_encoder = nn.Sequential(
            nn.Linear(force_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # 动作解码器
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
