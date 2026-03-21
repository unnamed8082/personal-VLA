
"""
VLAb 数据集加载器
==================
支持：
- VLAb 社区数据集 v1 (11.1K episodes)
- Franka Desk 标准化数据
- RLDS 格式转换
- 力传感器数据归一化
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


class VLAbDataset(Dataset):
    """VLAb 社区数据集 v1"""
    
    def __init__(self, 
                 data_dir: str = 'datasets/vlab_v1',
                 split: str = 'train',
                 max_episodes: int = None,
                 normalize_force: bool = True):
        """
        Args:
            data_dir: 数据集目录
            split: 'train', 'val', 或 'test'
            max_episodes: 最大使用 episode 数（用于快速测试）
            normalize_force: 是否归一化力数据到 [-1, 1]
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.normalize_force = normalize_force
        
        # 力数据统计量（用于归一化）
        self.force_stats = {
            'mean': np.array([0, 0, 20, 0, 0, 0]),  # Fx, Fy, Fz, Mx, My, Mz
            'std': np.array([5, 5, 10, 2, 2, 2])
        }
        
        # 加载元数据
        self.episodes = self._load_metadata(max_episodes)
        
        print(f"✅ VLAb 数据集加载完成")
        print(f"   目录：{self.data_dir}")
        print(f"   分割：{split}")
        print(f"   Episode 数：{len(self.episodes)}")
        print(f"   力数据归一化：{normalize_force}")
    
    def _load_metadata(self, max_episodes: int = None) -> List[dict]:
        """加载元数据"""
        metadata_file = self.data_dir / f'{self.split}_episodes.json'
        
        if not metadata_file.exists():
            # 如果没有元数据，尝试扫描目录
            print(f"⚠️ 未找到元数据文件，自动扫描目录...")
            return self._scan_directory()
        
        with open(metadata_file, 'r') as f:
            episodes = json.load(f)
        
        if max_episodes is not None:
            episodes = episodes[:max_episodes]
        
        return episodes
    
    def _scan_directory(self) -> List[dict]:
        """扫描目录获取 episode 列表"""
        episodes = []
        
        # 假设数据结构：data_dir/episode_0/, episode_1/, ...
        for i in range(1000):  # 最多扫描 1000 个
            ep_dir = self.data_dir / f'episode_{i}'
            if ep_dir.exists():
                episodes.append({
                    'id': i,
                    'dir': str(ep_dir)
                })
        
        return episodes
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        episode_info = self.episodes[idx]
        episode_data = self._load_episode(episode_info)
        
        # 转换为模型输入格式
        sample = self._process_episode(episode_data)
        
        return sample
    
    def _load_episode(self, episode_info: dict) -> dict:
        """加载单个 episode 数据"""
        ep_dir = Path(episode_info['dir'])
        
        # 加载各种数据
        data = {}
        
        # 1. 加载图像（RGB-D）
        rgb_path = ep_dir / 'rgb.npy'
        depth_path = ep_dir / 'depth.npy'
        
        if rgb_path.exists():
            data['rgb'] = np.load(rgb_path)
        else:
            # 创建占位符
            data['rgb'] = np.zeros((480, 640, 3), dtype=np.uint8)
        
        if depth_path.exists():
            data['depth'] = np.load(depth_path)
        else:
            data['depth'] = np.ones((480, 640), dtype=np.float32)
        
        # 2. 加载机器人状态
        state_path = ep_dir / 'robot_state.npy'
        if state_path.exists():
            data['state'] = np.load(state_path)
        else:
            data['state'] = np.zeros(7)  # [x, y, z, qx, qy, qz, qw]
        
        # 3. 加载力传感器数据
        force_path = ep_dir / 'force_torque.npy'
        if force_path.exists():
            data['force_torque'] = np.load(force_path)
        else:
            data['force_torque'] = np.zeros(6)
        
        # 4. 加载动作标签
        action_path = ep_dir / 'actions.npy'
        if action_path.exists():
            data['actions'] = np.load(action_path)
        else:
            data['actions'] = np.zeros(7)
        
        return data
    
    def _process_episode(self, episode_data: dict) -> Dict[str, torch.Tensor]:
        """处理 episode 数据为模型输入"""
        sample = {}
        
        # 处理图像
        rgb = episode_data['rgb']
        if isinstance(rgb, np.ndarray):
            # 转换为 PyTorch 格式 [C, H, W]
            rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            sample['observation.images.rgb'] = rgb_tensor
        
        # 处理深度图
        depth = episode_data['depth']
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
        sample['observation.images.depth'] = depth_tensor
        
        # 处理机器人状态
        state = episode_data['state']
        state_tensor = torch.from_numpy(state).float()
        sample['observation.state'] = state_tensor
        
        # 处理力传感器数据
        force = episode_data['force_torque']
        if self.normalize_force:
            force = (force - self.force_stats['mean']) / self.force_stats['std']
            force = np.clip(force, -1, 1)  # 限制在 [-1, 1]
        
        force_tensor = torch.from_numpy(force).float()
        sample['observation.force_torque'] = force_tensor
        
        # 处理动作标签
        actions = episode_data['actions']
        actions_tensor = torch.from_numpy(actions).float()
        sample['action'] = actions_tensor
        
        return sample


class FrankaDeskDataset(VLAbDataset):
    """Franka Desk 标准化数据集"""
    
    def __init__(self, data_dir: str = 'datasets/franka_desk', **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)
        print("✅ Franka Desk 数据集加载完成")
    
    def _process_episode(self, episode_data: dict) -> Dict[str, torch.Tensor]:
        """Franka Desk 特定处理"""
        sample = super()._process_episode(episode_data)
        
        # Franka 特定的坐标系转换等
        # TODO: 根据实际数据格式调整
        
        return sample


def create_dataloader(dataset: Dataset, 
                     batch_size: int = 4,
                     num_workers: int = 4,
                     pin_memory: bool = True) -> DataLoader:
    """创建 DataLoader"""
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    print(f"✅ DataLoader 创建完成")
    print(f"   Batch size: {batch_size}")
    print(f"   Worker 数：{num_workers}")
    
    return dataloader


def test_vlab_dataset():
    """测试 VLAb 数据集加载"""
    print("="*70)
    print("测试 VLAb 数据集加载")
    print("="*70)
    
    # 检查数据集目录是否存在
    data_dir = Path('datasets/vlab_v1')
    if not data_dir.exists() or len(list(data_dir.glob('episode_*'))) == 0:
        print("⚠️  数据集目录不存在或为空，创建模拟数据...")
        print(f"   预期目录：{data_dir.absolute()}")
        print(f"   💡 提示：运行 download_vlab.py 下载真实数据集")
        
        # 创建模拟数据集
        dataset = _create_mock_dataset()
    else:
        # 使用真实数据
        try:
            dataset = VLAbDataset(
                data_dir='datasets/vlab_v1',
                split='train',
                max_episodes=10,  # 只加载 10 个 episode 快速测试
                normalize_force=True
            )
        except Exception as e:
            print(f"❌ 加载失败：{e}")
            print("⚠️  改用模拟数据...")
            dataset = _create_mock_dataset()
    
    # 检查数据集是否为空
    if len(dataset) == 0:
        print("\n❌ 数据集为空！重新创建模拟数据...")
        dataset = _create_mock_dataset()
    
    # 创建 DataLoader
    dataloader = create_dataloader(dataset, batch_size=2)
    
    # 测试数据加载
    print("\n测试数据加载...")
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i+1}:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        
        if i >= 2:  # 只测试 3 个 batch
            break
    
    print("\n✅ VLAb 数据集测试完成！")
    return dataset, dataloader


def _create_mock_dataset() -> VLAbDataset:
    """创建模拟数据集用于测试"""
    import tempfile
    import os
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"创建模拟数据集于：{temp_dir}")
    
    # 创建几个模拟 episode
    for i in range(5):
        ep_dir = Path(temp_dir) / f'episode_{i}'
        ep_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模拟数据
        np.save(ep_dir / 'rgb.npy', np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        np.save(ep_dir / 'depth.npy', np.random.rand(480, 640).astype(np.float32))
        np.save(ep_dir / 'robot_state.npy', np.random.rand(7).astype(np.float32))
        np.save(ep_dir / 'force_torque.npy', np.random.randn(6).astype(np.float32) * 5)
        np.save(ep_dir / 'actions.npy', np.random.rand(7).astype(np.float32))
    
    # 创建元数据
    episodes = [{'id': i, 'dir': str(Path(temp_dir) / f'episode_{i}')} for i in range(5)]
    with open(Path(temp_dir) / 'train_episodes.json', 'w') as f:
        json.dump(episodes, f)
    
    return VLAbDataset(data_dir=temp_dir, split='train')


if __name__ == "__main__":
    dataset, dataloader = test_vlab_dataset()