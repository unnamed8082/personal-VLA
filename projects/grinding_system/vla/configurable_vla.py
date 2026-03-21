
"""
任务 2 增强版：可配置的打磨 VLA 系统
=====================================
功能：
- 可调打磨力参数
- 可调路径密度
- 可调表面质量阈值
- 可视化不同参数的效果
"""

import torch
import numpy as np
import open3d as o3d
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import json


@dataclass
class GrindingConfig:
    """打磨配置参数"""
    
    # 力控制参数
    target_force: float = 20.0        # 目标打磨力 (N)
    force_threshold_contact: float = 2.0   # 接触阈值 (N)
    force_threshold_grinding: float = 15.0  # 打磨力阈值 (N)
    force_kp: float = 0.1             # 力控制 P 增益
    
    # 路径规划参数
    path_step_over: float = 0.01      # 路径行间距 (米)
    path_velocity: float = 0.05       # 打磨速度 (m/s)
    path_smooth_window: int = 3       # 平滑窗口大小
    
    # 视觉参数
    quality_threshold: float = 0.5    # 表面质量阈值
    grinding_area_percentile: int = 70  # 打磨区域百分位
    
    # 仿真参数
    simulation_timestep: float = 0.01  # 仿真时间步长 (秒)
    real_time_factor: float = 1.0     # 实时因子 (1.0=真实速度)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'target_force': self.target_force,
            'force_threshold_contact': self.force_threshold_contact,
            'force_threshold_grinding': self.force_threshold_grinding,
            'force_kp': self.force_kp,
            'path_step_over': self.path_step_over,
            'path_velocity': self.path_velocity,
            'path_smooth_window': self.path_smooth_window,
            'quality_threshold': self.quality_threshold,
            'grinding_area_percentile': self.grinding_area_percentile,
            'simulation_timestep': self.simulation_timestep,
            'real_time_factor': self.real_time_factor
        }


@dataclass
class GrindingAction:
    """打磨动作数据结构"""
    position: np.ndarray
    orientation: np.ndarray
    force_target: np.ndarray
    velocity: float
    timestamp: float


class ConfigurableVisionModule:
    """可配置的视觉模块"""
    
    def __init__(self, config: GrindingConfig):
        self.config = config
        print("✅ 可配置视觉模块初始化完成")
    
    def detect_workpiece(self, point_cloud) -> Dict:
        """检测工件"""
        points = np.asarray(point_cloud.points)
        
        center = points.mean(axis=0)
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        normals = np.asarray(point_cloud.normals)
        
        return {
            'position': center,
            'normal': normals.mean(axis=0),
            'bbox_min': points.min(axis=0),
            'bbox_max': points.max(axis=0),
            'detected': True
        }
    
    def estimate_surface_quality(self, point_cloud) -> float:
        """估计表面质量"""
        points = np.asarray(point_cloud.points)
        std_dev = points.std(axis=0).mean()
        quality = 1.0 / (1.0 + std_dev * 10)
        return quality
    
    def segment_grinding_area(self, point_cloud) -> np.ndarray:
        """分割打磨区域"""
        points = np.asarray(point_cloud.points)
        z_coords = points[:, 2]
        threshold = np.percentile(z_coords, self.config.grinding_area_percentile)
        mask = z_coords > threshold
        return mask
    
    def analyze_quality_distribution(self, point_cloud) -> Dict:
        """分析质量分布"""
        points = np.asarray(point_cloud.points)
        
        # 分区域分析
        n_points = len(points)
        n_regions = 4
        
        region_qualities = []
        for i in range(n_regions):
            start_idx = i * n_points // n_regions
            end_idx = (i + 1) * n_points // n_points
            
            region_points = points[start_idx:end_idx]
            if len(region_points) > 0:
                std_dev = region_points.std(axis=0).mean()
                quality = 1.0 / (1.0 + std_dev * 10)
                region_qualities.append(quality)
        
        return {
            'overall': self.estimate_surface_quality(point_cloud),
            'by_region': region_qualities,
            'needs_grinding': sum(q < self.config.quality_threshold for q in region_qualities)
        }


class ConfigurableForceModule:
    """可配置的力觉模块"""
    
    def __init__(self, config: GrindingConfig):
        self.config = config
        print("✅ 可配置力觉模块初始化完成")
    
    def analyze_contact_state(self, force_data: np.ndarray) -> str:
        """分析接触状态"""
        fz = abs(force_data[2])
        
        if fz < self.config.force_threshold_contact:
            return 'free'
        elif fz < self.config.force_threshold_grinding:
            return 'contact'
        elif fz < self.config.force_threshold_grinding * 1.5:
            return 'grinding'
        else:
            return 'overload'
    
    def compute_force_command(self, current_force: np.ndarray) -> float:
        """计算力控制命令"""
        error = self.config.target_force - current_force[2]
        command = self.config.force_kp * error
        return np.clip(command, -5.0, 5.0)
    
    def simulate_force_response(self, 
                               trajectory: List[GrindingAction],
                               surface_stiffness: float = 1000.0) -> List[np.ndarray]:
        """模拟力响应"""
        force_history = []
        
        for action in trajectory:
            # 简化的力响应模型
            base_force = self.config.target_force
            
            # 添加表面变化引起的力波动
            height_variation = np.random.normal(0, 0.001)
            force_variation = surface_stiffness * height_variation
            
            fz = base_force + force_variation
            fx = np.random.normal(0, 0.5)
            fy = np.random.normal(0, 0.5)
            
            force = np.array([fx, fy, fz, 0.1, 0.1, 0.05])
            force_history.append(force)
        
        return force_history


class ConfigurableMotionModule:
    """可配置的运动规划模块"""
    
    def __init__(self, config: GrindingConfig):
        self.config = config
        print("✅ 可配置运动规划模块初始化完成")
    
    def plan_coverage_path(self, 
                          workpiece_bbox: Tuple[np.ndarray, np.ndarray]) -> List[GrindingAction]:
        """规划全覆盖路径"""
        bbox_min, bbox_max = workpiece_bbox
        
        # 根据配置生成路径
        x_range = np.arange(bbox_min[0], bbox_max[0], self.config.path_step_over)
        y_range = np.arange(bbox_min[1], bbox_max[1], self.config.path_step_over)
        
        trajectory = []
        
        for i, y in enumerate(y_range):
            if i % 2 == 0:
                x_coords = x_range
            else:
                x_coords = x_range[::-1]
            
            for x in x_coords:
                action = GrindingAction(
                    position=np.array([x, y, bbox_max[2]]),
                    orientation=np.array([0, 0, 0, 1]),
                    force_target=np.array([0, 0, self.config.target_force]),
                    velocity=self.config.path_velocity,
                    timestamp=time.time()
                )
                trajectory.append(action)
        
        print(f"✅ 生成打磨路径：{len(trajectory)} 个航点")
        print(f"   行间距：{self.config.path_step_over*1000:.1f} mm")
        print(f"   打磨速度：{self.config.path_velocity*100:.1f} cm/s")
        
        return trajectory
    
    def optimize_trajectory(self, trajectory: List[GrindingAction]) -> List[GrindingAction]:
        """优化轨迹"""
        if len(trajectory) < 3:
            return trajectory
        
        optimized = []
        window = self.config.path_smooth_window
        
        for i in range(len(trajectory)):
            start = max(0, i - window // 2)
            end = min(len(trajectory), i + window // 2 + 1)
            
            positions = [trajectory[j].position for j in range(start, end)]
            avg_position = np.mean(positions, axis=0)
            
            new_action = GrindingAction(
                position=avg_position,
                orientation=trajectory[i].orientation.copy(),
                force_target=trajectory[i].force_target.copy(),
                velocity=trajectory[i].velocity,
                timestamp=trajectory[i].timestamp
            )
            optimized.append(new_action)
        
        return optimized
    
    def compute_path_metrics(self, trajectory: List[GrindingAction]) -> Dict:
        """计算路径指标"""
        if len(trajectory) < 2:
            return {'total_length': 0, 'estimated_time': 0}
        
        total_length = 0
        for i in range(1, len(trajectory)):
            dist = np.linalg.norm(
                trajectory[i].position - trajectory[i-1].position
            )
            total_length += dist
        
        avg_velocity = np.mean([a.velocity for a in trajectory])
        estimated_time = total_length / avg_velocity if avg_velocity > 0 else 0
        
        return {
            'total_length': total_length,
            'estimated_time': estimated_time,
            'waypoints': len(trajectory),
            'avg_segment_length': total_length / (len(trajectory) - 1) if len(trajectory) > 1 else 0
        }


class ConfigurableVLASystem:
    """可配置的 VLA 系统"""
    
    def __init__(self, config: GrindingConfig, device: str = 'cpu'):
        self.config = config
        self.device = torch.device(device)
        
        self.vision_module = ConfigurableVisionModule(config)
        self.force_module = ConfigurableForceModule(config)
        self.motion_module = ConfigurableMotionModule(config)
        
        print("✅ 可配置 VLA 系统初始化完成")
        print(f"   设备：{self.device}")
        print(f"   配置:")
        for key, value in config.to_dict().items():
            print(f"     {key}: {value}")
    
    def perceive_and_plan(self, 
                         rgb_image: np.ndarray,
                         point_cloud) -> Dict:
        """感知 - 规划循环"""
        result = {}
        
        # 视觉理解
        print("\n👁️ 视觉理解...")
        workpiece_info = self.vision_module.detect_workpiece(point_cloud)
        quality = self.vision_module.estimate_surface_quality(point_cloud)
        quality_dist = self.vision_module.analyze_quality_distribution(point_cloud)
        
        result['vision'] = {
            'workpiece': workpiece_info,
            'quality': quality,
            'quality_distribution': quality_dist,
            'grinding_area': int(self.vision_module.segment_grinding_area(point_cloud).sum())
        }
        
        print(f"   工件位置：{workpiece_info['position']}")
        print(f"   表面质量：{quality:.2f} (阈值：{self.config.quality_threshold})")
        print(f"   需要打磨的区域数：{quality_dist['needs_grinding']}/{len(quality_dist['by_region'])}")
        
        # 运动规划
        print("\n📍 运动规划...")
        if workpiece_info['detected']:
            bbox = (workpiece_info['bbox_min'], workpiece_info['bbox_max'])
            trajectory = self.motion_module.plan_coverage_path(bbox)
            trajectory = self.motion_module.optimize_trajectory(trajectory)
            
            path_metrics = self.motion_module.compute_path_metrics(trajectory)
            
            result['motion'] = {
                'trajectory': trajectory,
                'metrics': path_metrics
            }
            print(f"   路径总长度：{path_metrics['total_length']:.2f} m")
            print(f"   预计时间：{path_metrics['estimated_time']:.1f} s")
            print(f"   航点数：{path_metrics['waypoints']}")
        
        return result
    
    def simulate_execution(self, 
                          plan: Dict,
                          visualize: bool = False) -> Dict:
        """仿真执行"""
        print("\n🚀 开始仿真执行...")
        
        trajectory = plan.get('motion', {}).get('trajectory', [])
        
        if len(trajectory) == 0:
            print("⚠️ 无轨迹可执行")
            return {}
        
        # 模拟力响应
        force_history = self.force_module.simulate_force_response(trajectory)
        
        # 统计结果
        forces = np.array(force_history)
        
        execution_result = {
            'completed': True,
            'steps': len(trajectory),
            'force_stats': {
                'mean_fz': float(forces[:, 2].mean()),
                'std_fz': float(forces[:, 2].std()),
                'max_fz': float(forces[:, 2].max()),
                'min_fz': float(forces[:, 2].min())
            },
            'path_metrics': plan['motion']['metrics']
        }
        
        print(f"✅ 仿真执行完成")
        print(f"   总步数：{execution_result['steps']}")
        print(f"   平均 Fz: {execution_result['force_stats']['mean_fz']:.2f} N")
        print(f"   力波动：±{execution_result['force_stats']['std_fz']:.2f} N")
        
        return execution_result


def compare_configs():
    """比较不同配置的效果"""
    print("=" * 70)
    print("参数对比实验")
    print("=" * 70)
    
    # 定义不同的配置
    configs = {
        '精细打磨': GrindingConfig(
            target_force=15.0,
            path_step_over=0.005,
            path_velocity=0.03,
            quality_threshold=0.6
        ),
        '标准打磨': GrindingConfig(
            target_force=20.0,
            path_step_over=0.01,
            path_velocity=0.05,
            quality_threshold=0.5
        ),
        '快速打磨': GrindingConfig(
            target_force=25.0,
            path_step_over=0.02,
            path_velocity=0.10,
            quality_threshold=0.4
        )
    }
    
    # 创建示例点云
    points = np.random.rand(10000, 3).astype(np.float32)
    points[:, :2] *= 0.5
    points[:, 2] = 0.9 + np.random.rand(10000) * 0.1
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    results = []
    
    for name, config in configs.items():
        print(f"\n{'='*70}")
        print(f"测试配置：{name}")
        print(f"{'='*70}")
        
        system = ConfigurableVLASystem(config, device='cpu')
        plan = system.perceive_and_plan(rgb_image, point_cloud)
        result = system.simulate_execution(plan)
        
        results.append({
            'name': name,
            'config': config.to_dict(),
            'result': result
        })
    
    # 保存对比结果
    with open('projects/grinding_system/data/config_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 对比结果已保存：projects/grinding_system/data/config_comparison.json")
    
    # 打印总结
    print(f"\n{'='*70}")
    print("配置对比总结")
    print(f"{'='*70}")
    print(f"{'配置':<10} {'航点数':<8} {'路径长度':<10} {'预计时间':<10} {'平均力':<8}")
    print(f"{'-'*70}")
    
    for r in results:
        metrics = r['result']['path_metrics']
        force_stats = r['result']['force_stats']
        print(f"{r['name']:<10} {metrics['waypoints']:<8} {metrics['total_length']:<10.2f} "
              f"{metrics['estimated_time']:<10.1f} {force_stats['mean_fz']:<8.2f}")
    
    return results


def test_single_config():
    """测试单个配置"""
    print("=" * 70)
    print("单配置测试")
    print("=" * 70)
    
    # 创建自定义配置
    config = GrindingConfig(
        target_force=20.0,
        path_step_over=0.01,
        path_velocity=0.05,
        quality_threshold=0.5
    )
    
    # 创建系统
    system = ConfigurableVLASystem(config, device='cpu')
    
    # 创建测试数据
    points = np.random.rand(10000, 3).astype(np.float32)
    points[:, :2] *= 0.5
    points[:, 2] = 0.9 + np.random.rand(10000) * 0.1
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 运行
    plan = system.perceive_and_plan(rgb_image, point_cloud)
    result = system.simulate_execution(plan)
    
    # 保存详细结果
    detailed_result = {
        'config': config.to_dict(),
        'vision': {
            'quality': plan['vision']['quality'],
            'grinding_area': plan['vision']['grinding_area']
        },
        'motion': plan['motion']['metrics'],
        'execution': result['force_stats']
    }
    
    with open('projects/grinding_system/data/single_config_result.json', 'w') as f:
        json.dump(detailed_result, f, indent=2)
    
    print(f"\n💾 详细结果已保存：projects/grinding_system/data/single_config_result.json")
    
    return config, result


if __name__ == "__main__":
    # 选择运行模式
    mode = input("\n选择测试模式:\n1. 单配置测试\n2. 多配置对比\n\n输入选项 (1/2): ")
    
    if mode == '2':
        compare_configs()
    else:
        test_single_config()