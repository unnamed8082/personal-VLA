"""
任务 2: VLA 大模型系统 - 基于 SmolVLA 的打磨专用模型
=====================================================
功能模块：
1. 视觉理解：工件识别、位姿估计
2. 力觉分析：接触状态判断、力控制
3. 运动规划：打磨轨迹生成
"""

import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class GrindingAction:
    """打磨动作数据结构"""
    position: np.ndarray      # [x, y, z] 位置
    orientation: np.ndarray   # [qx, qy, qz, qw] 四元数
    force_target: np.ndarray  # [fx, fy, fz] 目标力
    velocity: float           # 打磨速度 (m/s)
    timestamp: float          # 时间戳


class VisionUnderstandingModule:
    """视觉理解模块 - 从图像和点云中提取信息"""
    
    def __init__(self):
        print("✅ 视觉理解模块初始化完成")
    
    def detect_workpiece(self, 
                        rgb_image: np.ndarray,
                        point_cloud) -> Dict:
        """
        检测工件位置和姿态
        Args:
            rgb_image: RGB 图像
            point_cloud: Open3D 点云对象
        Returns:
            workpiece_info: 工件信息字典
        """
        # 简化版本：假设工件在中心
        points = np.asarray(point_cloud.points)
        
        # 计算点云中心
        center = points.mean(axis=0)
        
        # 计算法线（用于估计表面朝向）
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        normals = np.asarray(point_cloud.normals)
        avg_normal = normals.mean(axis=0)
        
        workpiece_info = {
            'position': center,
            'normal': avg_normal,
            'bbox_min': points.min(axis=0),
            'bbox_max': points.max(axis=0),
            'detected': True
        }
        
        return workpiece_info
    
    def estimate_surface_quality(self, 
                                point_cloud) -> float:
        """
        估计表面质量（粗糙度）
        Returns:
            quality_score: 0-1 之间，1 表示非常光滑
        """
        points = np.asarray(point_cloud.points)
        
        # 计算点的分布标准差（简化指标）
        std_dev = points.std(axis=0).mean()
        
        # 转换为质量分数（假设标准差越小越光滑）
        quality = 1.0 / (1.0 + std_dev * 10)
        
        return quality
    
    def segment_grinding_area(self, 
                             point_cloud,
                             target_area: str = 'center') -> np.ndarray:
        """
        分割打磨区域
        Returns:
            area_mask: 布尔掩码，True 表示需要打磨的区域
        """
        points = np.asarray(point_cloud.points)
        
        # 简单版本：选择 Z 坐标最高的区域（最靠近相机的区域）
        z_coords = points[:, 2]
        threshold = np.percentile(z_coords, 70)
        
        mask = z_coords > threshold
        
        return mask


class ForceAnalysisModule:
    """力觉分析模块 - 分析力数据并提供控制建议"""
    
    def __init__(self):
        self.force_threshold_contact = 2.0  # 接触阈值 (N)
        self.force_threshold_grinding = 15.0  # 打磨力阈值 (N)
        print("✅ 力觉分析模块初始化完成")
    
    def analyze_contact_state(self, force_data: np.ndarray) -> str:
        """
        分析接触状态
        Args:
            force_data: [Fx, Fy, Fz, Mx, My, Mz]
        Returns:
            state: 'free', 'contact', 'grinding', 'overload'
        """
        fz = abs(force_data[2])  # Z 方向力
        
        if fz < self.force_threshold_contact:
            return 'free'
        elif fz < self.force_threshold_grinding:
            return 'contact'
        elif fz < self.force_threshold_grinding * 1.5:
            return 'grinding'
        else:
            return 'overload'
    
    def compute_force_error(self, 
                           current_force: np.ndarray,
                           target_force: float) -> float:
        """
        计算力误差
        Returns:
            error: 当前力与目标力的差值
        """
        current_fz = current_force[2]
        error = target_force - current_fz
        return error
    
    def generate_force_command(self, 
                              error: float,
                              kp: float = 0.1,
                              ki: float = 0.01,
                              kd: float = 0.05) -> float:
        """
        PID 力控制器
        Returns:
            force_command: 力控制命令
        """
        # 简化版本：只有 P 控制
        command = kp * error
        return np.clip(command, -5.0, 5.0)  # 限制在±5N


class MotionPlanningModule:
    """运动规划模块 - 生成打磨轨迹"""
    
    def __init__(self):
        print("✅ 运动规划模块初始化完成")
    
    def plan_coverage_path(self, 
                          workpiece_bbox: Tuple[np.ndarray, np.ndarray],
                          step_over: float = 0.01) -> List[GrindingAction]:
        """
        规划全覆盖打磨路径
        Args:
            workpiece_bbox: (min, max) 工件边界框
            step_over: 行间距 (米)
        Returns:
            trajectory: 打磨动作列表
        """
        bbox_min, bbox_max = workpiece_bbox
        
        # 生成网格路径
        x_range = np.arange(bbox_min[0], bbox_max[0], step_over)
        y_range = np.arange(bbox_min[1], bbox_max[1], step_over)
        
        trajectory = []
        
        # 之字形路径
        for i, y in enumerate(y_range):
            # 交替方向
            if i % 2 == 0:
                x_coords = x_range
            else:
                x_coords = x_range[::-1]
            
            for x in x_coords:
                action = GrindingAction(
                    position=np.array([x, y, bbox_max[2]]),
                    orientation=np.array([0, 0, 0, 1]),  # 单位四元数
                    force_target=np.array([0, 0, 20.0]),  # 20N 向下力
                    velocity=0.05,  # 5 cm/s
                    timestamp=time.time()
                )
                trajectory.append(action)
        
        print(f"✅ 生成打磨路径：{len(trajectory)} 个航点")
        return trajectory
    
    def optimize_trajectory(self, 
                          trajectory: List[GrindingAction],
                          smooth_weight: float = 0.5) -> List[GrindingAction]:
        """
        优化轨迹（平滑）
        Returns:
            optimized_trajectory: 优化后的轨迹
        """
        if len(trajectory) < 3:
            return trajectory
        
        # 简单平滑：移动平均
        optimized = []
        window = 3
        
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


class GrindingVLASystem:
    """打磨 VLA 系统集成"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 初始化各个模块
        self.vision_module = VisionUnderstandingModule()
        self.force_module = ForceAnalysisModule()
        self.motion_module = MotionPlanningModule()
        
        # 加载 SmolVLA 模型（如果可用）
        self.vla_model = self._load_vla_model()
        
        print("✅ 打磨 VLA 系统初始化完成")
        print(f"   设备：{self.device}")
    
    def _load_vla_model(self):
        """加载 VLA 模型"""
        try:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            
            print("正在加载 SmolVLA 基础模型...")
            model = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(self.device)
            model.eval()
            
            print("✅ SmolVLA 模型加载成功")
            return model
            
        except Exception as e:
            print(f"⚠️ 无法加载 SmolVLA 模型：{e}")
            print("   将使用规则基系统代替")
            return None
    
    def perceive_and_plan(self, 
                         rgb_image: np.ndarray,
                         point_cloud,
                         force_data: Optional[np.ndarray] = None) -> Dict:
        """
        感知 - 规划循环
        Returns:
            plan: 包含视觉信息、力状态、规划轨迹的字典
        """
        result = {}
        
        # 1. 视觉理解
        print("\n👁️ 视觉理解...")
        workpiece_info = self.vision_module.detect_workpiece(rgb_image, point_cloud)
        quality = self.vision_module.estimate_surface_quality(point_cloud)
        grinding_mask = self.vision_module.segment_grinding_area(point_cloud)
        
        result['vision'] = {
            'workpiece': workpiece_info,
            'quality': quality,
            'grinding_area': grinding_mask.sum()
        }
        
        print(f"   工件位置：{workpiece_info['position']}")
        print(f"   表面质量：{quality:.2f}")
        
        # 2. 力觉分析
        print("\n🔧 力觉分析...")
        if force_data is not None:
            contact_state = self.force_module.analyze_contact_state(force_data)
            result['force'] = {
                'state': contact_state,
                'raw': force_data
            }
            print(f"   接触状态：{contact_state}")
        else:
            result['force'] = {'state': 'unknown'}
            print("   无外力数据")
        
        # 3. 运动规划
        print("\n📍 运动规划...")
        if workpiece_info['detected']:
            bbox = (workpiece_info['bbox_min'], workpiece_info['bbox_max'])
            trajectory = self.motion_module.plan_coverage_path(bbox)
            trajectory = self.motion_module.optimize_trajectory(trajectory)
            
            result['motion'] = {
                'trajectory': trajectory,
                'waypoints': len(trajectory)
            }
            print(f"   规划航点数：{len(trajectory)}")
        else:
            result['motion'] = {'trajectory': [], 'waypoints': 0}
            print("   未检测到工件，无法规划")
        
        return result
    
    def execute_simulation(self, 
                          plan: Dict,
                          real_time: bool = False) -> List[Dict]:
        """
        执行仿真
        Returns:
            execution_log: 执行日志
        """
        print("\n🚀 开始仿真执行...")
        
        trajectory = plan.get('motion', {}).get('trajectory', [])
        execution_log = []
        
        for i, action in enumerate(trajectory):
            log_entry = {
                'step': i,
                'position': action.position.tolist(),
                'force_target': action.force_target.tolist(),
                'timestamp': action.timestamp,
                'status': 'executed'
            }
            execution_log.append(log_entry)
            
            if real_time:
                time.sleep(action.velocity * 0.1)  # 仿真执行时间
            
            # 进度显示
            if i % 10 == 0:
                print(f"\r  进度：{i}/{len(trajectory)} ({i/len(trajectory)*100:.1f}%)", end='')
        
        print(f"\n✅ 仿真执行完成，总步数：{len(execution_log)}")
        return execution_log


def test_vla_system():
    """测试 VLA 系统"""
    print("=" * 70)
    print("任务 2: 测试打磨 VLA 系统")
    print("=" * 70)
    
    # 创建系统 - 强制使用 CPU
    vla_system = GrindingVLASystem(device='cpu')
    
    # 加载之前的测试数据
    import pickle
    try:
        with open('projects/grinding_system/data/sample_frame.pkl', 'rb') as f:
            frame_data = pickle.load(f)
        
        rgb_image = frame_data['rgb']
        point_cloud = frame_data['point_cloud']
        
        print("\n✅ 加载测试数据成功")
        
    except FileNotFoundError:
        print("⚠️ 未找到 sample_frame.pkl，使用示例数据")
        # 创建示例数据
        import cv2
        
        rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(rgb_image, (50, 200), (590, 430), (100, 100, 100), -1)
        
        points = np.random.rand(10000, 3).astype(np.float32)
        points[:, :2] *= 0.5
        points[:, 2] = 0.9 + np.random.rand(10000) * 0.1
        
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
    
    # 感知 - 规划
    print("\n开始感知 - 规划循环...")
    plan = vla_system.perceive_and_plan(
        rgb_image=rgb_image,
        point_cloud=point_cloud,
        force_data=np.array([0, 0, 5.0, 0, 0, 0])  # 示例力数据
    )
    
    # 执行仿真
    execution_log = vla_system.execute_simulation(plan, real_time=False)
    
    # 保存结果 - 修复 JSON 序列化问题
    import json
    
    # 将所有 numpy 数组转换为列表
    def convert_to_serializable(obj):
        """递归地将 numpy 数组转换为列表"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_plan = convert_to_serializable({
        'vision': {
            'workpiece': plan['vision']['workpiece'],
            'quality': plan['vision']['quality'],
            'grinding_area': int(plan['vision']['grinding_area'])
        },
        'force': plan['force'],
        'motion': {
            'waypoints': plan['motion']['waypoints']
        }
    })
    
    with open('projects/grinding_system/data/vla_plan.json', 'w') as f:
        json.dump(serializable_plan, f, indent=2)
    
    print(f"\n💾 规划结果已保存：projects/grinding_system/data/vla_plan.json")
    
    print("\n✅ VLA 系统测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    test_vla_system()