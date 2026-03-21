"""
任务 1.1: 仿真 3D 视觉相机模块 - 快速测试版
"""

import numpy as np
import open3d as o3d
import cv2
from typing import Dict, Optional
from dataclasses import dataclass
import time


@dataclass
class CameraConfig:
    """仿真相机配置"""
    width: int = 640
    height: int = 480
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 320.0
    cy: float = 240.0
    depth_scale: float = 1000.0
    z_near: float = 0.1
    z_far: float = 5.0
    noise_std: float = 0.001


class SimulatedRGBCamera:
    """仿真 RGB 相机"""
    
    def __init__(self, config: CameraConfig = None):
        self.config = config or CameraConfig()
        
    def generate_synthetic_image(self, scene_type: str = 'grinding') -> np.ndarray:
        """生成合成 RGB 图像"""
        w, h = self.config.width, self.config.height
        image = np.zeros((h, w, 3), dtype=np.uint8)
        
        if scene_type == 'grinding':
            # 绘制工作台
            cv2.rectangle(image, (50, 200), (w-50, h-50), (100, 100, 100), -1)
            
            # 绘制工件（金属色）
            center = (w//2, h//2)
            axes = (150, 80)
            cv2.ellipse(image, center, axes, 0, 0, 360, (180, 170, 150), -1)
            
            # 添加高光
            cv2.ellipse(image, center, (axes[0]-20, axes[1]-20), 0, 0, 360, 
                       (220, 210, 190), -1)
            
            # 机械臂末端
            cv2.circle(image, (w-100, 150), 30, (50, 50, 50), -1)
        
        return image


class SimulatedDepthCamera:
    """仿真深度相机"""
    
    def __init__(self, config: CameraConfig = None):
        self.config = config or CameraConfig()
        
    def generate_depth_image(self) -> np.ndarray:
        """生成深度图像"""
        w, h = self.config.width, self.config.height
        
        # 基础平面（1 米远）
        depth = np.ones((h, w), dtype=np.float32) * 1.0
        
        # 添加一个凸起的工件
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        mask = dist_from_center <= 150
        depth[mask] = 0.9 - 0.05 * np.sin(dist_from_center[mask] / 30)
        
        return depth


class SimulatedPointCloudGenerator:
    """点云生成器"""
    
    def __init__(self, camera_config: CameraConfig = None):
        self.camera_config = camera_config or CameraConfig()
        
    def depth_to_pointcloud(self, 
                           depth_image: np.ndarray,
                           rgb_image: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
        """将深度图转换为点云"""
        h, w = depth_image.shape
        
        # 创建网格
        v = np.arange(h)
        u = np.arange(w)
        u_grid, v_grid = np.meshgrid(u, v)
        
        # 反投影
        fx, fy = self.camera_config.fx, self.camera_config.fy
        cx, cy = self.camera_config.cx, self.camera_config.cy
        
        x = (u_grid - cx) * depth_image / fx
        y = (v_grid - cy) * depth_image / fy
        z = depth_image
        
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if rgb_image is not None:
            colors = rgb_image.reshape(-1, 3) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd


def test_camera():
    """测试相机功能"""
    print("=" * 60)
    print("📷 测试仿真相机")
    print("=" * 60)
    
    config = CameraConfig(width=640, height=480)
    rgb_cam = SimulatedRGBCamera(config)
    depth_cam = SimulatedDepthCamera(config)
    pc_gen = SimulatedPointCloudGenerator(config)
    
    # 生成 RGB 图像
    print("\n1️⃣ 生成 RGB 图像...")
    rgb_image = rgb_cam.generate_synthetic_image('grinding')
    print(f"   ✅ RGB 形状：{rgb_image.shape}")
    
    # 保存 RGB 图像
    cv2.imwrite('projects/grinding_system/data/test_rgb.png', rgb_image)
    print(f"   💾 已保存：projects/grinding_system/data/test_rgb.png")
    
    # 生成深度图
    print("\n2️⃣ 生成深度图像...")
    depth_image = depth_cam.generate_depth_image()
    print(f"   ✅ 深度形状：{depth_image.shape}")
    print(f"   📏 深度范围：{depth_image.min():.3f}m - {depth_image.max():.3f}m")
    
    # 生成点云
    print("\n3️⃣ 生成点云...")
    point_cloud = pc_gen.depth_to_pointcloud(depth_image, rgb_image)
    print(f"   ✅ 点数：{len(point_cloud.points)}")
    
    # 保存点云
    o3d.io.write_point_cloud('projects/grinding_system/data/test_pointcloud.ply', point_cloud)
    print(f"   💾 已保存：projects/grinding_system/data/test_pointcloud.ply")
    
    # 简单统计
    points = np.asarray(point_cloud.points)
    print(f"\n📊 点云统计:")
    print(f"   X 范围：[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"   Y 范围：[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"   Z 范围：[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    print("\n✅ 相机测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_camera()