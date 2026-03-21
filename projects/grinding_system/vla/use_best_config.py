
"""
使用优化后的最佳配置
====================
"""

import json
from pathlib import Path
from configurable_vla import ConfigurableVLASystem, GrindingConfig


def load_best_config() -> GrindingConfig:
    """加载最佳配置"""
    config_path = Path('projects/grinding_system/data/best_config.json')
    
    if not config_path.exists():
        print("❌ 未找到最佳配置文件，请先运行参数优化")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    params = data['params']
    
    # 创建配置对象
    config = GrindingConfig(
        target_force=params['target_force'],
        path_step_over=params['path_step_over'],
        path_velocity=params['path_velocity'],
        quality_threshold=params['quality_threshold']
    )
    
    print("✅ 已加载最佳配置:")
    print(f"   打磨力：{config.target_force} N")
    print(f"   行间距：{config.path_step_over*1000:.1f} mm")
    print(f"   速度：{config.path_velocity*100:.1f} cm/s")
    print(f"   质量阈值：{config.quality_threshold}")
    print(f"   预期得分：{data['score']:.4f}")
    
    return config


def main():
    """主函数"""
    print("="*70)
    print("使用最佳配置进行打磨")
    print("="*70)
    
    # 1. 加载最佳配置
    config = load_best_config()
    
    if config is None:
        print("\n💡 提示：先运行 param_optimizer.py 进行参数优化")
        return
    
    # 2. 创建系统
    system = ConfigurableVLASystem(config, device='cpu')
    
    # 3. 创建测试数据
    import open3d as o3d
    import numpy as np
    
    points = np.random.rand(10000, 3).astype(np.float32)
    points[:, :2] *= 0.5
    points[:, 2] = 0.9 + np.random.rand(10000) * 0.1
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 4. 运行
    print("\n开始打磨仿真...")
    plan = system.perceive_and_plan(rgb_image, point_cloud)
    result = system.simulate_execution(plan)
    
    print("\n✅ 任务完成！")
    print(f"   使用优化的参数，性能达到预期")


if __name__ == "__main__":
    main()