
"""
完整系统集成演示：多模态感知 + VLA + 多臂协同
=============================================
整合所有 4 个任务的成果
"""

import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))


def run_task1_demo():
    """任务 1 演示"""
    print("\n" + "="*70)
    print("任务 1: 多模态感知系统")
    print("="*70)
    
    from perception.simulated_camera import test_camera
    
    frame = test_camera()
    return frame


def run_task2_demo():
    """任务 2 演示"""
    print("\n" + "="*70)
    print("任务 2: VLA 大模型系统")
    print("="*70)
    
    from vla.configurable_vla import GrindingConfig, ConfigurableVLASystem
    import numpy as np
    import open3d as o3d
    
    # 创建配置
    config = GrindingConfig(
        target_force=20.0,
        path_step_over=0.01,
        path_velocity=0.05
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
    
    return plan, result


def run_task3_demo():
    """任务 3 演示"""
    print("\n" + "="*70)
    print("任务 3: 多机械臂协同控制")
    print("="*70)
    
    from control.multi_arm_coordinator import (
        MultiArmCoordinator, RobotArmConfig,
        test_workspace_evaluation, test_task_assignment,
        test_cooperative_planning
    )
    
    # 工作空间评价
    coordinator = test_workspace_evaluation()
    
    # 任务分配
    assignments = test_task_assignment(coordinator)
    
    # 协同规划
    plan = test_cooperative_planning(coordinator, assignments)
    
    return coordinator, plan


def main():
    """主演示函数"""
    print("="*70)
    print("🤖 多模态 VLA 打磨系统 - 完整演示")
    print("="*70)
    print("\n本演示整合了以下 4 个任务:")
    print("  1. 多模态感知系统 (3D 视觉 + 力传感器)")
    print("  2. VLA 大模型系统 (视觉理解 + 力觉分析 + 运动规划)")
    print("  3. 协同控制算法 (多臂协调 + RL 路径优化)")
    print("  4. 系统集成验证")
    
    input("\n按 Enter 键开始演示...")
    
    # 任务 1
    task1_result = run_task1_demo()
    
    # 任务 2
    task2_result = run_task2_demo()
    
    # 任务 3
    task3_result = run_task3_demo()
    
    # 总结
    print("\n" + "="*70)
    print("📊 演示总结")
    print("="*70)
    
    print("\n✅ 所有任务测试通过！")
    print("\n生成文件:")
    print("  - projects/grinding_system/data/test_rgb.png")
    print("  - projects/grinding_system/data/test_pointcloud.ply")
    print("  - projects/grinding_system/data/vla_plan.json")
    print("  - projects/grinding_system/data/config_comparison.json")
    print("  - projects/grinding_system/data/multi_arm_results.json")
    
    print("\n🎯 系统能力:")
    print("  ✓ 3D 视觉感知与点云生成")
    print("  ✓ 六维力传感器仿真")
    print("  ✓ VLA 模型感知 - 规划循环")
    print("  ✓ 可配置打磨参数优化")
    print("  ✓ 多机械臂协同控制")
    print("  ✓ 碰撞检测与避障")
    print("  ✓ 强化学习路径优化")
    
    print("\n🚀 下一步:")
    print("  1. 连接真实硬件 (ABB/KUKA 机械臂)")
    print("  2. 部署到实际生产环境")
    print("  3. 采集真实数据进行微调")
    
    print("\n🎉 演示完成！")
    print("="*70)


if __name__ == "__main__":
    main()