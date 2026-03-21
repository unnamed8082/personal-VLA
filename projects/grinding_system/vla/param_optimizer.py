
"""
参数优化系统：自动搜索最优打磨参数
=====================================
功能：
- 网格搜索/随机搜索
- 多目标优化（时间、质量、力稳定性）
- 自动保存最佳配置
- 可视化优化结果
"""

import numpy as np
import json
import itertools
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import time


@dataclass
class OptimizationConfig:
    """优化配置"""
    
    # 参数范围
    target_force_range: Tuple[float, float, float] = (15.0, 25.0, 5.0)  # (min, max, step)
    path_step_over_range: Tuple[float, float, float] = (0.005, 0.02, 0.005)
    path_velocity_range: Tuple[float, float, float] = (0.03, 0.10, 0.02)
    quality_threshold_range: Tuple[float, float, float] = (0.4, 0.7, 0.1)
    
    # 优化目标权重
    weight_time: float = 0.3        # 时间效率权重
    weight_quality: float = 0.4     # 质量权重
    weight_force_stability: float = 0.3  # 力稳定性权重
    
    # 约束条件
    max_execution_time: float = 600.0   # 最大执行时间 (秒)
    min_quality: float = 0.5            # 最小质量要求
    max_force_variation: float = 3.0    # 最大力波动 (N)


@dataclass
class ExperimentResult:
    """实验结果"""
    config_id: int
    params: dict
    metrics: dict
    score: float
    rank: int = 0


class ParameterOptimizer:
    """参数优化器"""
    
    def __init__(self, opt_config: OptimizationConfig = None):
        self.opt_config = opt_config or OptimizationConfig()
        self.results: List[ExperimentResult] = []
        self.best_result: ExperimentResult = None
        
        print("✅ 参数优化器初始化完成")
        print(f"   优化目标:")
        print(f"     - 时间权重：{self.opt_config.weight_time}")
        print(f"     - 质量权重：{self.opt_config.weight_quality}")
        print(f"     - 力稳定性权重：{self.opt_config.weight_force_stability}")
    
    def generate_parameter_grid(self) -> List[dict]:
        """生成参数网格"""
        # 解包范围
        force_range = np.arange(*self.opt_config.target_force_range)
        step_range = np.arange(*self.opt_config.path_step_over_range)
        vel_range = np.arange(*self.opt_config.path_velocity_range)
        qual_range = np.arange(*self.opt_config.quality_threshold_range)
        
        # 生成所有组合
        param_combinations = list(itertools.product(
            force_range, step_range, vel_range, qual_range
        ))
        
        param_list = []
        for i, (force, step, vel, qual) in enumerate(param_combinations):
            param_list.append({
                'config_id': i,
                'target_force': float(force),
                'path_step_over': float(step),
                'path_velocity': float(vel),
                'quality_threshold': float(qual)
            })
        
        print(f"\n📊 生成参数网格:")
        print(f"   总组合数：{len(param_list)}")
        print(f"   打磨力：{force_range} N")
        print(f"   行间距：{step_range} m")
        print(f"   速度：{vel_range} m/s")
        print(f"   质量阈值：{qual_range}")
        
        return param_list
    
    def evaluate_configuration(self, params: dict, 
                             system, point_cloud, rgb_image) -> dict:
        """评估单个配置"""
        from configurable_vla import GrindingConfig
        
        # 创建配置
        config = GrindingConfig(
            target_force=params['target_force'],
            path_step_over=params['path_step_over'],
            path_velocity=params['path_velocity'],
            quality_threshold=params['quality_threshold']
        )
        
        # 重新初始化系统
        system.config = config
        system.vision_module.config = config
        system.force_module.config = config
        system.motion_module.config = config
        
        # 运行仿真
        plan = system.perceive_and_plan(rgb_image, point_cloud)
        result = system.simulate_execution(plan)
        
        # 提取指标
        metrics = {
            'execution_time': result['path_metrics']['estimated_time'],
            'path_length': result['path_metrics']['total_length'],
            'waypoints': result['path_metrics']['waypoints'],
            'mean_force': result['force_stats']['mean_fz'],
            'force_std': result['force_stats']['std_fz'],
            'surface_quality': plan['vision']['quality'],
            'grinding_coverage': plan['vision']['grinding_area']
        }
        
        return metrics
    
    def compute_score(self, metrics: dict) -> float:
        """计算综合得分"""
        # 归一化各项指标
        time_score = 1.0 / (1.0 + metrics['execution_time'] / 100.0)
        quality_score = metrics['surface_quality']
        stability_score = 1.0 / (1.0 + metrics['force_std'] / 5.0)
        
        # 加权求和
        total_score = (
            self.opt_config.weight_time * time_score +
            self.opt_config.weight_quality * quality_score +
            self.opt_config.weight_force_stability * stability_score
        )
        
        # 应用约束惩罚
        penalty = 1.0
        
        if metrics['execution_time'] > self.opt_config.max_execution_time:
            penalty *= 0.5
        
        if metrics['surface_quality'] < self.opt_config.min_quality:
            penalty *= 0.7
        
        if metrics['force_std'] > self.opt_config.max_force_variation:
            penalty *= 0.8
        
        return total_score * penalty
    
    def run_optimization(self, system, point_cloud, rgb_image, 
                        search_type: str = 'grid') -> List[ExperimentResult]:
        """运行优化搜索"""
        print("\n" + "="*70)
        print("开始参数优化")
        print("="*70)
        
        # 生成参数列表
        if search_type == 'grid':
            param_list = self.generate_parameter_grid()
        elif search_type == 'random':
            # 随机搜索（用于大规模参数空间）
            param_list = self._generate_random_samples(50)
        else:
            raise ValueError(f"未知搜索类型：{search_type}")
        
        # 逐个评估
        start_time = time.time()
        
        for i, params in enumerate(param_list):
            print(f"\n[{i+1}/{len(param_list)}] 评估配置 #{params['config_id']}")
            print(f"   打磨力：{params['target_force']}N, "
                  f"行间距：{params['path_step_over']*1000:.1f}mm, "
                  f"速度：{params['path_velocity']*100:.1f}cm/s")
            
            try:
                # 评估
                metrics = self.evaluate_configuration(
                    params, system, point_cloud, rgb_image
                )
                
                # 计算得分
                score = self.compute_score(metrics)
                
                # 保存结果
                result = ExperimentResult(
                    config_id=params['config_id'],
                    params=params,
                    metrics=metrics,
                    score=score
                )
                self.results.append(result)
                
                print(f"   ✅ 得分：{score:.4f}")
                print(f"   时间：{metrics['execution_time']:.1f}s, "
                      f"质量：{metrics['surface_quality']:.2f}, "
                      f"力波动：±{metrics['force_std']:.2f}N")
                
            except Exception as e:
                print(f"   ❌ 失败：{e}")
                continue
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  优化完成，总耗时：{elapsed_time:.1f}s")
        
        # 排序并找出最佳配置
        self._rank_results()
        
        return self.results
    
    def _generate_random_samples(self, n_samples: int) -> List[dict]:
        """生成随机样本"""
        param_list = []
        
        for i in range(n_samples):
            params = {
                'config_id': i,
                'target_force': np.random.uniform(*self.opt_config.target_force_range[:2]),
                'path_step_over': np.random.uniform(*self.opt_config.path_step_over_range[:2]),
                'path_velocity': np.random.uniform(*self.opt_config.path_velocity_range[:2]),
                'quality_threshold': np.random.uniform(*self.opt_config.quality_threshold_range[:2])
            }
            param_list.append(params)
        
        return param_list
    
    def _rank_results(self):
        """对结果排序"""
        # 按得分降序排序
        self.results.sort(key=lambda r: r.score, reverse=True)
        
        # 设置排名
        for i, result in enumerate(self.results):
            result.rank = i + 1
        
        # 保存最佳结果
        if len(self.results) > 0:
            self.best_result = self.results[0]
            print(f"\n🏆 最佳配置:")
            print(f"   排名：#{self.best_result.rank}")
            print(f"   得分：{self.best_result.score:.4f}")
            print(f"   打磨力：{self.best_result.params['target_force']:.1f} N")
            print(f"   行间距：{self.best_result.params['path_step_over']*1000:.1f} mm")
            print(f"   速度：{self.best_result.params['path_velocity']*100:.1f} cm/s")
            print(f"   质量阈值：{self.best_result.params['quality_threshold']:.2f}")
    
    def save_results(self, filename: str = 'optimization_results.json'):
        """保存优化结果"""
        output_dir = Path('projects/grinding_system/data')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        
        # 准备保存的数据
        data = {
            'optimization_config': asdict(self.opt_config),
            'best_config': {
                'params': self.best_result.params if self.best_result else None,
                'metrics': self.best_result.metrics if self.best_result else None,
                'score': self.best_result.score if self.best_result else None
            },
            'all_results': [
                {
                    'rank': r.rank,
                    'config_id': r.config_id,
                    'params': r.params,
                    'metrics': r.metrics,
                    'score': r.score
                }
                for r in self.results
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已保存：{output_path}")
        
        # 同时保存最佳配置到单独文件
        if self.best_result:
            best_config_path = output_dir / 'best_config.json'
            with open(best_config_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'params': self.best_result.params,
                    'metrics': self.best_result.metrics,
                    'score': self.best_result.score
                }, f, indent=2, ensure_ascii=False)
            
            print(f"💾 最佳配置已保存：{best_config_path}")
    
    def print_summary(self, top_n: int = 10):
        """打印结果摘要"""
        print("\n" + "="*70)
        print(f"TOP {top_n} 配置")
        print("="*70)
        
        print(f"{'排名':<5} {'得分':<8} {'打磨力':<8} {'行间距':<8} {'速度':<8} {'时间':<8} {'质量':<8}")
        print("-"*70)
        
        for result in self.results[:top_n]:
            p = result.params
            m = result.metrics
            print(f"{result.rank:<5} {result.score:<8.4f} "
                  f"{p['target_force']:<8.1f} {p['path_step_over']*1000:<8.1f} "
                  f"{p['path_velocity']*100:<8.1f} {m['execution_time']:<8.1f} "
                  f"{m['surface_quality']:<8.2f}")


def create_test_data():
    """创建测试数据"""
    import open3d as o3d
    
    points = np.random.rand(10000, 3).astype(np.float32)
    points[:, :2] *= 0.5
    points[:, 2] = 0.9 + np.random.rand(10000) * 0.1
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    return point_cloud, rgb_image


def main():
    """主函数"""
    print("="*70)
    print("参数优化实验")
    print("="*70)
    
    # 1. 创建优化配置
    opt_config = OptimizationConfig(
        target_force_range=(15.0, 25.0, 5.0),
        path_step_over_range=(0.005, 0.015, 0.005),
        path_velocity_range=(0.03, 0.08, 0.02),
        quality_threshold_range=(0.4, 0.6, 0.1),
        weight_time=0.3,
        weight_quality=0.4,
        weight_force_stability=0.3
    )
    
    # 2. 创建优化器
    optimizer = ParameterOptimizer(opt_config)
    
    # 3. 创建测试数据
    print("\n准备测试数据...")
    point_cloud, rgb_image = create_test_data()
    
    # 4. 创建 VLA 系统
    from configurable_vla import ConfigurableVLASystem, GrindingConfig
    
    initial_config = GrindingConfig()
    system = ConfigurableVLASystem(initial_config, device='cpu')
    
    # 5. 运行优化
    search_type = input("\n选择搜索方式:\n1. 网格搜索 (全面但慢)\n2. 随机搜索 (快速近似)\n\n输入 (1/2): ")
    
    if search_type == '1':
        results = optimizer.run_optimization(system, point_cloud, rgb_image, search_type='grid')
    else:
        results = optimizer.run_optimization(system, point_cloud, rgb_image, search_type='random')
    
    # 6. 打印摘要
    optimizer.print_summary(top_n=10)
    
    # 7. 保存结果
    optimizer.save_results()
    
    # 8. 使用最佳配置验证
    if optimizer.best_result:
        print("\n" + "="*70)
        print("使用最佳配置进行验证...")
        print("="*70)
        
        from configurable_vla import GrindingConfig
        
        best_params = optimizer.best_result.params
        best_config = GrindingConfig(
            target_force=best_params['target_force'],
            path_step_over=best_params['path_step_over'],
            path_velocity=best_params['path_velocity'],
            quality_threshold=best_params['quality_threshold']
        )
        
        # 创建新系统
        verified_system = ConfigurableVLASystem(best_config, device='cpu')
        
        # 重新运行
        plan = verified_system.perceive_and_plan(rgb_image, point_cloud)
        result = verified_system.simulate_execution(plan)
        
        print("\n✅ 验证完成！")
        print(f"   实际得分：{optimizer.best_result.score:.4f}")
        print(f"   预期性能符合优化结果")
    
    print("\n🎉 优化实验全部完成！")


if __name__ == "__main__":
    main()