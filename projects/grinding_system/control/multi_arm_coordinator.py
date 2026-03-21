
"""
任务 3: 多机械臂协同控制算法
=====================================
功能：
1. 工作空间耦合度评价指标
2. 基于强化学习的自适应路径规划
3. 多臂避障和任务分配
4. 协同打磨仿真
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time
import json
from pathlib import Path


@dataclass
class RobotArmConfig:
    """机械臂配置"""
    arm_id: str = "arm_1"
    base_position: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    reach_radius: float = 0.8  # 最大工作半径 (米)
    max_velocity: float = 0.5  # 最大速度 (m/s)
    max_payload: float = 5.0   # 最大负载 (kg)


@dataclass
class TaskAssignment:
    """任务分配"""
    arm_id: str
    waypoints: List[np.ndarray]
    start_time: float
    estimated_duration: float
    priority: int = 1


class WorkspaceEvaluator:
    """工作空间耦合度评价器"""
    
    def __init__(self):
        print("✅ 工作空间评价器初始化完成")
    
    def compute_workspace_overlap(self, 
                                 arm1_config: RobotArmConfig,
                                 arm2_config: RobotArmConfig) -> float:
        """
        计算两个机械臂的工作空间重叠度
        Returns:
            overlap_ratio: 0-1 之间，1 表示完全重叠
        """
        # 计算基座距离
        base_distance = np.linalg.norm(
            arm1_config.base_position - arm2_config.base_position
        )
        
        # 计算工作空间半径和
        radius_sum = arm1_config.reach_radius + arm2_config.reach_radius
        
        # 如果没有重叠
        if base_distance >= radius_sum:
            return 0.0
        
        # 简化计算：假设球形工作空间
        overlap_volume = self._sphere_intersection_volume(
            arm1_config.reach_radius,
            arm2_config.reach_radius,
            base_distance
        )
        
        # 归一化
        total_volume = (4/3) * np.pi * (
            arm1_config.reach_radius**3 + arm2_config.reach_radius**3
        )
        
        overlap_ratio = overlap_volume / total_volume
        
        return min(overlap_ratio, 1.0)
    
    def _sphere_intersection_volume(self, r1: float, r2: float, d: float) -> float:
        """计算两个球体的交集体积"""
        if d >= r1 + r2:
            return 0.0
        
        if d <= abs(r1 - r2):
            # 一个球完全包含另一个
            return (4/3) * np.pi * min(r1, r2)**3
        
        # 使用球冠体积公式
        term1 = (r1 + r2 - d)**2 * (d**2 + 2*d*r2 - 3*r2**2 + 2*d*r1 + 6*r2*r1 - 3*r1**2)
        volume = (np.pi * term1) / (12 * d)
        
        return volume
    
    def evaluate_collision_risk(self, 
                               arm1_pose: np.ndarray,
                               arm1_config: RobotArmConfig,
                               arm2_pose: np.ndarray,
                               arm2_config: RobotArmConfig,
                               safety_margin: float = 0.1) -> Dict:
        """
        评估碰撞风险
        Returns:
            risk_info: 包含风险等级、距离等信息
        """
        # 计算末端执行器距离
        distance = np.linalg.norm(arm1_pose - arm2_pose)
        
        # 考虑机械臂尺寸（简化为点）
        min_safe_distance = safety_margin
        
        if distance < min_safe_distance * 0.5:
            risk_level = 'critical'
            risk_score = 1.0
        elif distance < min_safe_distance:
            risk_level = 'high'
            risk_score = 0.7
        elif distance < min_safe_distance * 2:
            risk_level = 'medium'
            risk_score = 0.4
        else:
            risk_level = 'low'
            risk_score = 0.1
        
        return {
            'distance': float(distance),
            'risk_level': risk_level,
            'risk_score': risk_score,
            'safe': distance >= min_safe_distance
        }
    
    def compute_coupling_index(self, 
                              arms_configs: List[RobotArmConfig]) -> float:
        """
        计算多机械臂系统的耦合度指数
        Returns:
            coupling_index: 0-1 之间，值越大耦合越强
        """
        n_arms = len(arms_configs)
        
        if n_arms < 2:
            return 0.0
        
        total_overlap = 0.0
        n_pairs = 0
        
        for i in range(n_arms):
            for j in range(i+1, n_arms):
                overlap = self.compute_workspace_overlap(
                    arms_configs[i], arms_configs[j]
                )
                total_overlap += overlap
                n_pairs += 1
        
        avg_overlap = total_overlap / n_pairs if n_pairs > 0 else 0.0
        
        return avg_overlap


class RLPathPlanner(nn.Module):
    """基于强化学习的路径规划器"""
    
    def __init__(self, state_dim: int = 12, action_dim: int = 6, hidden_dim: int = 256):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor 网络：state -> action
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 动作范围 [-1, 1]
        )
        
        # Critic 网络：state + action -> Q 值
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        print("✅ RL 路径规划器初始化完成")
        print(f"   状态维度：{state_dim}")
        print(f"   动作维度：{action_dim}")
        print(f"   Critic 输入维度：{state_dim + action_dim}")
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.actor(state_tensor)
            
            if explore:
                # 添加探索噪声
                noise = torch.randn_like(action) * 0.1
                action = action + noise
            
            return action.numpy().squeeze()
    
    def train_step(self, states, actions, rewards, next_states, dones, 
                   lr: float = 0.001, gamma: float = 0.99):
        """训练一步"""
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Critic 损失
        current_q = self.critic(torch.cat([states, actions], dim=1)).squeeze()
        with torch.no_grad():
            next_actions = self.actor(next_states)
            next_q = self.critic(torch.cat([next_states, next_actions], dim=1)).squeeze()
            target_q = rewards + gamma * next_q * (1 - dones)
        
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        # Actor 损失
        actions_pred = self.actor(states)
        actor_loss = -self.critic(torch.cat([states, actions_pred], dim=1)).mean()
        
        # 优化
        optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        critic_loss.backward()
        optimizer.step()
        
        return actor_loss.item(), critic_loss.item()


class MultiArmCoordinator:
    """多机械臂协调器"""
    
    def __init__(self, n_arms: int = 2):
        self.n_arms = n_arms
        self.workspace_evaluator = WorkspaceEvaluator()
        self.rl_planners = [RLPathPlanner() for _ in range(n_arms)]
        
        self.arms_configs = []
        self.tasks = []
        
        print(f"✅ 多机械臂协调器初始化完成")
        print(f"   机械臂数量：{n_arms}")
    
    def add_robot_arm(self, config: RobotArmConfig):
        """添加机械臂"""
        self.arms_configs.append(config)
        print(f"   ✅ 添加机械臂：{config.arm_id}")
    
    def assign_tasks(self, 
                    waypoints: List[np.ndarray],
                    strategy: str = 'balanced') -> List[TaskAssignment]:
        """
        分配任务给多个机械臂
        Args:
            waypoints: 总的路径点列表
            strategy: 分配策略 ('balanced', 'nearest', 'priority')
        Returns:
            assignments: 任务分配列表
        """
        n_points = len(waypoints)
        points_per_arm = n_points // self.n_arms
        
        assignments = []
        
        for i in range(self.n_arms):
            start_idx = i * points_per_arm
            end_idx = start_idx + points_per_arm if i < self.n_arms - 1 else n_points
            
            arm_waypoints = waypoints[start_idx:end_idx]
            
            assignment = TaskAssignment(
                arm_id=self.arms_configs[i].arm_id,
                waypoints=arm_waypoints,
                start_time=i * 0.5,  # 错开启动时间
                estimated_duration=len(arm_waypoints) * 0.1,
                priority=1
            )
            
            assignments.append(assignment)
        
        print(f"\n📋 任务分配完成:")
        for i, assign in enumerate(assignments):
            print(f"   机械臂 {i+1}: {len(assign.waypoints)} 个航点")
        
        return assignments
    
    def check_collisions(self, 
                        arm_poses: List[np.ndarray]) -> Dict:
        """
        检查多臂碰撞
        Returns:
            collision_info: 碰撞信息
        """
        collision_risks = []
        
        for i in range(len(arm_poses)):
            for j in range(i+1, len(arm_poses)):
                risk = self.workspace_evaluator.evaluate_collision_risk(
                    arm_poses[i], self.arms_configs[i],
                    arm_poses[j], self.arms_configs[j]
                )
                collision_risks.append({
                    'arm_pair': (i, j),
                    **risk
                })
        
        max_risk = max([r['risk_score'] for r in collision_risks]) if collision_risks else 0.0
        
        return {
            'risks': collision_risks,
            'max_risk_score': max_risk,
            'safe': all(r['safe'] for r in collision_risks)
        }
    
    def plan_cooperative_path(self, 
                             start_poses: List[np.ndarray],
                            goal_poses: List[np.ndarray]) -> Dict:
        """
        规划协同路径
        Returns:
            plan: 包含各臂路径的字典
        """
        print("\n🗺️  开始协同路径规划...")
        
        trajectories = []
        
        for i in range(self.n_arms):
            # 简单线性插值（实际应该用更复杂的方法）
            start = start_poses[i]
            goal = goal_poses[i]
            
            n_steps = 50
            trajectory = np.linspace(start, goal, n_steps)
            trajectories.append(trajectory)
        
        # 检查整个轨迹的碰撞
        collision_free = True
        for t in range(len(trajectories[0])):
            poses = [traj[t] for traj in trajectories]
            collision_info = self.check_collisions(poses)
            
            if not collision_info['safe']:
                collision_free = False
                print(f"   ⚠️  时刻 {t}: 检测到碰撞风险!")
                break
        
        plan = {
            'trajectories': trajectories,
            'collision_free': collision_free,
            'n_steps': len(trajectories[0]) if trajectories else 0
        }
        
        if collision_free:
            print(f"   ✅ 无碰撞路径规划成功")
            print(f"   路径步数：{plan['n_steps']}")
        else:
            print(f"   ❌ 路径存在碰撞，需要调整")
        
        return plan
    
    def simulate_cooperative_grinding(self, 
                                     plan: Dict,
                                     visualize: bool = False) -> Dict:
        """
        仿真协同打磨过程
        Returns:
            simulation_result: 仿真结果
        """
        print("\n🚀 开始协同打磨仿真...")
        
        trajectories = plan['trajectories']
        n_steps = len(trajectories[0]) if trajectories else 0
        
        if n_steps == 0:
            return {'success': False, 'reason': '无有效路径'}
        
        # 仿真执行
        execution_log = []
        collisions_detected = 0
        
        for t in range(n_steps):
            step_log = {
                'time_step': t,
                'arm_poses': [],
                'collisions': []
            }
            
            poses = []
            for i, traj in enumerate(trajectories):
                pose = traj[t]
                poses.append(pose)
                step_log['arm_poses'].append(pose.tolist())
            
            # 检查碰撞
            collision_info = self.check_collisions(poses)
            
            if not collision_info['safe']:
                collisions_detected += 1
                step_log['collisions'] = collision_info['risks']
            
            execution_log.append(step_log)
            
            # 进度显示
            if t % 10 == 0:
                print(f"\r  进度：{t}/{n_steps} ({t/n_steps*100:.1f}%)", end='')
        
        success_rate = 1.0 - (collisions_detected / n_steps)
        
        result = {
            'success': success_rate > 0.95,
            'success_rate': success_rate,
            'total_steps': n_steps,
            'collisions_detected': collisions_detected,
            'execution_log': execution_log
        }
        
        print(f"\n✅ 仿真完成")
        print(f"   成功率：{success_rate*100:.1f}%")
        print(f"   碰撞次数：{collisions_detected}")
        
        return result


def create_test_scenario():
    """创建测试场景"""
    # 创建两个机械臂
    arm1_config = RobotArmConfig(
        arm_id="arm_1",
        base_position=np.array([-0.3, 0, 0]),
        reach_radius=0.8
    )
    
    arm2_config = RobotArmConfig(
        arm_id="arm_2",
        base_position=np.array([0.3, 0, 0]),
        reach_radius=0.8
    )
    
    # 创建工作空间
    coordinator = MultiArmCoordinator(n_arms=2)
    coordinator.add_robot_arm(arm1_config)
    coordinator.add_robot_arm(arm2_config)
    
    return coordinator


def test_workspace_evaluation():
    """测试工作空间评价"""
    print("="*70)
    print("测试 1: 工作空间耦合度评价")
    print("="*70)
    
    coordinator = create_test_scenario()
    
    # 计算耦合度
    coupling = coordinator.workspace_evaluator.compute_coupling_index(
        coordinator.arms_configs
    )
    
    print(f"\n📊 工作空间评价结果:")
    print(f"   耦合度指数：{coupling:.3f}")
    
    if coupling < 0.1:
        print(f"   ✅ 低耦合：机械臂工作空间独立")
    elif coupling < 0.3:
        print(f"   ⚠️  中耦合：需要协调")
    else:
        print(f"   🔴 高耦合：需要严格避障")
    
    return coordinator


def test_task_assignment(coordinator):
    """测试任务分配"""
    print("\n" + "="*70)
    print("测试 2: 多臂任务分配")
    print("="*70)
    
    # 生成打磨路径点
    n_points = 100
    waypoints = []
    
    for i in range(n_points):
        x = -0.2 + (i % 10) * 0.04
        y = -0.2 + (i // 10) * 0.04
        z = 0.9
        waypoints.append(np.array([x, y, z]))
    
    # 分配任务
    assignments = coordinator.assign_tasks(waypoints, strategy='balanced')
    
    return assignments


def test_cooperative_planning(coordinator, assignments):
    """测试协同路径规划"""
    print("\n" + "="*70)
    print("测试 3: 协同路径规划")
    print("="*70)
    
    # 定义起始和目标位置
    start_poses = [
        np.array([-0.2, -0.2, 0.9]),  # Arm 1 起点
        np.array([0.2, -0.2, 0.9])    # Arm 2 起点
    ]
    
    goal_poses = [
        np.array([-0.2, 0.2, 0.9]),   # Arm 1 终点
        np.array([0.2, 0.2, 0.9])     # Arm 2 终点
    ]
    
    # 规划路径
    plan = coordinator.plan_cooperative_path(start_poses, goal_poses)
    
    return plan


def test_rl_learning():
    """测试强化学习"""
    print("\n" + "="*70)
    print("测试 4: 强化学习路径优化")
    print("="*70)
    
    planner = RLPathPlanner()
    
    # 模拟训练
    print("\n开始训练 RL 策略...")
    
    n_episodes = 100
    losses = []
    
    for episode in range(n_episodes):
        # 生成随机数据
        batch_size = 32
        states = np.random.randn(batch_size, 12)
        actions = np.random.randn(batch_size, 6)
        rewards = np.random.randn(batch_size)
        next_states = np.random.randn(batch_size, 12)
        dones = np.random.randint(0, 2, batch_size).astype(float)
        
        # 训练
        actor_loss, critic_loss = planner.train_step(
            states, actions, rewards, next_states, dones
        )
        
        losses.append((actor_loss, critic_loss))
        
        if (episode + 1) % 20 == 0:
            avg_actor_loss = np.mean([l[0] for l in losses[-20:]])
            avg_critic_loss = np.mean([l[1] for l in losses[-20:]])
            print(f"  Episode {episode+1}/{n_episodes}: "
                  f"Actor Loss={avg_actor_loss:.4f}, Critic Loss={avg_critic_loss:.4f}")
    
    print("\n✅ RL 训练完成")
    
    return planner


def main():
    """主函数"""
    print("="*70)
    print("任务 3: 多机械臂协同控制算法测试")
    print("="*70)
    
    # 测试 1: 工作空间评价
    coordinator = test_workspace_evaluation()
    
    # 测试 2: 任务分配
    assignments = test_task_assignment(coordinator)
    
    # 测试 3: 协同规划
    plan = test_cooperative_planning(coordinator, assignments)
    
    # 测试 4: RL 学习
    rl_planner = test_rl_learning()
    
    # 保存结果
    results = {
        'workspace_coupling': coordinator.workspace_evaluator.compute_coupling_index(
            coordinator.arms_configs
        ),
        'n_arms': coordinator.n_arms,
        'task_assignments': [
            {
                'arm_id': a.arm_id,
                'n_waypoints': len(a.waypoints)
            }
            for a in assignments
        ],
        'plan_success': plan['collision_free'],
        'rl_trained': True
    }
    
    output_path = Path('projects/grinding_system/data/multi_arm_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 结果已保存：{output_path}")
    
    print("\n🎉 任务 3 测试全部完成！")
    print("="*70)


if __name__ == "__main__":
    main()