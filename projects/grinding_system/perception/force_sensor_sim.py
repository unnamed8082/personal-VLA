
"""
任务 1.2: 仿真六维力传感器 - 快速测试版
"""

import numpy as np
from typing import List
from dataclasses import dataclass
import time


@dataclass
class ForceTorqueData:
    """力/力矩数据"""
    fx: float
    fy: float
    fz: float
    mx: float
    my: float
    mz: float
    timestamp: float


class SimulatedForceTorqueSensor:
    """仿真六维力扭矩传感器"""
    
    def __init__(self):
        self.bias = np.zeros(6)
        self.in_contact = False
        self.contact_force = np.zeros(6)
        print("✅ 仿真力传感器初始化完成")
    
    def calibrate_zero(self, samples: int = 50):
        """零点标定"""
        print("正在进行零点标定...")
        measurements = []
        for i in range(samples):
            raw = self._generate_raw_data()
            measurements.append(raw)
            time.sleep(0.01)
        
        self.bias = np.mean(measurements, axis=0)
        print(f"✅ 零点标定完成")
        print(f"   偏置：[{self.bias[0]:.3f}, {self.bias[1]:.3f}, {self.bias[2]:.3f}] N")
    
    def set_contact_state(self, in_contact: bool, contact_force: np.ndarray = None):
        """设置接触状态"""
        self.in_contact = in_contact
        if contact_force is not None:
            self.contact_force = contact_force
    
    def read(self) -> ForceTorqueData:
        """读取力传感器数据"""
        raw_data = self._generate_raw_data()
        calibrated = raw_data - self.bias
        
        return ForceTorqueData(
            fx=calibrated[0], fy=calibrated[1], fz=calibrated[2],
            mx=calibrated[3], my=calibrated[4], mz=calibrated[5],
            timestamp=time.time()
        )
    
    def _generate_raw_data(self) -> np.ndarray:
        """生成原始数据"""
        noise = np.random.normal(0, 0.1, 6)
        
        if self.in_contact:
            contact_noise = np.random.normal(0, 0.05, 6)
            data = self.contact_force + contact_noise + noise
        else:
            data = noise
        
        return data
    
    def simulate_grinding(self, target_force: float = 20.0, n_samples: int = 100) -> List[ForceTorqueData]:
        """模拟打磨过程"""
        print(f"🔧 模拟打磨过程：目标力={target_force}N, 采样数={n_samples}")
        
        force_history = []
        
        for i in range(n_samples):
            t = i * 0.01  # 100Hz
            
            self.in_contact = True
            
            # 生成时变的接触力
            fz = target_force + 2.0 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.5)
            fx = 3.0 * np.sin(2 * np.pi * t * 0.5) + np.random.normal(0, 0.3)
            fy = 3.0 * np.cos(2 * np.pi * t * 0.5) + np.random.normal(0, 0.3)
            
            contact_force = np.array([fx, fy, fz, 0.1, 0.1, 0.05])
            self.contact_force = contact_force
            
            data = self.read()
            force_history.append(data)
            
            # 进度显示
            if i % 20 == 0:
                print(f"\r  进度：{i/n_samples*100:.1f}%, Fz={fz:.2f}N", end='')
        
        print(f"\n✅ 完成，采样数：{len(force_history)}")
        return force_history


def analyze_and_plot(force_history: List[ForceTorqueData]):
    """分析并绘图"""
    try:
        import matplotlib.pyplot as plt
        
        # 提取数据
        timestamps = [d.timestamp - force_history[0].timestamp for d in force_history]
        forces = np.array([[d.fx, d.fy, d.fz] for d in force_history])
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(timestamps, forces[:, 0], label='Fx', alpha=0.7)
        ax.plot(timestamps, forces[:, 1], label='Fy', alpha=0.7)
        ax.plot(timestamps, forces[:, 2], label='Fz', alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force (N)')
        ax.set_title('Simulated Grinding Force')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig('projects/grinding_system/data/force_plot.png', dpi=150, bbox_inches='tight')
        print(f"📊 力图已保存：projects/grinding_system/data/force_plot.png")
        
    except ImportError:
        print("⚠️ 未安装 matplotlib，跳过绘图")


def test_force_sensor():
    """测试力传感器"""
    print("=" * 60)
    print("🔧 测试仿真力传感器")
    print("=" * 60)
    
    sensor = SimulatedForceTorqueSensor()
    
    # 零点标定
    sensor.calibrate_zero()
    
    # 模拟打磨
    print("\n开始模拟打磨过程...")
    force_history = sensor.simulate_grinding(target_force=20.0, n_samples=100)
    
    # 统计分析
    forces = np.array([[d.fx, d.fy, d.fz] for d in force_history])
    print(f"\n📊 力数据统计:")
    print(f"   Fx: 均值={forces[:, 0].mean():.2f}N, 标准差={forces[:, 0].std():.2f}N")
    print(f"   Fy: 均值={forces[:, 1].mean():.2f}N, 标准差={forces[:, 1].std():.2f}N")
    print(f"   Fz: 均值={forces[:, 2].mean():.2f}N, 标准差={forces[:, 2].std():.2f}N")
    
    # 绘图
    analyze_and_plot(force_history)
    
    print("\n✅ 力传感器测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_force_sensor()