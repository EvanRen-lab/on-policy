"""
终端移动性管理 - 固定轨迹生成
"""
import numpy as np
import os

class FixedTrajectoryGenerator:
    """固定轨迹生成器，确保不同方法使用相同的轨迹"""
    
    def __init__(self, num_terminals, radius=200.0, max_speed=10.0, seed=42):
        self.num_terminals = num_terminals
        self.radius = radius
        self.max_speed = max_speed
        self.seed = seed
        self.trajectory = []
        
        np.random.seed(seed)
        
    def generate(self, num_steps=100):
        """生成固定轨迹"""
        # 初始化位置
        positions = []
        for _ in range(self.num_terminals):
            r = self.radius * np.sqrt(np.random.rand())
            theta = 2 * np.pi * np.random.rand()
            positions.append([r * np.cos(theta), r * np.sin(theta)])
        
        positions = np.array(positions)
        self.trajectory.append(positions.copy())
        
        # 初始化速度
        velocities = []
        for _ in range(self.num_terminals):
            speed = np.random.rand() * self.max_speed
            angle = 2 * np.pi * np.random.rand()
            velocities.append([speed * np.cos(angle), speed * np.sin(angle)])
        
        velocities = np.array(velocities)
        
        # 生成轨迹
        for _ in range(num_steps - 1):
            positions += velocities
            
            # 边界反射
            for i in range(self.num_terminals):
                dist = np.linalg.norm(positions[i])
                if dist > self.radius:
                    normal = positions[i] / dist
                    velocities[i] = velocities[i] - 2 * np.dot(velocities[i], normal) * normal
            
            # 速度衰减
            velocities *= 0.9
            
            # 添加随机扰动
            velocities += np.random.randn(self.num_terminals, 2) * 0.1
            
            self.trajectory.append(positions.copy())
        
        return self.trajectory
    
    def get_step(self, step_idx):
        """获取指定步骤的位置"""
        if step_idx < len(self.trajectory):
            return self.trajectory[step_idx].copy()
        else:
            return self.trajectory[-1].copy()
    
    def save(self, filepath):
        """保存轨迹"""
        np.save(filepath, np.array(self.trajectory))
    
    def load(self, filepath):
        """加载轨迹"""
        self.trajectory = list(np.load(filepath))