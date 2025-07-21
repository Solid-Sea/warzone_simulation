"""
简化后的模拟控制器 - 移除冗余功能，专注于RL环境集成
"""
import threading
import time
import random
from typing import Dict, List

class SimulationController:
    """轻量级模拟控制器，用于RL环境"""
    
    def __init__(self, width=80, height=60):
        self.width = width
        self.height = height
        self.running = False
        self.step_count = 0
        self.max_steps = 1000
        
    def reset(self):
        """重置模拟状态"""
        self.step_count = 0
        self.running = True
        return self.get_state()
    
    def step(self, actions: Dict[int, int]) -> Dict:
        """执行一步模拟"""
        if not self.running:
            return {"done": True, "state": None, "reward": 0}
            
        self.step_count += 1
        
        # 处理动作
        rewards = self._process_actions(actions)
        
        # 检查终止条件
        done = self._check_termination()
        
        # 获取新状态
        state = self.get_state()
        
        return {
            "state": state,
            "reward": rewards,
            "done": done,
            "info": {"step": self.step_count}
        }
    
    def _process_actions(self, actions: Dict[int, int]) -> float:
        """处理单位动作"""
        # 简化的动作处理，返回总奖励
        return 0.0
    
    def _check_termination(self) -> bool:
        """检查是否终止"""
        return self.step_count >= self.max_steps
    
    def get_state(self) -> Dict:
        """获取当前状态"""
        # 返回简化的状态表示
        return {
            "step": self.step_count,
            "width": self.width,
            "height": self.height
        }
    
    def close(self):
        """清理资源"""
        self.running = False
