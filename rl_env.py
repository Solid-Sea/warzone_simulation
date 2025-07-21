"""
优化的强化学习环境 - 专注于高效训练和性能
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import torch
import os
from typing import Dict, List, Tuple, Any

class WarzoneEnv(gym.Env):
    """高性能战区强化学习环境"""
    
    def __init__(self, width=80, height=60):
        super(WarzoneEnv, self).__init__()
        
        self.width = width
        self.height = height
        self.max_steps = 500
        
        # 动作空间：6个基础动作
        self.action_space = spaces.Discrete(6)
        
        # 优化观察空间：降低维度提升训练效率
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(5 * height * width,), 
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, **kwargs):
        """重置环境"""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.units = self._initialize_units()
        self.terrain = self._initialize_terrain()
        
        return self._get_observation(), {}
    
    def _initialize_units(self) -> Dict[int, Dict]:
        """初始化单位 - 平衡兵种配置"""
        units = {}
        unit_id = 1
        
        # 红方单位 (10步兵 + 2坦克 + 1炮兵)
        red_config = [
            ("infantry", 10, (10, 10)),
            ("tank", 2, (15, 15)),
            ("artillery", 1, (12, 18))
        ]
        
        for unit_type, count, (x_offset, y_offset) in red_config:
            for i in range(count):
                units[unit_id] = self._create_unit(unit_type, "red", 
                                                 (x_offset + i % 3, y_offset + i // 3))
                unit_id += 1
        
        # 蓝方单位 (10步兵 + 2坦克 + 1炮兵)
        blue_config = [
            ("infantry", 10, (60, 50)),
            ("tank", 2, (55, 45)),
            ("artillery", 1, (65, 55))
        ]
        
        for unit_type, count, (x_offset, y_offset) in blue_config:
            for i in range(count):
                units[unit_id] = self._create_unit(unit_type, "blue", 
                                                 (x_offset + i % 3, y_offset + i // 3))
                unit_id += 1
            
        return units
    
    def _create_unit(self, unit_type, camp, position):
        """创建单位实例"""
        config = {
            "infantry": {"health": 100, "attack_range": 3, "damage": 25, "speed": 2},
            "tank": {"health": 150, "attack_range": 4, "damage": 40, "speed": 1},
            "artillery": {"health": 80, "attack_range": 6, "damage": 60, "speed": 1}
        }[unit_type]
        
        return {
            "type": unit_type,
            "position": position,
            "camp": camp,
            "health": config["health"],
            "attack_range": config["attack_range"],
            "damage": config["damage"],
            "speed": config["speed"]
        }
    
    def _initialize_terrain(self) -> np.ndarray:
        """初始化地形 - 随机障碍物"""
        terrain = np.zeros((self.height, self.width), dtype=np.float32)
        
        # 添加随机障碍物 (10%的地图)
        for _ in range(int(self.width * self.height * 0.1)):
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            terrain[y, x] = 1
            
        return terrain
    
    def _get_observation(self) -> np.ndarray:
        """获取观察状态 - 优化性能"""
        obs = np.zeros((5, self.height, self.width), dtype=np.float32)
        
        # 通道0: 红方单位位置
        # 通道1: 蓝方单位位置
        # 通道2: 地形障碍
        # 通道3: 红方单位血量
        # 通道4: 蓝方单位血量
        
        for unit in self.units.values():
            x, y = unit["position"]
            if 0 <= x < self.width and 0 <= y < self.height:
                if unit["camp"] == "red":
                    obs[0, y, x] = 1.0
                    obs[3, y, x] = unit["health"] / 100.0
                else:
                    obs[1, y, x] = 1.0
                    obs[4, y, x] = unit["health"] / 100.0
        
        obs[2] = self.terrain
        
        return obs.flatten()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步环境更新"""
        self.step_count += 1
        
        # 执行动作
        reward = self._execute_action(action)
        
        # 检查终止条件
        terminated = self._check_termination()
        truncated = self.step_count >= self.max_steps
        
        # 获取新观察
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, {}
    
    def _execute_action(self, action: int) -> float:
        """执行动作并计算奖励"""
        reward = 0.0
        
        # 动作映射: 0-3: 移动, 4: 攻击, 5: 防御
        if action < 4:  # 移动动作
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
            for unit in self.units.values():
                if unit["camp"] == "red":  # 只控制红方
                    new_x = max(0, min(self.width-1, unit["position"][0] + dx))
                    new_y = max(0, min(self.height-1, unit["position"][1] + dy))
                    
                    # 检查地形障碍
                    if self.terrain[new_y, new_x] == 0:
                        unit["position"] = (new_x, new_y)
                        reward += 0.01  # 小奖励鼓励移动
        
        elif action == 4:  # 攻击
            reward += self._process_combat()
        
        elif action == 5:  # 防御
            reward += 0.05  # 防御奖励
        
        return reward
    
    def _process_combat(self) -> float:
        """处理战斗逻辑并返回奖励"""
        reward = 0.0
        
        # 红方攻击蓝方
        for red_unit in self.units.values():
            if red_unit["camp"] != "red" or red_unit["health"] <= 0:
                continue
                
            for blue_id, blue_unit in list(self.units.items()):
                if blue_unit["camp"] != "blue" or blue_unit["health"] <= 0:
                    continue
                
                # 计算曼哈顿距离
                dist = abs(red_unit["position"][0] - blue_unit["position"][0]) + \
                       abs(red_unit["position"][1] - blue_unit["position"][1])
                
                if dist <= red_unit["attack_range"]:
                    # 造成伤害
                    blue_unit["health"] -= red_unit["damage"]
                    reward += 0.5  # 攻击奖励
                    
                    if blue_unit["health"] <= 0:
                        reward += 10.0  # 击杀奖励
                        del self.units[blue_id]
        
        return reward
    
    def _check_termination(self) -> bool:
        """检查终止条件"""
        red_alive = any(u["health"] > 0 and u["camp"] == "red" for u in self.units.values())
        blue_alive = any(u["health"] > 0 and u["camp"] == "blue" for u in self.units.values())
        
        return not red_alive or not blue_alive

def train_model(total_timesteps=100000, learning_rate=3e-4):
    """训练强化学习模型"""
    print("初始化强化学习环境...")
    env = WarzoneEnv()
    
    print("配置PPO模型参数...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./logs/"
    )
    
    print(f"开始训练 (共 {total_timesteps} 步)...")
    model.learn(total_timesteps=total_timesteps)
    
    # 保存模型
    os.makedirs("models", exist_ok=True)
    model_path = f"models/warzone_ppo_model"
    model.save(model_path)
    print(f"训练完成! 模型已保存至: {model_path}")
    
    return model

def evaluate_model(model, episodes=10):
    """评估训练好的模型"""
    print("评估模型性能...")
    env = WarzoneEnv()
    total_rewards = []
    wins = 0
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        
        # 检查胜利条件 (红方是否有存活单位)
        if any(u["health"] > 0 and u["camp"] == "red" for u in env.units.values()):
            wins += 1
        
        print(f"回合 {episode+1}: 奖励={episode_reward:.1f}, 胜={'是' if wins > episode else '否'}")
    
    avg_reward = np.mean(total_rewards)
    win_rate = wins / episodes
    print(f"平均奖励: {avg_reward:.2f}, 胜率: {win_rate:.1%}")
    
    return avg_reward, win_rate

if __name__ == "__main__":
    # 训练模型
    model = train_model(total_timesteps=100000, learning_rate=1e-4)
    
    # 评估模型
    avg_reward, win_rate = evaluate_model(model)
