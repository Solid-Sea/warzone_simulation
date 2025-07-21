"""
高性能战区强化学习环境 v2.1 - 优化版
修复循环依赖，增强地形影响，优化奖励函数
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import os
from typing import Dict, List, Tuple, Any, Optional
import json

class Config:
    """配置管理类 - 消除硬编码"""
    def __init__(self, config_path: Optional[str] = None):
        self.config = {
            "env": {
                "width": 80,
                "height": 60,
                "max_steps": 500,
                "obstacle_ratio": 0.1
            },
            "units": {
                "infantry": {"health": 100, "attack_range": 3, "damage": 25, "speed": 2},
                "tank": {"health": 150, "attack_range": 4, "damage": 40, "speed": 1},
                "artillery": {"health": 80, "attack_range": 6, "damage": 60, "speed": 1}
            },
            "rewards": {
                "move": 0.01,
                "attack": 0.5,
                "kill": 10.0,
                "defense": 0.05,
                "terrain_bonus": 0.1,
                "formation_bonus": 0.2
            },
            "terrain": {
                "obstacle_penalty": -0.1,
                "cover_bonus": 0.2,
                "high_ground_bonus": 0.3
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config.update(json.load(f))
    
    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default)
        return value

class WarzoneEnv(gym.Env):
    """优化版战区强化学习环境"""
    
    def __init__(self, width: int = None, height: int = None, config_path: Optional[str] = None):
        super(WarzoneEnv, self).__init__()
        
        self.config = Config(config_path)
        self.width = width or self.config.get("env.width")
        self.height = height or self.config.get("env.height")
        self.max_steps = self.config.get("env.max_steps")
        
        # 动作空间扩展：6个基础动作 + 2个战术动作
        self.action_space = spaces.Discrete(8)
        
        # 优化观察空间：包含地形信息
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(7 * self.height * self.width,),  # 增加地形通道
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, **kwargs):
        """重置环境 - 支持随机种子"""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.units = self._initialize_units()
        self.terrain = self._initialize_terrain()
        self.terrain_features = self._calculate_terrain_features()
        
        return self._get_observation(), {}
    
    def _initialize_units(self) -> Dict[int, Dict]:
        """初始化单位 - 使用配置系统"""
        units = {}
        unit_id = 1
        
        # 红方单位配置
        red_positions = [(10 + i % 5, 10 + i // 5) for i in range(13)]
        unit_types = ["infantry"] * 10 + ["tank"] * 2 + ["artillery"] * 1
        
        for unit_type, pos in zip(unit_types, red_positions):
            units[unit_id] = self._create_unit(unit_type, "red", pos)
            unit_id += 1
        
        # 蓝方单位配置
        blue_positions = [(60 + i % 5, 50 + i // 5) for i in range(13)]
        for unit_type, pos in zip(unit_types, blue_positions):
            units[unit_id] = self._create_unit(unit_type, "blue", pos)
            unit_id += 1
            
        return units
    
    def _create_unit(self, unit_type: str, camp: str, position: Tuple[int, int]) -> Dict:
        """创建单位实例 - 使用配置"""
        config = self.config.get(f"units.{unit_type}")
        return {
            "type": unit_type,
            "position": position,
            "camp": camp,
            "health": config["health"],
            "max_health": config["health"],
            "attack_range": config["attack_range"],
            "damage": config["damage"],
            "speed": config["speed"]
        }
    
    def _initialize_terrain(self) -> np.ndarray:
        """初始化地形 - 智能障碍物分布"""
        terrain = np.zeros((self.height, self.width), dtype=np.float32)
        
        # 添加障碍物集群而非随机分布
        num_clusters = int(self.width * self.height * self.config.get("env.obstacle_ratio") / 25)
        
        for _ in range(num_clusters):
            cluster_x = np.random.randint(10, self.width - 10)
            cluster_y = np.random.randint(10, self.height - 10)
            cluster_size = np.random.randint(3, 8)
            
            for dx in range(-cluster_size, cluster_size + 1):
                for dy in range(-cluster_size, cluster_size + 1):
                    x, y = cluster_x + dx, cluster_y + dy
                    if 0 <= x < self.width and 0 <= y < self.height:
                        if np.random.random() < 0.7:  # 70%概率放置障碍物
                            terrain[y, x] = 1
        
        return terrain
    
    def _calculate_terrain_features(self) -> Dict[str, np.ndarray]:
        """计算地形特征：高地、掩体等"""
        features = {
            "high_ground": np.zeros((self.height, self.width), dtype=np.float32),
            "cover": np.zeros((self.height, self.width), dtype=np.float32)
        }
        
        # 高地：边缘区域
        for y in range(self.height):
            for x in range(self.width):
                if self.terrain[y, x] == 0:  # 非障碍物
                    # 高地：靠近地图边缘
                    distance_to_edge = min(x, y, self.width - 1 - x, self.height - 1 - y)
                    if distance_to_edge <= 5:
                        features["high_ground"][y, x] = 1.0
                    
                    # 掩体：靠近障碍物
                    nearby_obstacles = 0
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                if self.terrain[ny, nx] == 1:
                                    nearby_obstacles += 1
                    
                    features["cover"][y, x] = min(nearby_obstacles / 8.0, 1.0)
        
        return features
    
    def _get_observation(self) -> np.ndarray:
        """获取观察状态 - 包含地形特征"""
        obs = np.zeros((7, self.height, self.width), dtype=np.float32)
        
        # 通道0-1: 红蓝方单位位置
        # 通道2-3: 红蓝方单位血量（标准化）
        # 通道4: 地形障碍
        # 通道5: 高地优势
        # 通道6: 掩体覆盖
        
        for unit in self.units.values():
            x, y = unit["position"]
            if 0 <= x < self.width and 0 <= y < self.height:
                channel = 0 if unit["camp"] == "red" else 1
                health_channel = 2 if unit["camp"] == "red" else 3
                
                obs[channel, y, x] = 1.0
                obs[health_channel, y, x] = unit["health"] / unit["max_health"]
        
        obs[4] = self.terrain
        obs[5] = self.terrain_features["high_ground"]
        obs[6] = self.terrain_features["cover"]
        
        return obs.flatten()
    
    def _calculate_terrain_bonus(self, position: Tuple[int, int], camp: str) -> float:
        """计算地形奖励加成"""
        x, y = position
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0.0
        
        bonus = 0.0
        
        # 高地奖励
        if self.terrain_features["high_ground"][y, x] > 0:
            bonus += self.config.get("terrain.high_ground_bonus")
        
        # 掩体奖励
        cover_value = self.terrain_features["cover"][y, x]
        bonus += cover_value * self.config.get("terrain.cover_bonus")
        
        return bonus
    
    def _calculate_formation_bonus(self, unit: Dict) -> float:
        """计算阵型奖励（单位协同）"""
        x, y = unit["position"]
        friendly_nearby = 0
        
        for other in self.units.values():
            if other["camp"] == unit["camp"] and other != unit:
                dist = abs(x - other["position"][0]) + abs(y - other["position"][1])
                if dist <= 3:  # 3格范围内的友军
                    friendly_nearby += 1
        
        return min(friendly_nearby * self.config.get("rewards.formation_bonus"), 1.0)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步环境更新 - 优化奖励计算"""
        self.step_count += 1
        
        # 执行动作
        reward = self._execute_action(action)
        
        # 检查终止条件
        terminated = self._check_termination()
        truncated = self.step_count >= self.max_steps
        
        # 获取新观察
        obs = self._get_observation()
        
        # 添加额外信息
        info = {
            "step": self.step_count,
            "red_units": sum(1 for u in self.units.values() if u["camp"] == "red"),
            "blue_units": sum(1 for u in self.units.values() if u["camp"] == "blue")
        }
        
        return obs, reward, terminated, truncated, info
    
    def _execute_action(self, action: int) -> float:
        """执行动作并计算奖励 - 包含地形和阵型影响"""
        reward = 0.0
        
        # 扩展动作空间：0-3移动，4-5攻击，6-7战术动作
        if action < 4:  # 移动动作
            reward += self._execute_move_action(action)
        elif action < 6:  # 攻击动作
            reward += self._execute_attack_action(action - 4)
        else:  # 战术动作
            reward += self._execute_tactical_action(action - 6)
        
        return reward
    
    def _execute_move_action(self, direction: int) -> float:
        """执行移动动作"""
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][direction]
        total_reward = 0.0
        
        for unit in self.units.values():
            if unit["camp"] == "red":
                new_x = max(0, min(self.width-1, unit["position"][0] + dx * unit["speed"]))
                new_y = max(0, min(self.height-1, unit["position"][1] + dy * unit["speed"]))
                
                # 检查地形障碍
                if self.terrain[new_y, new_x] == 0:
                    unit["position"] = (new_x, new_y)
                    
                    # 基础移动奖励
                    total_reward += self.config.get("rewards.move")
                    
                    # 地形奖励
                    total_reward += self._calculate_terrain_bonus((new_x, new_y), "red")
                    
                    # 阵型奖励
                    total_reward += self._calculate_formation_bonus(unit)
                else:
                    # 障碍物惩罚
                    total_reward += self.config.get("terrain.obstacle_penalty")
        
        return total_reward
    
    def _execute_attack_action(self, attack_type: int) -> float:
        """执行攻击动作 - 区分近战和远程"""
        reward = 0.0
        
        for red_unit in self.units.values():
            if red_unit["camp"] != "red" or red_unit["health"] <= 0:
                continue
                
            for blue_id, blue_unit in list(self.units.items()):
                if blue_unit["camp"] != "blue" or blue_unit["health"] <= 0:
                    continue
                
                dist = abs(red_unit["position"][0] - blue_unit["position"][0]) + \
                       abs(red_unit["position"][1] - blue_unit["position"][1])
                
                # 根据攻击类型调整有效范围
                effective_range = red_unit["attack_range"] * (1.2 if attack_type == 1 else 1.0)
                
                if dist <= effective_range:
                    # 计算伤害（考虑地形）
                    damage = red_unit["damage"]
                    
                    # 防守方掩体减伤
                    cover_bonus = self.terrain_features["cover"][blue_unit["position"][1], 
                                                               blue_unit["position"][0]]
                    damage *= (1 - cover_bonus * 0.3)
                    
                    blue_unit["health"] -= int(damage)
                    reward += self.config.get("rewards.attack")
                    
                    if blue_unit["health"] <= 0:
                        reward += self.config.get("rewards.kill")
                        del self.units[blue_id]
        
        return reward
    
    def _execute_tactical_action(self, tactic_type: int) -> float:
        """执行战术动作：0=防御姿态，1=集结"""
        reward = 0.0
        
        if tactic_type == 0:  # 防御姿态
            for unit in self.units.values():
                if unit["camp"] == "red":
                    terrain_bonus = self._calculate_terrain_bonus(unit["position"], "red")
                    reward += terrain_bonus * 2  # 防御时地形奖励翻倍
        
        else:  # 集结 - 向友军靠拢
            for unit in self.units.values():
                if unit["camp"] == "red":
                    formation_bonus = self._calculate_formation_bonus(unit)
                    reward += formation_bonus
        
        return reward
    
    def _check_termination(self) -> bool:
        """检查终止条件 - 优化判断"""
        red_alive = sum(1 for u in self.units.values() if u["camp"] == "red" and u["health"] > 0)
        blue_alive = sum(1 for u in self.units.values() if u["camp"] == "blue" and u["health"] > 0)
        
        return red_alive == 0 or blue_alive == 0

def train_model(total_timesteps=100000, learning_rate=1e-4, config_path=None):
    """训练强化学习模型 - 优化版"""
    print("🚀 初始化优化版强化学习环境...")
    env = WarzoneEnv(config_path=config_path)
    
    print("⚙️  配置PPO模型参数...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./logs/",
        device="auto"
    )
    
    print(f"🎯 开始训练 (共 {total_timesteps} 步)...")
    model.learn(total_timesteps=total_timesteps)
    
    # 保存模型和配置
    os.makedirs("models", exist_ok=True)
    model_path = f"models/warzone_ppo_v2"
    model.save(model_path)
    
    # 保存配置
    with open(f"{model_path}_config.json", 'w') as f:
        json.dump(env.config.config, f, indent=2)
    
    print(f"✅ 训练完成! 模型已保存至: {model_path}")
    return model

def evaluate_model(model, episodes=10, render=False):
    """评估训练好的模型 - 增强版"""
    print("📊 评估模型性能...")
    env = WarzoneEnv()
    total_rewards = []
    wins = 0
