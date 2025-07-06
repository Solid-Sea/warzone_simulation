import gymnasium as gym
import torch
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from simulation_controller import SimulationController

class TensorboardCallback(BaseCallback):
    """自定义回调函数用于TensorBoard可视化"""
    def __init__(self, check_freq: int = 1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # 记录训练指标
            self.logger.record("train/mean_reward", np.mean(self.episode_rewards[-100:]))
            self.logger.record("train/epsilon", self.model.exploration_rate)
            
            # 生成战场热力图
            fig = plt.figure(figsize=(10, 6))
            state = self.training_env.get_attr("sim_controller")[0].get_state()
            if state:
                terrain = np.array(state['terrain'])
                plt.imshow(terrain, cmap='terrain')
                plt.colorbar()
                self.logger.record("battlefield", Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
                plt.close()
        return True

    def _on_rollout_end(self) -> None:
        # 记录回合数据
        rewards = self.model.ep_info_buffer
        if len(rewards) > 0:
            self.episode_rewards.extend(rewards)

class WarzoneEnv(gym.Env):
    """强化学习环境"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.sim_controller = SimulationController()
        self.action_space = spaces.Discrete(6)  # 基础动作空间
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.sim_controller.map_height, 
                   self.sim_controller.map_width, 5),
            dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        self.sim_controller.init_simulation()
        observation = self._get_observation()
        info = {}
        return observation, info
    
    def step(self, action):
        # 执行动作逻辑
        self._take_action(action)
        
        # 推进模拟
        self.sim_controller.auto_simulation_step()
        self.sim_controller.process_events()
        
        # 获取新状态
        obs = self._get_observation()
        reward = self._calculate_reward()
        terminated = self._check_done()
        truncated = False  # 暂时不处理时间限制
        info = {}
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """生成包含地形、单位、建筑的状态观察"""
        state = np.zeros((self.sim_controller.map_height, 
                        self.sim_controller.map_width, 5))
        
        # 从模拟控制器获取实时数据
        sim_state = self.sim_controller.get_state()
        
        # 填充各通道数据
        # [地形, 红方存在, 蓝方存在, 红方血量, 蓝方血量]
        if sim_state:
            # 地形数据
            for y in range(self.sim_controller.map_height):
                for x in range(self.sim_controller.map_width):
                    state[y,x,0] = sim_state['terrain'][y][x]
            
            # 单位数据
            for unit in sim_state['units']:
                x, y = unit['position']
                if unit['camp'] == "red":
                    state[y,x,1] = 1
                    state[y,x,3] = unit['health']/100
                else:
                    state[y,x,2] = 1 
                    state[y,x,4] = unit['health']/100
        
        return state
    
    def _take_action(self, action):
        """将离散动作转换为游戏指令"""
        # 更新后的动作映射
        if action == 0:  # 无操作
            pass
        elif action == 1: 
            self._move_units_randomly()
        elif action == 2: 
            self._attack_nearest_enemy()
        elif action == 3:
            self._defend_position()
        elif action == 4:
            self._retreat_units()
        elif action == 5:
            self._call_artillery_support()

    def _move_units_randomly(self):
        """随机移动友方单位"""
        state = self.sim_controller.get_state()
        if state and state.get("units"):
            # 获取所有活跃的友方单位
            friendly_units = [u for u in state["units"] if u["camp"] == "red" and u["status"] == "active"]
            
            for unit in friendly_units:
                # 生成随机移动方向
                dx = np.random.randint(-3, 4)
                dy = np.random.randint(-3, 4)
                new_x = max(0, min(79, unit["position"][0] + dx))
                new_y = max(0, min(59, unit["position"][1] + dy))
                
                # 调用控制器移动单位
                self.sim_controller.move_unit(unit["id"], new_x, new_y)

    def _defend_position(self):
        """防御模式：单位向最近的建筑移动"""
        state = self.sim_controller.get_state()
        if state and state.get("units") and state.get("structures"):
            friendly_units = [u for u in state["units"] if u["camp"] == "red" and u["status"] == "active"]
            buildings = [s for s in state["structures"] if s["type"] in ["bunker", "house"]]
            
            for unit in friendly_units:
                # 找到最近的建筑
                nearest_building = min(buildings, 
                    key=lambda b: abs(b["position"][0]-unit["position"][0]) + abs(b["position"][1]-unit["position"][1]))
                
                # 移动到建筑周围
                target_x = nearest_building["position"][0] + np.random.randint(-2,3)
                target_y = nearest_building["position"][1] + np.random.randint(-2,3)
                self.sim_controller.move_unit(unit["id"], target_x, target_y)

    def _retreat_units(self):
        """撤退模式：单位向地图边缘移动"""
        state = self.sim_controller.get_state()
        if state and state.get("units"):
            friendly_units = [u for u in state["units"] if u["camp"] == "red" and u["status"] == "active"]
            
            for unit in friendly_units:
                # 生成撤退方向（向左侧移动）
                new_x = max(0, unit["position"][0] - 5)
                new_y = unit["position"][1] + np.random.randint(-2,3)
                self.sim_controller.move_unit(unit["id"], new_x, new_y)

    def _call_artillery_support(self):
        """请求炮兵支援"""
        state = self.sim_controller.get_state()
        if state and state.get("units"):
            # 找到最近的敌方单位集群
            enemies = [u for u in state["units"] if u["camp"] == "blue" and u["status"] == "active"]
            if len(enemies) > 3:
                cluster_center = np.mean([(u["position"][0], u["position"][1]) for u in enemies[:3]], axis=0)
                self.sim_controller.call_artillery((int(cluster_center[0]), int(cluster_center[1])))
        
    def _attack_nearest_enemy(self):
        """攻击最近的敌方单位"""
        state = self.sim_controller.get_state()
        if state and state.get("units"):
            # 获取所有活跃单位
            active_units = [u for u in state["units"] if u.get("status") == "active"]
            if len(active_units) < 2:
                return
                
            # 找到最近的敌人
            attacker = active_units[0]  # 简化逻辑，实际需选择友军单位
            enemies = [u for u in active_units if u["camp"] != attacker["camp"]]
            if enemies:
                # 按距离排序并选择最近的
                enemies.sort(key=lambda u: abs(u["position"][0]-attacker["position"][0]) + 
                            abs(u["position"][1]-attacker["position"][1]))
                target = enemies[0]
                
                # 调用控制器发起攻击
                self.sim_controller.attack(attacker["id"], target["id"])

    def _calculate_reward(self):
        """计算即时奖励"""
        # 示例奖励函数
        state = self.sim_controller.get_state()
        reward = 0
        
        # 歼敌奖励
        reward += len([u for u in state['units'] 
            if u['camp'] == "blue" and u['status'] != "active"]) * 10
        
        # 时间惩罚
        reward -= 0.1
        
        return reward
    
    def _check_done(self):
        """检查是否结束"""
        state = self.sim_controller.get_state()
        active_red = len([u for u in state['units'] 
                    if u['camp'] == "red" and u['status'] == "active"])
        active_blue = len([u for u in state['units'] 
                     if u['camp'] == "blue" and u['status'] == "active"])
        return active_red == 0 or active_blue == 0
    
    def render(self, mode='human'):
        """实时渲染战场画面"""
        if mode == 'human':
            # 通过模拟控制器获取最新状态并渲染
            state = self.sim_controller.get_state()
            if state:
                self.sim_controller.renderer.render_all(
                    state['terrain'],
                    state['units'],
                    state['structures'],
                    self.sim_controller.shared_vision['red']
                )
            return True
        return super().render(mode=mode)

def train():
    """启动训练流程"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    # 创建环境
    env = WarzoneEnv()
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # 配置模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tb_logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2
    )
    
    # 开始训练
    model.learn(
        total_timesteps=1_000_000,
        callback=TensorboardCallback(),
        tb_log_name="warzone_ppo",
        progress_bar=True
    )
    
    # 保存最终模型
    model.save("warzone_ai")
    print("训练完成！模型已保存为 warzone_ai.zip")

if __name__ == "__main__":
    train()
