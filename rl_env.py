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
        # 示例动作映射（需根据实际需求完善）
        if action == 0: pass  # 无操作
        elif action == 1: self._move_units()
        elif action == 2: self._attack_nearest()
        
    def _calculate_reward(self):
        """计算即时奖励"""
        # 示例奖励函数
        state = self.sim_controller.get_state()
        reward = 0
        
        # 歼敌奖励
        reward += len([u for u in state['units'] 
                     if u['camp'] == "blue" and not u['is_active']]) * 10
        
        # 时间惩罚
        reward -= 0.1
        
        return reward
    
    def _check_done(self):
        """检查是否结束"""
        state = self.sim_controller.get_state()
        active_red = len([u for u in state['units'] 
                        if u['camp'] == "red" and u['is_active']])
        active_blue = len([u for u in state['units'] 
                         if u['camp'] == "blue" and u['is_active']])
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
