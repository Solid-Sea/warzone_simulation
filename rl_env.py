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
            mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            self.logger.record("train/mean_reward", mean_reward)
            
            # 生成战场热力图（仅记录不显示）
            state = self.training_env.get_attr("sim_controller")[0].get_state()
            if state:
                terrain = np.array(state['terrain'])
                fig = plt.figure(figsize=(10, 6))
                plt.imshow(terrain, cmap='terrain')
                plt.colorbar()
                plt.title(f"Step {self.n_calls}")
                self.logger.record("battlefield", Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
                plt.close(fig)
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
        self.current_step = 0  # 初始化步数计数器
        self.initial_red_units = 0
        self.initial_blue_units = 0
        self.last_action = None
        self.action_space = spaces.Discrete(6)  # 基础动作空间
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.sim_controller.map_height, 
                   self.sim_controller.map_width, 5),
            dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        self.sim_controller.init_simulation()
        # 获取初始单位数量
        init_state = self.sim_controller.get_state()
        self.initial_red_units = len([u for u in init_state['units'] if u['camp'] == "red"])
        self.initial_blue_units = len([u for u in init_state['units'] if u['camp'] == "blue"])
        self.current_step = 0
        self.last_action = None
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
        state = self.sim_controller.get_state()
        if not state:
            return 0
            
        reward = 0
        
        # 实时战场分析
        active_red = len([u for u in state['units'] if u['camp'] == "red" and u['status'] == "active"])
        active_blue = len([u for u in state['units'] if u['camp'] == "blue" and u['status'] == "active"])
        
        # 基础奖励（歼敌差异）
        reward += (self.initial_blue_units - active_blue) * 15  # 歼敌奖励
        reward -= (self.initial_red_units - active_red) * 10  # 友军损失惩罚
        
        # 建筑控制奖励
        central_buildings = [s for s in state.get('structures', []) 
                            if s['type'] in ['bunker', 'house'] 
                            and 25 <= s['position'][0] <= 45 
                            and 15 <= s['position'][1] <= 35]
        
        for building in central_buildings:
            red_nearby = sum(1 for u in state['units'] 
                            if u['camp'] == "red" and u['status'] == "active"
                            and abs(u['position'][0] - building['position'][0]) <= 2
                            and abs(u['position'][1] - building['position'][1]) <= 2)
            blue_nearby = sum(1 for u in state['units'] 
                             if u['camp'] == "blue" and u['status'] == "active"
                             and abs(u['position'][0] - building['position'][0]) <= 2
                             and abs(u['position'][1] - building['position'][1]) <= 2)
            
            # 建筑控制奖励
            if red_nearby >= 2 and red_nearby > blue_nearby:
                reward += 5  # 控制关键建筑奖励
            elif blue_nearby >= 2 and blue_nearby > red_nearby:
                reward -= 3  # 敌方控制惩罚
                
        # 战略行动奖励
        if self.last_action == 2:  # 攻击
            reward += 1
        elif self.last_action == 3:  # 防御
            reward += 0.5
            
        # 时间惩罚（随步数增加）
        reward -= min(0.2, self.current_step * 0.001)
        
        return reward
    
    def _check_done(self):
        """综合胜利条件判断"""
        state = self.sim_controller.get_state()
        if not state:
            return False

        # 获取存活单位
        active_red = len([u for u in state['units'] if u['camp'] == "red" and u['status'] == "active"])
        active_blue = len([u for u in state['units'] if u['camp'] == "blue" and u['status'] == "active"])

        # 优先判断全歼情况：红方全灭则蓝方胜利；蓝方全灭且红方存活则红方胜利
        if active_red == 0:
            return True  # 蓝方胜利（包括双方全灭情况）
        if active_blue == 0 and active_red > 0:
            return True  # 红方胜利（仅当红方仍有单位时）

        # 建筑占领检查（需要至少3个关键建筑）
        central_buildings = [s for s in state.get('structures', []) 
                            if s['type'] in ['bunker', 'house'] 
                            and 25 <= s['position'][0] <= 45 
                            and 15 <= s['position'][1] <= 35]
        if len(central_buildings) < 3:
            return False  # 没有足够关键建筑时不触发占领胜利
            
        red_controlled = 0
        blue_controlled = 0

        for building in central_buildings:
            red_nearby = sum(1 for u in state['units'] 
                            if u['camp'] == "red" and u['status'] == "active"
                            and abs(u['position'][0] - building['position'][0]) <= 2  # 缩小控制范围
                            and abs(u['position'][1] - building['position'][1]) <= 2)
            blue_nearby = sum(1 for u in state['units'] 
                             if u['camp'] == "blue" and u['status'] == "active"
                             and abs(u['position'][0] - building['position'][0]) <= 2
                             and abs(u['position'][1] - building['position'][1]) <= 2)

            # 需要至少2个单位才能控制建筑
            if red_nearby >= 2 and red_nearby > blue_nearby:
                red_controlled += 1
            elif blue_nearby >= 2 and blue_nearby > red_nearby:
                blue_controlled += 1

        # 胜利条件（控制超过60%关键建筑且保持2个回合）
        max_steps = 300  # 缩短最大步数
        self.current_step += 1
        
        # 红方占领胜利
        if red_controlled >= len(central_buildings) * 0.6:
            return True
        # 蓝方坚守胜利（步数过半且控制多数建筑）
        if self.current_step >= max_steps and blue_controlled >= len(central_buildings) * 0.5:
            return True
        # 强制结束
        if self.current_step >= max_steps * 1.5:  
            return True
            
        return False
    
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
    """启动训练流程（简化输出版）"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    # 创建环境
    env = WarzoneEnv()
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # 配置模型（关闭详细输出）
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,  # 关闭详细输出
        tensorboard_log="./tb_logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2
    )
    
    # 自定义回调优化输出
    class SimpleCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.red_wins = 0
            self.blue_wins = 0
            self.total_rollouts = 0
            
        def _on_step(self) -> bool:
            """必需实现方法，但不执行具体操作"""
            return True
            
        def _on_rollout_end(self) -> None:
            self.total_rollouts += 1
            state = self.training_env.get_attr("sim_controller")[0].get_state()
            
            if state:
                active_red = len([u for u in state['units'] if u['camp']=="red" and u['status']=="active"])
                active_blue = len([u for u in state['units'] if u['camp']=="blue" and u['status']=="active"])
                
                # 判断胜方
                if active_red == 0 and active_blue > 0:
                    winner = "蓝方"
                    self.blue_wins += 1
                elif active_blue == 0:
                    winner = "红方"
                    self.red_wins += 1
                else:
                    winner = "平局"
                
                # 计算胜率
                red_win_rate = (self.red_wins / self.total_rollouts) * 100
                blue_win_rate = (self.blue_wins / self.total_rollouts) * 100
                
                print(f"轮次 {self.total_rollouts}: {winner}胜利 | "
                      f"红方胜率: {red_win_rate:.1f}% | "
                      f"蓝方胜率: {blue_win_rate:.1f}% | "
                      f"红方剩余: {active_red} | "
                      f"蓝方剩余: {active_blue}")
            else:
                print(f"轮次 {self.total_rollouts}: 无法获取状态")
    
    # 开始训练
    model.learn(
        total_timesteps=1_000_000,
        callback=SimpleCallback(),  # 使用优化后的回调
        tb_log_name="warzone_ppo"
    )
    
    # 保存最终模型
    model.save("warzone_ai")
    print("训练完成！模型已保存为 warzone_ai.zip")

if __name__ == "__main__":
    train()
