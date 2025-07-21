"""
增强的测试和训练脚本 - 集成优化后的环境
"""
import os
import numpy as np
import torch
from rl_env import WarzoneEnv, train_model, evaluate_model
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import time

def create_enhanced_model(env, total_timesteps=10000):
    """创建增强的PPO模型"""
    print("创建增强PPO模型...")
    
    # 增强的PPO配置
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./logs/enhanced_training"
    )
    
    return model

def run_comprehensive_training():
    """运行综合训练流程"""
    print("=" * 60)
    print("启动综合训练流程...")
    print("=" * 60)
    
    # 创建环境
    env = WarzoneEnv()
    env = Monitor(env)  # 监控环境
    
    # 创建模型
    model = create_enhanced_model(env, total_timesteps=20000)
    
    # 设置回调
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="./models/checkpoints/",
        name_prefix="warzone_model"
    )
    
    eval_env = WarzoneEnv()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model",
        log_path="./logs/eval",
        eval_freq=500,
        deterministic=True,
        render=False
    )
    
    # 开始训练
    print("开始增强训练...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=20000,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"训练完成！耗时: {training_time:.2f}秒")
    
    # 保存最终模型
    os.makedirs("models", exist_ok=True)
    final_model_path = "models/warzone_final_model"
    model.save(final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    return model

def test_model_performance(model, test_episodes=10):
    """测试模型性能"""
    print("\n" + "=" * 60)
    print("开始模型性能测试...")
    print("=" * 60)
    
    env = WarzoneEnv()
    
    # 运行测试
    test_rewards = []
    win_rates = []
    
    for episode in range(test_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        # 记录初始单位数量
        initial_red = len([u for u in env.units.values() if u["camp"] == "red"])
        initial_blue = len([u for u in env.units.values() if u["camp"] == "blue"])
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step_count += 1
            done = terminated or truncated
        
        # 计算胜率
        final_red = len([u for u in env.units.values() if u["camp"] == "red"])
        final_blue = len([u for u in env.units.values() if u["camp"] == "blue"])
        
        win = final_red > 0 and final_blue == 0
        win_rates.append(1.0 if win else 0.0)
        
        test_rewards.append(episode_reward)
        
        print(f"测试回合 {episode + 1}:")
        print(f"  奖励: {episode_reward:.2f}")
        print(f"  步数: {step_count}")
        print(f"  红方存活: {final_red}/{initial_red}")
        print(f"  蓝方存活: {final_blue}/{initial_blue}")
        print(f"  胜利: {'是' if win else '否'}")
        print("-" * 40)
    
    # 统计结果
    avg_reward = np.mean(test_rewards)
    avg_win_rate = np.mean(win_rates)
    
    print("\n性能测试结果:")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"胜率: {avg_win_rate:.2%}")
    print(f"标准差: {np.std(test_rewards):.2f}")
    
    return {
        "average_reward": avg_reward,
        "win_rate": avg_win_rate,
        "std_reward": np.std(test_rewards)
    }

def benchmark_training_speed():
    """基准测试训练速度"""
    print("\n" + "=" * 60)
    print("开始训练速度基准测试...")
    print("=" * 60)
    
    env = WarzoneEnv()
    model = create_enhanced_model(env, total_timesteps=5000)
    
    start_time = time.time()
    model.learn(total_timesteps=5000, progress_bar=False)
    training_time = time.time() - start_time
    
    steps_per_second = 5000 / training_time
    
    print(f"训练速度: {steps_per_second:.2f} steps/second")
    print(f"总训练时间: {training_time:.2f}秒")
    
    return steps_per_second

def main():
    """主测试流程"""
    print("战区AI训练测试系统 v2.0")
    print("=" * 60)
    
    # 检查环境
    print("1. 环境检查...")
    env = WarzoneEnv()
    obs, _ = env.reset()
    print(f"   观察空间形状: {obs.shape}")
    print(f"   动作空间: {env.action_space}")
    print("   ✓ 环境正常")
    
    # 运行训练
    print("\n2. 开始训练...")
    model = run_comprehensive_training()
    
    # 性能测试
    print("\n3. 性能测试...")
    results = test_model_performance(model, test_episodes=5)
    
    # 速度基准
    print("\n4. 速度基准测试...")
    speed = benchmark_training_speed()
    
    # 总结报告
    print("\n" + "=" * 60)
    print("训练测试总结报告")
    print("=" * 60)
    print(f"平均奖励: {results['average_reward']:.2f}")
    print(f"胜率: {results['win_rate']:.2%}")
    print(f"训练速度: {speed:.2f} steps/second")
    print("=" * 60)
    
    return model, results

if __name__ == "__main__":
    model, results = main()
