#!/usr/bin/env python3
"""
战区AI训练测试脚本 - 完整版
包含模型训练、评估、保存和加载功能
"""

import os
import sys
import json
import time
import numpy as np
from rl_env import WarzoneEnv, train_model, evaluate_model
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import warnings
warnings.filterwarnings("ignore")

class TrainingConfig:
    """训练配置类"""
    def __init__(self):
        self.config = {
            "training": {
                "total_timesteps": 50000,
                "learning_rate": 1e-4,
                "n_steps": 2048,
                "batch_size": 256,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2
            },
            "evaluation": {
                "eval_freq": 10000,
                "n_eval_episodes": 10,
                "save_freq": 25000
            },
            "environment": {
                "width": 80,
                "height": 60,
                "max_steps": 500
            }
        }
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.config.update(json.load(f))

def create_env(config_path=None):
    """创建环境包装器"""
    def _init():
        env = WarzoneEnv(config_path=config_path)
        env = Monitor(env)  # 添加监控
        return env
    return _init

def setup_callbacks(model_dir, eval_env):
    """设置训练回调"""
    callbacks = []
    
    # 评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best_model"),
        log_path=os.path.join(model_dir, "eval_logs"),
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    # 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=os.path.join(model_dir, "checkpoints"),
        name_prefix="warzone_model"
    )
    callbacks.append(checkpoint_callback)
    
    return callbacks

def run_training():
    """运行完整训练流程"""
    print("🎯 启动战区AI训练系统...")
    
    # 创建目录
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "eval_logs"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "logs"), exist_ok=True)
    
    # 保存配置
    config = TrainingConfig()
    config.save(os.path.join(model_dir, "training_config.json"))
    
    # 创建环境
    print("🌍 创建训练环境...")
    train_env = DummyVecEnv([create_env() for _ in range(1)])
    eval_env = DummyVecEnv([create_env() for _ in range(1)])
    
    # 创建模型
    print("🤖 初始化PPO模型...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config.config["training"]["learning_rate"],
        n_steps=config.config["training"]["n_steps"],
        batch_size=config.config["training"]["batch_size"],
        n_epochs=config.config["training"]["n_epochs"],
        gamma=config.config["training"]["gamma"],
        gae_lambda=config.config["training"]["gae_lambda"],
        clip_range=config.config["training"]["clip_range"],
        verbose=1,
        tensorboard_log=os.path.join(model_dir, "logs")
    )
    
    # 设置回调
    callbacks = setup_callbacks(model_dir, eval_env)
    
    # 开始训练
    print("🚀 开始训练...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=config.config["training"]["total_timesteps"],
        callback=callbacks,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"✅ 训练完成! 耗时: {training_time:.2f}秒")
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, "warzone_final_model")
    model.save(final_model_path)
    print(f"💾 最终模型已保存: {final_model_path}")
    
    return model

def quick_test():
    """快速测试训练流程"""
    print("🧪 运行快速测试...")
    
    # 创建环境
    env = WarzoneEnv()
    
    # 测试环境
    obs, _ = env.reset()
    print(f"观察空间形状: {obs.shape}")
    
    # 测试随机动作
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"步骤 {step+1}: 动作={action}, 奖励={reward:.2f}, 红方={info['red_units']}, 蓝方={info['blue_units']}")
        
        if terminated or truncated:
            break
    
    print("✅ 快速测试完成")

def load_and_evaluate(model_path=None):
    """加载并评估模型"""
    if model_path is None:
        model_path = "models/warzone_final_model"
    
    if not os.path.exists(f"{model_path}.zip"):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    print("📊 加载并评估模型...")
    
    # 加载模型
    model = PPO.load(model_path)
    
    # 评估
    evaluate_model(model, episodes=20)
    
    print("✅ 评估完成")

def benchmark_training():
    """训练性能基准测试"""
    print("⚡ 运行训练性能基准测试...")
    
    # 测试不同配置
    configs = [
        {"total_timesteps": 1000, "learning_rate": 1e-3},
        {"total_timesteps": 5000, "learning_rate": 1e-4},
        {"total_timesteps": 10000, "learning_rate": 1e-4}
    ]
    
    results = []
    
    for i, cfg in enumerate(configs):
        print(f"\n测试配置 {i+1}: {cfg}")
        
        start_time = time.time()
        model = train_model(**cfg)
        training_time = time.time() - start_time
        
        # 快速评估
        env = WarzoneEnv()
        obs, _ = env.reset()
        total_reward = 0
        
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        result = {
            "config": cfg,
            "training_time": training_time,
            "final_reward": float(total_reward)
        }
        results.append(result)
        
        print(f"训练时间: {training_time:.2f}s, 最终奖励: {total_reward:.2f}")
    
    # 保存结果
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        return obj
    
    with open("benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=convert_to_serializable)
    
    print("✅ 基准测试完成，结果已保存")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="战区AI训练系统")
    parser.add_argument("--mode", choices=["train", "test", "evaluate", "benchmark"], 
                        default="test", help="运行模式")
    parser.add_argument("--model-path", type=str, help="模型路径")
    parser.add_argument("--timesteps", type=int, default=50000, help="训练步数")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        config = TrainingConfig()
        config.config["training"]["total_timesteps"] = args.timesteps
        run_training()
    elif args.mode == "test":
        quick_test()
    elif args.mode == "evaluate":
        load_and_evaluate(args.model_path)
    elif args.mode == "benchmark":
        benchmark_training()

if __name__ == "__main__":
    # 如果没有参数，运行快速测试
    if len(sys.argv) == 1:
        quick_test()
    else:
        main()
