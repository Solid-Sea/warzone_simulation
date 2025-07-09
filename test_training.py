from rl_env import WarzoneEnv

def test_training():
    # 覆盖原训练函数，减少步数
    def short_train():
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        env = WarzoneEnv()
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            tensorboard_log="./tb_logs_test/",
            learning_rate=3e-4,
            n_steps=256,  # 减少步数
            batch_size=64,
            n_epochs=3,   # 减少epoch
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2
        )
        
        model.learn(total_timesteps=100)  # 仅100步训练
        model.save("test_model")
        print("测试训练完成! 模型已保存为 test_model.zip")
    
    # 运行测试训练
    short_train()

if __name__ == "__main__":
    test_training()
