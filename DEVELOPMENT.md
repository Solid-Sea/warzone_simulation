# 战区模拟器开发文档

## 项目结构
```
warzone_simulation/
├── core/                # 核心模拟逻辑
│   ├── map_generator.py
│   ├── pathfinding.py
│   └── terrain_engine.py
├── entities/            # 游戏实体定义
│   ├── structure.py
│   └── unit.py
├── visualization/       # 可视化模块
│   └── renderer.py
├── rl_env.py            # 强化学习环境
├── simulation_controller.py # 模拟控制器
├── main.py              # 主程序入口
└── requirements.txt     # 依赖库
```

## 训练流程
1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 启动训练：
```bash
python rl_env.py
```

3. 监控训练：
- TensorBoard日志保存在`tb_logs/`目录
- 模型每1000步自动保存

## 配置参数
```python
# rl_env.py 中的PPO配置
PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2
)
```

## 变更日志

### 2025/7/9
- 修复训练中异常平局问题：修改`rl_env.py`中`_check_done`方法，当双方单位同时全灭时判蓝方胜利
