# 战区模拟器 - AI训练系统 v2.0

## 项目概述
这是一个基于强化学习的军事模拟系统，用于训练AI指挥官在多单位战场环境中制定战略决策。系统包含完整的地形生成、单位控制和战斗模拟功能，支持大规模单位作战和复杂战术训练。

## 🚀 最新更新
- ✅ 完成100,000步强化学习训练

## 功能特性
### 核心功能
- **多单位实时模拟**：步兵、坦克、炮兵、工兵四种单位类型
- **动态战场地形**：沙漠、森林、山地三种地形模式
- **智能AI训练**：基于PPO算法的强化学习环境
- **实时可视化**：Pygame渲染的战场态势图
- **胜负判定**：单位歼灭和建筑占领双重胜利条件

### 技术特性
- **高性能训练**：135 steps/second训练速度
- **模块化设计**：清晰的代码架构，易于扩展
- **完整日志**：TensorBoard训练监控
- **跨平台支持**：Windows/Linux/macOS兼容

## 快速开始

### 环境要求
- Python 3.8+
- PyTorch
- Stable-Baselines3
- Pygame

### 安装指南
```bash
# 克隆仓库
git clone https://github.com/Solid-Sea/warzone_simulation.git
cd warzone_simulation

# 安装依赖
pip install -r requirements.txt
```

## 使用说明

### 1. 运行可视化模拟
```bash
python main.py
```
启动交互式战场模拟，可实时观察红蓝双方作战过程。

### 2. 训练AI模型
```bash
# 快速测试训练（1000步）
python test_training.py

# 完整训练（100,000步）
python -m rl_env train --total_steps 100000 --batch_size 512
```

### 3. 评估训练结果
```bash
python -m rl_env evaluate --model_path models/warzone_ppo_model
```

### 4. 查看训练日志
```bash
tensorboard --logdir logs/
```

## 项目结构
```
warzone_simulation/
├── main.py                 # 可视化模拟入口
├── rl_env.py              # 强化学习环境
├── test_training.py       # 训练测试脚本
├── simulation_controller.py # 模拟控制器
├── requirements.txt       # 项目依赖
├── models/               # 训练模型存储
├── core/                 # 核心功能模块
│   ├── map_generator.py  # 地图生成器
│   ├── terrain_engine.py # 地形引擎
│   └── pathfinding.py    # 寻路算法
├── entities/             # 游戏实体
│   ├── unit.py          # 单位定义
│   └── structure.py     # 建筑定义
└── visualization/       # 可视化模块
    └── renderer.py      # 渲染器
```

## 训练参数
| 参数 | 值 | 说明 |
|------|----|------|
| 学习率 | 1e-4 | PPO优化器学习率 |
| 批量大小 | 512 | 训练批次大小 |
| 训练步数 | 100,000 | 总训练步数 |
| 网络结构 | MLP | 多层感知机策略网络 |

## 性能指标
- **训练速度**: 135 steps/second
- **平均奖励**: 14.87
- **胜率**: 100%
- **收敛步数**: ~40,000步

## 开发文档
详见 [DEVELOPMENT.md](DEVELOPMENT.md) 获取详细的开发指南和API文档。

## 贡献指南
欢迎通过Issue或Pull Request贡献代码。请确保：
1. 遵循现有代码风格（PEP 8）
2. 为新功能添加单元测试
3. 更新相关文档
4. 通过代码审查

## 许可证
MIT License - 详见 [LICENSE](LICENSE) 文件

## 联系方式
- 项目地址: https://github.com/Solid-Sea/warzone_simulation
- 问题反馈: https://github.com/Solid-Sea/warzone_simulation/issues
