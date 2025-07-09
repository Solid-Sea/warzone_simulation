# 战区模拟器 - AI训练系统

## 项目概述
这是一个基于强化学习的军事模拟系统，用于训练AI指挥官在多单位战场环境中制定战略决策。系统包含完整的地形生成、单位控制和战斗模拟功能。

## 功能特性
- 多单位实时模拟（步兵、坦克、炮兵、工兵）
- 动态战场地形生成（沙漠、森林、山地）
- 强化学习训练环境（PPO算法）
- 实时战场可视化（Pygame渲染）
- 胜负判定系统（单位歼灭/建筑占领）

## 安装指南
```bash
# 克隆仓库
git clone https://github.com/your-username/warzone_simulation.git
cd warzone_simulation

# 安装依赖
pip install -r requirements.txt
```

## 使用说明
### 运行模拟
```bash
python main.py
```

### 训练AI模型
```bash
python rl_env.py
```

### 查看训练日志
```bash
tensorboard --logdir tb_logs/
```

## 开发文档
详见 [DEVELOPMENT.md](DEVELOPMENT.md)

## 贡献指南
欢迎通过Issue或Pull Request贡献代码。请确保：
1. 遵循现有代码风格
2. 添加必要的单元测试
3. 更新相关文档
