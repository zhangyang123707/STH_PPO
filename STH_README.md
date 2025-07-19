# STH (Spatio-Temporal Hybrid Attention Network) + Hierarchical PPO

## 概述

STH是一个基于时空混合注意力网络和分层PPO算法的交通信号控制系统。该系统结合了空间注意力（邻居路口间的关系）和时间注意力（历史状态序列），通过分层策略实现高效的交通信号控制。

## 主要特性

- **时空混合注意力**: 融合空间和时间特征，实现多路口协作
- **分层PPO**: 高层策略（信号阶段选择）+ 低层策略（具体动作）
- **奖励塑形**: 多目标奖励函数，平衡各种交通指标
- **特征融合**: 智能的特征处理和融合机制
- **可扩展架构**: 支持不同规模的交通网络

## 文件结构

```
STH_*.py
├── STH_model.py              # 时空混合注意力网络模型
├── STH_ppo.py                # 分层PPO算法实现
├── STH_agent.py              # 主智能体类
├── STH_utils.py              # 工具函数（奖励塑形、特征处理等）
├── STH_config.py             # 配置文件
├── STH_dataset_config.py     # 数据集配置文件
├── STH_test.py               # 测试脚本
├── STH_example.py            # 使用示例
├── STH_dataset_example.py    # 数据集使用示例
├── STH_jinan_example.py      # 济南数据集专用示例
└── STH_README.md             # 说明文档
```

## 核心组件

### 1. STHANModel (时空混合注意力网络)

**主要模块:**
- `MultiHeadAttention`: 多头注意力机制
- `SpatialAttention`: 空间注意力（处理邻居关系）
- `TemporalAttention`: 时间注意力（处理历史序列）
- `FeatureFusion`: 特征融合模块

**输入:**
- `obs`: 当前观测 [batch, num_agents, obs_dim]
- `neighbor_obs`: 邻居观测 [batch, num_agents, num_neighbors, obs_dim]
- `history_obs`: 历史观测 [batch, num_agents, history_len, obs_dim]

**输出:**
- `high_level_logits`: 高层策略logits [batch, num_agents, 4]
- `low_level_logits`: 低层策略logits [batch, num_agents, action_dim]
- `value`: 状态价值 [batch, num_agents, 1]

### 2. HierarchicalPPO (分层PPO算法)

**主要功能:**
- 分层动作选择（高层+低层）
- GAE (Generalized Advantage Estimation) 计算
- PPO更新（带裁剪）
- 经验缓冲区管理

**训练流程:**
1. 收集经验数据
2. 计算GAE优势函数
3. 多轮PPO更新
4. 策略和价值网络优化

### 3. STHAgent (主智能体)

**主要功能:**
- 环境交互管理
- 训练流程控制
- 模型保存/加载
- 性能评估

## 快速开始

### 1. 安装依赖

```bash
pip install torch numpy
```

### 2. 基本使用

```python
from STH_agent import STHAgent
from STH_config import get_config

# 创建环境（需要实现reset()和step()方法）
env = YourTrafficEnvironment()

# 创建智能体
agent = STHAgent(env, config_name='small')

# 训练
agent.train()

# 推理
obs = env.reset()
action = agent.get_action(obs)
```

### 3. 使用特定数据集

```python
from STH_agent import STHAgent
from STH_dataset_config import list_available_datasets, get_dataset_info

# 查看可用数据集
datasets = list_available_datasets()
print(f"Available datasets: {datasets}")

# 获取数据集信息
info = get_dataset_info('hangzhou_4_4')
print(f"Dataset: {info['name']}, Agents: {info['num_agents']}")

# 使用特定数据集创建智能体
agent = STHAgent(env, config_name='default', dataset_name='hangzhou_4_4')

# 训练（配置会自动适配数据集）
agent.train()
```

### 4. 使用济南数据集

```python
from STH_agent import STHAgent
from STH_config import get_config

# 创建环境
env = YourTrafficEnvironment()

# 使用济南3x4数据集创建智能体
agent = STHAgent(env, config_name='default', dataset_name='jinan_3_4')

# 训练（自动适配12个智能体，3x4网格）
agent.train()
```

### 3. 运行测试

```bash
python STH_test.py
```

### 4. 运行示例

```bash
python STH_example.py
```

### 5. 运行数据集示例

```bash
python STH_dataset_example.py
```

### 6. 运行济南数据集示例

```bash
python STH_jinan_example.py
```

## 配置说明

### 预定义配置

- `default`: 默认配置（4个路口，128维嵌入）
- `small`: 小网络配置（4个路口，64维嵌入）
- `large`: 大网络配置（16个路口，256维嵌入）
- `fast`: 快速训练配置（减少更新轮数）
- `aggressive`: 激进奖励塑形
- `conservative`: 保守奖励塑形

### 可用数据集

系统支持以下数据集：

#### 真实城市数据
- **杭州数据集**: `hangzhou_4_4`, `hangzhou_4_4_5734`, `hangzhou_4_4_5816`
- **济南数据集**: `jinan_3_4`, `jinan_3_4_2000`, `jinan_3_4_2500`
- **纽约数据集**: `newyork_16_3`, `newyork_28_7`

#### 合成数据
- **模板数据集**: `template_1_1`, `template_2_2`, `template_6_6`

#### 数据集特性
- **网格规模**: 从1x1到28x7不等
- **智能体数量**: 从1个到196个不等
- **交通流量**: 不同车辆密度配置
- **城市类型**: 真实城市和合成场景

### 主要参数

```python
config = {
    # 基础参数
    'obs_dim': 16,           # 状态维度
    'action_dim': 4,         # 动作维度
    'num_agents': 4,         # 路口数量
    'history_len': 4,        # 历史状态长度
    'num_neighbors': 4,      # 最大邻居数量
    
    # 模型参数
    'embed_dim': 128,        # 嵌入维度
    'n_heads': 4,            # 注意力头数
    'num_layers': 2,         # 注意力层数
    'dropout': 0.1,          # Dropout率
    
    # PPO参数
    'ppo_clip': 0.2,         # PPO裁剪参数
    'ppo_epochs': 10,        # PPO每轮更新步数
    'lr': 1e-3,              # 学习率
    'gamma': 0.99,           # 折扣因子
    'lam': 0.95,             # GAE参数
    
    # 奖励塑形参数
    'queue_length_weight': -0.1,      # 队列长度权重
    'waiting_time_weight': -0.05,     # 等待时间权重
    'throughput_weight': 0.2,         # 吞吐量权重
    'pressure_weight': -0.15,         # 压力权重
    'coordination_weight': 0.1,       # 协调权重
}
```

## 奖励塑形

系统支持多目标奖励塑形，包括：

1. **队列长度惩罚**: 减少车辆排队
2. **等待时间惩罚**: 减少车辆等待时间
3. **吞吐量奖励**: 鼓励车辆通过
4. **压力惩罚**: 基于队列长度和等待时间的综合指标
5. **协调奖励**: 鼓励相邻路口协调
6. **平滑奖励**: 避免频繁相位切换
7. **效率奖励**: 基于车辆速度

## 特征处理

### 特征标准化
- 自动标准化观测特征
- 支持邻居和历史特征标准化

### 特征增强
- 统计特征提取（均值、标准差、最大值、最小值）
- 时间差分特征
- 空间关系特征

### 邻接矩阵
支持多种网络拓扑：
- `grid`: 网格拓扑
- `ring`: 环形拓扑
- `random`: 随机拓扑

## 训练流程

1. **初始化**: 创建模型、PPO算法、历史缓冲区
2. **环境交互**: 收集观测、选择动作、执行动作
3. **数据处理**: 特征处理、奖励塑形、GAE计算
4. **模型更新**: PPO更新、策略优化
5. **评估**: 定期评估模型性能
6. **保存**: 保存最佳模型

## 性能监控

系统提供详细的性能监控：

- **训练统计**: 奖励、步数、熵等
- **交通指标**: 队列长度、等待时间、吞吐量等
- **模型指标**: 策略熵、价值损失等
- **日志记录**: 详细的训练日志

## 扩展指南

### 1. 自定义环境

实现以下接口：
```python
class YourEnvironment:
    def reset(self):
        # 返回初始观测 [num_agents, obs_dim]
        pass
    
    def step(self, actions):
        # 执行动作，返回 (obs, reward, done, info)
        pass
```

### 2. 使用特定数据集

```python
# 方法1: 在创建智能体时指定数据集
agent = STHAgent(env, config_name='default', dataset_name='hangzhou_4_4')

# 方法2: 获取数据集特定配置
config = get_config('default', dataset_name='hangzhou_4_4')
agent = STHAgent(env, config)

# 方法3: 查看数据集信息
from STH_dataset_config import get_dataset_info, print_dataset_summary
info = get_dataset_info('hangzhou_4_4')
print_dataset_summary()
```

### 3. 自定义奖励函数

修改 `STH_utils.py` 中的 `reward_shaping` 函数：
```python
def custom_reward_shaping(raw_reward, state, next_state, info, config):
    # 实现你的奖励塑形逻辑
    return shaped_reward
```

### 4. 自定义特征处理

修改 `STH_utils.py` 中的特征处理函数：
```python
def custom_feature_processing(obs, neighbor_obs, history_obs, config):
    # 实现你的特征处理逻辑
    return processed_obs, processed_neighbor_obs, processed_history_obs
```

## 故障排除

### 常见问题

1. **内存不足**: 减少 `batch_size` 或 `history_len`
2. **训练不稳定**: 调整学习率或PPO参数
3. **收敛慢**: 检查奖励函数设计
4. **维度不匹配**: 确认配置参数正确

### 调试模式

启用调试模式：
```python
config['debug'] = True
config['verbose'] = True
```

## 引用

如果您使用了这个STH模型，请引用相关论文：

```
@article{sth2024,
  title={Spatio-Temporal Hybrid Attention Network for Traffic Signal Control},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至 [your-email@example.com]

---

**注意**: 这是一个研究原型，在实际部署前请进行充分的测试和验证。 