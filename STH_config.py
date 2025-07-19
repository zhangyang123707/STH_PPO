# STH (Spatio-Temporal Hybrid Attention Network) 配置文件

config = {
    # 基础参数
    'obs_dim': 16,           # 状态维度（需根据实际环境调整）
    'action_dim': 4,         # 动作维度（信号相位数）
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
    'batch_size': 64,        # 批量大小
    'buffer_size': 10000,    # 经验缓冲区大小
    
    # 损失权重
    'value_loss_coef': 0.5,  # 价值损失系数
    'entropy_coef': 0.01,    # 熵损失系数
    'high_level_weight': 1.0, # 高层策略权重
    'low_level_weight': 1.0,  # 低层策略权重
    
    # 奖励塑形参数
    'queue_length_weight': -0.1,      # 队列长度权重
    'waiting_time_weight': -0.05,     # 等待时间权重
    'throughput_weight': 0.2,         # 吞吐量权重
    'pressure_weight': -0.15,         # 压力权重
    'coordination_weight': 0.1,       # 协调权重
    
    # 特征处理参数
    'normalize_features': True,       # 是否标准化特征
    'feature_augmentation': True,     # 是否增强特征
    'temporal_features': True,        # 是否提取时间特征
    'spatial_features': True,         # 是否提取空间特征
    
    # 训练参数
    'max_episodes': 1000,             # 最大训练回合数
    'max_steps_per_episode': 1000,    # 每回合最大步数
    'log_interval': 10,               # 日志记录间隔
    'save_interval': 100,             # 模型保存间隔
    'eval_interval': 50,              # 评估间隔
    
    # 环境参数
    'topology': 'grid',               # 网络拓扑 ('grid', 'ring', 'random')
    'yellow_time': 3,                 # 黄灯时间
    'min_green_time': 10,             # 最小绿灯时间
    'max_green_time': 60,             # 最大绿灯时间
    
    # 路径配置
    'model_save_path': './models/STH_model.pth',
    'log_save_path': './logs/STH_training.log',
    'tensorboard_path': './runs/STH_experiment',
    
    # 设备配置
    'device': 'auto',                 # 设备选择 ('auto', 'cpu', 'cuda')
    'num_workers': 4,                 # 数据加载器工作进程数
    
    # 调试参数
    'debug': False,                   # 调试模式
    'verbose': True,                  # 详细输出
    'seed': 42,                       # 随机种子
}

# 不同场景的配置变体
config_small_network = config.copy()
config_small_network.update({
    'num_agents': 4,
    'obs_dim': 12,
    'action_dim': 2,
    'embed_dim': 64,
    'n_heads': 2,
})

config_large_network = config.copy()
config_large_network.update({
    'num_agents': 16,
    'obs_dim': 20,
    'action_dim': 8,
    'embed_dim': 256,
    'n_heads': 8,
    'num_layers': 3,
    'batch_size': 128,
})

config_fast_training = config.copy()
config_fast_training.update({
    'ppo_epochs': 5,
    'batch_size': 32,
    'lr': 2e-3,
    'max_episodes': 500,
})

# 奖励塑形配置变体
config_aggressive = config.copy()
config_aggressive.update({
    'queue_length_weight': -0.2,
    'waiting_time_weight': -0.1,
    'pressure_weight': -0.3,
    'coordination_weight': 0.2,
})

config_conservative = config.copy()
config_conservative.update({
    'queue_length_weight': -0.05,
    'waiting_time_weight': -0.02,
    'pressure_weight': -0.1,
    'coordination_weight': 0.05,
})

def get_config(config_name='default', dataset_name=None):
    """
    获取指定配置
    
    Args:
        config_name: 配置名称
        dataset_name: 数据集名称（可选）
        
    Returns:
        config: 配置字典
    """
    configs = {
        'default': config,
        'small': config_small_network,
        'large': config_large_network,
        'fast': config_fast_training,
        'aggressive': config_aggressive,
        'conservative': config_conservative,
    }
    
    base_config = configs.get(config_name, config).copy()
    
    # 如果指定了数据集，则更新配置
    if dataset_name:
        try:
            from STH_dataset_config import get_recommended_config_for_dataset
            dataset_config = get_recommended_config_for_dataset(dataset_name)
            base_config.update(dataset_config)
        except ImportError:
            print(f"Warning: Could not import dataset config for {dataset_name}")
        except Exception as e:
            print(f"Warning: Error loading dataset config: {e}")
    
    return base_config

def validate_config(config):
    """
    验证配置参数的有效性
    
    Args:
        config: 配置字典
        
    Returns:
        is_valid: 是否有效
        errors: 错误信息列表
    """
    errors = []
    
    # 检查必要参数
    required_params = ['obs_dim', 'action_dim', 'num_agents', 'embed_dim', 'n_heads']
    for param in required_params:
        if param not in config:
            errors.append(f"Missing required parameter: {param}")
    
    # 检查参数有效性
    if config.get('embed_dim', 0) % config.get('n_heads', 1) != 0:
        errors.append("embed_dim must be divisible by n_heads")
    
    if config.get('num_agents', 0) <= 0:
        errors.append("num_agents must be positive")
    
    if config.get('lr', 0) <= 0:
        errors.append("learning rate must be positive")
    
    if config.get('gamma', 0) <= 0 or config.get('gamma', 0) >= 1:
        errors.append("gamma must be between 0 and 1")
    
    return len(errors) == 0, errors 