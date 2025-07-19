import numpy as np
import torch
from collections import deque
import math

class HistoryBuffer:
    """历史状态缓冲区，用于存储每个智能体的历史观测"""
    def __init__(self, max_len, num_agents, obs_dim):
        self.max_len = max_len
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.buffers = [deque(maxlen=max_len) for _ in range(num_agents)]
        
    def push(self, obs):
        """
        添加新的观测到历史缓冲区
        obs: [num_agents, obs_dim]
        """
        for i in range(self.num_agents):
            self.buffers[i].append(obs[i].copy())
    
    def get_history(self, history_len=None):
        """
        获取历史观测
        Returns: [num_agents, history_len, obs_dim]
        """
        if history_len is None:
            history_len = self.max_len
        
        history = np.zeros((self.num_agents, history_len, self.obs_dim))
        
        for i in range(self.num_agents):
            buffer_len = len(self.buffers[i])
            if buffer_len >= history_len:
                # 如果缓冲区足够长，取最近的history_len个
                history[i] = np.array(list(self.buffers[i])[-history_len:])
            else:
                # 如果缓冲区不够长，用零填充前面
                history[i, -buffer_len:] = np.array(list(self.buffers[i]))
        
        return history
    
    def clear(self):
        """清空所有缓冲区"""
        for buffer in self.buffers:
            buffer.clear()

def reward_shaping(raw_reward, state, next_state, info, config):
    """
    奖励塑形函数：结合全局与局部交通指标
    
    Args:
        raw_reward: 原始奖励
        state: 当前状态
        next_state: 下一个状态
        info: 环境信息
        config: 配置参数
    
    Returns:
        shaped_reward: 塑形后的奖励
    """
    shaped_reward = raw_reward.copy()
    
    # 获取配置参数
    queue_length_weight = config.get('queue_length_weight', -0.1)
    waiting_time_weight = config.get('waiting_time_weight', -0.05)
    throughput_weight = config.get('throughput_weight', 0.2)
    pressure_weight = config.get('pressure_weight', -0.15)
    coordination_weight = config.get('coordination_weight', 0.1)
    
    # 1. 队列长度奖励
    if 'queue_length' in info:
        queue_penalty = queue_length_weight * np.sum(info['queue_length'])
        shaped_reward += queue_penalty
    
    # 2. 等待时间奖励
    if 'waiting_time' in info:
        waiting_penalty = waiting_time_weight * np.sum(info['waiting_time'])
        shaped_reward += waiting_penalty
    
    # 3. 吞吐量奖励
    if 'throughput' in info:
        throughput_reward = throughput_weight * np.sum(info['throughput'])
        shaped_reward += throughput_reward
    
    # 4. 压力奖励（基于路口压力）
    if 'pressure' in info:
        pressure_penalty = pressure_weight * np.sum(info['pressure'])
        shaped_reward += pressure_penalty
    
    # 5. 协调奖励（鼓励相邻路口协调）
    if 'neighbor_coordination' in info:
        coordination_reward = coordination_weight * info['neighbor_coordination']
        shaped_reward += coordination_reward
    
    # 6. 平滑奖励（避免频繁切换）
    if 'phase_change' in info:
        change_penalty = -0.01 * info['phase_change']  # 轻微惩罚相位切换
        shaped_reward += change_penalty
    
    # 7. 效率奖励（基于车辆速度）
    if 'avg_speed' in info:
        speed_reward = 0.05 * np.mean(info['avg_speed'])
        shaped_reward += speed_reward
    
    return shaped_reward

def process_features(obs, neighbor_obs, history_obs, config):
    """
    特征处理与融合函数
    
    Args:
        obs: 当前观测 [num_agents, obs_dim]
        neighbor_obs: 邻居观测 [num_agents, num_neighbors, obs_dim]
        history_obs: 历史观测 [num_agents, history_len, obs_dim]
        config: 配置参数
    
    Returns:
        processed_obs: 处理后的当前观测
        processed_neighbor_obs: 处理后的邻居观测
        processed_history_obs: 处理后的历史观测
    """
    # 1. 特征标准化
    if config.get('normalize_features', True):
        obs = normalize_features(obs)
        neighbor_obs = normalize_features(neighbor_obs)
        history_obs = normalize_features(history_obs)
    
    # 2. 特征增强
    if config.get('feature_augmentation', True):
        obs = augment_features(obs)
        neighbor_obs = augment_features(neighbor_obs)
        history_obs = augment_features(history_obs)
    
    # 3. 时间特征提取
    if config.get('temporal_features', True):
        history_obs = extract_temporal_features(history_obs)
    
    # 4. 空间特征提取
    if config.get('spatial_features', True):
        neighbor_obs = extract_spatial_features(neighbor_obs)
    
    return obs, neighbor_obs, history_obs

def normalize_features(features):
    """特征标准化"""
    if len(features.shape) == 3:  # [num_agents, num_neighbors, obs_dim] or [num_agents, history_len, obs_dim]
        mean = np.mean(features, axis=(0, 1), keepdims=True)
        std = np.std(features, axis=(0, 1), keepdims=True) + 1e-8
        return (features - mean) / std
    else:  # [num_agents, obs_dim]
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-8
        return (features - mean) / std

def augment_features(features):
    """特征增强"""
    # 添加统计特征
    if len(features.shape) == 3:
        # 对于邻居或历史特征，添加统计信息
        mean_feat = np.mean(features, axis=1, keepdims=True)
        std_feat = np.std(features, axis=1, keepdims=True)
        max_feat = np.max(features, axis=1, keepdims=True)
        min_feat = np.min(features, axis=1, keepdims=True)
        
        # 将统计特征添加到原始特征中
        stats = np.concatenate([mean_feat, std_feat, max_feat, min_feat], axis=-1)
        # 这里可以根据需要调整特征维度
        return features
    
    return features

def extract_temporal_features(history_obs):
    """提取时间特征"""
    # 计算时间差分
    if history_obs.shape[1] > 1:
        temporal_diff = np.diff(history_obs, axis=1)
        # 可以添加更多时间特征，如趋势、周期性等
        return history_obs
    
    return history_obs

def extract_spatial_features(neighbor_obs):
    """提取空间特征"""
    # 计算空间关系特征
    if neighbor_obs.shape[1] > 1:
        # 计算邻居间的相似性
        neighbor_similarity = np.mean(neighbor_obs, axis=1, keepdims=True)
        # 可以添加更多空间特征
        return neighbor_obs
    
    return neighbor_obs

def create_adjacency_matrix(num_agents, topology='grid'):
    """
    创建邻接矩阵
    
    Args:
        num_agents: 智能体数量
        topology: 拓扑结构 ('grid', 'random', 'ring')
    
    Returns:
        adjacency_matrix: [num_agents, num_agents]
    """
    if topology == 'grid':
        # 网格拓扑
        grid_size = int(math.sqrt(num_agents))
        adjacency_matrix = np.zeros((num_agents, num_agents))
        
        for i in range(num_agents):
            row, col = i // grid_size, i % grid_size
            
            # 上下左右邻居
            neighbors = []
            if row > 0:  # 上
                neighbors.append((row - 1) * grid_size + col)
            if row < grid_size - 1:  # 下
                neighbors.append((row + 1) * grid_size + col)
            if col > 0:  # 左
                neighbors.append(row * grid_size + col - 1)
            if col < grid_size - 1:  # 右
                neighbors.append(row * grid_size + col + 1)
            
            for neighbor in neighbors:
                if neighbor < num_agents:
                    adjacency_matrix[i, neighbor] = 1
                    adjacency_matrix[neighbor, i] = 1
    
    elif topology == 'ring':
        # 环形拓扑
        adjacency_matrix = np.zeros((num_agents, num_agents))
        for i in range(num_agents):
            adjacency_matrix[i, (i + 1) % num_agents] = 1
            adjacency_matrix[i, (i - 1) % num_agents] = 1
    
    elif topology == 'random':
        # 随机拓扑
        adjacency_matrix = np.random.randint(0, 2, (num_agents, num_agents))
        np.fill_diagonal(adjacency_matrix, 0)  # 移除自环
        adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)  # 确保对称
    
    return adjacency_matrix

def get_neighbor_obs(obs, adjacency_matrix, num_neighbors):
    """
    根据邻接矩阵获取邻居观测
    
    Args:
        obs: 当前观测 [num_agents, obs_dim]
        adjacency_matrix: 邻接矩阵 [num_agents, num_agents]
        num_neighbors: 最大邻居数量
    
    Returns:
        neighbor_obs: 邻居观测 [num_agents, num_neighbors, obs_dim]
    """
    num_agents, obs_dim = obs.shape
    neighbor_obs = np.zeros((num_agents, num_neighbors, obs_dim))
    
    for i in range(num_agents):
        # 获取邻居索引
        neighbors = np.where(adjacency_matrix[i] == 1)[0]
        
        # 如果邻居数量不足，用自身填充
        if len(neighbors) < num_neighbors:
            neighbors = np.concatenate([neighbors, [i] * (num_neighbors - len(neighbors))])
        else:
            # 如果邻居数量过多，随机选择
            neighbors = np.random.choice(neighbors, num_neighbors, replace=False)
        
        neighbor_obs[i] = obs[neighbors]
    
    return neighbor_obs

def compute_traffic_metrics(state, info):
    """
    计算交通指标
    
    Args:
        state: 当前状态
        info: 环境信息
    
    Returns:
        metrics: 交通指标字典
    """
    metrics = {}
    
    # 计算队列长度
    if 'queue_length' in info:
        metrics['total_queue_length'] = np.sum(info['queue_length'])
        metrics['avg_queue_length'] = np.mean(info['queue_length'])
    
    # 计算等待时间
    if 'waiting_time' in info:
        metrics['total_waiting_time'] = np.sum(info['waiting_time'])
        metrics['avg_waiting_time'] = np.mean(info['waiting_time'])
    
    # 计算吞吐量
    if 'throughput' in info:
        metrics['total_throughput'] = np.sum(info['throughput'])
    
    # 计算平均速度
    if 'avg_speed' in info:
        metrics['avg_speed'] = np.mean(info['avg_speed'])
    
    # 计算压力（基于队列长度和等待时间）
    if 'queue_length' in info and 'waiting_time' in info:
        pressure = info['queue_length'] * info['waiting_time']
        metrics['total_pressure'] = np.sum(pressure)
        metrics['avg_pressure'] = np.mean(pressure)
    
    return metrics

def log_training_info(episode, total_reward, metrics, entropy, config):
    """
    记录训练信息
    
    Args:
        episode: 当前回合数
        total_reward: 总奖励
        metrics: 交通指标
        entropy: 动作熵
        config: 配置参数
    """
    if episode % config.get('log_interval', 10) == 0:
        print(f"Episode {episode}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  High-level Entropy: {entropy[0]:.4f}")
        print(f"  Low-level Entropy: {entropy[1]:.4f}")
        
        for key, value in metrics.items():
            print(f"  {key}: {value:.2f}")
        print("-" * 50) 