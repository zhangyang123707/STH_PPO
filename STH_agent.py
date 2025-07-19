import numpy as np
import torch
import os
import time
from collections import deque
import logging

from STH_model import STHANModel
from STH_ppo import HierarchicalPPO
from STH_utils import (
    reward_shaping, process_features, HistoryBuffer, 
    create_adjacency_matrix, get_neighbor_obs, 
    compute_traffic_metrics, log_training_info
)
from STH_config import get_config, validate_config

class STHAgent:
    """
    STHAgent: 时空混合注意力+分层PPO智能体主流程
    """
    def __init__(self, env, config_name='default', dataset_name=None):
        self.env = env
        self.config = get_config(config_name, dataset_name)
        
        # 如果指定了数据集，记录数据集信息
        if dataset_name:
            self.dataset_name = dataset_name
            try:
                from STH_dataset_config import get_dataset_info
                self.dataset_info = get_dataset_info(dataset_name)
                self.logger.info(f"Using dataset: {self.dataset_info['name']}")
            except ImportError:
                self.dataset_info = None
        else:
            self.dataset_name = None
            self.dataset_info = None
        
        # 验证配置
        is_valid, errors = validate_config(self.config)
        if not is_valid:
            raise ValueError(f"Invalid config: {errors}")
        
        # 设置随机种子
        if self.config.get('seed') is not None:
            np.random.seed(self.config['seed'])
            torch.manual_seed(self.config['seed'])
        
        # 创建邻接矩阵
        self.adjacency_matrix = create_adjacency_matrix(
            self.config['num_agents'], 
            self.config['topology']
        )
        
        # 初始化模型
        self.model = STHANModel(
            obs_dim=self.config['obs_dim'],
            action_dim=self.config['action_dim'],
            num_agents=self.config['num_agents'],
            history_len=self.config['history_len'],
            embed_dim=self.config['embed_dim'],
            n_heads=self.config['n_heads'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        )
        
        # 初始化PPO算法
        self.ppo = HierarchicalPPO(self.model, self.config)
        
        # 初始化历史缓冲区
        self.history_buffer = HistoryBuffer(
            max_len=self.config['history_len'],
            num_agents=self.config['num_agents'],
            obs_dim=self.config['obs_dim']
        )
        
        # 训练状态
        self.episode_count = 0
        self.total_steps = 0
        self.best_reward = float('-inf')
        
        # 创建保存目录
        os.makedirs(os.path.dirname(self.config['model_save_path']), exist_ok=True)
        os.makedirs(os.path.dirname(self.config['log_save_path']), exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # 训练历史
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'high_level_entropies': [],
            'low_level_entropies': [],
            'traffic_metrics': []
        }
        
        self.logger.info(f"STHAgent initialized with config: {config_name}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_logging(self):
        """设置日志记录"""
        self.logger = logging.getLogger('STHAgent')
        self.logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        fh = logging.FileHandler(self.config['log_save_path'])
        fh.setLevel(logging.INFO)
        
        # 创建控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def reset_episode(self):
        """重置回合状态"""
        # 重置环境
        obs = self.env.reset()
        
        # 重置历史缓冲区
        self.history_buffer.clear()
        
        # 初始化历史状态
        for _ in range(self.config['history_len']):
            self.history_buffer.push(obs)
        
        return obs
    
    def get_processed_observations(self, obs):
        """
        获取处理后的观测数据
        
        Args:
            obs: 原始观测 [num_agents, obs_dim]
            
        Returns:
            processed_obs: 处理后的当前观测
            neighbor_obs: 邻居观测
            history_obs: 历史观测
        """
        # 获取历史观测
        history_obs = self.history_buffer.get_history(self.config['history_len'])
        
        # 获取邻居观测
        neighbor_obs = get_neighbor_obs(
            obs, 
            self.adjacency_matrix, 
            self.config['num_neighbors']
        )
        
        # 处理特征
        processed_obs, processed_neighbor_obs, processed_history_obs = process_features(
            obs, neighbor_obs, history_obs, self.config
        )
        
        return processed_obs, processed_neighbor_obs, processed_history_obs
    
    def run_episode(self, training=True):
        """
        运行一轮交互，收集数据，训练模型
        
        Args:
            training: 是否处于训练模式
            
        Returns:
            episode_info: 回合信息字典
        """
        # 重置环境
        obs = self.reset_episode()
        
        # 初始化回合数据
        episode_data = {
            'obs': [],
            'neighbor_obs': [],
            'history_obs': [],
            'high_level_actions': [],
            'low_level_actions': [],
            'high_level_log_probs': [],
            'low_level_log_probs': [],
            'rewards': [],
            'values': [],
            'dones': [],
            'traffic_metrics': []
        }
        
        total_reward = 0
        step_count = 0
        
        while step_count < self.config['max_steps_per_episode']:
            # 获取处理后的观测
            processed_obs, neighbor_obs, history_obs = self.get_processed_observations(obs)
            
            # 选择动作
            high_level_actions, low_level_actions, high_level_log_probs, low_level_log_probs, values = \
                self.ppo.select_action(
                    processed_obs[np.newaxis, ...],  # 添加batch维度
                    neighbor_obs[np.newaxis, ...],
                    history_obs[np.newaxis, ...],
                    training=training
                )
            
            # 移除batch维度
            high_level_actions = high_level_actions[0]
            low_level_actions = low_level_actions[0]
            high_level_log_probs = high_level_log_probs[0]
            low_level_log_probs = low_level_log_probs[0]
            values = values[0]
            
            # 执行动作
            next_obs, raw_reward, done, info = self.env.step(low_level_actions)
            
            # 奖励塑形
            shaped_reward = reward_shaping(raw_reward, obs, next_obs, info, self.config)
            
            # 计算交通指标
            metrics = compute_traffic_metrics(obs, info)
            
            # 更新历史缓冲区
            self.history_buffer.push(next_obs)
            
            # 存储数据
            episode_data['obs'].append(obs.copy())
            episode_data['neighbor_obs'].append(neighbor_obs.copy())
            episode_data['history_obs'].append(history_obs.copy())
            episode_data['high_level_actions'].append(high_level_actions.copy())
            episode_data['low_level_actions'].append(low_level_actions.copy())
            episode_data['high_level_log_probs'].append(high_level_log_probs.copy())
            episode_data['low_level_log_probs'].append(low_level_log_probs.copy())
            episode_data['rewards'].append(shaped_reward.copy())
            episode_data['values'].append(values.copy())
            episode_data['dones'].append(done.copy())
            episode_data['traffic_metrics'].append(metrics)
            
            total_reward += np.sum(shaped_reward)
            step_count += 1
            self.total_steps += 1
            
            obs = next_obs
            
            if np.all(done):
                break
        
        # 计算GAE
        if training and len(episode_data['rewards']) > 0:
            # 获取最终状态的价值
            final_obs, final_neighbor_obs, final_history_obs = self.get_processed_observations(obs)
            _, _, _, _, final_value = self.ppo.select_action(
                final_obs[np.newaxis, ...],
                final_neighbor_obs[np.newaxis, ...],
                final_history_obs[np.newaxis, ...],
                training=False
            )
            final_value = final_value[0]
            
            # 转换为张量
            rewards = torch.FloatTensor(episode_data['rewards'])
            values = torch.FloatTensor(episode_data['values'])
            dones = torch.FloatTensor(episode_data['dones'])
            
            # 计算GAE
            advantages, returns = self.ppo.compute_gae(
                rewards, values, dones, final_value
            )
            
            episode_data['advantages'] = advantages.numpy()
            episode_data['returns'] = returns.numpy()
        
        # 记录训练历史
        self.training_history['episode_rewards'].append(total_reward)
        self.training_history['episode_lengths'].append(step_count)
        
        # 计算平均熵
        if len(episode_data['high_level_log_probs']) > 0:
            high_level_entropy = -np.mean(episode_data['high_level_log_probs'])
            low_level_entropy = -np.mean(episode_data['low_level_log_probs'])
            self.training_history['high_level_entropies'].append(high_level_entropy)
            self.training_history['low_level_entropies'].append(low_level_entropy)
        
        # 计算平均交通指标
        if episode_data['traffic_metrics']:
            avg_metrics = {}
            for key in episode_data['traffic_metrics'][0].keys():
                avg_metrics[key] = np.mean([m[key] for m in episode_data['traffic_metrics']])
            self.training_history['traffic_metrics'].append(avg_metrics)
        
        episode_info = {
            'episode': self.episode_count,
            'total_reward': total_reward,
            'step_count': step_count,
            'avg_reward': total_reward / step_count if step_count > 0 else 0,
            'traffic_metrics': avg_metrics if episode_data['traffic_metrics'] else {},
            'data': episode_data if training else None
        }
        
        return episode_info
    
    def train(self):
        """训练智能体"""
        self.logger.info("Starting training...")
        
        for episode in range(self.config['max_episodes']):
            self.episode_count = episode
            
            # 运行回合
            episode_info = self.run_episode(training=True)
            
            # 训练模型
            if episode_info['data'] is not None:
                self.ppo.update(episode_info['data'])
            
            # 记录日志
            if episode % self.config['log_interval'] == 0:
                entropy = (
                    self.training_history['high_level_entropies'][-1] if self.training_history['high_level_entropies'] else 0,
                    self.training_history['low_level_entropies'][-1] if self.training_history['low_level_entropies'] else 0
                )
                log_training_info(
                    episode, 
                    episode_info['total_reward'], 
                    episode_info['traffic_metrics'], 
                    entropy, 
                    self.config
                )
            
            # 保存模型
            if episode % self.config['save_interval'] == 0:
                self.save_model()
            
            # 评估模型
            if episode % self.config['eval_interval'] == 0:
                self.evaluate()
            
            # 检查是否达到最佳性能
            if episode_info['total_reward'] > self.best_reward:
                self.best_reward = episode_info['total_reward']
                self.save_model('best')
        
        self.logger.info("Training completed!")
    
    def evaluate(self, num_episodes=5):
        """评估模型性能"""
        self.logger.info("Evaluating model...")
        
        eval_rewards = []
        eval_metrics = []
        
        for _ in range(num_episodes):
            episode_info = self.run_episode(training=False)
            eval_rewards.append(episode_info['total_reward'])
            eval_metrics.append(episode_info['traffic_metrics'])
        
        avg_reward = np.mean(eval_rewards)
        avg_metrics = {}
        if eval_metrics:
            for key in eval_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in eval_metrics])
        
        self.logger.info(f"Evaluation - Avg Reward: {avg_reward:.2f}")
        for key, value in avg_metrics.items():
            self.logger.info(f"  {key}: {value:.2f}")
        
        return avg_reward, avg_metrics
    
    def save_model(self, suffix=''):
        """保存模型"""
        if suffix:
            save_path = self.config['model_save_path'].replace('.pth', f'_{suffix}.pth')
        else:
            save_path = self.config['model_save_path']
        
        self.ppo.save_model(save_path)
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, suffix=''):
        """加载模型"""
        if suffix:
            load_path = self.config['model_save_path'].replace('.pth', f'_{suffix}.pth')
        else:
            load_path = self.config['model_save_path']
        
        if os.path.exists(load_path):
            self.ppo.load_model(load_path)
            self.logger.info(f"Model loaded from {load_path}")
        else:
            self.logger.warning(f"Model file not found: {load_path}")
    
    def get_action(self, obs):
        """
        获取动作（用于推理）
        
        Args:
            obs: 观测 [num_agents, obs_dim]
            
        Returns:
            actions: 动作 [num_agents]
        """
        processed_obs, neighbor_obs, history_obs = self.get_processed_observations(obs)
        
        _, low_level_actions, _, _, _ = self.ppo.select_action(
            processed_obs[np.newaxis, ...],
            neighbor_obs[np.newaxis, ...],
            history_obs[np.newaxis, ...],
            training=False
        )
        
        return low_level_actions[0]
    
    def get_training_stats(self):
        """获取训练统计信息"""
        if not self.training_history['episode_rewards']:
            return {}
        
        recent_rewards = self.training_history['episode_rewards'][-100:]  # 最近100回合
        
        stats = {
            'total_episodes': len(self.training_history['episode_rewards']),
            'total_steps': self.total_steps,
            'avg_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'max_reward': np.max(recent_rewards),
            'min_reward': np.min(recent_rewards),
            'best_reward': self.best_reward,
        }
        
        if self.training_history['high_level_entropies']:
            stats['avg_high_level_entropy'] = np.mean(self.training_history['high_level_entropies'][-100:])
            stats['avg_low_level_entropy'] = np.mean(self.training_history['low_level_entropies'][-100:])
        
        return stats 