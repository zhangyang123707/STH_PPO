import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class ExperienceBuffer:
    """经验缓冲区，用于存储交互数据"""
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
    
    def push(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()

class HierarchicalPPO:
    """
    分层PPO算法实现：高层策略（如阶段选择）、低层策略（具体动作），支持多智能体。
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        
        # 经验缓冲区
        self.experience_buffer = ExperienceBuffer(max_size=config.get('buffer_size', 10000))
        
        # 训练参数
        self.ppo_clip = config['ppo_clip']
        self.ppo_epochs = config['ppo_epochs']
        self.gamma = config['gamma']
        self.lam = config['lam']
        self.batch_size = config['batch_size']
        
        # 价值损失系数
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        
        # 分层策略权重
        self.high_level_weight = config.get('high_level_weight', 1.0)
        self.low_level_weight = config.get('low_level_weight', 1.0)
        
    def select_action(self, obs, neighbor_obs, history_obs, training=True):
        """
        根据当前观测选择高层和低层动作。
        
        Args:
            obs: 当前观测 [batch, num_agents, obs_dim]
            neighbor_obs: 邻居观测 [batch, num_agents, num_neighbors, obs_dim]
            history_obs: 历史观测 [batch, num_agents, history_len, obs_dim]
            training: 是否处于训练模式
            
        Returns:
            high_level_actions: 高层动作 [batch, num_agents]
            low_level_actions: 低层动作 [batch, num_agents]
            high_level_log_probs: 高层动作对数概率 [batch, num_agents]
            low_level_log_probs: 低层动作对数概率 [batch, num_agents]
            value: 状态价值 [batch, num_agents, 1]
        """
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            neighbor_obs = torch.FloatTensor(neighbor_obs).to(self.device)
            history_obs = torch.FloatTensor(history_obs).to(self.device)
            
            # 获取动作概率分布
            high_level_probs, low_level_probs, value = self.model.get_action_probs(
                obs, neighbor_obs, history_obs
            )
            
            # 采样动作
            if training:
                # 训练时使用概率采样
                high_level_dist = torch.distributions.Categorical(high_level_probs)
                low_level_dist = torch.distributions.Categorical(low_level_probs)
                
                high_level_actions = high_level_dist.sample()
                low_level_actions = low_level_dist.sample()
                
                high_level_log_probs = high_level_dist.log_prob(high_level_actions)
                low_level_log_probs = low_level_dist.log_prob(low_level_actions)
            else:
                # 推理时使用贪婪策略
                high_level_actions = torch.argmax(high_level_probs, dim=-1)
                low_level_actions = torch.argmax(low_level_probs, dim=-1)
                
                high_level_log_probs = torch.log(high_level_probs + 1e-8)
                low_level_log_probs = torch.log(low_level_probs + 1e-8)
                
                # 收集对应动作的对数概率
                batch_size, num_agents = obs.size(0), obs.size(1)
                high_level_log_probs = torch.gather(
                    high_level_log_probs.view(-1, 4), 1,
                    high_level_actions.view(-1, 1)
                ).view(batch_size, num_agents)
                
                low_level_log_probs = torch.gather(
                    low_level_log_probs.view(-1, low_level_probs.size(-1)), 1,
                    low_level_actions.view(-1, 1)
                ).view(batch_size, num_agents)
            
            return (
                high_level_actions.cpu().numpy(),
                low_level_actions.cpu().numpy(),
                high_level_log_probs.cpu().numpy(),
                low_level_log_probs.cpu().numpy(),
                value.cpu().numpy()
            )
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        计算广义优势估计（GAE）
        
        Args:
            rewards: 奖励序列 [T, batch, num_agents]
            values: 价值序列 [T, batch, num_agents]
            dones: 终止标志 [T, batch, num_agents]
            next_value: 下一个状态的价值 [batch, num_agents]
            
        Returns:
            advantages: 优势函数 [T, batch, num_agents]
            returns: 回报 [T, batch, num_agents]
        """
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update(self, batch):
        """
        使用采样数据进行PPO更新。
        
        Args:
            batch: 包含以下字段的字典：
                - obs: 观测序列 [T, batch, num_agents, obs_dim]
                - neighbor_obs: 邻居观测序列 [T, batch, num_agents, num_neighbors, obs_dim]
                - history_obs: 历史观测序列 [T, batch, num_agents, history_len, obs_dim]
                - high_level_actions: 高层动作序列 [T, batch, num_agents]
                - low_level_actions: 低层动作序列 [T, batch, num_agents]
                - high_level_log_probs: 高层动作对数概率 [T, batch, num_agents]
                - low_level_log_probs: 低层动作对数概率 [T, batch, num_agents]
                - rewards: 奖励序列 [T, batch, num_agents]
                - values: 价值序列 [T, batch, num_agents]
                - dones: 终止标志 [T, batch, num_agents]
                - advantages: 优势函数 [T, batch, num_agents]
                - returns: 回报 [T, batch, num_agents]
        """
        # 将数据转换为张量
        obs = torch.FloatTensor(batch['obs']).to(self.device)
        neighbor_obs = torch.FloatTensor(batch['neighbor_obs']).to(self.device)
        history_obs = torch.FloatTensor(batch['history_obs']).to(self.device)
        high_level_actions = torch.LongTensor(batch['high_level_actions']).to(self.device)
        low_level_actions = torch.LongTensor(batch['low_level_actions']).to(self.device)
        old_high_level_log_probs = torch.FloatTensor(batch['high_level_log_probs']).to(self.device)
        old_low_level_log_probs = torch.FloatTensor(batch['low_level_log_probs']).to(self.device)
        advantages = torch.FloatTensor(batch['advantages']).to(self.device)
        returns = torch.FloatTensor(batch['returns']).to(self.device)
        
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 重塑数据用于批量处理
        T, batch_size, num_agents = obs.size(0), obs.size(1), obs.size(2)
        obs_flat = obs.view(-1, obs.size(-1))
        neighbor_obs_flat = neighbor_obs.view(-1, neighbor_obs.size(-2), neighbor_obs.size(-1))
        history_obs_flat = history_obs.view(-1, history_obs.size(-2), history_obs.size(-1))
        high_level_actions_flat = high_level_actions.view(-1)
        low_level_actions_flat = low_level_actions.view(-1)
        old_high_level_log_probs_flat = old_high_level_log_probs.view(-1)
        old_low_level_log_probs_flat = old_low_level_log_probs.view(-1)
        advantages_flat = advantages.view(-1)
        returns_flat = returns.view(-1)
        
        # PPO更新
        for epoch in range(self.ppo_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(obs_flat))
            
            for start_idx in range(0, len(obs_flat), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(obs_flat))
                batch_indices = indices[start_idx:end_idx]
                
                # 获取当前批次数据
                obs_batch = obs_flat[batch_indices]
                neighbor_obs_batch = neighbor_obs_flat[batch_indices]
                history_obs_batch = history_obs_flat[batch_indices]
                high_level_actions_batch = high_level_actions_flat[batch_indices]
                low_level_actions_batch = low_level_actions_flat[batch_indices]
                old_high_level_log_probs_batch = old_high_level_log_probs_flat[batch_indices]
                old_low_level_log_probs_batch = old_low_level_log_probs_flat[batch_indices]
                advantages_batch = advantages_flat[batch_indices]
                returns_batch = returns_flat[batch_indices]
                
                # 重塑为正确的维度
                obs_batch = obs_batch.view(-1, num_agents, obs.size(-1))
                neighbor_obs_batch = neighbor_obs_batch.view(-1, num_agents, neighbor_obs.size(-2), neighbor_obs.size(-1))
                history_obs_batch = history_obs_batch.view(-1, num_agents, history_obs.size(-2), history_obs.size(-1))
                high_level_actions_batch = high_level_actions_batch.view(-1, num_agents)
                low_level_actions_batch = low_level_actions_batch.view(-1, num_agents)
                old_high_level_log_probs_batch = old_high_level_log_probs_batch.view(-1, num_agents)
                old_low_level_log_probs_batch = old_low_level_log_probs_batch.view(-1, num_agents)
                advantages_batch = advantages_batch.view(-1, num_agents)
                returns_batch = returns_batch.view(-1, num_agents)
                
                # 前向传播
                high_level_log_probs, low_level_log_probs, values = self.model.get_action_log_probs(
                    obs_batch, neighbor_obs_batch, history_obs_batch,
                    high_level_actions_batch, low_level_actions_batch
                )
                
                # 计算比率
                high_level_ratio = torch.exp(high_level_log_probs - old_high_level_log_probs_batch)
                low_level_ratio = torch.exp(low_level_log_probs - old_low_level_log_probs_batch)
                
                # 计算裁剪后的目标
                high_level_clipped_ratio = torch.clamp(high_level_ratio, 1 - self.ppo_clip, 1 + self.ppo_clip)
                low_level_clipped_ratio = torch.clamp(low_level_ratio, 1 - self.ppo_clip, 1 + self.ppo_clip)
                
                # 计算策略损失
                high_level_policy_loss = -torch.min(
                    high_level_ratio * advantages_batch,
                    high_level_clipped_ratio * advantages_batch
                ).mean()
                
                low_level_policy_loss = -torch.min(
                    low_level_ratio * advantages_batch,
                    low_level_clipped_ratio * advantages_batch
                ).mean()
                
                # 计算价值损失
                value_loss = F.mse_loss(values.squeeze(-1), returns_batch)
                
                # 计算熵损失（用于鼓励探索）
                high_level_probs, low_level_probs, _ = self.model.get_action_probs(
                    obs_batch, neighbor_obs_batch, history_obs_batch
                )
                high_level_entropy = -(high_level_probs * torch.log(high_level_probs + 1e-8)).sum(dim=-1).mean()
                low_level_entropy = -(low_level_probs * torch.log(low_level_probs + 1e-8)).sum(dim=-1).mean()
                entropy_loss = -(high_level_entropy + low_level_entropy)
                
                # 总损失
                total_loss = (
                    self.high_level_weight * high_level_policy_loss +
                    self.low_level_weight * low_level_policy_loss +
                    self.value_loss_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def get_action_entropy(self, obs, neighbor_obs, history_obs):
        """计算动作分布的熵，用于监控探索程度"""
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            neighbor_obs = torch.FloatTensor(neighbor_obs).to(self.device)
            history_obs = torch.FloatTensor(history_obs).to(self.device)
            
            high_level_probs, low_level_probs, _ = self.model.get_action_probs(
                obs, neighbor_obs, history_obs
            )
            
            high_level_entropy = -(high_level_probs * torch.log(high_level_probs + 1e-8)).sum(dim=-1).mean()
            low_level_entropy = -(low_level_probs * torch.log(low_level_probs + 1e-8)).sum(dim=-1).mean()
            
            return high_level_entropy.item(), low_level_entropy.item() 