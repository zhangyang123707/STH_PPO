#!/usr/bin/env python3
"""
STH模型使用示例
展示如何使用时空混合注意力网络+分层PPO进行交通信号控制
"""

import numpy as np
import torch
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from STH_model import STHANModel
from STH_ppo import HierarchicalPPO
from STH_agent import STHAgent
from STH_config import get_config

class MockTrafficEnvironment:
    """
    模拟交通环境，用于演示STH模型的使用
    在实际应用中，这里应该替换为真实的交通仿真环境（如SUMO）
    """
    def __init__(self, num_agents=4, obs_dim=16, action_dim=4):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.step_count = 0
        self.max_steps = 100
        
    def reset(self):
        """重置环境"""
        self.step_count = 0
        # 返回随机初始状态
        return np.random.randn(self.num_agents, self.obs_dim)
    
    def step(self, actions):
        """
        执行动作
        
        Args:
            actions: 动作数组 [num_agents]
            
        Returns:
            obs: 新的观测
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        self.step_count += 1
        
        # 模拟状态转移
        obs = np.random.randn(self.num_agents, self.obs_dim)
        
        # 模拟奖励（基于动作的简单奖励）
        reward = -np.abs(actions - 2)  # 偏好动作2
        
        # 模拟交通指标
        info = {
            'queue_length': np.random.randn(self.num_agents),
            'waiting_time': np.random.randn(self.num_agents),
            'throughput': np.random.randn(self.num_agents),
            'pressure': np.random.randn(self.num_agents),
            'avg_speed': np.random.randn(self.num_agents),
            'phase_change': np.random.randint(0, 2, self.num_agents),
            'neighbor_coordination': np.random.randn()
        }
        
        # 检查是否结束
        done = np.array([self.step_count >= self.max_steps] * self.num_agents)
        
        return obs, reward, done, info

def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("STH Model Basic Usage Example")
    print("=" * 60)
    
    # 1. 获取配置（使用济南3x4数据集）
    config = get_config('small', dataset_name='jinan_3_4')
    print(f"Using config: small network with Jinan 3x4 dataset ({config['num_agents']} agents)")
    
    # 2. 创建模拟环境
    env = MockTrafficEnvironment(
        num_agents=config['num_agents'],
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim']
    )
    
    # 3. 创建STH智能体（使用济南3x4数据集）
    agent = STHAgent(env, config_name='default', dataset_name='jinan_3_4')
    
    # 4. 运行几个回合进行演示
    print("\nRunning demonstration episodes...")
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        
        # 运行一个回合
        episode_info = agent.run_episode(training=True)
        
        print(f"  Total Reward: {episode_info['total_reward']:.2f}")
        print(f"  Steps: {episode_info['step_count']}")
        print(f"  Avg Reward: {episode_info['avg_reward']:.2f}")
        
        # 显示交通指标
        if episode_info['traffic_metrics']:
            print("  Traffic Metrics:")
            for key, value in episode_info['traffic_metrics'].items():
                print(f"    {key}: {value:.2f}")
    
    # 5. 获取训练统计
    stats = agent.get_training_stats()
    print(f"\nTraining Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    print("\n✓ Basic usage example completed!")

def example_model_inference():
    """模型推理示例"""
    print("\n" + "=" * 60)
    print("STH Model Inference Example")
    print("=" * 60)
    
    config = get_config('small', dataset_name='jinan_3_4')
    
    # 创建模型
    model = STHANModel(
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'],
        num_agents=config['num_agents'],
        history_len=config['history_len'],
        embed_dim=config['embed_dim'],
        n_heads=config['n_heads']
    )
    
    # 创建PPO
    ppo = HierarchicalPPO(model, config)
    
    # 准备输入数据
    batch_size = 1
    obs = np.random.randn(batch_size, config['num_agents'], config['obs_dim'])
    neighbor_obs = np.random.randn(batch_size, config['num_agents'], config['num_neighbors'], config['obs_dim'])
    history_obs = np.random.randn(batch_size, config['num_agents'], config['history_len'], config['obs_dim'])
    
    print(f"Input shapes:")
    print(f"  obs: {obs.shape}")
    print(f"  neighbor_obs: {neighbor_obs.shape}")
    print(f"  history_obs: {history_obs.shape}")
    
    # 进行推理
    with torch.no_grad():
        high_level_actions, low_level_actions, high_level_log_probs, low_level_log_probs, value = \
            ppo.select_action(obs, neighbor_obs, history_obs, training=False)
    
    print(f"\nOutput shapes:")
    print(f"  high_level_actions: {high_level_actions.shape}")
    print(f"  low_level_actions: {low_level_actions.shape}")
    print(f"  value: {value.shape}")
    
    print(f"\nSelected actions:")
    print(f"  High-level: {high_level_actions[0]}")
    print(f"  Low-level: {low_level_actions[0]}")
    print(f"  Value: {value[0, :, 0]}")
    
    print("\n✓ Model inference example completed!")

def example_custom_config():
    """自定义配置示例"""
    print("\n" + "=" * 60)
    print("STH Model Custom Configuration Example")
    print("=" * 60)
    
    # 获取默认配置（使用济南3x4数据集）
    config = get_config('default', dataset_name='jinan_3_4')
    
    # 自定义配置
    custom_config = config.copy()
    custom_config.update({
        'num_agents': 6,
        'obs_dim': 20,
        'action_dim': 6,
        'embed_dim': 256,
        'n_heads': 8,
        'num_layers': 3,
        'history_len': 8,
        'ppo_epochs': 15,
        'batch_size': 128,
        'lr': 5e-4,
        # 自定义奖励塑形参数
        'queue_length_weight': -0.15,
        'waiting_time_weight': -0.08,
        'throughput_weight': 0.25,
        'pressure_weight': -0.2,
        'coordination_weight': 0.15,
    })
    
    print("Custom configuration:")
    for key, value in custom_config.items():
        if key in ['queue_length_weight', 'waiting_time_weight', 'throughput_weight', 
                  'pressure_weight', 'coordination_weight']:
            print(f"  {key}: {value}")
    
    # 创建模型
    model = STHANModel(
        obs_dim=custom_config['obs_dim'],
        action_dim=custom_config['action_dim'],
        num_agents=custom_config['num_agents'],
        history_len=custom_config['history_len'],
        embed_dim=custom_config['embed_dim'],
        n_heads=custom_config['n_heads'],
        num_layers=custom_config['num_layers']
    )
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    print("\n✓ Custom configuration example completed!")

def example_training_workflow():
    """训练工作流示例"""
    print("\n" + "=" * 60)
    print("STH Model Training Workflow Example")
    print("=" * 60)
    
    # 创建环境（使用济南3x4数据集配置）
    config = get_config('fast', dataset_name='jinan_3_4')
    env = MockTrafficEnvironment(
        num_agents=config['num_agents'],
        obs_dim=config['obs_dim'], 
        action_dim=config['action_dim']
    )
    
    # 创建智能体（使用济南3x4数据集）
    agent = STHAgent(env, config_name='fast', dataset_name='jinan_3_4')
    
    print("Training workflow:")
    print("1. Initialize agent ✓")
    print("2. Run training episodes...")
    
    # 运行几个训练回合
    for episode in range(5):
        episode_info = agent.run_episode(training=True)
        
        if episode % 2 == 0:
            print(f"   Episode {episode + 1}: Reward = {episode_info['total_reward']:.2f}")
    
    print("3. Save model...")
    agent.save_model()
    
    print("4. Evaluate model...")
    avg_reward, avg_metrics = agent.evaluate(num_episodes=2)
    print(f"   Evaluation: Avg Reward = {avg_reward:.2f}")
    
    print("5. Load model...")
    agent.load_model()
    
    print("\n✓ Training workflow example completed!")

def main():
    """主函数"""
    print("STH (Spatio-Temporal Hybrid Attention Network) Model Examples")
    print("This demonstrates the usage of STH model for traffic signal control")
    
    try:
        # 运行各种示例
        example_basic_usage()
        example_model_inference()
        example_custom_config()
        example_training_workflow()
        
        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("=" * 60)
        
        print("\nNext steps:")
        print("1. Replace MockTrafficEnvironment with your actual traffic simulation environment")
        print("2. Adjust the configuration parameters based on your specific use case")
        print("3. Run the training with your environment")
        print("4. Monitor the training progress and adjust hyperparameters as needed")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 