#!/usr/bin/env python3
"""
STH数据集使用示例
展示如何在STH模型中使用特定的数据集
"""

import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from STH_dataset_config import print_dataset_summary, get_dataset_info, list_available_datasets
from STH_config import get_config
from STH_agent import STHAgent

class MockTrafficEnvironment:
    """模拟交通环境"""
    def __init__(self, num_agents=4, obs_dim=16, action_dim=4):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.step_count = 0
        self.max_steps = 100
        
    def reset(self):
        self.step_count = 0
        return np.random.randn(self.num_agents, self.obs_dim)
    
    def step(self, actions):
        self.step_count += 1
        obs = np.random.randn(self.num_agents, self.obs_dim)
        reward = -np.abs(actions - 2)
        info = {
            'queue_length': np.random.randn(self.num_agents),
            'waiting_time': np.random.randn(self.num_agents),
            'throughput': np.random.randn(self.num_agents),
            'pressure': np.random.randn(self.num_agents),
            'avg_speed': np.random.randn(self.num_agents),
            'phase_change': np.random.randint(0, 2, self.num_agents),
            'neighbor_coordination': np.random.randn()
        }
        done = np.array([self.step_count >= self.max_steps] * self.num_agents)
        return obs, reward, done, info

def example_list_datasets():
    """列出所有可用数据集"""
    print("=" * 60)
    print("Available Datasets")
    print("=" * 60)
    
    datasets = list_available_datasets()
    for i, dataset_name in enumerate(datasets, 1):
        print(f"{i}. {dataset_name}")
    
    print(f"\nTotal: {len(datasets)} datasets available")
    return datasets

def example_dataset_info(dataset_name):
    """显示数据集详细信息"""
    print(f"\n" + "=" * 60)
    print(f"Dataset Info: {dataset_name}")
    print("=" * 60)
    
    try:
        info = get_dataset_info(dataset_name)
        
        print(f"Name: {info['name']}")
        print(f"City: {info['city']}")
        print(f"Grid Size: {info['grid_size']}")
        print(f"Number of Agents: {info['num_agents']}")
        print(f"Observation Dimension: {info['obs_dim']}")
        print(f"Action Dimension: {info['action_dim']}")
        print(f"Traffic File: {info['traffic_file']}")
        print(f"Roadnet File: {info['roadnet_file']}")
        print(f"Data Path: {info['data_path']}")
        print(f"Description: {info['description']}")
        print(f"Files Exist: {'✓' if info['files_exist'] else '✗'}")
        
        return info
    except Exception as e:
        print(f"Error getting dataset info: {e}")
        return None

def example_config_with_dataset(dataset_name):
    """展示使用特定数据集的配置"""
    print(f"\n" + "=" * 60)
    print(f"Configuration with Dataset: {dataset_name}")
    print("=" * 60)
    
    try:
        # 获取数据集配置
        config = get_config('default', dataset_name)
        
        print("Key configuration parameters:")
        print(f"  num_agents: {config['num_agents']}")
        print(f"  obs_dim: {config['obs_dim']}")
        print(f"  action_dim: {config['action_dim']}")
        print(f"  dataset_name: {config.get('dataset_name', 'None')}")
        print(f"  traffic_file: {config.get('traffic_file', 'None')}")
        print(f"  roadnet_file: {config.get('roadnet_file', 'None')}")
        print(f"  data_path: {config.get('data_path', 'None')}")
        
        return config
    except Exception as e:
        print(f"Error getting config: {e}")
        return None

def example_agent_with_dataset(dataset_name):
    """展示使用特定数据集创建智能体"""
    print(f"\n" + "=" * 60)
    print(f"Creating Agent with Dataset: {dataset_name}")
    print("=" * 60)
    
    try:
        # 获取数据集信息
        info = get_dataset_info(dataset_name)
        if not info:
            print("Could not get dataset info")
            return None
        
        # 创建模拟环境
        env = MockTrafficEnvironment(
            num_agents=info['num_agents'],
            obs_dim=info['obs_dim'],
            action_dim=info['action_dim']
        )
        
        # 创建智能体
        agent = STHAgent(env, config_name='default', dataset_name=dataset_name)
        
        print(f"✓ Agent created successfully")
        print(f"  Dataset: {agent.dataset_name}")
        print(f"  Number of agents: {agent.config['num_agents']}")
        print(f"  Observation dimension: {agent.config['obs_dim']}")
        print(f"  Action dimension: {agent.config['action_dim']}")
        
        return agent
    except Exception as e:
        print(f"Error creating agent: {e}")
        return None

def example_training_with_dataset(dataset_name):
    """展示使用特定数据集进行训练"""
    print(f"\n" + "=" * 60)
    print(f"Training with Dataset: {dataset_name}")
    print("=" * 60)
    
    try:
        # 获取数据集信息
        info = get_dataset_info(dataset_name)
        if not info:
            print("Could not get dataset info")
            return None
        
        # 创建模拟环境
        env = MockTrafficEnvironment(
            num_agents=info['num_agents'],
            obs_dim=info['obs_dim'],
            action_dim=info['action_dim']
        )
        
        # 创建智能体
        agent = STHAgent(env, config_name='fast', dataset_name=dataset_name)
        
        print("Running a few training episodes...")
        
        # 运行几个训练回合
        for episode in range(3):
            episode_info = agent.run_episode(training=True)
            print(f"  Episode {episode + 1}: Reward = {episode_info['total_reward']:.2f}")
        
        print("✓ Training example completed")
        return agent
    except Exception as e:
        print(f"Error during training: {e}")
        return None

def example_dataset_comparison():
    """比较不同数据集"""
    print("\n" + "=" * 60)
    print("Dataset Comparison")
    print("=" * 60)
    
    # 选择几个不同的数据集进行比较
    datasets_to_compare = ['hangzhou_4_4', 'jinan_3_4', 'newyork_16_3']
    
    comparison_data = []
    
    for dataset_name in datasets_to_compare:
        try:
            info = get_dataset_info(dataset_name)
            if info:
                comparison_data.append({
                    'name': dataset_name,
                    'city': info['city'],
                    'grid_size': info['grid_size'],
                    'num_agents': info['num_agents'],
                    'files_exist': info['files_exist']
                })
        except Exception as e:
            print(f"Error getting info for {dataset_name}: {e}")
    
    # 打印比较表格
    print(f"{'Dataset':<20} {'City':<10} {'Grid':<10} {'Agents':<8} {'Files':<6}")
    print("-" * 60)
    
    for data in comparison_data:
        status = "✓" if data['files_exist'] else "✗"
        print(f"{data['name']:<20} {data['city']:<10} {data['grid_size']:<10} {data['num_agents']:<8} {status:<6}")

def main():
    """主函数"""
    print("STH Dataset Usage Examples")
    print("This demonstrates how to use specific datasets with STH model")
    
    try:
        # 1. 列出所有数据集
        datasets = example_list_datasets()
        
        # 2. 显示数据集摘要
        print_dataset_summary()
        
        # 3. 数据集比较
        example_dataset_comparison()
        
        # 4. 使用特定数据集
        if datasets:
            # 使用第一个可用数据集作为示例
            example_dataset = datasets[0]
            
            # 显示数据集信息
            example_dataset_info(example_dataset)
            
            # 显示配置
            example_config_with_dataset(example_dataset)
            
            # 创建智能体
            agent = example_agent_with_dataset(example_dataset)
            
            # 训练示例
            if agent:
                example_training_with_dataset(example_dataset)
        
        print("\n" + "=" * 60)
        print("🎉 Dataset examples completed successfully!")
        print("=" * 60)
        
        print("\nUsage tips:")
        print("1. Use 'STH_dataset_config.print_dataset_summary()' to see all datasets")
        print("2. Use 'get_config(config_name, dataset_name)' to get dataset-specific config")
        print("3. Use 'STHAgent(env, config_name, dataset_name)' to create agent with dataset")
        print("4. Check dataset files exist before using: 'check_dataset_files(dataset_name)'")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 