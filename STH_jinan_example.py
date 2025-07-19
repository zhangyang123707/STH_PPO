#!/usr/bin/env python3
"""
STH模型使用济南3x4数据集示例
专门展示如何在STH模型中使用济南数据集
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
from STH_dataset_config import get_dataset_info, check_dataset_files

class MockTrafficEnvironment:
    """模拟交通环境（适配济南3x4数据集）"""
    def __init__(self, num_agents=12, obs_dim=16, action_dim=4):
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

def check_jinan_dataset():
    """检查济南数据集状态"""
    print("=" * 60)
    print("济南3x4数据集状态检查")
    print("=" * 60)
    
    dataset_name = 'jinan_3_4'
    
    # 检查数据集信息
    try:
        info = get_dataset_info(dataset_name)
        print(f"数据集名称: {info['name']}")
        print(f"城市: {info['city']}")
        print(f"网格大小: {info['grid_size']}")
        print(f"智能体数量: {info['num_agents']}")
        print(f"观测维度: {info['obs_dim']}")
        print(f"动作维度: {info['action_dim']}")
        print(f"交通文件: {info['traffic_file']}")
        print(f"路网文件: {info['roadnet_file']}")
        print(f"数据路径: {info['data_path']}")
        print(f"文件存在: {'✓' if info['files_exist'] else '✗'}")
        
        return info['files_exist']
    except Exception as e:
        print(f"获取数据集信息失败: {e}")
        return False

def example_jinan_config():
    """展示济南数据集的配置"""
    print("\n" + "=" * 60)
    print("济南3x4数据集配置")
    print("=" * 60)
    
    dataset_name = 'jinan_3_4'
    
    # 获取不同配置下的济南数据集设置
    configs = ['small', 'default', 'large']
    
    for config_name in configs:
        config = get_config(config_name, dataset_name)
        print(f"\n{config_name.upper()} 配置:")
        print(f"  智能体数量: {config['num_agents']}")
        print(f"  观测维度: {config['obs_dim']}")
        print(f"  动作维度: {config['action_dim']}")
        print(f"  嵌入维度: {config['embed_dim']}")
        print(f"  注意力头数: {config['n_heads']}")
        print(f"  数据集名称: {config.get('dataset_name', '未指定')}")
        print(f"  交通文件: {config.get('traffic_file', '未指定')}")
        print(f"  路网文件: {config.get('roadnet_file', '未指定')}")

def example_jinan_agent():
    """使用济南数据集创建智能体"""
    print("\n" + "=" * 60)
    print("使用济南3x4数据集创建STH智能体")
    print("=" * 60)
    
    dataset_name = 'jinan_3_4'
    
    try:
        # 获取数据集配置
        config = get_config('default', dataset_name)
        
        # 创建模拟环境
        env = MockTrafficEnvironment(
            num_agents=config['num_agents'],
            obs_dim=config['obs_dim'],
            action_dim=config['action_dim']
        )
        
        # 创建智能体
        agent = STHAgent(env, config_name='default', dataset_name=dataset_name)
        
        print(f"✓ 智能体创建成功")
        print(f"  数据集: {agent.dataset_name}")
        print(f"  智能体数量: {agent.config['num_agents']}")
        print(f"  观测维度: {agent.config['obs_dim']}")
        print(f"  动作维度: {agent.config['action_dim']}")
        
        if agent.dataset_info:
            print(f"  数据集信息: {agent.dataset_info['name']}")
            print(f"  交通文件: {agent.dataset_info['traffic_file']}")
            print(f"  路网文件: {agent.dataset_info['roadnet_file']}")
        
        return agent
    except Exception as e:
        print(f"✗ 创建智能体失败: {e}")
        return None

def example_jinan_training():
    """使用济南数据集进行训练"""
    print("\n" + "=" * 60)
    print("使用济南3x4数据集进行训练")
    print("=" * 60)
    
    dataset_name = 'jinan_3_4'
    
    try:
        # 获取配置
        config = get_config('fast', dataset_name)
        
        # 创建环境
        env = MockTrafficEnvironment(
            num_agents=config['num_agents'],
            obs_dim=config['obs_dim'],
            action_dim=config['action_dim']
        )
        
        # 创建智能体
        agent = STHAgent(env, config_name='fast', dataset_name=dataset_name)
        
        print("开始训练...")
        print(f"使用配置: {config['num_agents']}个智能体, {config['obs_dim']}维观测, {config['action_dim']}维动作")
        
        # 运行几个训练回合
        for episode in range(5):
            episode_info = agent.run_episode(training=True)
            print(f"  回合 {episode + 1}: 总奖励 = {episode_info['total_reward']:.2f}, 步数 = {episode_info['step_count']}")
        
        # 评估
        print("\n开始评估...")
        avg_reward, avg_metrics = agent.evaluate(num_episodes=2)
        print(f"  平均奖励: {avg_reward:.2f}")
        
        # 保存模型
        print("\n保存模型...")
        agent.save_model()
        
        print("✓ 训练完成")
        return agent
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        return None

def example_jinan_model_inference():
    """使用济南数据集进行模型推理"""
    print("\n" + "=" * 60)
    print("使用济南3x4数据集进行模型推理")
    print("=" * 60)
    
    dataset_name = 'jinan_3_4'
    
    try:
        # 获取配置
        config = get_config('default', dataset_name)
        
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
        
        print(f"输入数据形状:")
        print(f"  观测: {obs.shape}")
        print(f"  邻居观测: {neighbor_obs.shape}")
        print(f"  历史观测: {history_obs.shape}")
        
        # 进行推理
        with torch.no_grad():
            high_level_actions, low_level_actions, high_level_log_probs, low_level_log_probs, value = \
                ppo.select_action(obs, neighbor_obs, history_obs, training=False)
        
        print(f"\n输出数据形状:")
        print(f"  高层动作: {high_level_actions.shape}")
        print(f"  低层动作: {low_level_actions.shape}")
        print(f"  价值: {value.shape}")
        
        print(f"\n选择的动作:")
        print(f"  高层动作: {high_level_actions[0]}")
        print(f"  低层动作: {low_level_actions[0]}")
        print(f"  价值: {value[0, :, 0]}")
        
        print("✓ 推理完成")
        return model
    except Exception as e:
        print(f"✗ 推理失败: {e}")
        return None

def main():
    """主函数"""
    print("STH模型使用济南3x4数据集示例")
    print("This demonstrates how to use STH model with Jinan 3x4 dataset")
    
    try:
        # 1. 检查数据集状态
        dataset_available = check_jinan_dataset()
        
        if not dataset_available:
            print("\n⚠️  警告: 济南数据集文件不存在，将使用模拟数据")
            print("请确保以下文件存在:")
            print("  - data/Jinan/3_4/anon_3_4_jinan_real.json")
            print("  - data/Jinan/3_4/roadnet_3_4.json")
        
        # 2. 展示配置
        example_jinan_config()
        
        # 3. 创建智能体
        agent = example_jinan_agent()
        
        # 4. 模型推理
        model = example_jinan_model_inference()
        
        # 5. 训练示例
        if agent:
            trained_agent = example_jinan_training()
        
        print("\n" + "=" * 60)
        print("🎉 济南数据集示例完成!")
        print("=" * 60)
        
        print("\n使用说明:")
        print("1. 当前使用模拟环境，如需真实环境请替换MockTrafficEnvironment")
        print("2. 济南3x4数据集包含12个智能体，3x4网格布局")
        print("3. 数据集文件: anon_3_4_jinan_real.json, roadnet_3_4.json")
        print("4. 配置会自动适配数据集参数")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 