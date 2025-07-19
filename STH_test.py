#!/usr/bin/env python3
"""
STH模型测试脚本
用于验证时空混合注意力网络和分层PPO算法的各个组件
"""

import numpy as np
import torch
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from STH_model import STHANModel
from STH_ppo import HierarchicalPPO
from STH_utils import (
    HistoryBuffer, create_adjacency_matrix, get_neighbor_obs,
    process_features, reward_shaping, compute_traffic_metrics
)
from STH_config import get_config

def test_model_forward():
    """测试模型前向传播"""
    print("Testing model forward pass...")
    
    config = get_config('small')
    
    # 创建模型
    model = STHANModel(
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'],
        num_agents=config['num_agents'],
        history_len=config['history_len'],
        embed_dim=config['embed_dim'],
        n_heads=config['n_heads']
    )
    
    # 创建测试数据
    batch_size = 2
    obs = torch.randn(batch_size, config['num_agents'], config['obs_dim'])
    neighbor_obs = torch.randn(batch_size, config['num_agents'], config['num_neighbors'], config['obs_dim'])
    history_obs = torch.randn(batch_size, config['num_agents'], config['history_len'], config['obs_dim'])
    
    # 前向传播
    high_level_logits, low_level_logits, value, spatial_attn, temporal_attn = model(
        obs, neighbor_obs, history_obs
    )
    
    # 检查输出形状
    assert high_level_logits.shape == (batch_size, config['num_agents'], 4), f"High level logits shape: {high_level_logits.shape}"
    assert low_level_logits.shape == (batch_size, config['num_agents'], config['action_dim']), f"Low level logits shape: {low_level_logits.shape}"
    assert value.shape == (batch_size, config['num_agents'], 1), f"Value shape: {value.shape}"
    
    print("✓ Model forward pass test passed!")
    return True

def test_ppo_action_selection():
    """测试PPO动作选择"""
    print("Testing PPO action selection...")
    
    config = get_config('small')
    
    # 创建模型和PPO
    model = STHANModel(
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'],
        num_agents=config['num_agents'],
        history_len=config['history_len'],
        embed_dim=config['embed_dim'],
        n_heads=config['n_heads']
    )
    
    ppo = HierarchicalPPO(model, config)
    
    # 创建测试数据
    batch_size = 1
    obs = np.random.randn(batch_size, config['num_agents'], config['obs_dim'])
    neighbor_obs = np.random.randn(batch_size, config['num_agents'], config['num_neighbors'], config['obs_dim'])
    history_obs = np.random.randn(batch_size, config['num_agents'], config['history_len'], config['obs_dim'])
    
    # 测试训练模式
    high_level_actions, low_level_actions, high_level_log_probs, low_level_log_probs, value = \
        ppo.select_action(obs, neighbor_obs, history_obs, training=True)
    
    # 检查输出形状
    assert high_level_actions.shape == (batch_size, config['num_agents']), f"High level actions shape: {high_level_actions.shape}"
    assert low_level_actions.shape == (batch_size, config['num_agents']), f"Low level actions shape: {low_level_actions.shape}"
    assert high_level_log_probs.shape == (batch_size, config['num_agents']), f"High level log probs shape: {high_level_log_probs.shape}"
    assert low_level_log_probs.shape == (batch_size, config['num_agents']), f"Low level log probs shape: {low_level_log_probs.shape}"
    assert value.shape == (batch_size, config['num_agents'], 1), f"Value shape: {value.shape}"
    
    # 测试推理模式
    high_level_actions, low_level_actions, high_level_log_probs, low_level_log_probs, value = \
        ppo.select_action(obs, neighbor_obs, history_obs, training=False)
    
    print("✓ PPO action selection test passed!")
    return True

def test_utils():
    """测试工具函数"""
    print("Testing utility functions...")
    
    config = get_config('small')
    
    # 测试历史缓冲区
    history_buffer = HistoryBuffer(
        max_len=config['history_len'],
        num_agents=config['num_agents'],
        obs_dim=config['obs_dim']
    )
    
    # 添加一些观测
    for i in range(5):
        obs = np.random.randn(config['num_agents'], config['obs_dim'])
        history_buffer.push(obs)
    
    history = history_buffer.get_history()
    assert history.shape == (config['num_agents'], config['history_len'], config['obs_dim']), f"History shape: {history.shape}"
    
    # 测试邻接矩阵创建
    adjacency_matrix = create_adjacency_matrix(config['num_agents'], 'grid')
    assert adjacency_matrix.shape == (config['num_agents'], config['num_agents']), f"Adjacency matrix shape: {adjacency_matrix.shape}"
    
    # 测试邻居观测获取
    obs = np.random.randn(config['num_agents'], config['obs_dim'])
    neighbor_obs = get_neighbor_obs(obs, adjacency_matrix, config['num_neighbors'])
    assert neighbor_obs.shape == (config['num_agents'], config['num_neighbors'], config['obs_dim']), f"Neighbor obs shape: {neighbor_obs.shape}"
    
    # 测试特征处理
    processed_obs, processed_neighbor_obs, processed_history_obs = process_features(
        obs, neighbor_obs, history, config
    )
    
    # 测试奖励塑形
    raw_reward = np.random.randn(config['num_agents'])
    state = obs
    next_state = obs
    info = {
        'queue_length': np.random.randn(config['num_agents']),
        'waiting_time': np.random.randn(config['num_agents']),
        'throughput': np.random.randn(config['num_agents']),
        'pressure': np.random.randn(config['num_agents'])
    }
    
    shaped_reward = reward_shaping(raw_reward, state, next_state, info, config)
    assert shaped_reward.shape == raw_reward.shape, f"Shaped reward shape: {shaped_reward.shape}"
    
    # 测试交通指标计算
    metrics = compute_traffic_metrics(state, info)
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    
    print("✓ Utility functions test passed!")
    return True

def test_gae_computation():
    """测试GAE计算"""
    print("Testing GAE computation...")
    
    config = get_config('small')
    ppo = HierarchicalPPO(None, config)  # 不需要模型来测试GAE
    
    # 创建测试数据
    T = 10
    batch_size = 2
    num_agents = config['num_agents']
    
    rewards = torch.randn(T, batch_size, num_agents)
    values = torch.randn(T, batch_size, num_agents)
    dones = torch.randint(0, 2, (T, batch_size, num_agents)).float()
    next_value = torch.randn(batch_size, num_agents)
    
    # 计算GAE
    advantages, returns = ppo.compute_gae(rewards, values, dones, next_value)
    
    # 检查输出形状
    assert advantages.shape == (T, batch_size, num_agents), f"Advantages shape: {advantages.shape}"
    assert returns.shape == (T, batch_size, num_agents), f"Returns shape: {returns.shape}"
    
    print("✓ GAE computation test passed!")
    return True

def test_config():
    """测试配置系统"""
    print("Testing configuration system...")
    
    # 测试默认配置
    config = get_config('default')
    assert 'obs_dim' in config, "Default config should contain obs_dim"
    assert 'action_dim' in config, "Default config should contain action_dim"
    
    # 测试小网络配置
    small_config = get_config('small')
    assert small_config['num_agents'] == 4, f"Small config num_agents: {small_config['num_agents']}"
    assert small_config['embed_dim'] == 64, f"Small config embed_dim: {small_config['embed_dim']}"
    
    # 测试大网络配置
    large_config = get_config('large')
    assert large_config['num_agents'] == 16, f"Large config num_agents: {large_config['num_agents']}"
    assert large_config['embed_dim'] == 256, f"Large config embed_dim: {large_config['embed_dim']}"
    
    print("✓ Configuration system test passed!")
    return True

def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("Running STH Model Tests")
    print("=" * 50)
    
    tests = [
        test_config,
        test_model_forward,
        test_ppo_action_selection,
        test_utils,
        test_gae_computation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The STH model is ready to use.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 