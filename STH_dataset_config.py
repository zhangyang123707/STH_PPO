# STH数据集配置文件
# 定义可用的数据集和相应的配置

import os

# 数据集根目录
DATA_ROOT = "data"

# 可用数据集配置
DATASET_CONFIGS = {
    # 杭州数据集
    "hangzhou_4_4": {
        "name": "Hangzhou 4x4 Grid",
        "city": "Hangzhou",
        "grid_size": "4_4",
        "num_agents": 16,  # 4x4网格
        "obs_dim": 16,
        "action_dim": 4,
        "traffic_file": "anon_4_4_hangzhou_real.json",
        "roadnet_file": "roadnet_4_4.json",
        "data_path": os.path.join(DATA_ROOT, "Hangzhou", "4_4"),
        "description": "杭州4x4网格交通网络，真实交通数据"
    },
    
    "hangzhou_4_4_5734": {
        "name": "Hangzhou 4x4 Grid (5734)",
        "city": "Hangzhou", 
        "grid_size": "4_4",
        "num_agents": 16,
        "obs_dim": 16,
        "action_dim": 4,
        "traffic_file": "anon_4_4_hangzhou_real_5734.json",
        "roadnet_file": "roadnet_4_4.json",
        "data_path": os.path.join(DATA_ROOT, "Hangzhou", "4_4"),
        "description": "杭州4x4网格交通网络，5734车辆流量"
    },
    
    "hangzhou_4_4_5816": {
        "name": "Hangzhou 4x4 Grid (5816)",
        "city": "Hangzhou",
        "grid_size": "4_4", 
        "num_agents": 16,
        "obs_dim": 16,
        "action_dim": 4,
        "traffic_file": "anon_4_4_hangzhou_real_5816.json",
        "roadnet_file": "roadnet_4_4.json",
        "data_path": os.path.join(DATA_ROOT, "Hangzhou", "4_4"),
        "description": "杭州4x4网格交通网络，5816车辆流量"
    },
    
    # 济南数据集
    "jinan_3_4": {
        "name": "Jinan 3x4 Grid",
        "city": "Jinan",
        "grid_size": "3_4",
        "num_agents": 12,  # 3x4网格
        "obs_dim": 16,
        "action_dim": 4,
        "traffic_file": "anon_3_4_jinan_real.json",
        "roadnet_file": "roadnet_3_4.json",
        "data_path": os.path.join(DATA_ROOT, "Jinan", "3_4"),
        "description": "济南3x4网格交通网络，真实交通数据"
    },
    
    "jinan_3_4_2000": {
        "name": "Jinan 3x4 Grid (2000)",
        "city": "Jinan",
        "grid_size": "3_4",
        "num_agents": 12,
        "obs_dim": 16,
        "action_dim": 4,
        "traffic_file": "anon_3_4_jinan_real_2000.json",
        "roadnet_file": "roadnet_3_4.json",
        "data_path": os.path.join(DATA_ROOT, "Jinan", "3_4"),
        "description": "济南3x4网格交通网络，2000车辆流量"
    },
    
    "jinan_3_4_2500": {
        "name": "Jinan 3x4 Grid (2500)",
        "city": "Jinan",
        "grid_size": "3_4",
        "num_agents": 12,
        "obs_dim": 16,
        "action_dim": 4,
        "traffic_file": "anon_3_4_jinan_real_2500.json",
        "roadnet_file": "roadnet_3_4.json",
        "data_path": os.path.join(DATA_ROOT, "Jinan", "3_4"),
        "description": "济南3x4网格交通网络，2500车辆流量"
    },
    
    # 纽约数据集
    "newyork_16_3": {
        "name": "New York 16x3 Grid",
        "city": "NewYork",
        "grid_size": "16_3",
        "num_agents": 48,  # 16x3网格
        "obs_dim": 16,
        "action_dim": 4,
        "traffic_file": "anon_16_3_newyork_real.json",
        "roadnet_file": "roadnet_16_3.json",
        "data_path": os.path.join(DATA_ROOT, "NewYork", "16_3"),
        "description": "纽约16x3网格交通网络，真实交通数据"
    },
    
    "newyork_28_7": {
        "name": "New York 28x7 Grid",
        "city": "NewYork",
        "grid_size": "28_7",
        "num_agents": 196,  # 28x7网格
        "obs_dim": 16,
        "action_dim": 4,
        "traffic_file": "anon_28_7_newyork_real_double.json",
        "roadnet_file": "roadnet_28_7.json",
        "data_path": os.path.join(DATA_ROOT, "NewYork", "28_7"),
        "description": "纽约28x7网格交通网络，双倍流量"
    },
    
    # 模板数据集（合成数据）
    "template_1_1": {
        "name": "Template 1x1 Grid",
        "city": "template",
        "grid_size": "1_1",
        "num_agents": 1,  # 1x1网格
        "obs_dim": 16,
        "action_dim": 4,
        "traffic_file": "cross.2phases_rou01_equal_300.xml",
        "roadnet_file": "roadnet_1_1.json",
        "data_path": os.path.join(DATA_ROOT, "template", "1"),
        "description": "模板1x1网格，合成交通数据"
    },
    
    "template_2_2": {
        "name": "Template 2x2 Grid",
        "city": "template",
        "grid_size": "2_2",
        "num_agents": 4,  # 2x2网格
        "obs_dim": 16,
        "action_dim": 4,
        "traffic_file": "2_intersections_300_0.3_uni.xml",
        "roadnet_file": "roadnet_2_2.json",
        "data_path": os.path.join(DATA_ROOT, "template", "2"),
        "description": "模板2x2网格，合成交通数据"
    },
    
    "template_6_6": {
        "name": "Template 6x6 Grid",
        "city": "template",
        "grid_size": "6_6",
        "num_agents": 36,  # 6x6网格
        "obs_dim": 16,
        "action_dim": 4,
        "traffic_file": "6_intersections_300_0.3_uni.xml",
        "roadnet_file": "roadnet_6_6.json",
        "data_path": os.path.join(DATA_ROOT, "template", "6"),
        "description": "模板6x6网格，合成交通数据"
    }
}

def get_dataset_config(dataset_name):
    """
    获取指定数据集的配置
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        config: 数据集配置字典
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {list(DATASET_CONFIGS.keys())}")
    
    return DATASET_CONFIGS[dataset_name]

def list_available_datasets():
    """
    列出所有可用的数据集
    
    Returns:
        list: 数据集名称列表
    """
    return list(DATASET_CONFIGS.keys())

def get_dataset_info(dataset_name):
    """
    获取数据集详细信息
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        info: 数据集信息字典
    """
    config = get_dataset_config(dataset_name)
    
    info = {
        "name": config["name"],
        "city": config["city"],
        "grid_size": config["grid_size"],
        "num_agents": config["num_agents"],
        "obs_dim": config["obs_dim"],
        "action_dim": config["action_dim"],
        "traffic_file": config["traffic_file"],
        "roadnet_file": config["roadnet_file"],
        "data_path": config["data_path"],
        "description": config["description"],
        "files_exist": check_dataset_files(dataset_name)
    }
    
    return info

def check_dataset_files(dataset_name):
    """
    检查数据集文件是否存在
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        bool: 文件是否存在
    """
    config = get_dataset_config(dataset_name)
    
    traffic_file_path = os.path.join(config["data_path"], config["traffic_file"])
    roadnet_file_path = os.path.join(config["data_path"], config["roadnet_file"])
    
    return os.path.exists(traffic_file_path) and os.path.exists(roadnet_file_path)

def print_dataset_summary():
    """
    打印数据集摘要信息
    """
    print("=" * 80)
    print("Available Datasets Summary")
    print("=" * 80)
    
    for dataset_name, config in DATASET_CONFIGS.items():
        files_exist = check_dataset_files(dataset_name)
        status = "✓" if files_exist else "✗"
        
        print(f"{status} {dataset_name}")
        print(f"    Name: {config['name']}")
        print(f"    City: {config['city']}")
        print(f"    Grid: {config['grid_size']} ({config['num_agents']} agents)")
        print(f"    Traffic: {config['traffic_file']}")
        print(f"    Roadnet: {config['roadnet_file']}")
        print(f"    Description: {config['description']}")
        print()

def get_recommended_config_for_dataset(dataset_name):
    """
    根据数据集获取推荐的STH配置
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        config: 推荐的STH配置
    """
    dataset_config = get_dataset_config(dataset_name)
    
    # 根据智能体数量选择配置
    num_agents = dataset_config["num_agents"]
    
    if num_agents <= 4:
        config_name = "small"
    elif num_agents <= 16:
        config_name = "default"
    else:
        config_name = "large"
    
    # 导入STH配置
    from STH_config import get_config
    sth_config = get_config(config_name)
    
    # 更新数据集相关参数
    sth_config.update({
        "num_agents": dataset_config["num_agents"],
        "obs_dim": dataset_config["obs_dim"],
        "action_dim": dataset_config["action_dim"],
        "dataset_name": dataset_name,
        "traffic_file": dataset_config["traffic_file"],
        "roadnet_file": dataset_config["roadnet_file"],
        "data_path": dataset_config["data_path"]
    })
    
    return sth_config

if __name__ == "__main__":
    # 打印数据集摘要
    print_dataset_summary()
    
    # 示例：获取特定数据集配置
    try:
        config = get_dataset_config("hangzhou_4_4")
        print(f"Dataset config: {config['name']}")
        print(f"Files exist: {check_dataset_files('hangzhou_4_4')}")
    except ValueError as e:
        print(f"Error: {e}") 