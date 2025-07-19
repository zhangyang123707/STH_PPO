#!/bin/bash

# STH模型Docker运行脚本
# 包含完整的运行顺序和步骤

set -e  # 遇到错误时退出

echo "🚀 STH模型Docker运行脚本"
echo "=================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 函数：打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker是否安装
check_docker() {
    print_info "检查Docker安装..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    print_success "Docker已安装"
}

# 检查Docker Compose是否安装
check_docker_compose() {
    print_info "检查Docker Compose安装..."
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    print_success "Docker Compose已安装"
}

# 创建必要的目录
create_directories() {
    print_info "创建必要的目录..."
    mkdir -p models logs runs data
    print_success "目录创建完成"
}

# 构建Docker镜像
build_image() {
    print_info "构建Docker镜像..."
    docker build -t sth-model .
    print_success "Docker镜像构建完成"
}

# 运行测试
run_tests() {
    print_info "运行STH模型测试..."
    docker run --rm -v $(pwd):/app sth-model python STH_test.py
    print_success "测试完成"
}

# 检查数据集
check_dataset() {
    print_info "检查数据集配置..."
    docker run --rm -v $(pwd):/app sth-model python -c "
from STH_dataset_config import print_dataset_summary
print_dataset_summary()
"
    print_success "数据集检查完成"
}

# 运行基本示例
run_basic_example() {
    print_info "运行STH基本示例..."
    docker run --rm -v $(pwd):/app sth-model python STH_example.py
    print_success "基本示例运行完成"
}

# 运行济南数据集示例
run_jinan_example() {
    print_info "运行济南数据集示例..."
    docker run --rm -v $(pwd):/app sth-model python STH_jinan_example.py
    print_success "济南数据集示例运行完成"
}

# 运行数据集示例
run_dataset_example() {
    print_info "运行数据集使用示例..."
    docker run --rm -v $(pwd):/app sth-model python STH_dataset_example.py
    print_success "数据集示例运行完成"
}

# 启动TensorBoard
start_tensorboard() {
    print_info "启动TensorBoard..."
    docker run -d --name sth-tensorboard \
        -p 6006:6006 \
        -v $(pwd)/runs:/app/runs \
        sth-model \
        tensorboard --logdir=/app/runs --host=0.0.0.0 --port=6006
    print_success "TensorBoard已启动，访问 http://localhost:6006"
}

# 运行训练
run_training() {
    print_info "开始训练STH模型..."
    docker run --rm \
        -v $(pwd):/app \
        sth-model \
        python -c "
from STH_agent import STHAgent
from STH_config import get_config
import numpy as np

class MockTrafficEnvironment:
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

# 创建环境和智能体
config = get_config('default', dataset_name='jinan_3_4')
env = MockTrafficEnvironment(
    num_agents=config['num_agents'],
    obs_dim=config['obs_dim'],
    action_dim=config['action_dim']
)
agent = STHAgent(env, config_name='default', dataset_name='jinan_3_4')

# 运行训练
print('开始训练...')
for episode in range(10):
    episode_info = agent.run_episode(training=True)
    print(f'Episode {episode + 1}: Reward = {episode_info[\"total_reward\"]:.2f}')

# 保存模型
agent.save_model()
print('训练完成，模型已保存')
"
    print_success "训练完成"
}

# 清理容器
cleanup() {
    print_info "清理容器..."
    docker stop sth-tensorboard 2>/dev/null || true
    docker rm sth-tensorboard 2>/dev/null || true
    print_success "清理完成"
}

# 主函数
main() {
    case "$1" in
        "build")
            check_docker
            create_directories
            build_image
            ;;
        "test")
            check_docker
            run_tests
            ;;
        "check-data")
            check_docker
            check_dataset
            ;;
        "example")
            check_docker
            run_basic_example
            ;;
        "jinan")
            check_docker
            run_jinan_example
            ;;
        "dataset")
            check_docker
            run_dataset_example
            ;;
        "train")
            check_docker
            run_training
            ;;
        "tensorboard")
            check_docker
            start_tensorboard
            ;;
        "full")
            check_docker
            check_docker_compose
            create_directories
            build_image
            run_tests
            check_dataset
            run_basic_example
            run_jinan_example
            run_dataset_example
            run_training
            start_tensorboard
            ;;
        "cleanup")
            cleanup
            ;;
        *)
            echo "用法: $0 {build|test|check-data|example|jinan|dataset|train|tensorboard|full|cleanup}"
            echo ""
            echo "命令说明:"
            echo "  build      - 构建Docker镜像"
            echo "  test       - 运行测试"
            echo "  check-data - 检查数据集"
            echo "  example    - 运行基本示例"
            echo "  jinan      - 运行济南数据集示例"
            echo "  dataset    - 运行数据集示例"
            echo "  train      - 运行训练"
            echo "  tensorboard- 启动TensorBoard"
            echo "  full       - 完整流程（推荐）"
            echo "  cleanup    - 清理容器"
            echo ""
            echo "推荐运行顺序:"
            echo "  1. ./docker-run.sh full    # 完整流程"
            echo "  2. 访问 http://localhost:6006 查看TensorBoard"
            echo "  3. ./docker-run.sh cleanup # 清理"
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@" 