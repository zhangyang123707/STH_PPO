# 🐳 STH模型Docker运行指南

本指南介绍如何在Docker环境中运行STH（时空混合注意力网络+分层PPO）交通信号控制模型。

## 📋 前置要求

### 1. 安装Docker
- **Windows**: 安装 [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: 安装 [Docker Engine](https://docs.docker.com/engine/install/)
- **macOS**: 安装 [Docker Desktop](https://www.docker.com/products/docker-desktop)

### 2. 安装Docker Compose（可选）
- 用于更复杂的容器编排
- Docker Desktop通常已包含

## 🚀 快速开始

### 方法1: 使用自动化脚本（推荐）

#### Linux/macOS:
```bash
# 给脚本执行权限
chmod +x docker-run.sh

# 运行完整流程
./docker-run.sh full
```

#### Windows:
```cmd
# 运行完整流程
docker-run.bat full
```

### 方法2: 手动执行

```bash
# 1. 构建镜像
docker build -t sth-model .

# 2. 运行测试
docker run --rm -v $(pwd):/app sth-model python STH_test.py

# 3. 检查数据集
docker run --rm -v $(pwd):/app sth-model python -c "from STH_dataset_config import print_dataset_summary; print_dataset_summary()"

# 4. 运行示例
docker run --rm -v $(pwd):/app sth-model python STH_example.py

# 5. 运行训练
docker run --rm -v $(pwd):/app sth-model python STH_jinan_example.py
```

## 📁 文件结构

```
colight-master/
├── Dockerfile              # Docker镜像构建文件
├── docker-compose.yml      # Docker Compose配置
├── docker-run.sh          # Linux/macOS运行脚本
├── docker-run.bat         # Windows运行脚本
├── requirements.txt       # Python依赖
├── STH_*.py              # STH模型文件
├── data/                 # 数据集目录
├── models/               # 模型保存目录
├── logs/                 # 日志目录
└── runs/                 # TensorBoard日志
```

## 🔧 运行顺序详解

### 1. 构建阶段
```bash
./docker-run.sh build
```
- 检查Docker安装
- 创建必要目录
- 构建Docker镜像
- 安装所有依赖

### 2. 验证阶段
```bash
./docker-run.sh test
./docker-run.sh check-data
```
- 运行单元测试
- 检查数据集配置
- 验证环境正确性

### 3. 示例阶段
```bash
./docker-run.sh example
./docker-run.sh jinan
./docker-run.sh dataset
```
- 运行基本示例
- 运行济南数据集示例
- 运行数据集使用示例

### 4. 训练阶段
```bash
./docker-run.sh train
```
- 创建智能体
- 运行训练过程
- 保存训练模型

### 5. 监控阶段
```bash
./docker-run.sh tensorboard
```
- 启动TensorBoard
- 访问 http://localhost:6006
- 查看训练进度

## 🎯 具体运行步骤

### 步骤1: 环境准备
```bash
# 确保在项目根目录
cd colight-master

# 检查Docker状态
docker --version
docker-compose --version
```

### 步骤2: 构建镜像
```bash
# 构建STH模型镜像
docker build -t sth-model .
```

### 步骤3: 验证环境
```bash
# 运行测试
docker run --rm -v $(pwd):/app sth-model python STH_test.py

# 检查数据集
docker run --rm -v $(pwd):/app sth-model python -c "
from STH_dataset_config import print_dataset_summary
print_dataset_summary()
"
```

### 步骤4: 运行示例
```bash
# 基本示例
docker run --rm -v $(pwd):/app sth-model python STH_example.py

# 济南数据集示例
docker run --rm -v $(pwd):/app sth-model python STH_jinan_example.py

# 数据集示例
docker run --rm -v $(pwd):/app sth-model python STH_dataset_example.py
```

### 步骤5: 开始训练
```bash
# 运行训练
docker run --rm -v $(pwd):/app sth-model python -c "
from STH_agent import STHAgent
from STH_config import get_config
import numpy as np

# 创建模拟环境
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
```

### 步骤6: 启动监控
```bash
# 启动TensorBoard
docker run -d --name sth-tensorboard \
    -p 6006:6006 \
    -v $(pwd)/runs:/app/runs \
    sth-model \
    tensorboard --logdir=/app/runs --host=0.0.0.0 --port=6006

# 访问TensorBoard
# 打开浏览器访问: http://localhost:6006
```

## 🔍 常用命令

### 查看容器状态
```bash
# 查看运行中的容器
docker ps

# 查看所有容器
docker ps -a

# 查看镜像
docker images
```

### 进入容器调试
```bash
# 进入运行中的容器
docker exec -it sth-traffic-control bash

# 运行交互式Python
docker run -it --rm -v $(pwd):/app sth-model python
```

### 查看日志
```bash
# 查看容器日志
docker logs sth-tensorboard

# 实时查看日志
docker logs -f sth-tensorboard
```

### 清理资源
```bash
# 停止容器
docker stop sth-tensorboard

# 删除容器
docker rm sth-tensorboard

# 删除镜像
docker rmi sth-model

# 清理所有未使用的资源
docker system prune
```

## 🐛 故障排除

### 常见问题

#### 1. Docker权限问题
```bash
# Linux: 添加用户到docker组
sudo usermod -aG docker $USER
# 重新登录或重启
```

#### 2. 端口被占用
```bash
# 检查端口占用
netstat -tulpn | grep 6006

# 使用不同端口
docker run -p 6007:6006 ...
```

#### 3. 内存不足
```bash
# 增加Docker内存限制
# 在Docker Desktop设置中调整内存限制
```

#### 4. 网络问题
```bash
# 检查网络连接
docker network ls
docker network inspect bridge
```

### 调试技巧

#### 1. 查看详细构建日志
```bash
docker build -t sth-model . --progress=plain --no-cache
```

#### 2. 检查容器内部
```bash
docker run -it --rm -v $(pwd):/app sth-model bash
```

#### 3. 查看文件挂载
```bash
docker run --rm -v $(pwd):/app sth-model ls -la /app
```

## 📊 性能优化

### 1. GPU支持
```bash
# 使用GPU版本（需要NVIDIA Docker）
docker run --gpus all -v $(pwd):/app sth-model python STH_example.py
```

### 2. 多容器并行
```bash
# 使用Docker Compose
docker-compose up -d
```

### 3. 资源限制
```bash
# 限制内存和CPU
docker run --memory=4g --cpus=2 -v $(pwd):/app sth-model python STH_example.py
```

## 📝 注意事项

1. **数据持久化**: 使用卷挂载保存模型和日志
2. **环境隔离**: 每个容器都是独立的环境
3. **资源管理**: 注意内存和CPU使用情况
4. **网络访问**: 确保端口映射正确
5. **文件权限**: 注意挂载目录的权限设置

## 🎉 成功标志

当看到以下输出时，说明运行成功：

```
🎉 All examples completed successfully!
============================================================
✓ 智能体创建成功
✓ 训练完成，模型已保存
TensorBoard已启动，访问 http://localhost:6006
```

## 📞 获取帮助

如果遇到问题，请：

1. 检查Docker和Docker Compose版本
2. 查看容器日志
3. 确认文件权限和路径
4. 参考故障排除部分
5. 检查系统资源使用情况 