@echo off
REM STH模型Docker运行脚本 (Windows版本)
REM 包含完整的运行顺序和步骤

echo 🚀 STH模型Docker运行脚本 (Windows)
echo ==================================

setlocal enabledelayedexpansion

REM 检查参数
if "%1"=="" goto usage

REM 检查Docker是否安装
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker未安装，请先安装Docker Desktop
    exit /b 1
)

REM 创建必要的目录
if not exist models mkdir models
if not exist logs mkdir logs
if not exist runs mkdir runs
if not exist data mkdir data

if "%1"=="build" goto build
if "%1"=="test" goto test
if "%1"=="check-data" goto check-data
if "%1"=="example" goto example
if "%1"=="jinan" goto jinan
if "%1"=="dataset" goto dataset
if "%1"=="train" goto train
if "%1"=="tensorboard" goto tensorboard
if "%1"=="full" goto full
if "%1"=="cleanup" goto cleanup
goto usage

:build
echo [INFO] 构建Docker镜像...
docker build -t sth-model .
echo [SUCCESS] Docker镜像构建完成
goto end

:test
echo [INFO] 运行STH模型测试...
docker run --rm -v %cd%:/app sth-model python STH_test.py
echo [SUCCESS] 测试完成
goto end

:check-data
echo [INFO] 检查数据集配置...
docker run --rm -v %cd%:/app sth-model python -c "from STH_dataset_config import print_dataset_summary; print_dataset_summary()"
echo [SUCCESS] 数据集检查完成
goto end

:example
echo [INFO] 运行STH基本示例...
docker run --rm -v %cd%:/app sth-model python STH_example.py
echo [SUCCESS] 基本示例运行完成
goto end

:jinan
echo [INFO] 运行济南数据集示例...
docker run --rm -v %cd%:/app sth-model python STH_jinan_example.py
echo [SUCCESS] 济南数据集示例运行完成
goto end

:dataset
echo [INFO] 运行数据集使用示例...
docker run --rm -v %cd%:/app sth-model python STH_dataset_example.py
echo [SUCCESS] 数据集示例运行完成
goto end

:train
echo [INFO] 开始训练STH模型...
docker run --rm -v %cd%:/app sth-model python -c "from STH_agent import STHAgent; from STH_config import get_config; import numpy as np; class MockTrafficEnvironment: def __init__(self, num_agents=12, obs_dim=16, action_dim=4): self.num_agents = num_agents; self.obs_dim = obs_dim; self.action_dim = action_dim; self.step_count = 0; self.max_steps = 100; def reset(self): self.step_count = 0; return np.random.randn(self.num_agents, self.obs_dim); def step(self, actions): self.step_count += 1; obs = np.random.randn(self.num_agents, self.obs_dim); reward = -np.abs(actions - 2); info = {'queue_length': np.random.randn(self.num_agents), 'waiting_time': np.random.randn(self.num_agents), 'throughput': np.random.randn(self.num_agents), 'pressure': np.random.randn(self.num_agents), 'avg_speed': np.random.randn(self.num_agents), 'phase_change': np.random.randint(0, 2, self.num_agents), 'neighbor_coordination': np.random.randn()}; done = np.array([self.step_count >= self.max_steps] * self.num_agents); return obs, reward, done, info; config = get_config('default', dataset_name='jinan_3_4'); env = MockTrafficEnvironment(num_agents=config['num_agents'], obs_dim=config['obs_dim'], action_dim=config['action_dim']); agent = STHAgent(env, config_name='default', dataset_name='jinan_3_4'); print('开始训练...'); [agent.run_episode(training=True) for _ in range(10)]; agent.save_model(); print('训练完成，模型已保存')"
echo [SUCCESS] 训练完成
goto end

:tensorboard
echo [INFO] 启动TensorBoard...
docker run -d --name sth-tensorboard -p 6006:6006 -v %cd%/runs:/app/runs sth-model tensorboard --logdir=/app/runs --host=0.0.0.0 --port=6006
echo [SUCCESS] TensorBoard已启动，访问 http://localhost:6006
goto end

:full
echo [INFO] 开始完整流程...
call :build
call :test
call :check-data
call :example
call :jinan
call :dataset
call :train
call :tensorboard
echo [SUCCESS] 完整流程执行完成
goto end

:cleanup
echo [INFO] 清理容器...
docker stop sth-tensorboard 2>nul
docker rm sth-tensorboard 2>nul
echo [SUCCESS] 清理完成
goto end

:usage
echo 用法: %0 {build^|test^|check-data^|example^|jinan^|dataset^|train^|tensorboard^|full^|cleanup}
echo.
echo 命令说明:
echo   build      - 构建Docker镜像
echo   test       - 运行测试
echo   check-data - 检查数据集
echo   example    - 运行基本示例
echo   jinan      - 运行济南数据集示例
echo   dataset    - 运行数据集示例
echo   train      - 运行训练
echo   tensorboard- 启动TensorBoard
echo   full       - 完整流程（推荐）
echo   cleanup    - 清理容器
echo.
echo 推荐运行顺序:
echo   1. %0 full    # 完整流程
echo   2. 访问 http://localhost:6006 查看TensorBoard
echo   3. %0 cleanup # 清理
goto end

:end
echo.
echo 脚本执行完成
pause 