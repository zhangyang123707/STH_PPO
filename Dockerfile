# STH模型专用Dockerfile
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装STH模型所需的额外依赖
RUN pip install --no-cache-dir \
    torch==1.12.1+cpu \
    torchvision==0.13.1+cpu \
    torchaudio==0.12.1+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    numpy \
    scipy \
    scikit-learn \
    tqdm \
    tensorboard

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p models logs runs data

# 设置环境变量
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=""

# 暴露端口（如果需要Web界面）
EXPOSE 6006

# 设置默认命令
CMD ["python", "STH_example.py"] 