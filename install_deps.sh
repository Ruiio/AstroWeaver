#!/bin/bash
# AstroWeaver 依赖安装脚本

echo "安装 AstroWeaver 依赖..."

# 检查 Python 版本
python3 --version || { echo "Python 3 未安装"; exit 1; }

# 安装核心依赖
echo "安装核心依赖..."
pip3 install -r requirements-minimal.txt

# 检查是否安装成功
echo "检查依赖安装..."
python3 -c "import pandas, openai, httpx, yaml, requests, bs4; print('✅ 核心依赖安装成功')"

# 可选：安装完整依赖
echo "如需安装完整依赖，请运行: pip3 install -r requirements.txt"

echo "安装完成！"
echo "请复制 .env.template 为 .env 并填写您的 API 密钥"