# astroWeaver/utils/config.py

import os
import re
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

# 定义配置文件的默认路径，相对于项目根目录
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    从指定的YAML文件加载配置，支持环境变量替换。

    Args:
        config_path (str, optional): 配置文件的路径。
                                     如果为None，则使用默认路径。

    Returns:
        Dict[str, Any]: 包含所有配置项的字典。

    Raises:
        FileNotFoundError: 如果配置文件不存在。
        yaml.YAMLError: 如果配置文件格式不正确。
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.error(f"Configuration file not found at: {path}")
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    logger.info(f"Loading configuration from: {path}")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 环境变量替换：支持 ${VAR_NAME} 或 $VAR_NAME 格式
        def replace_env_var(match):
            var_name = match.group(1) or match.group(2)
            # 尝试从环境变量获取，如果不存在则使用原字符串
            return os.environ.get(var_name, match.group(0))
        
        # 替换 ${VAR} 和 $VAR 格式
        content = re.sub(r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)', replace_env_var, content)
        
        config = yaml.safe_load(content)

        # 可以在这里添加对配置项的验证逻辑，例如检查必要的键是否存在
        _validate_config(config)

        return config
    except yaml.YAMLError as e:
        logger.exception(f"Error parsing YAML configuration file: {path}")
        raise
    except Exception as e:
        logger.exception(f"An unexpected error occurred while loading config from {path}")
        raise


def _validate_config(config: Dict[str, Any]):
    """
    一个简单的配置验证函数（可根据需要扩展）。
    """
    required_keys = ['api_keys', 'paths', 'llm', 'embedding', 'vector_db', 'pipeline']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required top-level key '{key}' in config file.")

    if 'llm' not in config['api_keys']:
        raise ValueError("Missing 'llm' API key under 'api_keys'.")


# 创建一个全局配置对象，以便在项目的任何地方方便地导入和使用
# 这种模式可以简化代码，但要注意它在应用启动时只加载一次
try:
    config = load_config()
except (FileNotFoundError, ValueError) as e:
    logger.critical(f"Could not load initial configuration. Please check your config file. Error: {e}")
    config = {}  # 在加载失败时提供一个空字典，以避免导入时出错

# 示例用法
if __name__ == '__main__':
    # 如果直接运行此文件，将打印加载的配置
    from pprint import pprint

    if config:
        print("Configuration loaded successfully:")
        pprint(config)
    else:
        print("Failed to load configuration.")