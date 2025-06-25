# astroWeaver/utils/logging_setup.py

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
        log_level: str = "INFO",
        log_dir: str = "logs",
        log_filename: str = "astroWeaver.log"
):
    """
    配置项目范围内的日志记录。

    此函数会设置两种日志处理器：
    1. StreamHandler: 将日志输出到控制台（stdout）。
    2. RotatingFileHandler: 将日志写入文件，并支持日志轮转。

    Args:
        log_level (str): 日志级别 (e.g., "DEBUG", "INFO", "WARNING").
        log_dir (str): 存储日志文件的目录。
        log_filename (str): 日志文件的名称。
    """
    log_level_upper = log_level.upper()
    level = getattr(logging, log_level_upper, logging.INFO)

    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    log_file = log_path / log_filename

    # 获取根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 如果已经有处理器，则先移除，避免重复添加
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 定义日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 2. 文件处理器 (支持轮转)
    # 每个文件最大5MB，保留5个备份文件
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.info(f"Logging configured. Level: {log_level_upper}. Log file: {log_file}")


# 示例用法
if __name__ == '__main__':
    # 直接运行此文件以测试日志设置
    setup_logging(log_level="DEBUG")

    logging.debug("This is a debug message.")
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.critical("This is a critical message.")