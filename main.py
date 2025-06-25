# main.py
import logging
from weaver.utils.logging_setup import setup_logging
from weaver.utils.config import config  # 直接导入已加载的配置对象
from weaver.core.pipeline import run_pipeline


def main():
    # 1. 设置日志
    setup_logging()

    # 2. 配置已通过导入自动加载，可以直接使用
    if not config:
        logging.critical("Exiting due to configuration loading failure.")
        return

    # 3. 运行主管道
    try:
        run_pipeline(config)
    except Exception as e:
        logging.exception("The pipeline terminated with an unhandled exception.")


if __name__ == "__main__":
    main()