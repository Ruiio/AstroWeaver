# astroWeaver/storage/file_handler.py

import logging
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from ..utils.text_utils import sanitize_filename

logger = logging.getLogger(__name__)


class FileHandler:
    """
    处理所有与文件系统相关的读写操作。
    """

    def __init__(self, output_dir: str):
        """
        初始化FileHandler。

        Args:
            output_dir (str): 所有输出文件的根目录。
        """
        self.base_output_path = Path(output_dir)
        self.entity_data_path = self.base_output_path / "entity_data"
        self.simbad_data_path = self.base_output_path / "simbad_data"

        # 创建基础输出目录
        self.base_output_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Base output directory initialized at: {self.base_output_path}")

    def load_entity_list(self, file_path: str) -> List[str]:
        """
        从Excel或CSV文件中加载待处理的实体列表。
        假设实体名称在第一列。

        Args:
            file_path (str): 输入文件的路径。

        Returns:
            List[str]: 实体名称列表。
        """
        p = Path(file_path)
        if not p.exists():
            logger.error(f"Input entity file not found: {file_path}")
            return []

        try:
            if p.suffix == '.xlsx':
                df = pd.read_excel(p, header=None)
            elif p.suffix == '.csv':
                df = pd.read_csv(p, header=None)
            else:
                logger.error(f"Unsupported file format for entity list: {p.suffix}")
                return []

            # 提取第一列，去除空值和前后空格
            entities = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
            logger.info(f"Loaded {len(entities)} entities from {file_path}")
            return entities
        except Exception as e:
            logger.exception(f"Failed to load entity list from {file_path}")
            return []

    def save_json(self, file_path: Path, data: Dict[str, Any]):
        """
        将字典数据保存为格式化的JSON文件。

        Args:
            file_path (Path): 目标文件路径。
            data (Dict[str, Any]): 要保存的数据。
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logger.debug(f"Successfully saved JSON to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON to {file_path}: {e}")

    def save_entity_data(self, entity_name: str, data: Dict[str, Any]):
        """
        保存单个实体的最终处理结果。

        Args:
            entity_name (str): 实体名称。
            data (Dict[str, Any]): 实体的数据。
        """
        # 只在需要保存数据时才创建目录
        self.entity_data_path.mkdir(parents=True, exist_ok=True)
        safe_name = sanitize_filename(entity_name)
        file_path = self.entity_data_path / f"{safe_name}.json"
        self.save_json(file_path, data)
        logger.debug(f"Saved entity data for '{entity_name}' to {file_path}")

    def save_simbad_data(self, entity_name: str, data: Dict[str, Any]):
        """
        保存从SIMBAD获取的原始数据。

        Args:
            entity_name (str): 实体名称。
            data (Dict[str, Any]): SIMBAD返回的数据。
        """
        # 只在需要保存数据时才创建目录
        self.simbad_data_path.mkdir(parents=True, exist_ok=True)
        safe_name = sanitize_filename(entity_name)
        file_path = self.simbad_data_path / f"{safe_name}.json"
        self.save_json(file_path, data)
        logger.debug(f"Saved SIMBAD data for '{entity_name}' to {file_path}")