# astroWeaver/data_sources/simbad_client.py

import logging
from typing import Dict, Any, Optional

from weaver.utils.get_simbads import get_simbad_data





logger = logging.getLogger(__name__)

class SimbadClient:
    """
    一个简单的客户端，封装对 get_simbad_data 函数的调用。
    """
    def get_data(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        根据实体名称查询SIMBAD数据库。

        Args:
            entity_name (str): 要查询的天体名称。

        Returns:
            Optional[Dict[str, Any]]: 包含查询结果的字典，如果未找到或出错则返回None。
        """
        logger.info(f"Querying SIMBAD for entity: '{entity_name}'")
        try:
            simbad_data = get_simbad_data(entity_name)
            if simbad_data and isinstance(simbad_data, dict) and simbad_data:
                logger.info(f"Successfully retrieved data from SIMBAD for '{entity_name}'.")
                return simbad_data
            else:
                logger.warning(f"No results found in SIMBAD for '{entity_name}'.")
                return None
        except Exception as e:
            logger.error(f"An error occurred while querying SIMBAD for '{entity_name}': {e}")
            return None