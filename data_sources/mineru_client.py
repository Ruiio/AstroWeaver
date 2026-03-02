# astroWeaver/data_sources/simbad_client.py

import logging
from typing import Dict, Any, Optional, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from weaver.utils.MinerU import parse_file
from weaver.utils.config import config






logger = logging.getLogger(__name__)

class MinerUClient:
    """
    一个简单的客户端，封装对 get_simbad_data 函数的调用。
    """
    def parseFile(self, file_paths: str) -> Optional[List[str]]:
        """
        根据实体名称查询SIMBAD数据库。

        Args:
            entity_name (str): 要查询的天体名称。

        Returns:
            Optional[Dict[str, Any]]: 包含查询结果的字典，如果未找到或出错则返回None。
        """
        logger.info(f"Parsing File for entity: '{file_paths}'")
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                # Set a really small chunk size, just to show.
                chunk_size=config['chunk']['chunk_size'],
                chunk_overlap=config['chunk']['chunk_overlap'],
                length_function=len,
            )
            md_data = parse_file(file_paths)
            if md_data and isinstance(md_data, dict) and md_data:
                for item in md_data:
                    texts = text_splitter.create_documents([md_data[item]])
                    trunked_texts = []
                    for txt in texts:
                        trunked_texts.append(txt.page_content)
                    logger.info(f"Successfully retrieved markdown data from MinerU.")
                    return trunked_texts
            else:
                logger.warning(f"No results retrieved from MinerU.")
                return None
        except Exception as e:
            logger.error(f"An error occurred while Parsing File for '{file_paths}': {e}")
            return None