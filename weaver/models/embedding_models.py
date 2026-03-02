# astroWeaver/models/embedding_models.py

import logging
from typing import List, Optional
import requests
import time

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """
    通过HTTP API与远程嵌入模型服务交互的客户端。
    """

    def __init__(self, api_base_url: str, request_timeout: int = 60, max_retries: int = 3, lazy_init: bool = True):
        """
        初始化API客户端。

        Args:
            api_base_url (str): 嵌入模型FastAPI服务的URL (e.g., "http://localhost:8005").
            request_timeout (int): 请求超时时间（秒）。
            max_retries (int): 请求失败时的最大重试次数。
            lazy_init (bool): 是否延迟初始化，True表示不立即检查服务健康状态，等到第一次使用时再检查。
        """
        # --- 保持不变 ---
        if not api_base_url.endswith('/'):
            api_base_url += '/'
        self.api_base_url = api_base_url # 保存原始URL，用于CustomEmbeddingFunction的name()方法
        self.api_url = f"{api_base_url}embeddings"
        self.timeout = request_timeout
        self.max_retries = max_retries
        self.service_checked = False

        logger.info(f"Initializing EmbeddingClient for API endpoint: {self.api_url}")
        if not lazy_init:
            self._check_service_health()

    def _check_service_health(self):
        """
        检查远程服务是否可用。
        """
        try:
            response = requests.get(self.api_base_url, timeout=5)
            if response.status_code == 200:
                logger.info("Embedding service seems to be running.")
                return True
            else:
                logger.warning(f"Embedding service might be down. Health check returned status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Could not connect to embedding service at {self.api_url}. Error: {e}")
            raise ConnectionError("Failed to connect to the embedding service.") from e

    def _make_request(self, texts: List[str], task: str) -> Optional[List[List[float]]]:
        """
        向远程API发送嵌入请求，并处理重试逻辑。
        """
        # 如果服务尚未检查，先检查服务健康状态
        if not getattr(self, 'service_checked', False):
            try:
                self._check_service_health()
                self.service_checked = True
            except ConnectionError:
                logger.warning("Embedding service is not available, but continuing without it.")
                # 继续执行，不抛出异常
        
        # 根据API格式构建payload
        payload = {
            "texts": texts
        }
        # 只有当task不为None时才添加到payload
        if task is not None:
            payload["prompt_name"] = task

        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.api_url, json=payload, timeout=self.timeout)
                response.raise_for_status()

                data = response.json()
                if "embeddings" in data and isinstance(data["embeddings"], list):
                    return data["embeddings"]
                else:
                    logger.error(f"Invalid response format from embedding API: {data}")
                    return None

            except requests.exceptions.RequestException as e:
                logger.warning(f"Embedding API request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("Max retries reached for embedding API request.")
                    return None
        return None

    def get_embedding(self, text: str, prompt_name: str = None) -> Optional[List[float]]:
        """
        获取单个文本的嵌入向量。
        """
        embeddings = self._make_request([text], task=prompt_name)
        if embeddings and len(embeddings) == 1:
            return embeddings[0]
        return None

    def get_embeddings(self, texts: List[str], prompt_name: str = None) -> Optional[List[List[float]]]:
        """
        批量获取多个文本的嵌入向量。
        """
        return self._make_request(texts, task=prompt_name)

    # --- 新增方法以修复错误 ---
    def encode(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        ChromaDB兼容方法，用于获取文本嵌入。
        
        Args:
            texts: 要嵌入的文本列表。
            
        Returns:
            嵌入向量列表，如果失败则返回None。
        """
        return self.get_embeddings(texts)
    # --- 修改结束 ---


# 示例用法 (保持不变)
if __name__ == '__main__':
    # ...
    pass