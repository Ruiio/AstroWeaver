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

    def __init__(self, api_base_url: str, request_timeout: int = 60, max_retries: int = 3):
        """
        初始化API客户端。

        Args:
            api_base_url (str): 嵌入模型FastAPI服务的URL (e.g., "http://localhost:8005").
            request_timeout (int): 请求超时时间（秒）。
            max_retries (int): 请求失败时的最大重试次数。
        """
        # --- 保持不变 ---
        if not api_base_url.endswith('/'):
            api_base_url += '/'
        self.api_base_url = api_base_url # 保存原始URL，用于CustomEmbeddingFunction的name()方法
        self.api_url = f"{api_base_url}embeddings"
        self.timeout = request_timeout
        self.max_retries = max_retries

        logger.info(f"Initializing EmbeddingClient for API endpoint: {self.api_url}")
        self._check_service_health()

    def _check_service_health(self):
        # --- 保持不变 ---
        """
        检查远程服务是否可用。
        """
        try:
            response = requests.get(self.api_url.rsplit('/', 1)[0], timeout=5)
            if response.status_code == 200 or response.status_code == 404:
                logger.info("Embedding service seems to be running.")
            else:
                logger.warning(f"Embedding service might be down. Health check returned status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Could not connect to embedding service at {self.api_url}. Error: {e}")
            raise ConnectionError("Failed to connect to the embedding service.") from e

    def _make_request(self, texts: List[str], task: str) -> Optional[List[List[float]]]:
        # --- 保持不变 ---
        """
        向远程API发送嵌入请求，并处理重试逻辑。
        """
        payload = {
            "text_list": texts,
            "task": task
        }

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

    def get_embedding(self, text: str, is_query: bool = False) -> Optional[List[float]]:
        # --- 保持不变 ---
        """
        获取单个文本的嵌入向量。
        """
        task = "retrieval-query" if is_query else "retrieval-doc"
        embeddings = self._make_request([text], task=task)
        if embeddings and len(embeddings) == 1:
            return embeddings[0]
        return None

    def get_embeddings(self, texts: List[str], is_query: bool = False) -> Optional[List[List[float]]]:
        # --- 保持不变 ---
        """
        批量获取多个文本的嵌入向量。
        """
        task = "retrieval-query" if is_query else "retrieval-doc"
        return self._make_request(texts, task=task)

    # --- 新增方法以修复错误 ---
    def encode(self, texts: List[str], is_query: bool = False) -> Optional[List[List[float]]]:
        """
        一个通用的编码方法，作为get_embeddings的别名，以兼容ChromaDB的EmbeddingFunction。
        ChromaDB在调用时不会区分查询和文档，所以我们默认is_query=False。
        """
        return self.get_embeddings(texts, is_query=is_query)
    # --- 修改结束 ---


# 示例用法 (保持不变)
if __name__ == '__main__':
    # ...
    pass