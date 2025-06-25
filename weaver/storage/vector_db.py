# astroWeaver/storage/vector_db.py

import logging
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

# 确保从正确的相对路径导入
from ..models.embedding_models import EmbeddingClient

logger = logging.getLogger(__name__)


class CustomEmbeddingFunction(EmbeddingFunction):
    """
    将我们的EmbeddingClient包装成ChromaDB可用的EmbeddingFunction。
    """

    def __init__(self, client: EmbeddingClient):
        self._client = client

    def name(self) -> str:
        """
        返回嵌入函数的唯一名称。
        """
        # 使用EmbeddingClient中保存的api_base_url来确保名称唯一且稳定
        return f"custom-api-embedding-{self._client.api_base_url}"

    def __call__(self, input: Documents) -> Embeddings:
        """
        当ChromaDB需要嵌入时，这个方法会被调用。
        """
        # --- 核心修正：调用新增的 .encode() 方法 ---
        embeddings = self._client.encode(input)
        # --- 修改结束 ---

        if embeddings is None:
            logger.error(f"Failed to get embeddings for {len(input)} texts from the remote API.")
            raise ValueError("Failed to get embeddings from the remote API.")
        return embeddings


class VectorDBClient:
    """
    封装与ChromaDB的交互，提供高级接口用于知识图谱规范化。
    """

    def __init__(self, path: str, embedding_client: EmbeddingClient):
        try:
            self.client = chromadb.PersistentClient(path=path)
            self.custom_embedding_function = CustomEmbeddingFunction(embedding_client)
            logger.info(f"ChromaDB client initialized at path: {path}.")
        except Exception as e:
            logger.exception("Failed to initialize ChromaDB client.")
            raise

    def create_collection_if_not_exists(self, collection_name: str):
        # --- 保持不变 ---
        try:
            self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.custom_embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection '{collection_name}' is ready.")
        except Exception as e:
            logger.exception(f"Failed to get or create collection '{collection_name}'.")
            raise

    def add(
            self,
            collection_name: str,
            documents: List[str],
            metadata: List[Dict[str, Any]],
            ids: List[str]
    ):
        # --- 保持不变 ---
        try:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.custom_embedding_function
            )
            collection.add(
                documents=documents,
                metadatas=metadata,
                ids=ids
            )
            logger.info(f"Added {len(ids)} items to collection '{collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to add items to collection '{collection_name}': {e}")

    def search(
            self,
            collection_name: str,
            query_texts: List[str],
            top_k: int,
            filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        # --- 保持不变 ---
        try:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.custom_embedding_function
            )

            results = collection.query(
                query_texts=query_texts,
                n_results=top_k,
                where=filter_dict,
                include=['metadatas', 'distances']
            )

            # --- 健壮性优化：处理多查询结果 ---
            # ChromaDB为每个查询文本返回一个结果列表，所以结果是二维的
            all_formatted_results = []
            if not results or not results.get('ids'):
                return []

            num_queries = len(results['ids'])
            for i in range(num_queries):
                query_results = []
                # 检查当前查询是否有返回结果
                if not results['ids'][i]:
                    continue

                ids = results['ids'][i]
                metadatas = results['metadatas'][i]
                distances = results['distances'][i]

                for j in range(len(ids)):
                    query_results.append({
                        "id": ids[j],
                        "metadata": metadatas[j],
                        "score": 1 - distances[j]  # 将cosine distance转换为similarity
                    })
                all_formatted_results.append(query_results)

            # 为了与旧接口兼容，如果只有一个查询，我们返回一维列表
            if len(all_formatted_results) == 1:
                return all_formatted_results[0]
            # 如果有多个查询，返回二维列表
            return all_formatted_results

        except Exception as e:
            logger.error(f"Error querying collection '{collection_name}': {e}")
            return []

    def reset(self):
        # --- 保持不变 ---
        try:
            self.client.reset()
            logger.info("ChromaDB client has been reset.")
        except Exception as e:
            logger.error(f"Failed to reset ChromaDB client: {e}")