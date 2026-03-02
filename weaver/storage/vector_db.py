# weaver/storage/vector_db.py (Final, Corrected Version)

import logging
import os
from typing import List, Dict, Any, Optional

# 设置环境变量以避免ChromaDB默认嵌入函数的问题
os.environ['ALLOW_RESET'] = 'TRUE'

# 延迟导入chromadb以避免初始化问题
# import chromadb
# from chromadb import Settings
# from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from weaver.models.embedding_models import EmbeddingClient
from weaver.utils.config import config


logger = logging.getLogger(__name__)


class CustomEmbeddingFunction:
    """
    将我们的EmbeddingClient包装成ChromaDB可用的EmbeddingFunction。
    """

    def __init__(self, client: EmbeddingClient):
        self._client = client
    
    def name(self) -> str:
        """
        返回嵌入函数的名称，ChromaDB要求的方法。
        """
        return "custom_embedding_function"

    def __call__(self, input):
        """
        当ChromaDB需要嵌入时，这个方法会被调用。
        增强错误处理和内存管理。
        """
        if not input:
            logger.warning("Empty input provided to embedding function")
            return []
        
        # 限制单次处理的文档数量以避免内存问题
        max_batch_size = 32
        if len(input) > max_batch_size:
            logger.info(f"Large batch detected ({len(input)} docs), processing in smaller chunks")
            all_embeddings = []
            for i in range(0, len(input), max_batch_size):
                batch = input[i:i + max_batch_size]
                batch_embeddings = self._process_batch(batch)
                all_embeddings.extend(batch_embeddings)
            return all_embeddings
        else:
            return self._process_batch(input)
    
    def _process_batch(self, batch):
        """
        处理单个批次的文档嵌入。
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 调用您的 EmbeddingClient 的 encode 方法
                embeddings = self._client.encode(batch)

                # **关键修复**: 严格处理来自 EmbeddingClient 的失败（None返回值）
                if embeddings is None:
                    if attempt < max_retries - 1:
                        logger.warning(f"EmbeddingClient returned None for batch (attempt {attempt + 1}/{max_retries}), retrying...")
                        import time
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        logger.error(
                            f"EmbeddingClient failed to return embeddings for {len(batch)} documents after {max_retries} attempts. "
                            "This will cause the current ChromaDB operation to fail."
                        )
                        raise ValueError("Failed to get embeddings from the remote API via EmbeddingClient after multiple retries.")

                # 验证返回的嵌入向量格式
                if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                    raise ValueError(f"Invalid embeddings format: expected {len(batch)} embeddings, got {len(embeddings) if isinstance(embeddings, list) else 'non-list'}")
                
                return embeddings
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Embedding request failed (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                    import time
                    time.sleep(2 ** attempt)
                    continue
                else:
                    logger.error(f"Embedding request failed after {max_retries} attempts: {e}")
                    raise ValueError(f"Failed to get embeddings after {max_retries} attempts: {e}") from e
        
        # 这行代码不应该被执行到，但为了类型安全
        raise ValueError("Unexpected error in embedding processing")


class VectorDBClient:
    """
    封装与ChromaDB的交互，提供高级接口用于知识图谱规范化。
    """

    def __init__(self, path: str, embedding_client: EmbeddingClient):
        """
        初始化ChromaDB客户端。
        """
        try:
            # 在这里导入chromadb以避免初始化问题
            import chromadb
            from chromadb import Settings
            
            # 允许重置，便于测试
            myChromaSetting = Settings(allow_reset=True)
            self.client = chromadb.PersistentClient(path=path, settings=myChromaSetting)
            # 使用您的 EmbeddingClient 创建自定义嵌入函数
            self.custom_embedding_function = CustomEmbeddingFunction(embedding_client)
            logger.info(f"ChromaDB client initialized at path: {path}.")
        except Exception as e:
            logger.exception("Failed to initialize ChromaDB client.")
            raise

    def create_collection_if_not_exists(self, collection_name: str):
        """
        如果集合不存在，则创建它。
        处理'_type'错误和集合损坏问题。
        """
        try:
            self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.custom_embedding_function,
                metadata={"hnsw:space": "cosine"}  # Cosine similarity is good for text
            )
            logger.info(f"Collection '{collection_name}' is ready.")
        except KeyError as e:
            if "'_type'" in str(e):
                logger.warning(f"Collection '{collection_name}' appears corrupted (missing '_type'). Attempting to delete and recreate.")
                try:
                    # 尝试删除损坏的集合
                    self.client.delete_collection(collection_name)
                    logger.info(f"Deleted corrupted collection '{collection_name}'.")
                except Exception as delete_error:
                    logger.warning(f"Could not delete corrupted collection: {delete_error}")
                
                # 重新创建集合
                try:
                    self.client.create_collection(
                        name=collection_name,
                        embedding_function=self.custom_embedding_function,
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info(f"Successfully recreated collection '{collection_name}'.")
                except Exception as recreate_error:
                    logger.error(f"Failed to recreate collection '{collection_name}': {recreate_error}")
                    raise
            else:
                logger.exception(f"KeyError in collection '{collection_name}': {e}")
                raise
        except Exception as e:
            logger.exception(f"Failed to get or create collection '{collection_name}'.")
            raise

    def add(self, collection_name: str, documents: List[str], metadata: List[Dict[str, Any]], ids: List[str]):
        """
        向指定集合中添加文档（批量）。
        增强错误处理和内存管理。
        """
        if not documents:
            logger.info("No documents to add. Skipping.")
            return
        
        # 验证输入参数
        if len(documents) != len(metadata) or len(documents) != len(ids):
            raise ValueError("documents, metadata, and ids must have the same length")
        
        # 分批处理大量数据以避免内存问题
        batch_size = 50  # 减小批次大小
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    collection = self.client.get_collection(
                        name=collection_name,
                        embedding_function=self.custom_embedding_function
                    )
                    collection.add(
                        documents=batch_docs,
                        metadatas=batch_metadata,
                        ids=batch_ids
                    )
                    logger.info(f"Added batch {batch_num}/{total_batches} ({len(batch_ids)} items) to collection '{collection_name}'.")
                    break  # 成功则跳出重试循环
                    
                except ValueError as ve:
                    if "embedding" in str(ve).lower():
                        logger.error(f"Embedding service failed for batch {batch_num}: {ve}")
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying batch {batch_num} (attempt {attempt + 2}/{max_retries})...")
                            import time
                            time.sleep(2 ** attempt)  # 指数退避
                            continue
                        else:
                            logger.error(f"Failed to process batch {batch_num} after {max_retries} attempts. Skipping this batch.")
                            break
                    else:
                        raise  # 重新抛出非嵌入相关的ValueError
                        
                except Exception as e:
                    logger.error(f"Failed to add batch {batch_num} to collection '{collection_name}': {e}", exc_info=True)
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying batch {batch_num} (attempt {attempt + 2}/{max_retries})...")
                        import time
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        logger.error(f"Failed to process batch {batch_num} after {max_retries} attempts. Skipping this batch.")
                        break

    def search(
            self,
            collection_name: str,
            query_texts: List[str],
            top_k: int,
            filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        (健壮版) 在指定集合中搜索与查询文本最相似的文档。
        返回一个列表的列表，对应每个查询文本。
        """
        if not query_texts:
            return []

        try:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.custom_embedding_function
            )

            # 这个调用会触发 CustomEmbeddingFunction
            results = collection.query(
                query_texts=query_texts,
                n_results=top_k,
                where=filter_dict,
                include=['metadatas', 'distances']
            )

            # ChromaDB 0.4.x+ 的返回格式是字典，包含与查询文本数量相等的列表
            all_formatted_results = []
            if not results or 'ids' not in results or not results['ids']:
                return [[] for _ in query_texts]

            num_queries = len(query_texts)
            for i in range(num_queries):
                query_results = []
                if not results['ids'][i]:
                    all_formatted_results.append(query_results)
                    continue

                ids = results['ids'][i]
                metadatas = results['metadatas'][i]
                distances = results['distances'][i]

                for j in range(len(ids)):
                    query_results.append({
                        "id": ids[j],
                        "metadata": metadatas[j],
                        "score": 1.0 - distances[j] if distances[j] is not None else 0.0
                    })
                all_formatted_results.append(query_results)

            return all_formatted_results

        except Exception as e:
            # 捕获所有异常，包括从 CustomEmbeddingFunction 抛出的 ValueError
            logger.error(f"Error querying collection '{collection_name}': {e}", exc_info=True)
            # 在任何错误情况下，都返回格式正确的空结果，防止下游代码崩溃
            return [[] for _ in query_texts]

    def delete_collection(self, collection_name: str):
        """删除一个集合，主要用于测试和清理。"""
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Collection '{collection_name}' has been deleted.")
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")

    def reset(self):
        """重置整个ChromaDB客户端，删除所有数据。"""
        try:
            self.client.reset()
            logger.info("ChromaDB client has been reset.")
        except Exception as e:
            logger.error(f"Failed to reset ChromaDB client: {e}")

if __name__ == "__main__":
    embdding_client = EmbeddingClient(api_base_url=config['embedding']['api_url'])
    # 创建一个示例向量数据库客户端
    vector_db_client = VectorDBClient(
        path=config.get('paths', {}).get('vector_db_path', ''),
        embedding_client=embdding_client
    )
    vector_db_client.delete_collection("relations")
    vector_db_client.delete_collection("entities")