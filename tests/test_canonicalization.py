# tests/test_canonicalization.py

import unittest
import asyncio
import os
import logging
import shutil
from pathlib import Path
import gc

# 假设你的项目结构是 weaver/weaver/core/...
# 如果你的根目录是 weaver/，那么导入路径应该是 from weaver.models...
from weaver.models.llm_models import LLMClient
from weaver.models.embedding_models import EmbeddingClient
from weaver.storage.vector_db import VectorDBClient
from weaver.core.canonicalization import canonicalize_graph
from weaver.utils.config import config

logging.basicConfig(level=logging.INFO)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

API_KEY = os.environ.get("DASHSCOPE_API_KEY") or config.get('api_keys', {}).get('llm')
EMBEDDING_API_URL = config.get('embedding', {}).get('api_url')
VECTOR_DB_PATH = config.get('paths', {}).get('vector_db_path')


@unittest.skipIf(not all([API_KEY, EMBEDDING_API_URL, VECTOR_DB_PATH]),
                 "Skipping integration test: Required configs not set.")
class TestCanonicalizationIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.info("Setting up clients for canonicalization integration test...")
        cls.llm_client = LLMClient(api_key=API_KEY, base_url=config['llm']['base_url'])
        cls.embedding_client = EmbeddingClient(api_base_url=EMBEDDING_API_URL)
        cls.test_db_path = Path(f"{VECTOR_DB_PATH}_test")
        if cls.test_db_path.exists():
            shutil.rmtree(cls.test_db_path)
        cls.test_db_path.mkdir(parents=True)
        cls.vector_db_client = VectorDBClient(path=str(cls.test_db_path), embedding_client=cls.embedding_client)
        cls.rel_collection = "test_canonical_relations"
        cls.ent_collection = "test_canonical_entities"
        cls.vector_db_client.create_collection_if_not_exists(cls.rel_collection)
        cls.vector_db_client.create_collection_if_not_exists(cls.ent_collection)
        cls.preload_db()

    @classmethod
    def tearDownClass(cls):
        logging.info("Tearing down test environment...")
        if hasattr(cls, 'vector_db_client') and cls.vector_db_client:
            cls.vector_db_client = None
            gc.collect()
        if hasattr(cls, 'test_db_path') and cls.test_db_path.exists():
            import time
            time.sleep(0.5)
            try:
                shutil.rmtree(cls.test_db_path)
                logging.info(f"Cleaned up test database at: {cls.test_db_path}")
            except PermissionError as e:
                logging.warning(f"Could not remove test database directory due to a permission error. "
                                f"This is common on Windows and can often be ignored. Error: {e}")
            except Exception as e:
                logging.error(f"An unexpected error occurred during teardown: {e}")

    @classmethod
    def preload_db(cls):
        logging.info("Preloading test vector database...")
        cls.vector_db_client.add(
            collection_name=cls.rel_collection,
            documents=["orbits"],
            metadata=[{"canonical_name": "Orbits", "original_term": "orbits"}],
            ids=["Orbits"]
        )
        cls.vector_db_client.add(
            collection_name=cls.ent_collection,
            documents=["Sun"],
            metadata=[{"canonical_name": "Sun", "original_term": "Sun"}],
            ids=["Sun"]
        )
        logging.info("Test database preloaded.")

    async def _run_canonicalize_graph_live_test(self):
        """
        这是一个包含核心测试逻辑的异步辅助函数。
        现在它会验证图谱和映射关系。
        """
        # 1. 准备输入数据
        raw_relations = {
            "revolves around": {"Sol"},
            "has moon": {"Phobos"}
        }

        # 2. 调用被测试的核心函数，现在它返回三个值
        final_graph, relation_map, entity_map = await canonicalize_graph(
            raw_relations=raw_relations,
            vector_db_client=self.vector_db_client,
            llm_client=self.llm_client,
            config=config,
            relation_collection=self.rel_collection,
            entity_collection=self.ent_collection
        )

        # 3. 定义预期结果
        # 3.1 预期图谱
        expected_graph = {
            "Orbits": {"Sun"},
            "Has Moon": {"Phobos"}
        }

        # 3.2 预期关系映射
        expected_relation_map = {
            "revolves around": "Orbits",
            "has moon": "Has Moon"  # 它被创建为新的规范关系
        }

        # 3.3 预期实体映射
        expected_entity_map = {
            "Sol": "Sun",
            "Phobos": "Phobos"  # 它被创建为新的规范实体
        }

        logging.info(f"Final canonicalized graph from test: {final_graph}")
        logging.info(f"Final relation map from test: {relation_map}")
        logging.info(f"Final entity map from test: {entity_map}")

        # 4. 断言
        # 4.1 验证图谱
        actual_graph_sets = {k: set(v) for k, v in final_graph.items()}
        self.assertEqual(actual_graph_sets, expected_graph)

        # 4.2 验证关系映射
        self.assertEqual(relation_map, expected_relation_map)

        # 4.3 验证实体映射
        self.assertEqual(entity_map, expected_entity_map)

    def test_canonicalization_live(self):
        """
        这是unittest会发现并运行的测试方法。
        """
        logging.info("--- Running test_canonicalization_live ---")
        asyncio.run(self._run_canonicalize_graph_live_test())


if __name__ == '__main__':
    # 确保你的项目根目录在Python路径中，以便可以找到 'weaver' 包
    # 如果直接从 tests/ 目录运行，可能需要调整路径
    # import sys
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    unittest.main()