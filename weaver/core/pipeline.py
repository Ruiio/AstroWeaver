# astroWeaver/core/pipeline.py

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from data_sources.simbad_client import SimbadClient
from data_sources.wikipedia_client import WikipediaClient
from .extraction import extract_relations_from_sections
from .canonicalization import canonicalize_graph
from ..models.llm_models import LLMClient
from ..models.embedding_models import EmbeddingClient
from ..storage.vector_db import VectorDBClient
from ..storage.file_handler import FileHandler

logger = logging.getLogger(__name__)


class EntityProcessor:
    """封装处理单个实体的所有逻辑和所需客户端。"""

    def __init__(self, config: Dict[str, Any], clients: Dict[str, Any]):
        self.config = config
        self.wiki_client = clients['wiki']
        self.simbad_client = clients['simbad']
        self.llm_client = clients['llm']
        # self.embedding_client 不再需要直接被持有，因为它被注入到了VectorDBClient中
        self.vector_db_client = clients['vector_db']
        self.file_handler = clients['file_handler']

    async def process_entity(self, entity_name: str):
        """处理单个实体的完整异步流程。"""
        logger.info(f"Starting processing for entity: {entity_name}")
        try:
            # 1. 数据获取
            sections = self.wiki_client.get_article_sections(entity_name)
            infobox = self.wiki_client.get_infobox(entity_name)
            simbad_data = self.simbad_client.get_data(entity_name)

            if not sections:
                logger.warning(f"No sections found for '{entity_name}'. Aborting.")
                return {"entity": entity_name, "status": "failed", "reason": "No Wikipedia content"}

            # 2. 初步抽取
            raw_relations = extract_relations_from_sections(
                entity_name, sections, self.llm_client, self.config['llm']['extraction_model']
            )
            relation_map, entity_map = {}, {}

            if raw_relations:
                final_graph, relation_map, entity_map = await canonicalize_graph(
                    raw_relations, self.vector_db_client, self.llm_client, self.config
                )
            else:
                final_graph = {}

            # 3. 规范化
            if raw_relations:
                # --- 修正2: 移除多余的 embedding_client 参数 ---
                final_graph = await canonicalize_graph(
                    raw_relations, self.vector_db_client, self.llm_client, self.config
                )
            else:
                final_graph = {}

            # 4. 整合与保存
            final_data = {
                "entity_name": entity_name,
                "infobox_attributes": infobox,
                "consolidated_relationships": final_graph
            }

            self.file_handler.save_entity_data(entity_name, final_data)
            if simbad_data:
                self.file_handler.save_simbad_data(entity_name, simbad_data)

            logger.info(f"Successfully processed and saved data for entity: {entity_name}")
            return {
                "entity": entity_name,
                "status": "success",
                "relation_map": relation_map,
                "entity_map": entity_map
            }
        except Exception as e:
            logger.exception(f"An unhandled error occurred while processing '{entity_name}'.")
            return {"entity": entity_name, "status": "failed", "reason": str(e)}


def run_pipeline(config: Dict[str, Any]):
    """
    运行整个批处理管道。
    """
    # 1. 初始化所有客户端
    # 将embedding_client的初始化提前，因为它被vector_db_client依赖
    embedding_client = EmbeddingClient(api_base_url=config['embedding']['api_url'])

    clients = {
        'wiki': WikipediaClient(),
        'simbad': SimbadClient(),
        'llm': LLMClient(api_key=config['api_keys']['llm'], base_url=config['llm']['base_url']),
        'embedding': embedding_client,  # 仍然可以保留它，以备将来使用
        # --- 修正3: 使用正确的参数初始化VectorDBClient ---
        'vector_db': VectorDBClient(
            path=config['paths']['vector_db_path'],
            embedding_client=embedding_client
        ),
        'file_handler': FileHandler(output_dir=config['paths']['output_dir'])
    }

    # 确保向量数据库集合存在
    clients['vector_db'].create_collection_if_not_exists("canonical_relations")
    clients['vector_db'].create_collection_if_not_exists("canonical_entities")

    # 2. 加载待处理实体列表
    entity_list = clients['file_handler'].load_entity_list(config['paths']['input_file'])
    if not entity_list:
        logger.error("Entity list is empty. Exiting.")
        return

    logger.info(f"Loaded {len(entity_list)} entities to process.")

    # 3. 使用线程池并行处理实体
    processor = EntityProcessor(config, clients)


    # --- 修正4: 简化异步执行逻辑 ---
    # ThreadPoolExecutor 和 asyncio.run() 的嵌套可能会导致问题，
    # 特别是当底层的库（如requests）不是真正的异步时。
    # 一个更简单、更健壮的模式是使用asyncio.gather来并发运行异步任务。
    async def main_task():
        # 创建一个任务列表
        tasks = [processor.process_entity(entity) for entity in entity_list]

        # 使用asyncio.gather并发运行所有任务
        # 可以通过一个信号量来限制并发数量，防止瞬间过载API
        semaphore = asyncio.Semaphore(config['pipeline']['num_workers'])

        async def run_with_semaphore(task):
            async with semaphore:
                return await task

        # 将带信号量的任务包装起来
        sem_tasks = [run_with_semaphore(task) for task in tasks]

        results = await asyncio.gather(*sem_tasks, return_exceptions=True)
        return results

    results = asyncio.run(main_task())



    # 4. 报告结果
    # 检查结果中是否有异常
    final_relation_map = {}
    final_entity_map = {}
    processed_results = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            entity = entity_list[i]
            logger.error(f"Task for entity '{entity}' failed with an exception: {res}")
            processed_results.append({"entity": entity, "status": "failed", "reason": str(res)})
        else:
            processed_results.append(res)
            if res and res.get('status') == 'success':
                final_relation_map.update(res.get('relation_map', {}))
                final_entity_map.update(res.get('entity_map', {}))
    canonical_relation_dict = {}
    for original, canonical in final_relation_map.items():
        if canonical not in canonical_relation_dict:
            canonical_relation_dict[canonical] = []
        if original != canonical:  # 只记录被合并的
            canonical_relation_dict[canonical].append(original)

    canonical_entity_dict = {}
    for original, canonical in final_entity_map.items():
        if canonical not in canonical_entity_dict:
            canonical_entity_dict[canonical] = []
        if original != canonical:
            canonical_entity_dict[canonical].append(original)

    # 使用 FileHandler 保存
    file_handler = clients['file_handler']
    output_path = file_handler.base_output_path / "canonical_maps"
    output_path.mkdir(exist_ok=True)

    file_handler.save_json(output_path / "relation_map.json", canonical_relation_dict)
    file_handler.save_json(output_path / "entity_map.json", canonical_entity_dict)
    logger.info(f"Saved final canonicalization maps to: {output_path}")
    success_count = sum(1 for r in processed_results if r and r.get('status') == 'success')
    failed_count = len(processed_results) - success_count
    logger.info("=" * 50)
    logger.info("Pipeline finished.")
    logger.info(f"Total entities attempted: {len(processed_results)}")
    logger.info(f"  - Success: {success_count}")
    logger.info(f"  - Failed: {failed_count}")
    logger.info("=" * 50)

    if failed_count > 0:
        logger.warning("Failed entities:")
        for r in processed_results:
            if r and r.get('status') == 'failed':
                logger.warning(f"  - {r.get('entity', 'Unknown')}: {r.get('reason', 'No reason provided')}")