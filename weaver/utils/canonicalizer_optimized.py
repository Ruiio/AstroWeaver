#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能优化版批内规范化器
主要优化：
1. 并发处理LLM调用
2. 智能缓存机制
3. 批量处理优化
4. 内存管理改进
"""

import asyncio
import json
import logging
import sys
import argparse
import gc
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Set, Optional
from collections import defaultdict
import hashlib

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from weaver.utils.config import load_config
from weaver.models.llm_models import LLMClient, create_llm_client, get_model_name
from weaver.models.embedding_models import EmbeddingClient
from weaver.storage.vector_db import VectorDBClient
from weaver.core.canonicalization import _normalize_term, to_camel_case, to_pascal_case, _llm_judge_synonym_async

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizedCanonicalization:
    """性能优化版规范化器"""
    
    def __init__(self, config_path: str = 'configs/config.yaml'):
        self.config = load_config(config_path)
        self.clients = {}
        self.similarity_threshold = 0.7
        self.top_k = 5
        self.relation_collection = "relations"
        self.entity_collection = "entities"
        
        # 性能优化配置
        self.max_concurrent_llm_calls = 5  # 最大并发LLM调用数
        self.batch_size = 20  # 批处理大小
        
        # 缓存机制
        self.llm_cache = {}  # LLM判断结果缓存
        self.similarity_cache = {}  # 相似度判断缓存
        
        logger.info("初始化性能优化版规范化器")
    
    def initialize_clients(self) -> bool:
        """初始化所有客户端"""
        try:
            # 禁用ChromaDB遥测功能
            import os
            os.environ['ANONYMIZED_TELEMETRY'] = 'False'
            
            # 初始化LLM客户端
            try:
                self.clients['llm'] = create_llm_client(self.config)
                logger.info(f"LLM客户端初始化成功，使用提供商: {self.config.get('llm', {}).get('provider', 'ali')}")
            except Exception as e:
                logger.error(f"LLM客户端初始化失败: {e}")
                return False
            
            # 初始化嵌入客户端
            embedding_config = self.config.get('embedding', {})
            api_url = embedding_config.get('api_url')
            if api_url:
                self.clients['embedding'] = EmbeddingClient(api_base_url=api_url)
            
            # 初始化向量数据库客户端
            vector_db_path = self.config.get('vector_db', {}).get('persist_directory', './data/vectordb')
            self.clients['vector_db'] = VectorDBClient(
                path=vector_db_path,
                embedding_client=self.clients['embedding']
            )
            
            logger.info("所有客户端初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"客户端初始化失败: {e}")
            return False
    
    def load_triples_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """从JSONL文件加载三元组"""
        try:
            triples = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            triple = json.loads(line)
                            triples.append(triple)
                        except json.JSONDecodeError as e:
                            logger.warning(f"第{line_num}行JSON解析失败: {e}")
                            continue
            
            logger.info(f"从文件加载三元组: {file_path}")
            logger.info(f"成功加载 {len(triples)} 个三元组")
            return triples
            
        except Exception as e:
            logger.error(f"加载三元组文件失败: {e}")
            return []
    
    def _get_cache_key(self, term_type: str, new_term: str, candidates: List[str]) -> str:
        """生成缓存键"""
        candidates_str = '|'.join(sorted(candidates))
        content = f"{term_type}:{new_term}:{candidates_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _cached_llm_judge(self, term_type: str, new_term: str, candidates: List[str]) -> Optional[str]:
        """带缓存的LLM判断"""
        if not candidates:
            return None
            
        cache_key = self._get_cache_key(term_type, new_term, candidates)
        
        # 检查缓存
        if cache_key in self.llm_cache:
            logger.debug(f"缓存命中: {new_term} -> {self.llm_cache[cache_key]}")
            return self.llm_cache[cache_key]
        
        # 调用LLM
        llm_model = get_model_name(self.config, 'judge_model')
        result = await _llm_judge_synonym_async(
            term_type, new_term, candidates, self.clients['llm'], llm_model
        )
        
        # 缓存结果
        self.llm_cache[cache_key] = result
        return result
    
    async def _normalize_term_optimized(
        self,
        term_type: str,
        term: str,
        collection_name: str,
        staged_canonical_names: Set[str],
        staged_term_mappings: Dict[str, str],
        semaphore: asyncio.Semaphore
    ) -> tuple[str, str]:
        """优化版术语规范化"""
        async with semaphore:  # 限制并发数
            try:
                # 1. 检查批内映射缓存
                if term in staged_term_mappings:
                    return term, staged_term_mappings[term]
                
                # 2. 向量搜索数据库中的候选项
                try:
                    # 确保集合存在
                    self.clients['vector_db'].create_collection_if_not_exists(collection_name)
                    search_results = self.clients['vector_db'].search(collection_name, [term], self.top_k)
                    db_candidates = [res['metadata']['canonical_name'] for res in search_results[0] if
                                   res['score'] >= self.similarity_threshold and 
                                   res['metadata']['canonical_name'] is not None and 
                                   isinstance(res['metadata']['canonical_name'], str)]
                except Exception as e:
                    logger.warning(f"VectorDB search failed for term '{term}': {e}")
                    db_candidates = []
                
                # 3. 检查批内相似术语
                batch_canonical = None
                if staged_term_mappings:
                    staged_canonical_list = [v for v in set(staged_term_mappings.values()) if v is not None and isinstance(v, str)]
                    if staged_canonical_list:
                        batch_canonical = await self._cached_llm_judge(
                            term_type, term, staged_canonical_list
                        )
                        if batch_canonical:
                            logger.info(f"Term '{term}' merged with batch canonical '{batch_canonical}'")
                            staged_term_mappings[term] = batch_canonical
                            return term, batch_canonical
                
                # 4. 检查数据库候选项
                canonical_name = None
                if db_candidates:
                    canonical_name = await self._cached_llm_judge(term_type, term, db_candidates)
                
                # 5. 生成新规范名 - 保持原始字符串
                if not canonical_name:
                    # 直接使用原始术语作为规范名称，不进行任何转换
                    canonical_name = term
                    if canonical_name not in staged_canonical_names:
                        staged_canonical_names.add(canonical_name)
                        logger.info(f"Staging new canonical {term_type}: '{canonical_name}' (original term preserved)")
                
                # 6. 更新映射
                staged_term_mappings[term] = canonical_name
                return term, canonical_name
                
            except Exception as e:
                logger.error(f"规范化术语 '{term}' 失败: {e}")
                # 后备方案 - 保持原始字符串
                fallback_canonical = term
                staged_term_mappings[term] = fallback_canonical
                return term, fallback_canonical
    
    async def normalize_terms_batch_optimized(
        self, 
        terms: List[str], 
        term_type: str, 
        collection_name: str
    ) -> Dict[str, str]:
        """优化版批量术语规范化"""
        term_map = {}
        staged_canonical_names = set()
        staged_term_mappings = {}
        
        # 创建信号量限制并发数
        semaphore = asyncio.Semaphore(self.max_concurrent_llm_calls)
        
        logger.info(f"开始优化版批量规范化 {len(terms)} 个{term_type} (并发数: {self.max_concurrent_llm_calls})")
        start_time = time.time()
        
        # 分批处理以控制内存使用
        for i in range(0, len(terms), self.batch_size):
            batch_terms = terms[i:i + self.batch_size]
            
            # 并发处理当前批次
            tasks = [
                self._normalize_term_optimized(
                    term_type, term, collection_name, 
                    staged_canonical_names, staged_term_mappings, semaphore
                )
                for term in batch_terms
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"批处理中出现异常: {result}")
                    continue
                
                original_term, canonical_name = result
                term_map[original_term] = canonical_name
            
            # 进度报告和内存清理
            processed = min(i + self.batch_size, len(terms))
            logger.info(f"已处理 {processed}/{len(terms)} 个{term_type}")
            
            if processed % (self.batch_size * 2) == 0:
                gc.collect()
        
        # 批量添加新项到向量数据库
        await self._batch_add_new_items(collection_name, staged_canonical_names, staged_term_mappings)
        
        # 统计和报告
        elapsed_time = time.time() - start_time
        unique_canonicals = len(set(term_map.values()))
        merge_count = len(terms) - unique_canonicals
        
        logger.info(f"{term_type}规范化完成: {len(term_map)} 个映射 -> {unique_canonicals} 个规范名")
        logger.info(f"批内合并: {merge_count} 个术语, 耗时: {elapsed_time:.2f}秒")
        logger.info(f"平均速度: {len(terms)/elapsed_time:.2f} 术语/秒")
        logger.info(f"缓存命中率: {len(self.llm_cache)} 个缓存条目")
        
        return term_map
    
    async def _batch_add_new_items(
        self, 
        collection_name: str, 
        staged_canonical_names: Set[str], 
        staged_term_mappings: Dict[str, str]
    ):
        """批量添加新项到向量数据库"""
        try:
            new_items = []
            seen_ids = set()
            
            for term, canonical_name in staged_term_mappings.items():
                if canonical_name in staged_canonical_names and term is not None and canonical_name is not None:
                    # 生成唯一ID，避免重复
                    unique_id = f"{canonical_name}_{hashlib.md5(str(term).encode()).hexdigest()[:8]}"
                    
                    if unique_id not in seen_ids:
                        seen_ids.add(unique_id)
                        new_items.append({
                            "document": str(term),
                            "metadata": {"canonical_name": str(canonical_name), "original_term": str(term)},
                            "id": unique_id
                        })
            
            if new_items:
                # 这里可以进一步优化为真正的批量插入
                new_items_dict = {collection_name: new_items}
                await self.batch_add_to_vector_db(new_items_dict)
                logger.info(f"批量添加 {len(new_items)} 个新项到向量数据库")
                
        except Exception as e:
            logger.error(f"批量添加到向量数据库失败: {e}")
    
    async def batch_add_to_vector_db(self, new_items_to_add: Dict[str, List[Dict]]):
        """批量添加项目到向量数据库"""
        try:
            for collection_name, items in new_items_to_add.items():
                if items:
                    # 确保集合存在
                    self.clients['vector_db'].create_collection_if_not_exists(collection_name)
                    
                    documents = [item['document'] for item in items]
                    metadatas = [item['metadata'] for item in items]
                    ids = [item['id'] for item in items]
                    
                    self.clients['vector_db'].add(
                        collection_name=collection_name,
                        documents=documents,
                        metadata=metadatas,
                        ids=ids
                    )
                    
                    logger.info(f"向 {collection_name} 集合添加了 {len(items)} 个新项")
                    
        except Exception as e:
            logger.error(f"批量添加到向量数据库失败: {e}")
    
    async def canonicalize_triples_batch(self, triples: List[Dict[str, Any]]) -> tuple:
        """优化版批量规范化三元组"""
        try:
            # 提取所有唯一的关系和实体，处理列表类型的字段
            all_relations = set()
            all_entities = set()
            
            for t in triples:
                # 处理predicate（通常是字符串）
                predicate = t['predicate']
                if isinstance(predicate, list):
                    all_relations.update(predicate)
                else:
                    all_relations.add(predicate)
                
                # 处理subject
                subject = t['subject']
                if isinstance(subject, list):
                    all_entities.update(subject)
                else:
                    all_entities.add(subject)
                
                # 处理object
                obj = t['object']
                if isinstance(obj, list):
                    all_entities.update(obj)
                else:
                    all_entities.add(obj)
            
            all_relations = list(all_relations)
            all_entities = list(all_entities)
            
            logger.info(f"开始优化版规范化: {len(all_relations)} 个关系, {len(all_entities)} 个实体")
            
            # 并发规范化关系和实体
            relation_task = self.normalize_terms_batch_optimized(
                all_relations, 'relation', self.relation_collection
            )
            entity_task = self.normalize_terms_batch_optimized(
                all_entities, 'entity', self.entity_collection
            )
            
            relation_map, entity_map = await asyncio.gather(relation_task, entity_task)
            
            # 应用映射到三元组，处理列表类型的字段
            canonicalized_triples = []
            for triple in triples:
                # 处理subject
                subject = triple['subject']
                if isinstance(subject, list):
                    canonicalized_subject = [entity_map.get(s, s) for s in subject]
                else:
                    canonicalized_subject = entity_map.get(subject, subject)
                
                # 处理predicate
                predicate = triple['predicate']
                if isinstance(predicate, list):
                    canonicalized_predicate = [relation_map.get(p, p) for p in predicate]
                else:
                    canonicalized_predicate = relation_map.get(predicate, predicate)
                
                # 处理object
                obj = triple['object']
                if isinstance(obj, list):
                    canonicalized_object = [entity_map.get(o, o) for o in obj]
                else:
                    canonicalized_object = entity_map.get(obj, obj)
                
                canonicalized_triple = {
                    'subject': canonicalized_subject,
                    'predicate': canonicalized_predicate,
                    'object': canonicalized_object
                }
                
                # 保留其他字段
                for key, value in triple.items():
                    if key not in ['subject', 'predicate', 'object']:
                        canonicalized_triple[key] = value
                
                canonicalized_triples.append(canonicalized_triple)
            
            logger.info(f"三元组规范化完成: {len(canonicalized_triples)} 个三元组")
            return canonicalized_triples, relation_map, entity_map
            
        except Exception as e:
            logger.error(f"批量规范化三元组失败: {e}")
            return [], {}, {}
    
    def save_canonicalization_results(
        self, 
        canonicalized_triples: List[Dict[str, Any]], 
        relation_map: Dict[str, str], 
        entity_map: Dict[str, str], 
        input_file: str
    ) -> None:
        """保存规范化结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存规范化后的三元组
            output_file = f"data/output/canonicalized_triples_optimized_{timestamp}.jsonl"
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for triple in canonicalized_triples:
                    f.write(json.dumps(triple, ensure_ascii=False) + '\n')
            
            # 保存映射关系
            mapping_file = f"data/output/canonicalization_mappings_optimized_{timestamp}.json"
            mapping_data = {
                'input_file': input_file,
                'timestamp': timestamp,
                'statistics': {
                    'total_triples': len(canonicalized_triples),
                    'total_relations': len(relation_map),
                    'unique_canonical_relations': len(set(relation_map.values())),
                    'total_entities': len(entity_map),
                    'unique_canonical_entities': len(set(entity_map.values())),
                    'relation_merge_rate': 1 - len(set(relation_map.values())) / len(relation_map) if relation_map else 0,
                    'entity_merge_rate': 1 - len(set(entity_map.values())) / len(entity_map) if entity_map else 0,
                    'cache_entries': len(self.llm_cache)
                },
                'relation_mappings': relation_map,
                'entity_mappings': entity_map
            }
            
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(mapping_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"规范化结果已保存:")
            logger.info(f"  三元组文件: {output_file}")
            logger.info(f"  映射文件: {mapping_file}")
            
        except Exception as e:
            logger.error(f"保存规范化结果失败: {e}")
    
    async def run_canonicalization(self, input_file: str) -> bool:
        """运行优化版规范化流程"""
        try:
            logger.info("🚀 开始优化版规范化流程")
            start_time = time.time()
            
            # 1. 初始化客户端
            if not self.initialize_clients():
                logger.error("客户端初始化失败，规范化终止")
                return False
            
            # 2. 加载三元组
            triples = self.load_triples_from_file(input_file)
            if not triples:
                logger.error("未能加载任何三元组，规范化终止")
                return False
            
            # 3. 执行优化版规范化
            canonicalized_triples, relation_map, entity_map = await self.canonicalize_triples_batch(triples)
            
            if not canonicalized_triples:
                logger.error("规范化失败，未生成任何结果")
                return False
            
            # 4. 保存结果
            self.save_canonicalization_results(canonicalized_triples, relation_map, entity_map, input_file)
            
            # 5. 性能报告
            total_time = time.time() - start_time
            logger.info(f"✅ 优化版规范化完成，总耗时: {total_time:.2f}秒")
            logger.info(f"处理速度: {len(triples)/total_time:.2f} 三元组/秒")
            
            return True
            
        except Exception as e:
            logger.error(f"规范化流程失败: {e}")
            return False
        finally:
            # 清理资源
            self.llm_cache.clear()
            gc.collect()

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='性能优化版独立规范化器')
    parser.add_argument('input_file', help='输入的JSONL三元组文件路径')
    parser.add_argument('--config', default='configs/config.yaml', help='配置文件路径')
    parser.add_argument('--concurrent', type=int, default=5, help='最大并发LLM调用数')
    parser.add_argument('--batch-size', type=int, default=20, help='批处理大小')
    
    args = parser.parse_args()
    
    # 创建规范化器实例
    canonicalizer = OptimizedCanonicalization(args.config)
    canonicalizer.max_concurrent_llm_calls = args.concurrent
    canonicalizer.batch_size = args.batch_size
    
    # 运行规范化
    success = await canonicalizer.run_canonicalization(args.input_file)
    
    if success:
        logger.info("🎉 规范化任务成功完成")
        sys.exit(0)
    else:
        logger.error("❌ 规范化任务失败")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())