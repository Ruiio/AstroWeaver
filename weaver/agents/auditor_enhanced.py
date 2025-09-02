#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版知识审核器
整合了批量审核、优化版规范化、中间结果保存等功能
"""

import asyncio
import json
import logging
import gc
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import defaultdict

from weaver.models.llm_models import LLMClient
from weaver.storage.vector_db import VectorDBClient
from weaver.core.canonicalization import _llm_judge_synonym_async
from weaver.agents.extractor import ExtractedTriple

logger = logging.getLogger(__name__)

class EnhancedKnowledgeAuditor:
    """增强版知识审核器"""
    
    def __init__(self, llm_client: LLMClient, vector_db_client: VectorDBClient, config: Dict[str, Any]):
        self.llm_client = llm_client
        self.vector_db_client = vector_db_client
        self.config = config
        
        # 配置参数
        vdb_config = config.get('vector_db', {})
        self.relation_collection = vdb_config.get('relation_collection', 'relations')
        self.entity_collection = vdb_config.get('entity_collection', 'entities')
        self.similarity_threshold = vdb_config.get('similarity_threshold', 0.7)
        self.top_k = vdb_config.get('top_k', 5)
        
        # 性能优化配置
        self.max_concurrent_llm_calls = config.get('performance', {}).get('max_concurrent_llm_calls', 5)
        self.batch_size = config.get('performance', {}).get('batch_size', 20)
        
        # 缓存机制
        self.llm_cache = {}  # LLM判断结果缓存
        self.similarity_cache = {}  # 相似度判断缓存
        
        # 中间结果存储
        self.intermediate_results = {
            'low_confidence_triples': [],
            'high_confidence_triples': [],
            'entity_mappings': {},
            'relation_mappings': {},
            'processing_stats': {}
        }
        
        logger.info("初始化增强版知识审核器")
    
    def save_intermediate_results(self, output_dir: str, stage: str) -> str:
        """保存中间结果"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{stage}_results_{timestamp}.json"
            filepath = output_path / filename
            
            # 添加时间戳和阶段信息
            results_with_metadata = {
                'timestamp': timestamp,
                'stage': stage,
                'results': self.intermediate_results
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_with_metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"中间结果已保存: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"保存中间结果失败: {e}")
            return ""
    
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
        llm_model = self.config.get('llm', {}).get('judge_model', 'deepseek-v3')
        result = await _llm_judge_synonym_async(
            term_type, new_term, candidates, self.llm_client, llm_model
        )
        
        # 缓存结果
        self.llm_cache[cache_key] = result
        return result
    
    async def _normalize_term_enhanced(
        self,
        term_type: str,
        term: str,
        collection_name: str,
        staged_canonical_names: Set[str],
        staged_term_mappings: Dict[str, str],
        semaphore: asyncio.Semaphore
    ) -> Tuple[str, str]:
        """增强版术语规范化"""
        async with semaphore:  # 限制并发数
            try:
                # 1. 检查批内映射缓存
                if term in staged_term_mappings:
                    return term, staged_term_mappings[term]
                
                # 2. 向量搜索数据库中的候选项
                try:
                    # 确保集合存在
                    self.vector_db_client.create_collection_if_not_exists(collection_name)
                    search_results = self.vector_db_client.search(collection_name, [term], self.top_k)
                    db_candidates = [res['metadata']['canonical_name'] for res in search_results[0] if
                                   res['score'] >= self.similarity_threshold]
                except Exception as e:
                    logger.warning(f"VectorDB search failed for term '{term}': {e}")
                    db_candidates = []
                
                # 3. 检查批内相似术语
                batch_canonical = None
                if staged_term_mappings:
                    staged_canonical_list = list(set(staged_term_mappings.values()))
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
    
    async def _batch_add_new_items(self, new_items: List[Dict[str, Any]], collection_name: str):
        """批量添加新项到向量数据库"""
        if not new_items:
            return
        
        try:
            documents = [item['original_term'] for item in new_items]
            metadatas = [{
                'canonical_name': item['canonical_name'],
                'original_term': item['original_term']
            } for item in new_items]
            
            # 为每个项目生成唯一ID
            ids = []
            for item in new_items:
                # 使用原始术语的哈希值生成唯一ID
                term_hash = hashlib.md5(item['original_term'].encode()).hexdigest()[:8]
                unique_id = f"{item['canonical_name']}_{term_hash}"
                ids.append(unique_id)
            
            self.vector_db_client.add(
                collection_name=collection_name,
                documents=documents,
                metadata=metadatas,
                ids=ids
            )
            
            logger.info(f"批量添加 {len(new_items)} 个新项到 {collection_name}")
            
        except Exception as e:
            logger.error(f"批量添加新项失败: {e}")
    
    async def audit_and_normalize_triples_enhanced(
        self, 
        triples: List[ExtractedTriple],
        output_dir: str = "./results/intermediate"
    ) -> Tuple[List[Dict[str, Any]], List[ExtractedTriple]]:
        """增强版三元组审核和规范化"""
        start_time = time.time()
        logger.info(f"开始增强版三元组审核和规范化，共 {len(triples)} 个三元组")
        
        # 1. 置信度过滤
        confidence_threshold = self.config.get('audit', {}).get('confidence_threshold', 0.7)
        high_confidence_triples = [t for t in triples if t.get('confidence', 0) >= confidence_threshold]
        low_confidence_triples = [t for t in triples if t.get('confidence', 0) < confidence_threshold]
        
        logger.info(f"高置信度三元组: {len(high_confidence_triples)}, 低置信度三元组: {len(low_confidence_triples)}")
        
        # 保存置信度过滤结果
        self.intermediate_results['low_confidence_triples'] = low_confidence_triples
        self.intermediate_results['high_confidence_triples'] = high_confidence_triples
        self.save_intermediate_results(output_dir, "confidence_filtering")
        
        if not high_confidence_triples:
            logger.warning("没有高置信度三元组需要处理")
            return [], low_confidence_triples
        
        # 2. 提取所有术语
        all_relations = list(set(t['predicate'] for t in high_confidence_triples))
        
        all_subjects = []
        all_objects = []
        
        for t in high_confidence_triples:
            # 确保subject和object是字符串
            subject = t['subject'] if isinstance(t['subject'], str) else str(t['subject'])
            obj = t['object'] if isinstance(t['object'], str) else str(t['object'])
            all_subjects.append(subject)
            all_objects.append(obj)
        
        all_entities = list(set(all_subjects + all_objects))
        
        logger.info(f"提取到 {len(all_relations)} 个关系, {len(all_entities)} 个实体")
        
        # 3. 批量规范化关系
        logger.info("开始批量规范化关系...")
        relation_mappings = await self._normalize_terms_batch(
            all_relations, "relation", self.relation_collection
        )
        
        # 4. 批量规范化实体
        logger.info("开始批量规范化实体...")
        entity_mappings = await self._normalize_terms_batch(
            all_entities, "entity", self.entity_collection
        )
        
        # 保存映射结果
        self.intermediate_results['entity_mappings'] = entity_mappings
        self.intermediate_results['relation_mappings'] = relation_mappings
        self.save_intermediate_results(output_dir, "term_mappings")
        
        # 5. 重构三元组
        normalized_triples = []
        for triple in high_confidence_triples:
            normalized_triple = {
                "subject": entity_mappings.get(triple['subject'], triple['subject']),
                "predicate": relation_mappings.get(triple['predicate'], triple['predicate']),
                "object": entity_mappings.get(triple['object'], triple['object']),
                "confidence": triple.get('confidence', 1.0),
                "original_triple": triple  # 保留原始三元组信息
            }
            normalized_triples.append(normalized_triple)
        
        # 6. 统计信息
        processing_time = time.time() - start_time
        stats = {
            'total_triples': len(triples),
            'high_confidence_triples': len(high_confidence_triples),
            'low_confidence_triples': len(low_confidence_triples),
            'normalized_triples': len(normalized_triples),
            'unique_relations': len(all_relations),
            'unique_entities': len(all_entities),
            'processing_time_seconds': processing_time,
            'cache_hits': len(self.llm_cache)
        }
        
        self.intermediate_results['processing_stats'] = stats
        logger.info(f"审核和规范化完成，耗时 {processing_time:.2f} 秒")
        logger.info(f"统计信息: {stats}")
        
        # 保存最终结果
        self.save_intermediate_results(output_dir, "final_normalized")
        
        return normalized_triples, low_confidence_triples
    
    async def _normalize_terms_batch(
        self, 
        terms: List[str], 
        term_type: str, 
        collection_name: str
    ) -> Dict[str, str]:
        """批量术语规范化"""
        term_mappings = {}
        staged_canonical_names = set()
        staged_term_mappings = {}
        new_items_to_add = []
        
        # 创建信号量限制并发数
        semaphore = asyncio.Semaphore(self.max_concurrent_llm_calls)
        
        logger.info(f"开始批量规范化 {len(terms)} 个{term_type} (并发数: {self.max_concurrent_llm_calls})")
        
        # 分批处理以控制内存使用
        for i in range(0, len(terms), self.batch_size):
            batch_terms = terms[i:i + self.batch_size]
            
            # 并发处理当前批次
            tasks = [
                self._normalize_term_enhanced(
                    term_type, term, collection_name, 
                    staged_canonical_names, staged_term_mappings, semaphore
                )
                for term in batch_terms
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"处理术语失败: {batch_terms[j]}, 错误: {result}")
                    term_mappings[batch_terms[j]] = batch_terms[j]  # 使用原始术语作为后备
                else:
                    original_term, canonical_name = result
                    term_mappings[original_term] = canonical_name
                    
                    # 收集需要添加到数据库的新项
                    if canonical_name in staged_canonical_names:
                        new_items_to_add.append({
                            'original_term': original_term,
                            'canonical_name': canonical_name
                        })
            
            # 内存清理
            gc.collect()
            
            # 进度报告
            processed = min(i + self.batch_size, len(terms))
            logger.info(f"已处理 {processed}/{len(terms)} 个{term_type}")
        
        # 批量添加新项到向量数据库
        if new_items_to_add:
            await self._batch_add_new_items(new_items_to_add, collection_name)
        
        logger.info(f"批量规范化完成: {len(term_mappings)} 个{term_type}映射")
        return term_mappings