#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合版AstroWeaver知识图谱构建Pipeline
集成所有优化功能：
1. 批量审核和规范化
2. 中间结果保存
3. 高置信度过滤
4. 内存安全处理
5. 性能优化
"""

import asyncio
import json
import logging
import sys
import argparse
import gc
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

# Windows asyncio修复
if sys.platform == 'win32':
    import asyncio
    if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONASYNCIODEBUG'] = '0'

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from neo4j import GraphDatabase
from weaver.agents.data_scout import DataScout, Orchestrator
from weaver.agents.extractor import InformationExtractor
from weaver.agents.auditor_enhanced import EnhancedKnowledgeAuditor
from weaver.agents.constructor import GraphArchitect
from weaver.models.llm_models import LLMClient, create_llm_client, get_model_name
from weaver.models.embedding_models import EmbeddingClient
from weaver.storage.vector_db import VectorDBClient
from weaver.storage.file_handler import FileHandler
from weaver.utils.config import load_config
from weaver.utils.logging_setup import setup_logging
from data_sources.wikipedia_client import WikipediaClient
from data_sources.mineru_client import MinerUClient

# 导入优化版规范化器
from weaver.utils.canonicalizer_optimized import OptimizedCanonicalization

# 配置日志
logger = logging.getLogger(__name__)

class IntegratedPipeline:
    """整合版Pipeline类"""
    
    def __init__(self, config_path: str, skip_audit: bool = False):
        """初始化整合版Pipeline"""
        self.config = load_config(config_path)
        
        # 设置日志
        setup_logging(
            log_level=self.config.get('logging', {}).get('level', 'INFO'),
            log_dir=self.config.get('logging', {}).get('dir', 'logs'),
            log_filename=self.config.get('logging', {}).get('filename', 'astroWeaver.log')
        )
        self.logger = logging.getLogger(__name__)
        
        # 配置参数
        self.batch_size = self.config.get('pipeline', {}).get('batch_size', 10)
        self.confidence_threshold = self.config.get('auditor', {}).get('confidence_threshold', 0.8)
        self.skip_audit = skip_audit or self.config.get('pipeline', {}).get('skip_audit', False)  # 添加跳过审核标志
        self.max_retries = 3
        self.retry_delay = 5
        
        # 中间结果保存路径
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.intermediate_dir = self.output_dir / 'intermediate_results'
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建按阶段组织的目录结构
        self.stage_dirs = {
            'data_acquisition': self.intermediate_dir / '01_data_acquisition',
            'information_extraction': self.intermediate_dir / '02_information_extraction', 
            'knowledge_auditing': self.intermediate_dir / '03_knowledge_auditing',
            'canonicalization': self.intermediate_dir / '04_canonicalization',
            'graph_construction': self.intermediate_dir / '05_graph_construction'
        }
        
        # 创建所有阶段目录
        for stage_dir in self.stage_dirs.values():
            stage_dir.mkdir(parents=True, exist_ok=True)
        
        # 时间戳用于文件命名
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 客户端和组件
        self.clients = {}
        self.agents = {}
        
        # 统计信息
        self.stats = {
            'total_entities': 0,
            'total_documents': 0,
            'total_extracted_triples': 0,
            'high_confidence_triples': 0,
            'low_confidence_triples': 0,
            'canonicalized_triples': 0,
            'inserted_triples': 0
        }
        
        self.logger.info(f"初始化整合版Pipeline，时间戳: {self.timestamp}")
        self.logger.info(f"批处理大小: {self.batch_size}")
        self.logger.info(f"置信度阈值: {self.confidence_threshold}")
        self.logger.info(f"跳过审核阶段: {self.skip_audit}")
        self.logger.info(f"中间结果保存目录: {self.intermediate_dir}")
    
    async def initialize_clients(self):
        """初始化所有客户端"""
        try:
            self.logger.info("=== 初始化客户端 ===")
            
            # LLM客户端 - 使用工厂函数自动选择提供商
            self.clients['llm'] = create_llm_client(self.config)
            
            # 嵌入模型客户端
            self.clients['embedding'] = EmbeddingClient(
                api_base_url=self.config['embedding']['api_url']
            )
            
            # 向量数据库客户端
            self.clients['vector_db'] = VectorDBClient(
                path=self.config['paths']['vector_db_path'],
                embedding_client=self.clients['embedding']
            )
            
            # 图数据库客户端
            self.clients['neo4j'] = GraphDatabase.driver(
                self.config['neo4j']['uri'],
                auth=(self.config['neo4j']['user'], self.config['neo4j']['password'])
            )
            
            # 文件处理客户端
            self.clients['file_handler'] = FileHandler(
                output_dir=str(self.output_dir)
            )
            
            # 数据源客户端
            self.clients['wikipedia'] = WikipediaClient()
            self.clients['mineru'] = MinerUClient()
            
            self.logger.info("✅ 所有客户端初始化完成")
            
        except Exception as e:
            self.logger.error(f"客户端初始化失败: {e}")
            raise
    
    async def initialize_agents(self):
        """初始化所有Agent"""
        try:
            self.logger.info("=== 初始化Agent ===")
            
            # 数据侦察员
            self.agents['data_scout'] = DataScout(
                wikiClient=self.clients['wikipedia'],
                minerUClient=self.clients['mineru']
            )
            
            # 编排器
            self.agents['orchestrator'] = Orchestrator(
                llm_client=self.clients['llm'],
                data_scout=self.agents['data_scout']
            )
            
            # 信息提取器
            self.agents['extractor'] = InformationExtractor(
                llm_client=self.clients['llm'],
                model_name=get_model_name(self.config),
                topic="astronomy"  # 默认主题
            )
            
            # 增强版知识审核器
            self.agents['auditor'] = EnhancedKnowledgeAuditor(
                llm_client=self.clients['llm'],
                vector_db_client=self.clients['vector_db'],
                config=self.config
            )
            
            # 图构建器
            self.agents['constructor'] = GraphArchitect(
                driver=self.clients['neo4j'],
                file_handler=self.clients['file_handler'],
                llm_client=self.clients['llm']
            )
            
            # 优化版规范化器
            self.agents['canonicalizer'] = OptimizedCanonicalization(
                config_path='configs/config.yaml'
            )
            
            self.logger.info("✅ 所有Agent初始化完成")
            
        except Exception as e:
            self.logger.error(f"Agent初始化失败: {e}")
            raise
    
    def save_intermediate_result(self, stage: str, data: Any, filename: str = None):
        """保存中间结果到对应的阶段目录"""
        try:
            # 获取对应阶段的目录
            stage_dir = self.stage_dirs.get(stage, self.intermediate_dir)
            
            if filename is None:
                filename = f"{stage}_result.json"
            
            filepath = stage_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                if isinstance(data, (list, dict)):
                    json.dump(data, f, ensure_ascii=False, indent=2)
                else:
                    json.dump({'data': str(data)}, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"💾 {stage}阶段结果已保存: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"保存{stage}阶段结果失败: {e}")
            return None
    
    def generate_agent_report(self, stage: str, stats: Dict[str, Any], details: Dict[str, Any] = None):
        """生成agent处理阶段报告"""
        try:
            stage_dir = self.stage_dirs.get(stage, self.intermediate_dir)
            report_file = stage_dir / f"{stage}_report.json"
            
            report = {
                'stage': stage,
                'timestamp': datetime.now().isoformat(),
                'execution_time': stats.get('execution_time', 0),
                'status': 'completed',
                'statistics': stats,
                'details': details or {},
                'output_files': []
            }
            
            # 列出该阶段生成的所有文件
            if stage_dir.exists():
                report['output_files'] = [f.name for f in stage_dir.iterdir() if f.is_file() and f.name != f"{stage}_report.json"]
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"📋 {stage}阶段报告已生成: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"生成{stage}阶段报告失败: {e}")
            return None
    
    async def stage_1_data_acquisition(self, input_file: str) -> List[Dict[str, Any]]:
        """阶段1: 数据获取"""
        self.logger.info("\n" + "="*50)
        self.logger.info("🚀 阶段1: 数据获取")
        self.logger.info("="*50)
        
        try:
            # 读取输入文件
            input_path = Path(input_file)
            if not input_path.exists():
                raise FileNotFoundError(f"输入文件不存在: {input_file}")
            
            # 根据文件类型读取数据
            if input_path.suffix.lower() == '.xlsx':
                df = pd.read_excel(input_path)
                if 'type' not in df.columns or 'value' not in df.columns:
                    # 如果没有type和value列，假设是实体列表
                    entity_column = df.columns[0]
                    entities = df[entity_column].dropna().tolist()
                    items = [{'type': 'topic', 'value': entity} for entity in entities]
                else:
                    items = df[['type', 'value']].dropna().to_dict('records')
            elif input_path.suffix.lower() == '.json':
                # JSON文件处理
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    # 如果是列表，检查每个元素的格式
                    items = []
                    for item in data:
                        if isinstance(item, dict) and 'type' in item and 'value' in item:
                            items.append(item)
                        elif isinstance(item, str):
                            items.append({'type': 'topic', 'value': item})
                        else:
                            items.append({'type': 'topic', 'value': str(item)})
                elif isinstance(data, dict):
                    # 如果是字典，检查是否有type和value字段
                    if 'type' in data and 'value' in data:
                        items = [data]
                    else:
                        # 否则将整个字典作为一个topic处理
                        items = [{'type': 'topic', 'value': json.dumps(data, ensure_ascii=False)}]
                else:
                    # 其他类型直接转为字符串处理
                    items = [{'type': 'topic', 'value': str(data)}]
            elif input_path.suffix.lower() == '.txt':
                # 文本文件处理
                with open(input_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                items = [{'type': 'topic', 'value': content}]
            elif input_path.suffix.lower() == '.jsonl':
                # JSONL文件处理 - 跳过数据获取阶段
                self.logger.info("检测到JSONL文件，跳过数据获取阶段")
                return []
            else:
                raise ValueError(f"不支持的文件格式: {input_path.suffix}")
            
            self.stats['total_entities'] = len(items)
            self.logger.info(f"读取到 {len(items)} 个待处理项目")
            
            # 批量处理数据获取
            all_documents = []
            all_structured_data = []
            
            for i in range(0, len(items), self.batch_size):
                batch = items[i:i+self.batch_size]
                batch_num = i // self.batch_size + 1
                total_batches = (len(items) - 1) // self.batch_size + 1
                
                self.logger.info(f"\n📦 处理批次 {batch_num}/{total_batches} ({len(batch)} 个项目)")
                
                for item in batch:
                    item_type = str(item['type']).lower().strip()
                    item_value = str(item['value'])
                    
                    self.logger.info(f"处理项目: {item_type} - {item_value}")
                    
                    try:
                        if item_type in ['topic', 'pdf']:
                            # 使用Orchestrator处理
                            result = self.agents['orchestrator'].run(item_value)
                            if result:
                                if result.get('text_chunks'):
                                    all_documents.extend(result['text_chunks'])
                                if result.get('structured_data'):
                                    all_structured_data.append(result['structured_data'])
                        
                        # 内存清理
                        if len(all_documents) % 50 == 0:
                            gc.collect()
                            
                    except Exception as e:
                        self.logger.error(f"处理项目 {item_value} 失败: {e}")
                        continue
            
            self.stats['total_documents'] = len(all_documents)
            
            # 保存中间结果
            acquisition_result = {
                'documents': all_documents,
                'structured_data': all_structured_data,
                'stats': {
                    'total_items': len(items),
                    'total_documents': len(all_documents),
                    'total_structured_data': len(all_structured_data)
                }
            }
            
            # 保存主要结果
            self.save_intermediate_result('data_acquisition', acquisition_result)
            
            # 分别保存文档和结构化数据
            stage_dir = self.stage_dirs['data_acquisition']
            if all_documents:
                docs_file = stage_dir / 'documents.jsonl'
                with open(docs_file, 'w', encoding='utf-8') as f:
                    for doc in all_documents:
                        f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
            if all_structured_data:
                struct_file = stage_dir / 'structured_data.json'
                with open(struct_file, 'w', encoding='utf-8') as f:
                    json.dump(all_structured_data, f, ensure_ascii=False, indent=2)
            
            # 生成agent报告
            stats = {
                'total_items': len(items),
                'total_documents': len(all_documents),
                'total_structured_data': len(all_structured_data),
                'execution_time': 0  # 可以在这里添加实际的执行时间
            }
            self.generate_agent_report('data_acquisition', stats)
            
            self.logger.info(f"\n✅ 数据获取完成:")
            self.logger.info(f"  - 处理项目数: {len(items)}")
            self.logger.info(f"  - 获取文档数: {len(all_documents)}")
            self.logger.info(f"  - 结构化数据数: {len(all_structured_data)}")
            
            return all_documents
            
        except Exception as e:
            self.logger.error(f"数据获取阶段失败: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    async def stage_2_information_extraction(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """阶段2: 信息提取"""
        self.logger.info("\n" + "="*50)
        self.logger.info("🚀 阶段2: 信息提取")
        self.logger.info("="*50)
        
        try:
            if not documents:
                self.logger.warning("没有文档可供提取，跳过信息提取阶段")
                return []
            
            # 批量提取信息
            all_triples = []
            
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i+self.batch_size]
                batch_num = i // self.batch_size + 1
                total_batches = (len(documents) - 1) // self.batch_size + 1
                
                self.logger.info(f"\n📦 提取批次 {batch_num}/{total_batches} ({len(batch)} 个文档)")
                
                try:
                    # 使用增强版信息提取器处理批次
                    batch_triples = await self.agents['extractor'].extract_from_text_blocks(
                        batch
                    )
                    all_triples.extend(batch_triples)
                    
                    self.logger.info(f"批次 {batch_num} 提取到 {len(batch_triples)} 个三元组")
                    
                    # 内存清理
                    if batch_num % 5 == 0:
                        gc.collect()
                        
                except Exception as e:
                    self.logger.error(f"批次 {batch_num} 提取失败: {e}")
                    continue
            
            self.stats['total_extracted_triples'] = len(all_triples)
            
            # 保存中间结果
            extraction_result = {
                'triples': all_triples,
                'stats': {
                    'total_documents': len(documents),
                    'total_triples': len(all_triples),
                    'avg_triples_per_doc': len(all_triples) / len(documents) if documents else 0
                }
            }
            
            self.save_intermediate_result('information_extraction', extraction_result)
            
            # 优化数据结构：分离source_text以避免冗余
            stage_dir = self.stage_dirs['information_extraction']
            
            # 创建source_text映射表
            source_texts = {}
            optimized_triples = []
            
            for triple in all_triples:
                # 确保保留完整的source_text而非preview
                source_text = triple.get('source_text', '')
                if not source_text and 'source_text_preview' in triple:
                    # 如果只有preview，尝试从原始文档中恢复完整文本
                    if 'source_block_index' in triple:
                        block_idx = triple['source_block_index']
                        if block_idx < len(documents):
                            source_text = documents[block_idx].get('content', triple.get('source_text_preview', ''))
                
                # 生成source_text的唯一ID
                if source_text:
                    import hashlib
                    source_id = hashlib.md5(source_text.encode('utf-8')).hexdigest()[:16]
                    source_texts[source_id] = source_text
                    
                    # 创建优化的三元组（不包含完整source_text）
                    optimized_triple = {k: v for k, v in triple.items() if k not in ['source_text', 'source_text_preview']}
                    optimized_triple['source_id'] = source_id
                    optimized_triples.append(optimized_triple)
                else:
                    # 如果没有source_text，保持原样
                    optimized_triples.append(triple)
            
            # 保存优化的三元组
            jsonl_file = stage_dir / "extracted_triples.jsonl"
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                for triple in optimized_triples:
                    f.write(json.dumps(triple, ensure_ascii=False) + '\n')
            
            # 保存source_text映射表
            source_texts_file = stage_dir / "source_texts.json"
            with open(source_texts_file, 'w', encoding='utf-8') as f:
                json.dump(source_texts, f, ensure_ascii=False, indent=2)
            
            # 分别保存高置信度和低置信度三元组（使用优化结构）
            high_conf_triples = []
            low_conf_triples = []
            high_conf_optimized = []
            low_conf_optimized = []
            
            for i, original_triple in enumerate(all_triples):
                confidence = original_triple.get('confidence', 0)
                optimized_triple = optimized_triples[i]
                
                if confidence >= self.confidence_threshold:
                    high_conf_triples.append(original_triple)
                    high_conf_optimized.append(optimized_triple)
                else:
                    low_conf_triples.append(original_triple)
                    low_conf_optimized.append(optimized_triple)
            
            if high_conf_optimized:
                high_conf_file = stage_dir / "high_confidence_triples.jsonl"
                with open(high_conf_file, 'w', encoding='utf-8') as f:
                    for triple in high_conf_optimized:
                        f.write(json.dumps(triple, ensure_ascii=False) + '\n')
            
            if low_conf_optimized:
                low_conf_file = stage_dir / "low_confidence_triples.jsonl"
                with open(low_conf_file, 'w', encoding='utf-8') as f:
                    for triple in low_conf_optimized:
                        f.write(json.dumps(triple, ensure_ascii=False) + '\n')
            
            # 生成agent报告
            stats = {
                'total_documents': len(documents),
                'total_triples': len(all_triples),
                'high_confidence_triples': len(high_conf_triples),
                'low_confidence_triples': len(low_conf_triples),
                'avg_triples_per_doc': len(all_triples) / len(documents) if documents else 0,
                'execution_time': 0
            }
            details = {
                'confidence_threshold': self.confidence_threshold,
                'batch_size': self.batch_size
            }
            self.generate_agent_report('information_extraction', stats, details)
            
            self.logger.info(f"\n✅ 信息提取完成:")
            self.logger.info(f"  - 处理文档数: {len(documents)}")
            self.logger.info(f"  - 提取三元组数: {len(all_triples)}")
            self.logger.info(f"  - 平均每文档三元组数: {len(all_triples) / len(documents) if documents else 0:.2f}")
            
            return all_triples
            
        except Exception as e:
            self.logger.error(f"信息提取阶段失败: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    async def stage_3_knowledge_auditing(self, triples: List[Dict[str, Any]]) -> tuple:
        """阶段3: 知识审核（仅审核，不规范化）"""
        self.logger.info("\n" + "="*50)
        self.logger.info("🚀 阶段3: 知识审核")
        self.logger.info("="*50)
        
        try:
            if not triples:
                self.logger.warning("没有三元组可供审核，跳过知识审核阶段")
                return [], []
            
            # 仅进行置信度过滤，不进行规范化
            self.logger.info(f"开始审核 {len(triples)} 个三元组")
            
            # 置信度过滤
            confidence_threshold = self.config.get('audit', {}).get('confidence_threshold', 0.7)
            high_confidence_triples = [t for t in triples if t.get('confidence', 0) >= confidence_threshold]
            low_confidence_triples = [t for t in triples if t.get('confidence', 0) < confidence_threshold]
            
            self.logger.info(f"高置信度三元组: {len(high_confidence_triples)}, 低置信度三元组: {len(low_confidence_triples)}")
            
            self.stats['high_confidence_triples'] = len(high_confidence_triples)
            self.stats['low_confidence_triples'] = len(low_confidence_triples)
            
            # 保存中间结果
            audit_result = {
                'high_confidence_triples': high_confidence_triples,
                'low_confidence_triples': low_confidence_triples,
                'stats': {
                    'total_input_triples': len(triples),
                    'high_confidence_count': len(high_confidence_triples),
                    'low_confidence_count': len(low_confidence_triples),
                    'high_confidence_rate': len(high_confidence_triples) / len(triples) if triples else 0
                }
            }
            
            self.save_intermediate_result('knowledge_auditing', audit_result)
            
            # 保存到阶段目录
            stage_dir = self.stage_dirs['knowledge_auditing']
            
            # 优化数据结构：分离source_text以避免冗余
            import hashlib
            source_texts = {}
            
            def optimize_triples(triples_list):
                optimized = []
                for triple in triples_list:
                    # 获取完整的source_text
                    source_text = triple.get('source_text', '')
                    if not source_text and 'source_text_preview' in triple:
                        source_text = triple.get('source_text_preview', '')
                    
                    if source_text:
                        # 生成source_text的唯一ID
                        source_id = hashlib.md5(source_text.encode('utf-8')).hexdigest()[:16]
                        source_texts[source_id] = source_text
                        
                        # 创建优化的三元组
                        optimized_triple = {k: v for k, v in triple.items() if k not in ['source_text', 'source_text_preview']}
                        optimized_triple['source_id'] = source_id
                        optimized.append(optimized_triple)
                    else:
                        optimized.append(triple)
                return optimized
            
            # 分别保存高低置信度三元组为JSONL（使用优化结构）
            if high_confidence_triples:
                optimized_high = optimize_triples(high_confidence_triples)
                high_conf_file = stage_dir / "audited_high_confidence_triples.jsonl"
                with open(high_conf_file, 'w', encoding='utf-8') as f:
                    for triple in optimized_high:
                        f.write(json.dumps(triple, ensure_ascii=False) + '\n')
                self.logger.info(f"💾 审核后高置信度三元组已保存: {high_conf_file}")
            
            if low_confidence_triples:
                optimized_low = optimize_triples(low_confidence_triples)
                low_conf_file = stage_dir / "audited_low_confidence_triples.jsonl"
                with open(low_conf_file, 'w', encoding='utf-8') as f:
                    for triple in optimized_low:
                        f.write(json.dumps(triple, ensure_ascii=False) + '\n')
                self.logger.info(f"💾 审核后低置信度三元组已保存: {low_conf_file}")
            
            # 保存source_text映射表
            if source_texts:
                source_texts_file = stage_dir / "source_texts.json"
                with open(source_texts_file, 'w', encoding='utf-8') as f:
                    json.dump(source_texts, f, ensure_ascii=False, indent=2)
                self.logger.info(f"💾 Source texts映射表已保存: {source_texts_file}")
            
            # 生成agent报告
            stats = {
                'total_input_triples': len(triples),
                'high_confidence_count': len(high_confidence_triples),
                'low_confidence_count': len(low_confidence_triples),
                'high_confidence_rate': len(high_confidence_triples) / len(triples) if triples else 0,
                'execution_time': 0
            }
            details = {
                'confidence_threshold': self.confidence_threshold,
                'audit_method': 'enhanced_knowledge_auditor'
            }
            self.generate_agent_report('knowledge_auditing', stats, details)
            
            self.logger.info(f"\n✅ 知识审核完成:")
            self.logger.info(f"  - 输入三元组数: {len(triples)}")
            self.logger.info(f"  - 高置信度三元组数: {len(high_confidence_triples)}")
            self.logger.info(f"  - 低置信度三元组数: {len(low_confidence_triples)}")
            self.logger.info(f"  - 高置信度率: {len(high_confidence_triples) / len(triples) * 100 if triples else 0:.2f}%")
            
            return high_confidence_triples, low_confidence_triples
            
        except Exception as e:
            self.logger.error(f"知识审核阶段失败: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    async def stage_4_canonicalization(self, high_confidence_triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """阶段4: 规范化处理"""
        self.logger.info("\n" + "="*50)
        self.logger.info("🚀 阶段4: 规范化处理")
        self.logger.info("="*50)
        
        try:
            # 在规范化之前进行资源清理
            self.logger.info("🧹 规范化前资源清理...")
            
            # 清理前面阶段的资源
            if 'auditor' in self.agents and hasattr(self.agents['auditor'], 'cleanup'):
                await self.agents['auditor'].cleanup()
            
            if 'extractor' in self.agents and hasattr(self.agents['extractor'], 'cleanup'):
                await self.agents['extractor'].cleanup()
                
            # 内存清理
            gc.collect()
            self.logger.info("✅ 资源清理完成")
            
            # 获取阶段目录
            stage_dir = self.stage_dirs['canonicalization']
            
            # 从stage3的输出文件中读取高置信度三元组，而不是使用传入参数
            audit_stage_dir = self.stage_dirs['knowledge_auditing']
            high_confidence_file = audit_stage_dir / "audited_high_confidence_triples.jsonl"
            
            if not high_confidence_file.exists():
                self.logger.warning("未找到stage3输出的高置信度三元组文件，跳过规范化阶段")
                return []
            
            self.logger.info(f"使用stage3输出文件: {high_confidence_file}")
            
            # 读取高置信度三元组
            stage3_high_confidence_triples = []
            with open(high_confidence_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        stage3_high_confidence_triples.append(json.loads(line.strip()))
            
            if not stage3_high_confidence_triples:
                self.logger.warning("stage3输出的高置信度三元组文件为空，跳过规范化阶段")
                return []
            
            self.logger.info(f"开始规范化 {len(stage3_high_confidence_triples)} 个高置信度三元组")
            
            # 直接使用stage3输出的文件进行规范化
            temp_file = high_confidence_file
            
            # 使用OptimizedCanonicalization的run_canonicalization方法
            try:
                self.logger.info(f"开始调用规范化器处理文件: {temp_file}")
                
                # 设置超时时间为10分钟（600秒）
                timeout_seconds = 600
                self.logger.info(f"规范化超时设置: {timeout_seconds}秒")
                
                success = await asyncio.wait_for(
                    self.agents['canonicalizer'].run_canonicalization(str(temp_file)),
                    timeout=timeout_seconds
                )
                
                if not success:
                    self.logger.error("规范化器返回失败状态")
                    raise Exception("规范化过程失败")
                    
                self.logger.info("规范化器执行成功")
                
            except asyncio.TimeoutError:
                self.logger.error(f"规范化过程超时（{timeout_seconds}秒）")
                raise Exception(f"规范化过程超时（{timeout_seconds}秒）")
            except Exception as e:
                self.logger.error(f"规范化过程中发生异常: {e}")
                import traceback
                self.logger.error(f"异常详情: {traceback.format_exc()}")
                raise Exception(f"规范化过程失败: {e}")
            
            # 查找生成的规范化结果文件
            canonicalized_files = list(Path("data/output").glob(f"canonicalized_triples_optimized_*.jsonl"))
            if not canonicalized_files:
                raise Exception("未找到规范化结果文件")
            
            # 获取最新的规范化结果文件
            latest_file = max(canonicalized_files, key=lambda x: x.stat().st_mtime)
            
            # 读取规范化后的三元组
            canonicalized_triples = []
            with open(latest_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        triple = json.loads(line.strip())
                        canonicalized_triple = {
                            'subject': triple['subject'],
                            'predicate': triple['predicate'], 
                            'object': triple['object'],
                            'confidence': triple.get('confidence', 1.0),
                            'source_text': triple.get('source_text', ''),
                            'text_id': triple.get('text_id', '')
                        }
                        canonicalized_triples.append(canonicalized_triple)
            
            self.stats['canonicalized_triples'] = len(canonicalized_triples)
            
            # 初始化映射字典（规范化器内部处理，这里暂时为空）
            relation_map = {}
            entity_map = {}
            
            # 保存中间结果
            canonicalization_result = {
                'canonicalized_triples': canonicalized_triples,
                'relation_map': relation_map,
                'entity_map': entity_map,
                'stats': {
                    'input_triples': len(stage3_high_confidence_triples),
                    'output_triples': len(canonicalized_triples),
                    'relation_mappings': len(relation_map),
                    'entity_mappings': len(entity_map)
                }
            }
            
            self.save_intermediate_result('canonicalization', canonicalization_result)
            
            # 保存到阶段目录
            stage_dir = self.stage_dirs['canonicalization']
            
            # 优化规范化三元组的数据结构
            import hashlib
            canon_source_texts = {}
            canon_optimized = []
            
            for triple in canonicalized_triples:
                # 获取完整的source_text
                source_text = triple.get('source_text', '')
                if not source_text and 'source_text_preview' in triple:
                    source_text = triple.get('source_text_preview', '')
                
                if source_text:
                    # 生成source_text的唯一ID
                    source_id = hashlib.md5(source_text.encode('utf-8')).hexdigest()[:16]
                    canon_source_texts[source_id] = source_text
                    
                    # 创建优化的三元组
                    optimized_triple = {k: v for k, v in triple.items() if k not in ['source_text', 'source_text_preview']}
                    optimized_triple['source_id'] = source_id
                    canon_optimized.append(optimized_triple)
                else:
                    canon_optimized.append(triple)
            
            # 保存优化后的规范化三元组
            canonicalized_file = stage_dir / "canonicalized_triples.jsonl"
            with open(canonicalized_file, 'w', encoding='utf-8') as f:
                for triple in canon_optimized:
                    f.write(json.dumps(triple, ensure_ascii=False) + '\n')
            
            # 保存source_text映射表
            if canon_source_texts:
                canon_source_texts_file = stage_dir / "source_texts.json"
                with open(canon_source_texts_file, 'w', encoding='utf-8') as f:
                    json.dump(canon_source_texts, f, ensure_ascii=False, indent=2)
            
            # 保存映射关系
            relation_map_file = stage_dir / "relation_mappings.json"
            with open(relation_map_file, 'w', encoding='utf-8') as f:
                json.dump(relation_map, f, ensure_ascii=False, indent=2)
            
            entity_map_file = stage_dir / "entity_mappings.json"
            with open(entity_map_file, 'w', encoding='utf-8') as f:
                json.dump(entity_map, f, ensure_ascii=False, indent=2)
            
            # 生成agent报告
            stats = {
                'input_triples': len(stage3_high_confidence_triples),
                'output_triples': len(canonicalized_triples),
                'relation_mappings': len(relation_map),
                'entity_mappings': len(entity_map),
                'canonicalization_rate': len(canonicalized_triples) / len(stage3_high_confidence_triples) if stage3_high_confidence_triples else 0,
                'execution_time': 0
            }
            details = {
                'canonicalizer': 'OptimizedCanonicalization',
                'method': 'batch_optimized'
            }
            self.generate_agent_report('canonicalization', stats, details)
            
            self.logger.info(f"\n✅ 规范化处理完成:")
            self.logger.info(f"  - 输入三元组数: {len(stage3_high_confidence_triples)}")
            self.logger.info(f"  - 规范化三元组数: {len(canonicalized_triples)}")
            self.logger.info(f"  - 关系映射数: {len(relation_map)}")
            self.logger.info(f"  - 实体映射数: {len(entity_map)}")
            
            return canonicalized_triples
            
        except Exception as e:
            self.logger.error(f"规范化处理阶段失败: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    async def stage_5_graph_construction(self, canonicalized_triples: List[Dict[str, Any]]) -> bool:
        """阶段5: 图数据库构建"""
        self.logger.info("\n" + "="*50)
        self.logger.info("🚀 阶段5: 图数据库构建")
        self.logger.info("="*50)
        
        try:
            if not canonicalized_triples:
                self.logger.warning("没有规范化三元组可供插入图数据库")
                return False
            
            # 只插入高置信度的规范化三元组
            self.logger.info(f"开始将 {len(canonicalized_triples)} 个规范化三元组插入图数据库")
            
            # 使用图构建器插入数据
            await self.agents['constructor'].build_and_persist(
                normalized_triples=canonicalized_triples,
                structured_data={},  # 结构化数据可以为空
                output_filename=f"graph_construction_{self.timestamp}.json"
            )
            
            self.stats['inserted_triples'] = len(canonicalized_triples)
            
            # 保存最终结果
            final_result = {
                'inserted_triples': canonicalized_triples,
                'stats': self.stats,
                'timestamp': self.timestamp
            }
            
            self.save_intermediate_result('graph_construction', final_result)
            
            # 保存到阶段目录
            stage_dir = self.stage_dirs['graph_construction']
            
            # 保存插入的三元组
            inserted_file = stage_dir / "inserted_triples.jsonl"
            with open(inserted_file, 'w', encoding='utf-8') as f:
                for triple in canonicalized_triples:
                    f.write(json.dumps(triple, ensure_ascii=False) + '\n')
            
            # 生成agent报告
            stats = {
                'inserted_triples': len(canonicalized_triples),
                'total_pipeline_stats': self.stats,
                'execution_time': 0
            }
            details = {
                'graph_constructor': 'GraphArchitect',
                'database': 'Neo4j',
                'connection_uri': self.config['neo4j']['uri']
            }
            self.generate_agent_report('graph_construction', stats, details)
            
            self.logger.info(f"\n✅ 图数据库构建完成:")
            self.logger.info(f"  - 插入三元组数: {len(canonicalized_triples)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"图数据库构建阶段失败: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    async def run_pipeline(self, input_file: str) -> bool:
        """运行完整的Pipeline"""
        start_time = time.time()
        
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("🚀 启动整合版AstroWeaver Pipeline")
            self.logger.info("="*80)
            
            # 初始化
            await self.initialize_clients()
            await self.initialize_agents()
            
            # 检查输入文件类型
            input_path = Path(input_file)
            
            if input_path.suffix.lower() == '.jsonl':
                # 如果是JSONL文件，直接从规范化阶段开始
                self.logger.info("检测到JSONL文件，跳过数据获取、信息提取和知识审核阶段")
                
                # 直接读取JSONL文件作为高置信度三元组
                high_confidence_triples = []
                with open(input_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            high_confidence_triples.append(json.loads(line.strip()))
                
                self.stats['high_confidence_triples'] = len(high_confidence_triples)
                self.stats['low_confidence_triples'] = 0
                self.stats['total_entities'] = 0
                self.stats['total_documents'] = 0
                self.stats['total_extracted_triples'] = len(high_confidence_triples)
                
                self.logger.info(f"从JSONL文件加载了 {len(high_confidence_triples)} 个三元组")
            else:
                # 正常流程
                # 阶段1: 数据获取
                documents = await self.stage_1_data_acquisition(input_file)
                
                # 阶段2: 信息提取
                triples = await self.stage_2_information_extraction(documents)
                
                # 阶段3: 知识审核（可选）
                if self.skip_audit:
                    self.logger.info("⏭️ 跳过知识审核阶段")
                    high_confidence_triples = triples  # 直接使用提取的三元组
                    low_confidence_triples = []
                    self.stats['high_confidence_triples'] = len(high_confidence_triples)
                    self.stats['low_confidence_triples'] = 0
                else:
                    high_confidence_triples, low_confidence_triples = await self.stage_3_knowledge_auditing(triples)
            
            # 阶段4: 规范化处理
            canonicalized_triples = await self.stage_4_canonicalization(high_confidence_triples)
            
            # 阶段5: 图数据库构建
            success = await self.stage_5_graph_construction(canonicalized_triples)
            
            # 最终统计
            end_time = time.time()
            duration = end_time - start_time
            
            self.logger.info("\n" + "="*80)
            self.logger.info("🎉 整合版Pipeline执行完成")
            self.logger.info("="*80)
            self.logger.info(f"总耗时: {duration:.2f}秒")
            self.logger.info(f"\n📊 执行统计:")
            self.logger.info(f"  - 处理实体数: {self.stats['total_entities']}")
            self.logger.info(f"  - 获取文档数: {self.stats['total_documents']}")
            self.logger.info(f"  - 提取三元组数: {self.stats['total_extracted_triples']}")
            self.logger.info(f"  - 高置信度三元组数: {self.stats['high_confidence_triples']}")
            self.logger.info(f"  - 低置信度三元组数: {self.stats['low_confidence_triples']}")
            self.logger.info(f"  - 规范化三元组数: {self.stats['canonicalized_triples']}")
            self.logger.info(f"  - 插入图数据库三元组数: {self.stats['inserted_triples']}")
            
            # 保存最终统计到根目录
            final_stats = {
                'pipeline_stats': self.stats,
                'execution_time': duration,
                'timestamp': self.timestamp,
                'success': success
            }
            
            # 保存到intermediate_results根目录
            final_stats_file = self.intermediate_dir / 'pipeline_summary.json'
            with open(final_stats_file, 'w', encoding='utf-8') as f:
                json.dump(final_stats, f, ensure_ascii=False, indent=2)
            
            # 生成总体pipeline报告
            pipeline_report = {
                'pipeline_name': 'AstroWeaver Integrated Pipeline',
                'execution_timestamp': self.timestamp,
                'total_execution_time': duration,
                'success': success,
                'stages_completed': [
                    'data_acquisition',
                    'information_extraction', 
                    'knowledge_auditing',
                    'canonicalization',
                    'graph_construction'
                ],
                'final_statistics': self.stats,
                'stage_directories': {
                    stage: str(path) for stage, path in self.stage_dirs.items()
                },
                'output_summary': {
                    'total_entities_processed': self.stats['total_entities'],
                    'documents_acquired': self.stats['total_documents'],
                    'triples_extracted': self.stats['total_extracted_triples'],
                    'high_confidence_triples': self.stats['high_confidence_triples'],
                    'canonicalized_triples': self.stats['canonicalized_triples'],
                    'triples_inserted_to_graph': self.stats['inserted_triples']
                }
            }
            
            pipeline_report_file = self.intermediate_dir / 'pipeline_report.json'
            with open(pipeline_report_file, 'w', encoding='utf-8') as f:
                json.dump(pipeline_report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"📋 Pipeline总体报告已生成: {pipeline_report_file}")
            self.logger.info(f"📊 Pipeline统计已保存: {final_stats_file}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Pipeline执行失败: {e}")
            self.logger.error(traceback.format_exc())
            return False
        
        finally:
            # 清理资源
            await self.cleanup()
    
    async def cleanup(self):
        """清理资源"""
        try:
            self.logger.info("🧹 清理资源...")
            
            # 关闭数据库连接
            if 'neo4j' in self.clients and self.clients['neo4j']:
                self.clients['neo4j'].close()
            
            # 清理规范化器
            if 'canonicalizer' in self.agents and hasattr(self.agents['canonicalizer'], 'cleanup'):
                await self.agents['canonicalizer'].cleanup()
            
            # 内存清理
            gc.collect()
            
            self.logger.info("✅ 资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='整合版AstroWeaver Pipeline')
    parser.add_argument('input_file', help='输入文件路径（Excel格式）')
    parser.add_argument('--config', default='configs/config.yaml', help='配置文件路径')
    parser.add_argument('--log-level', default='INFO', help='日志级别')
    parser.add_argument('--skip-audit', action='store_true', help='跳过知识审核阶段')
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    try:
        # 创建并运行Pipeline
        pipeline = IntegratedPipeline(args.config, skip_audit=args.skip_audit)
        success = await pipeline.run_pipeline(args.input_file)
        
        if success:
            logger.info("✅ 整合版Pipeline执行成功")
            sys.exit(0)
        else:
            logger.error("❌ 整合版Pipeline执行失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline运行异常: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())