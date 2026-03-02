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
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from weaver.utils.getWikidata import SimpleWikidataClient

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

    def _classify_source_authority(self, source_url: str) -> tuple[str, float]:
        """根据URL域名估计来源权威等级与分值。"""
        gcfg = self.config.get('graph_conflict', {})
        aw = gcfg.get('authority_weights', {})

        default_general = float(aw.get('general_web', 0.6))
        source_url = (source_url or '').lower()

        if any(k in source_url for k in ['simbad', 'cds.unistra.fr']):
            return 'simbad', float(aw.get('simbad', 1.0))
        if any(k in source_url for k in ['arxiv.org', 'nature.com', 'science.org', 'iopscience', 'aanda.org', 'apj', 'mnras']):
            return 'academic_paper', float(aw.get('academic_paper', 0.9))
        if 'wikipedia.org' in source_url:
            return 'wikipedia', float(aw.get('wikipedia', 0.7))
        if any(k in source_url for k in ['news', 'xinhuanet', 'bbc.com/news', 'cnn.com']):
            return 'web_news', float(aw.get('web_news', 0.5))
        return 'general_web', default_general

    @staticmethod
    def _extract_numeric_value(val: Any) -> Optional[float]:
        """从字符串中提取首个数值（用于容差比较）。"""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val)
        m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None

    def _compute_confidence_score(self, authority_score: float, ts_norm: float) -> float:
        gcfg = self.config.get('graph_conflict', {})
        w1 = float(gcfg.get('w1', 0.7))
        w2 = float(gcfg.get('w2', 0.3))
        return w1 * authority_score + w2 * ts_norm
    
    def __init__(self, config_path: str, skip_audit: bool = False, multi_entity_mode: bool = False, resume: bool = False, output_dir: str = None):
        """初始化整合版Pipeline
        
        Args:
            config_path: 配置文件路径
            skip_audit: 是否跳过审计阶段
            multi_entity_mode: 是否启用多实体属性抽取模式
            resume: 是否接续上一轮处理
            output_dir: 自定义输出文件夹路径
        """
        self.config = load_config(config_path)
        self.multi_entity_mode = multi_entity_mode
        self.resume = resume
        
        # 设置日志
        setup_logging(
            log_level=self.config.get('logging', {}).get('level', 'INFO'),
            log_dir=self.config.get('logging', {}).get('dir', 'logs'),
            log_filename=self.config.get('logging', {}).get('filename', 'astroWeaver.log')
        )
        self.logger = logging.getLogger(__name__)
        
        # 配置参数
        self.batch_size = self.config.get('pipeline', {}).get('batch_size', 30)
        self.confidence_threshold = self.config.get('confidence', {}).get('audit_threshold', 0.8)
        self.skip_audit = skip_audit or self.config.get('pipeline', {}).get('skip_audit', False)  # 添加跳过审核标志
        self.max_retries = 3
        self.retry_delay = 5
        
        # 中间结果保存路径
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
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
            'total_extracted_attributes': 0,
            'total_extracted_events': 0,
            'high_confidence_triples': 0,
            'low_confidence_triples': 0,
            'high_confidence_events': 0,
            'low_confidence_events': 0,
            'canonicalized_triples': 0,
            'inserted_triples': 0,
            'inserted_events': 0
        }
        
        self.logger.info(f"初始化整合版Pipeline，时间戳: {self.timestamp}")
        self.logger.info(f"批处理大小: {self.batch_size}")
        self.logger.info(f"置信度阈值: {self.confidence_threshold}")
        self.logger.info(f"跳过审核阶段: {self.skip_audit}")
        self.logger.info(f"中间结果保存目录: {self.intermediate_dir}")

    @staticmethod
    def _safe_load_json(path: Path, default):
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            return default
        return default

    def _infer_source_authority(self, source_url: str = '', source_type: str = '') -> str:
        url = (source_url or '').lower()
        st = (source_type or '').lower()

        if 'simbad' in url or 'simbad.cds.unistra.fr' in url:
            return 'simbad'
        if ('arxiv.org' in url or 'doi.org' in url or 'nature.com' in url or
            'science.org' in url or 'iopscience.iop.org' in url or 'academic.oup.com' in url):
            return 'academic_paper'
        if 'wikipedia.org' in url:
            return 'wikipedia'
        if st == 'news' or any(x in url for x in ['news', 'reuters.com', 'apnews.com', 'xinhuanet.com']):
            return 'web_news'
        return 'general_web'

    def _enrich_triples_for_conflict_resolution(self, triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为三元组补充冲突融合需要的元数据：来源权威、时间戳、CS分数。"""
        if not triples:
            return triples

        graph_cfg = self.config.get('graph_conflict', {})
        w1 = float(graph_cfg.get('w1', 0.7))
        w2 = float(graph_cfg.get('w2', 0.3))
        authority_weights = graph_cfg.get('authority_weights', {
            'simbad': 1.0,
            'academic_paper': 0.9,
            'wikipedia': 0.7,
            'web_news': 0.5,
            'general_web': 0.6
        })

        # 构建 source_id -> 元数据 映射（优先来自 stage1 文档）
        source_meta = {}
        docs_file = self.output_dir / 'intermediate_results' / '01_data_acquisition' / 'documents.jsonl'
        if docs_file.exists():
            try:
                import hashlib
                with open(docs_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        doc = json.loads(line)
                        txt = doc.get('text') or doc.get('content') or ''
                        candidates = [txt]
                        if isinstance(txt, str) and len(txt) > 200:
                            candidates.append(txt[:200])
                        for c in candidates:
                            if not c:
                                continue
                            sid = hashlib.md5(c.encode('utf-8')).hexdigest()[:16]
                            source_meta[sid] = {
                                'source_url': doc.get('source_url', ''),
                                'page_title': doc.get('page_title', ''),
                                'source_type': doc.get('source_type', 'web'),
                                'origin_type': doc.get('origin_type', ''),
                                'origin_value': doc.get('origin_value', ''),
                            }
            except Exception as e:
                self.logger.warning(f"构建 source_meta 失败: {e}")

        now_ts = time.time()
        enriched = []
        for t in triples:
            nt = dict(t)
            sid = nt.get('source_id', '')
            meta = source_meta.get(sid, {})

            source_url = nt.get('source_url') or meta.get('source_url', '')
            source_type = nt.get('source_type') or meta.get('source_type', 'web')
            source_authority = nt.get('source_authority') or self._infer_source_authority(source_url, source_type)
            authority_score = float(authority_weights.get(source_authority, authority_weights.get('general_web', 0.6)))

            ts_val = nt.get('timestamp')
            if ts_val is None:
                ts_val = now_ts
            try:
                ts_val = float(ts_val)
            except Exception:
                ts_val = now_ts
            normalized_t = max(0.0, min(1.0, ts_val / now_ts if now_ts > 0 else 1.0))

            confidence_score = w1 * authority_score + w2 * normalized_t

            nt['source_authority'] = source_authority
            nt['authority_score'] = authority_score
            nt['timestamp'] = ts_val
            nt['confidence_score'] = confidence_score
            if source_url:
                nt['source_url'] = source_url
            if meta.get('page_title'):
                nt['page_title'] = meta.get('page_title')

            enriched.append(nt)

        return enriched
    
    async def initialize_clients(self):
        """初始化所有客户端"""
        try:
            self.logger.info("=== 初始化客户端 ===")
            
            # LLM客户端 - 使用工厂函数自动选择提供商
            self.clients['llm'] = create_llm_client(self.config)
            
            # 嵌入模型客户端
            self.clients['embedding'] = EmbeddingClient(
                api_base_url=self.config['embedding']['api_url'],
                lazy_init=True  # 使用延迟初始化，不立即检查服务健康状态
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
            proxies = self.config.get('proxies', {})
            self.clients['wikidata'] = SimpleWikidataClient(proxies=proxies)
            self.logger.info("✅ Wikidata客户端初始化完成")
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
                llm_client=self.clients['llm'],
                config=self.config
            )
            
            # 优化版规范化器 - 包装以支持输出目录
            class WrappedOptimizedCanonicalization(OptimizedCanonicalization):
                def __init__(self, config_path, output_dir):
                    super().__init__(config_path)
                    self.output_dir = Path(output_dir)
                
                async def run_canonicalization(self, input_file_path):
                    # 调用父类方法
                    success = await super().run_canonicalization(input_file_path)
                    
                    # 如果成功，将输出文件移动到正确的输出目录
                    if success:
                        import shutil
                        import glob
                        import os
                        
                        # 查找当前目录下生成的规范化文件
                        for pattern in ["canonicalized_triples_optimized_*.jsonl", 
                                       "canonicalization_mappings_optimized_*.json"]:
                            for file in glob.glob(pattern):
                                src = Path(file)
                                dst = self.output_dir / src.name
                                if not dst.exists():
                                    shutil.move(str(src), str(dst))
                                    self.logger.info(f"移动规范化文件到输出目录: {dst}")
                    
                    return success
            
            self.agents['canonicalizer'] = WrappedOptimizedCanonicalization(
                config_path='configs/config.yaml',
                output_dir=str(self.output_dir)
            )
            
            # 从配置文件设置规范化器的并发度参数
            performance_config = self.config.get('performance', {})
            if 'max_concurrent_llm_calls' in performance_config:
                self.agents['canonicalizer'].max_concurrent_llm_calls = performance_config['max_concurrent_llm_calls']
                self.logger.info(f"设置规范化器并发度: {performance_config['max_concurrent_llm_calls']}")
            if 'batch_size' in performance_config:
                self.agents['canonicalizer'].batch_size = performance_config['batch_size']
                self.logger.info(f"设置规范化器批处理大小: {performance_config['batch_size']}")
            
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
    
    def check_intermediate_results(self) -> Dict[str, Any]:
        """检查中间结果文件，确定可以从哪个阶段恢复"""
        recovery_info = {
            'can_resume': False,
            'resume_from_stage': None,
            'available_data': {}
        }
        
        # 检查各阶段的输出文件
        stages_to_check = [
            ('stage_1_data_acquisition', '01_data_acquisition', 'data_acquisition_result.json'),
            ('stage_2_information_extraction', '02_information_extraction', ['extracted_relations.jsonl', 'high_confidence_triples.jsonl', 'extracted_events.jsonl']),
            ('stage_3_knowledge_auditing', '03_knowledge_auditing', 'high_confidence_triples.jsonl'),
            ('stage_4_canonicalization', '04_canonicalization', 'canonicalized_triples.jsonl')
        ]
        
        for stage_name, stage_dir, key_files in stages_to_check:
            stage_path = self.intermediate_dir / stage_dir
            
            # 处理多个可能的关键文件
            if isinstance(key_files, list):
                key_file_path = None
                for key_file in key_files:
                    potential_path = stage_path / key_file
                    if potential_path.exists():
                        key_file_path = potential_path
                        break
            else:
                key_file_path = stage_path / key_files
            
            if key_file_path and key_file_path.exists():
                recovery_info['available_data'][stage_name] = {
                    'stage_dir': stage_path,
                    'key_file': key_file_path,
                    'file_size': key_file_path.stat().st_size
                }
                recovery_info['can_resume'] = True
                recovery_info['resume_from_stage'] = stage_name
        
        return recovery_info
    
    def load_intermediate_data(self, stage_name: str, file_path: Path) -> List[Dict[str, Any]]:
        """从中间结果文件加载数据"""
        data = []
        
        try:
            if file_path.suffix == '.jsonl':
                # JSONL格式
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line.strip()))
            elif file_path.suffix == '.json':
                # JSON格式
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, list):
                        data = loaded_data
                    elif isinstance(loaded_data, dict) and 'documents' in loaded_data:
                        data = loaded_data['documents']
                    else:
                        data = [loaded_data]
            
            self.logger.info(f"从 {file_path} 加载了 {len(data)} 条记录")
            return data
            
        except Exception as e:
            self.logger.error(f"加载中间数据失败: {e}")
            return []
    
    async def _finalize_pipeline(self, success: bool, duration: float) -> bool:
        """完成Pipeline的最终统计和报告生成"""
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
        self.logger.info(f"  - 提取事件数: {self.stats['total_extracted_events']}")
        self.logger.info(f"  - 高置信度事件数: {self.stats['high_confidence_events']}")
        self.logger.info(f"  - 低置信度事件数: {self.stats['low_confidence_events']}")
        self.logger.info(f"  - 插入事件数: {self.stats['inserted_events']}")
        
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
                'triples_inserted_to_graph': self.stats['inserted_triples'],
                'events_extracted': self.stats['total_extracted_events'],
                'high_confidence_events': self.stats['high_confidence_events'],
                'low_confidence_events': self.stats['low_confidence_events'],
                'events_inserted': self.stats['inserted_events']
            }
        }
        
        pipeline_report_file = self.intermediate_dir / 'pipeline_report.json'
        with open(pipeline_report_file, 'w', encoding='utf-8') as f:
            json.dump(pipeline_report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📋 Pipeline总体报告已生成: {pipeline_report_file}")
        self.logger.info(f"📊 Pipeline统计已保存: {final_stats_file}")
        
        return success
    
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
                            # 旧格式：{"type": "pdf", "value": "path"}
                            items.append(item)
                        elif isinstance(item, dict) and len(item) == 1:
                            # 新格式：{"pdf": "path"}
                            key, value = next(iter(item.items()))
                            if key in ['topic', 'entity', 'direct_question', 'pdf']:
                                items.append({'type': key, 'value': value})
                            else:
                                items.append({'type': 'topic', 'value': json.dumps(item, ensure_ascii=False)})
                        elif isinstance(item, str):
                            items.append({'type': 'topic', 'value': item})
                        else:
                            items.append({'type': 'topic', 'value': str(item)})
                elif isinstance(data, dict):
                    # 如果是字典，检查是否有type和value字段
                    if 'type' in data and 'value' in data:
                        items = [data]
                    else:
                        # 检查是否是新格式（直接使用键名作为类型）
                        if len(data) == 1:
                            key, value = next(iter(data.items()))
                            if key in ['topic', 'entity', 'direct_question', 'pdf']:
                                items = [{'type': key, 'value': value}]
                            else:
                                # 否则将整个字典作为一个topic处理
                                items = [{'type': 'topic', 'value': json.dumps(data, ensure_ascii=False)}]
                        else:
                            # 否则将整个字典作为一个topic处理
                            items = [{'type': 'topic', 'value': json.dumps(data, ensure_ascii=False)}]
                else:
                    # 其他类型直接转为字符串处理
                    items = [{'type': 'topic', 'value': str(data)}]
                
                # 提取第一个实体名称用于后续抽取
                if items:
                    self.input_entity_name = items[0]['value']
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
                    
                    # 设置当前实体名称用于属性抽取
                    if item_type == 'entity':
                        self.current_entity_name = item_value
                    else:
                        self.current_entity_name = item_value  # 对于其他类型，也使用值作为实体名称
                    
                    self.logger.info(f"处理项目: {item_type} - {item_value!r}")
                    
                    try:
                        if item_type in ['topic', 'entity', 'direct_question', 'pdf']:
                            # 使用Orchestrator处理，转换为新的字典格式
                            if item_type == 'topic':
                                input_data = {"topic": item_value}
                            elif item_type == 'entity':
                                input_data = {"entity": item_value}
                            elif item_type == 'direct_question':
                                input_data = {"direct_question": item_value}
                            elif item_type == 'pdf':
                                input_data = {"pdf": item_value}
                            result = self.agents['orchestrator'].run(input_data)
                            if result:
                                if result.get('text_chunks'):
                                    all_documents.extend(result['text_chunks'])
                                if result.get('structured_data'):
                                    all_structured_data.append(result['structured_data'])
                        # 仅当类型为 entity 或 topic 时尝试获取 Wikidata
                        if item_type in ['entity', 'topic']:
                            self.logger.info(f"🔍 正在从 Wikidata 获取: {item_value} ...")
                            # 使用 asyncio.to_thread 避免阻塞主线程
                            wikidata_data = await asyncio.to_thread(
                                self.clients['wikidata'].process_entity, item_value
                            )

                            if wikidata_data:
                                # 构造带后缀的键名，例如 "Sun_wikidata"
                                key_name = f"{item_value}_wikidata"
                                all_structured_data.append({key_name: wikidata_data})
                                self.logger.info(
                                    f"✅ 成功获取 Wikidata 数据: {item_value} ({len(wikidata_data)} 字段)")
                            else:
                                self.logger.warning(f"⚠️ 未在 Wikidata 找到: {item_value}")
                        # 内存清理
                        if len(all_documents) % 50 == 0:
                            gc.collect()
                            
                    except Exception as e:
                        self.logger.error(f"处理项目 {item_value!r} 失败: {e}")
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
    
    async def process_text_block(self, text_block, entity_name=None, multi_entity_mode=False):
        """处理单个文本块，提取信息"""
        try:
            if multi_entity_mode:
                return await self.agents['extractor'].extract_comprehensive_information(
                    [text_block], multi_entity_mode=True
                )
            else:
                return await self.agents['extractor'].extract_comprehensive_information(
                    [text_block], entity_name=entity_name, multi_entity_mode=False
                )
        except Exception as e:
            self.logger.error(f"处理文本块时出错: {e}")
            return {'attributes': [], 'relations': [], 'events': []}

    async def process_batch_concurrent(self, text_blocks, entity_name=None, multi_entity_mode=False):
        """并发处理一批文本块"""
        # 获取并发数量，默认为3，可以在配置文件中调整
        concurrent_limit = self.config.get('performance', {}).get('extraction_concurrent_limit', 20)
        
        
        # 创建信号量限制并发数
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        async def process_with_semaphore(text):
            async with semaphore:
                return await self.process_text_block(text, entity_name, multi_entity_mode)
        
        # 创建所有文本块的任务
        tasks = [process_with_semaphore(text) for text in text_blocks]
        
        # 并发执行所有任务
        results = await asyncio.gather(*tasks)
        
        # 合并结果
        merged_result = {'attributes': [], 'relations': [], 'events': []}
        for result in results:
            if result['attributes']:
                merged_result['attributes'].extend(result['attributes'])
            if result['relations']:
                merged_result['relations'].extend(result['relations'])
            if result['events']:
                merged_result['events'].extend(result['events'])
        
        return merged_result

    async def stage_2_information_extraction(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """阶段2: 信息提取"""
        self.logger.info("\n" + "="*50)
        self.logger.info("🚀 阶段2: 信息提取 (并发优化版)")
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
                    # 使用新的综合抽取方法，区分属性和关系
                    # 检查是否启用多实体属性抽取模式
                    multi_entity_mode = getattr(self, 'multi_entity_mode', False)
                    
                    # 从文档中提取文本内容
                    text_blocks = []
                    for doc in batch:
                        if isinstance(doc, str):
                            # 如果doc是字符串，直接使用
                            text_blocks.append(doc)
                        elif isinstance(doc, dict):
                            # 如果doc是字典，提取文本内容
                            if 'content' in doc:
                                text_blocks.append(doc['content'])
                            elif 'text' in doc:
                                text_blocks.append(doc['text'])
                            else:
                                # 如果没有content或text字段，尝试获取其他文本字段
                                text_content = str(doc.get('title', '')) + ' ' + str(doc.get('summary', ''))
                                if text_content.strip():
                                    text_blocks.append(text_content)
                        else:
                            # 其他类型转为字符串
                            text_blocks.append(str(doc))
                    
                    if not text_blocks:
                        self.logger.warning(f"批次 {batch_num} 中没有找到可提取的文本内容")
                        continue
                    
                    # 获取实体名称（用于单实体模式）
                    entity_name = getattr(self, 'current_entity_name', None)
                    
                    # 使用并发处理文本块
                    self.logger.info(f"开始并发处理 {len(text_blocks)} 个文本块")
                    extraction_result = await self.process_batch_concurrent(
                        text_blocks, 
                        entity_name=entity_name, 
                        multi_entity_mode=multi_entity_mode
                    )
                    
                    # 处理提取结果（包含属性、关系和事件）
                    if extraction_result['attributes']:
                        if not hasattr(self, 'all_attributes'):
                            self.all_attributes = []
                        self.all_attributes.extend(extraction_result['attributes'])
                        self.logger.info(f"批次 {batch_num} 提取到 {len(extraction_result['attributes'])} 个属性")
                        self.stats['total_extracted_attributes'] += len(extraction_result['attributes'])
                    
                    if extraction_result['relations']:
                        # 将关系转换为三元组格式
                        for relation in extraction_result['relations']:
                            triple = {
                                'subject': relation['subject'],
                                'predicate': relation['predicate'],
                                'object': relation['object'],
                                'confidence': relation['confidence'],
                                'source_text': relation['source_text'],
                                'text_id': relation['text_id'],
                                'attributes': relation.get('attributes', {})
                            }
                            all_triples.append(triple)
                        self.logger.info(f"批次 {batch_num} 提取到 {len(extraction_result['relations'])} 个关系")
                    
                    # 处理事件抽取结果
                    if extraction_result['events']:
                        if not hasattr(self, 'all_events'):
                            self.all_events = []
                        self.all_events.extend(extraction_result['events'])
                        self.logger.info(f"批次 {batch_num} 提取到 {len(extraction_result['events'])} 个事件")
                        self.stats['total_extracted_events'] += len(extraction_result['events'])
                        
                        # 根据置信度分类事件
                        for event in extraction_result['events']:
                            if event['confidence'] >= self.confidence_threshold:
                                self.stats['high_confidence_events'] += 1
                            else:
                                self.stats['low_confidence_events'] += 1
                    
                    if not extraction_result['attributes'] and not extraction_result['relations'] and not extraction_result['events']:
                        self.logger.info(f"批次 {batch_num} 未提取到任何信息")
                    
                    # 内存清理
                    if batch_num % 5 == 0:
                        gc.collect()
                        
                except Exception as e:
                    self.logger.error(f"批次 {batch_num} 提取失败: {e}")
                    continue
            
            # 获取属性和事件数据（如果存在）
            all_attributes = getattr(self, 'all_attributes', [])
            all_events = getattr(self, 'all_events', [])
            
            self.stats['total_extracted_relations'] = len(all_triples)
            self.stats['total_extracted_attributes'] = len(all_attributes)
            self.stats['total_extracted_events'] = len(all_events)
            
            # 保存中间结果，区分属性、关系和事件
            extraction_result = {
                'relations': all_triples,  # 只有关系会进入规范化流程
                'attributes': all_attributes,  # 属性单独保存，不进入规范化
                'events': all_events,  # 事件单独保存，不进入规范化
                'stats': {
                    'total_documents': len(documents),
                    'total_relations': len(all_triples),
                    'total_attributes': len(all_attributes),
                    'total_events': len(all_events),
                    'avg_relations_per_doc': len(all_triples) / len(documents) if documents else 0,
                    'avg_attributes_per_doc': len(all_attributes) / len(documents) if documents else 0,
                    'avg_events_per_doc': len(all_events) / len(documents) if documents else 0
                }
            }
            
            self.logger.info("⚠️ 注意：属性(attributes)和事件(events)不会进入规范化流程，将保持原始形式")
            
            self.save_intermediate_result('information_extraction', extraction_result)
            
            # 优化数据结构：分离source_text以避免冗余
            stage_dir = self.stage_dirs['information_extraction']
            
            # 创建source_text映射表
            source_texts = {}
            optimized_relations = []
            optimized_attributes = []
            optimized_events = []
            
            # 处理关系三元组
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
                    
                    # 创建优化的关系三元组（不包含完整source_text）
                    optimized_triple = {k: v for k, v in triple.items() if k not in ['source_text', 'source_text_preview']}
                    optimized_triple['source_id'] = source_id
                    optimized_relations.append(optimized_triple)
                else:
                    # 如果没有source_text，保持原样
                    optimized_relations.append(triple)
            
            # 处理属性
            for attr in all_attributes:
                # 确保保留完整的source_text而非preview
                source_text = attr.get('source_text', '')
                if not source_text and 'source_text_preview' in attr:
                    # 如果只有preview，尝试从原始文档中恢复完整文本
                    if 'source_block_index' in attr:
                        block_idx = attr['source_block_index']
                        if block_idx < len(documents):
                            source_text = documents[block_idx].get('content', attr.get('source_text_preview', ''))
                
                # 生成source_text的唯一ID
                if source_text:
                    import hashlib
                    source_id = hashlib.md5(source_text.encode('utf-8')).hexdigest()[:16]
                    source_texts[source_id] = source_text
                    
                    # 创建优化的属性（不包含完整source_text）
                    optimized_attr = {k: v for k, v in attr.items() if k not in ['source_text', 'source_text_preview']}
                    optimized_attr['source_id'] = source_id
                    optimized_attributes.append(optimized_attr)
                else:
                    # 如果没有source_text，保持原样
                    optimized_attributes.append(attr)
            
            # 处理事件
            for event in all_events:
                # 确保保留完整的source_text而非preview
                source_text = event.get('source_text', '')
                if not source_text and 'source_text_preview' in event:
                    # 如果只有preview，尝试从原始文档中恢复完整文本
                    if 'source_block_index' in event:
                        block_idx = event['source_block_index']
                        if block_idx < len(documents):
                            source_text = documents[block_idx].get('content', event.get('source_text_preview', ''))
                
                # 生成source_text的唯一ID
                if source_text:
                    import hashlib
                    source_id = hashlib.md5(source_text.encode('utf-8')).hexdigest()[:16]
                    source_texts[source_id] = source_text
                    
                    # 创建优化的事件（不包含完整source_text）
                    optimized_evt = {k: v for k, v in event.items() if k not in ['source_text', 'source_text_preview']}
                    optimized_evt['source_id'] = source_id
                    optimized_events.append(optimized_evt)
                else:
                    # 如果没有source_text，保持原样
                    optimized_events.append(event)
            
            # 分别保存关系、属性和事件
            relations_file = stage_dir / "extracted_relations.jsonl"
            with open(relations_file, 'w', encoding='utf-8') as f:
                for relation in optimized_relations:
                    f.write(json.dumps(relation, ensure_ascii=False) + '\n')
            
            attributes_file = stage_dir / "extracted_attributes.jsonl"
            with open(attributes_file, 'w', encoding='utf-8') as f:
                for attribute in optimized_attributes:
                    f.write(json.dumps(attribute, ensure_ascii=False) + '\n')
            
            events_file = stage_dir / "extracted_events.jsonl"
            with open(events_file, 'w', encoding='utf-8') as f:
                for event in optimized_events:
                    f.write(json.dumps(event, ensure_ascii=False) + '\n')
            
            # 移除重复保存：不再保存extracted_triples.jsonl，统一使用extracted_relations.jsonl
            
            # 保存source_text映射表
            source_texts_file = stage_dir / "source_texts.json"
            with open(source_texts_file, 'w', encoding='utf-8') as f:
                json.dump(source_texts, f, ensure_ascii=False, indent=2)
            
            # 注意：不再在stage2输出high_confidence_triples.jsonl和low_confidence_triples.jsonl
            # 这些文件将在stage3知识审核阶段生成，避免重复输出
            
            # 生成agent报告
            stats = {
                'total_documents': len(documents),
                'total_relations': len(all_triples),
                'total_attributes': len(all_attributes),
                'avg_relations_per_doc': len(all_triples) / len(documents) if documents else 0,
                'avg_attributes_per_doc': len(all_attributes) / len(documents) if documents else 0,
                'execution_time': 0
            }
            details = {
                'confidence_threshold': self.confidence_threshold,
                'batch_size': self.batch_size
            }
            self.generate_agent_report('information_extraction', stats, details)
            
            self.logger.info(f"\n✅ 信息提取完成:")
            self.logger.info(f"  - 处理文档数: {len(documents)}")
            self.logger.info(f"  - 提取关系三元组数: {len(all_triples)}")
            self.logger.info(f"  - 提取属性数: {len(all_attributes)}")
            self.logger.info(f"  - 平均每文档关系数: {len(all_triples) / len(documents) if documents else 0:.2f}")
            self.logger.info(f"  - 平均每文档属性数: {len(all_attributes) / len(documents) if documents else 0:.2f}")
            self.logger.info(f"  - 关系文件: extracted_relations.jsonl")
            self.logger.info(f"  - 属性文件: extracted_attributes.jsonl")
            self.logger.info(f"  - 注意: 只有关系会进入后续的规范化流程")
            
            return all_triples  # 返回关系三元组用于后续规范化
            
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
            confidence_threshold = self.config.get('confidence', {}).get('audit_threshold', 0.7)
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
            
            # 加载UAT_processed.json文件进行同名实体消解
            self.logger.info("🔄 开始同名实体消解处理...")
            uat_file_path = self.output_dir / "UAT_processed.json"
            if not uat_file_path.exists():
                # 回退到项目根目录
                uat_file_path = Path("UAT_processed.json")
            # 默认使用原始三元组文件
            temp_file = high_confidence_file
            
            if not uat_file_path.exists():
                self.logger.warning("未找到UAT_processed.json文件，跳过同名实体消解")
            else:
                try:
                    # 读取UAT_processed.json文件
                    with open(uat_file_path, 'r', encoding='utf-8') as f:
                        uat_data = json.load(f)
                    
                    # 构建同名实体映射表
                    entity_mapping = {}
                    entity_definitions = {}
                    
                    # 处理UAT数据，构建映射表
                    for item in uat_data:
                        for key, value in item.items():
                            if key != "definition":
                                # 主实体名称
                                primary_entity = key
                                # 同名实体列表
                                aliases = value if isinstance(value, list) and value is not None else []
                                
                                # 将所有同名实体映射到主实体
                                for alias in aliases:
                                    if alias and alias != primary_entity:
                                        entity_mapping[alias] = primary_entity
                                
                                # 获取定义（如果存在）
                                definition = item.get("definition")
                                if definition:
                                    entity_definitions[primary_entity] = definition
                    
                    # 从structured_data.json中加载simbad数据的identifiers
                    self.logger.info("🔍 从structured_data.json中加载simbad数据的identifiers...")
                    structured_data_path = self.stage_dirs['data_acquisition'] / "structured_data.json"
                    
                    if structured_data_path.exists():
                        try:
                            with open(structured_data_path, 'r', encoding='utf-8') as f:
                                structured_data = json.load(f)
                            
                            # 处理每个实体的simbad数据中的identifiers
                            simbad_identifiers_count = 0
                            
                            for entity_data in structured_data:
                                for key, value in entity_data.items():
                                    if key.endswith("_simbad") and isinstance(value, dict):
                                        # 获取实体名称（去掉_simbad后缀）
                                        entity_name = key.replace("_simbad", "")
                                        
                                        # 获取identifiers列表
                                        identifiers = value.get("identifiers", [])
                                        
                                        if identifiers and isinstance(identifiers, list):
                                            # 将identifiers映射到实体名称
                                            for identifier in identifiers:
                                                if identifier and identifier != entity_name:
                                                    entity_mapping[identifier] = entity_name
                                                    simbad_identifiers_count += 1
                            
                            self.logger.info(f"✅ 从structured_data.json中加载了 {simbad_identifiers_count} 个simbad标识符")
                        except Exception as e:
                            self.logger.error(f"加载structured_data.json失败: {e}")
                            self.logger.error(traceback.format_exc())
                    else:
                        self.logger.warning(f"未找到structured_data.json文件: {structured_data_path}")
                        
                    
                    self.logger.info(f"总共加载了 {len(entity_mapping)} 个同名实体映射和 {len(entity_definitions)} 个实体定义")
                    
                    # 应用同名实体消解到三元组
                    resolved_triples = []
                    for triple in stage3_high_confidence_triples:
                        # 处理主语
                        subject = triple['subject']
                        if subject in entity_mapping:
                            triple['subject'] = entity_mapping[subject]
                            triple['original_subject'] = subject
                        
                        # 处理宾语
                        obj = triple['object']
                        if obj in entity_mapping:
                            triple['object'] = entity_mapping[obj]
                            triple['original_object'] = obj
                        
                        # 添加定义作为属性（如果存在）
                        if triple['subject'] in entity_definitions:
                            triple['subject_definition'] = entity_definitions[triple['subject']]
                        
                        if triple['object'] in entity_definitions and triple['predicate'].lower() != 'has definition':
                            triple['object_definition'] = entity_definitions[triple['object']]
                        
                        resolved_triples.append(triple)
                    
                    # 不再为有定义的实体添加definition三元组到resolved_triples中
                    # 定义三元组只添加到属性文件中，不添加到高置信度三元组文件
                    
                    # 将实体定义添加到属性文件中
                    self.logger.info("📝 将实体定义添加到属性文件中...")
                    attributes_file = self.stage_dirs['information_extraction'] / "extracted_attributes.jsonl"
                    
                    if attributes_file.exists():
                        try:
                            # === 修复开始：优化内存使用 ===
                            # 不再加载整个文件到 existing_attributes 列表
                            # 而是只记录已经存在 definition 的实体名称
                            existing_definition_entities = set()
                            
                            with open(attributes_file, 'r', encoding='utf-8') as f:
                                for line in f:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    try:
                                        # 逐行解析，检查完立即释放内存
                                        attr = json.loads(line)
                                        if attr.get('attribute') == 'definition':
                                            existing_definition_entities.add(attr.get('entity'))
                                    except json.JSONDecodeError:
                                        continue
                            
                            self.logger.info(f"已扫描现有属性文件，发现 {len(existing_definition_entities)} 个实体已有定义")
                            # === 修复结束 ===
                            
                            # 添加新的definition属性
                            new_attributes = []
                            for entity, definition in entity_definitions.items():
                                if entity not in existing_definition_entities and definition:
                                    new_attr = {
                                        'entity': entity,
                                        'attribute': 'definition',
                                        'value': definition,
                                        'confidence': 1.0,
                                        'text_id': 'UAT_definition',
                                        'source_id': 'uat_processed_json'
                                    }
                                    new_attributes.append(new_attr)
                            
                            # 将新属性添加到文件
                            if new_attributes:
                                with open(attributes_file, 'a', encoding='utf-8') as f:
                                    for attr in new_attributes:
                                        f.write(json.dumps(attr, ensure_ascii=False) + '\n')
                                
                                self.logger.info(f"✅ 已添加 {len(new_attributes)} 个实体定义到属性文件")
                            else:
                                self.logger.info("✅ 没有新的实体定义需要添加到属性文件")
                        
                        except Exception as e:
                            self.logger.error(f"添加实体定义到属性文件失败: {e}")
                            # 这里使用全局的 traceback，不要在函数内 import
                            self.logger.error(traceback.format_exc())
                    else:
                        self.logger.warning(f"未找到属性文件: {attributes_file}")
                    
                    # 更新三元组列表
                    stage3_high_confidence_triples = resolved_triples
                    
                    # 保存同名实体消解后的三元组
                    resolved_file = stage_dir / "resolved_triples.jsonl"
                    with open(resolved_file, 'w', encoding='utf-8') as f:
                        for triple in resolved_triples:
                            f.write(json.dumps(triple, ensure_ascii=False) + '\n')
                    
                    # 保存实体映射和定义
                    mapping_file = stage_dir / "entity_name_mapping.json"
                    with open(mapping_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'entity_mapping': entity_mapping,
                            'entity_definitions': entity_definitions
                        }, f, ensure_ascii=False, indent=2)
                    
                    self.logger.info(f"✅ 同名实体消解完成，处理了 {len(resolved_triples)} 个三元组")
                    self.logger.info(f"💾 同名实体消解结果已保存: {resolved_file}")
                    self.logger.info(f"💾 实体映射和定义已保存: {mapping_file}")
                    
                    # 使用原始高置信度三元组文件进行规范化，而不是消解后的三元组文件
                    # 这样可以确保不会将定义三元组加入规范化过程
                    temp_file = high_confidence_file
                    
                except Exception as e:
                    self.logger.error(f"同名实体消解处理失败: {e}")
                    self.logger.error(traceback.format_exc())
                    # 如果失败，使用原始三元组文件（已在开始设置为默认值）
            
            self.logger.info(f"开始规范化 {len(stage3_high_confidence_triples)} 个高置信度三元组")
            
            
            # 使用OptimizedCanonicalization的run_canonicalization方法
            try:
                self.logger.info(f"开始调用规范化器处理文件: {temp_file}")
                
                # 从配置文件读取规范化超时时间
                performance_config = self.config.get('performance', {})
                timeout_seconds = performance_config.get('canonicalization_timeout', 1200)
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
                # ❌ 删除下面这一行： import traceback 
                # import traceback  <-- 删除这行，因为它导致了 UnboundLocalError
                self.logger.error(f"异常详情: {traceback.format_exc()}")
                raise Exception(f"规范化过程失败: {e}")
            
            # 查找生成的规范化结果文件（优先在当前输出目录查找）
            canonicalized_files = list(self.output_dir.glob(f"canonicalized_triples_optimized_*.jsonl"))
            if not canonicalized_files:
                # 回退到默认路径
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
                        if 'confidence' not in triple:
                            triple['confidence'] = 1.0
                        if 'source_text' not in triple:
                            triple['source_text'] = ''
                        if 'text_id' not in triple:
                            triple['text_id'] = ''
                        canonicalized_triples.append(triple)
            
            self.stats['canonicalized_triples'] = len(canonicalized_triples)
            
            # 读取规范化映射文件（优先在当前输出目录查找）
            mapping_files = list(self.output_dir.glob(f"canonicalization_mappings_optimized_*.json"))
            if not mapping_files:
                # 回退到默认路径
                mapping_files = list(Path("data/output").glob(f"canonicalization_mappings_optimized_*.json"))
            
            relation_map = {}
            entity_map = {}
            
            if mapping_files:
                # 获取最新的映射文件
                latest_mapping_file = max(mapping_files, key=lambda x: x.stat().st_mtime)
                
                try:
                    with open(latest_mapping_file, 'r', encoding='utf-8') as f:
                        mapping_data = json.load(f)
                        relation_map = mapping_data.get('relation_mappings', {})
                        entity_map = mapping_data.get('entity_mappings', {})
                        
                    self.logger.info(f"成功读取映射文件: {latest_mapping_file}")
                    self.logger.info(f"关系映射数量: {len(relation_map)}, 实体映射数量: {len(entity_map)}")
                    
                    # 将规范化文件复制到当前阶段目录
                    stage_dir = self.stage_dirs['canonicalization']
                    target_file = stage_dir / latest_file.name
                    if not target_file.exists():
                        import shutil
                        shutil.copy2(latest_file, target_file)
                        self.logger.info(f"复制规范化结果文件到阶段目录: {target_file}")
                    
                    target_mapping = stage_dir / latest_mapping_file.name
                    if not target_mapping.exists():
                        import shutil
                        shutil.copy2(latest_mapping_file, target_mapping)
                        self.logger.info(f"复制映射文件到阶段目录: {target_mapping}")
                        
                except Exception as e:
                    self.logger.error(f"读取映射文件失败: {e}")
                    relation_map = {}
                    entity_map = {}
            else:
                self.logger.warning("未找到规范化映射文件，使用空映射")
            
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
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    async def stage_5_graph_construction(self, canonicalized_triples: List[Dict[str, Any]]) -> bool:
        """阶段5: 图数据库构建 (完整修复版)
        包含功能：
        1. 实体映射对齐：使用Stage4的映射表统一结构化数据的实体名
        2. 格式兼容：支持Wikidata/Simbad的非字典格式数据
        3. 孤立节点补全：为仅存在于结构化数据中的实体生成合成三元组，强制创建节点
        """
        self.logger.info("\n" + "="*50)
        self.logger.info("🚀 阶段5: 图数据库构建")
        self.logger.info("="*50)
        
        try:
            # =================================================================================
            # 步骤 0: 准备工作 & 加载映射
            # =================================================================================

            # 0.1 为三元组补齐来源权威与冲突融合打分元数据（V1）
            import hashlib
            now_ts = time.time()
            ts_norm_now = min(max(now_ts / 4102444800.0, 0.0), 1.0)  # 近似归一化到2100年

            source_meta_by_id = {}
            docs_file = self.output_dir / "intermediate_results" / "01_data_acquisition" / "documents.jsonl"
            if docs_file.exists():
                with open(docs_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        d = json.loads(line)
                        txt = d.get('text', '')
                        if txt:
                            sid = hashlib.md5(txt.encode('utf-8')).hexdigest()[:16]
                            source_meta_by_id[sid] = {
                                'source_url': d.get('source_url', ''),
                                'source_type': d.get('source_type', ''),
                                'origin_type': d.get('origin_type', ''),
                                'origin_value': d.get('origin_value', ''),
                                'page_title': d.get('page_title', '')
                            }

            for triple in canonicalized_triples:
                sid = triple.get('source_id', '')
                meta = source_meta_by_id.get(sid, {})
                source_url = meta.get('source_url', '')
                source_authority, authority_score = self._classify_source_authority(source_url)
                confidence_val = float(triple.get('confidence', 1.0) or 1.0)
                cs = self._compute_confidence_score(authority_score, ts_norm_now)
                # 融入抽取置信度，保证与现有审核链兼容
                cs = 0.7 * cs + 0.3 * confidence_val

                triple['source_url'] = source_url
                triple['source_authority'] = source_authority
                triple['authority_score'] = authority_score
                triple['timestamp'] = now_ts
                triple['confidence_score'] = cs

            # 1. 收集三元组中已有的所有节点
            # 用于后续判断哪些结构化数据是“孤立”的（即不在三元组关系网络中）
            existing_graph_nodes = set()
            if canonicalized_triples:
                for triple in canonicalized_triples:
                    if 'subject' in triple:
                        existing_graph_nodes.add(triple['subject'])
                    if 'object' in triple:
                        existing_graph_nodes.add(triple['object'])
            
            self.logger.info(f"从现有三元组中识别出 {len(existing_graph_nodes)} 个节点")

            # 2. 加载实体映射表 (从Stage 4输出获取)
            entity_map = {}
            try:
                # 尝试查找最新的映射文件
                mapping_files = list(self.stage_dirs['canonicalization'].glob("entity_mappings.json"))
                if mapping_files:
                    # 取最新的一个
                    latest_mapping_file = max(mapping_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_mapping_file, 'r', encoding='utf-8') as f:
                        entity_map = json.load(f)
                    self.logger.info(f"✅ 加载了 {len(entity_map)} 个实体映射规则用于对齐结构化数据")
                else:
                    self.logger.warning("⚠️ 未找到实体映射文件，结构化数据将使用原始名称")
            except Exception as e:
                self.logger.warning(f"加载实体映射失败: {e}")

            # =================================================================================
            # 步骤 1: 准备属性数据 (structured_data)
            # =================================================================================
            
            # structured_data 最终将包含所有合并后的属性
            # 格式: { "EntityName": { "attr1": "val1", ... } }
            structured_data = {} 
            
            # -----------------------------------------------------------------------------
            # 来源一：结构化文件 (InfoBox/Simbad/Wikidata) - 优先级高
            # -----------------------------------------------------------------------------
            stage1_structured_path = self.output_dir / "intermediate_results" / "01_data_acquisition" / "structured_data.json"
            
            if stage1_structured_path.exists():
                try:
                    with open(stage1_structured_path, 'r', encoding='utf-8') as f:
                        stage1_raw_data = json.load(f)
                    
                    processed_count = 0
                    
                    for item in stage1_raw_data:
                        for key, value in item.items():
                            # 识别后缀
                            suffix = ""
                            if key.endswith('_info_box'): suffix = '_info_box'
                            elif key.endswith('_simbad'): suffix = '_simbad'
                            elif key.endswith('_wikidata'): suffix = '_wikidata'
                            
                            if suffix:
                                # 1. 获取原始实体名
                                raw_entity_name = key[:-len(suffix)]
                                
                                # 2. 确定最终实体名 (应用映射)
                                # 逻辑：优先查表，查不到则保留原名（去除首尾空格）
                                # 注意：不使用 capitalize()，以免破坏专有名词大小写
                                if raw_entity_name in entity_map:
                                    entity_name = entity_map[raw_entity_name]
                                elif raw_entity_name.lower() in entity_map:
                                    entity_name = entity_map[raw_entity_name.lower()]
                                else:
                                    entity_name = raw_entity_name.strip()

                                if entity_name not in structured_data:
                                    structured_data[entity_name] = {}

                                # 3. 处理属性值 (兼容多种格式)
                                if isinstance(value, dict):
                                    for attr_k, attr_v in value.items():
                                        final_val = attr_v
                                        
                                        # Wikidata 特殊处理
                                        if suffix == '_wikidata':
                                            if not isinstance(final_val, str):
                                                final_val = str(final_val)
                                        
                                        # Simbad 特殊处理 (可能是嵌套字典或列表)
                                        if suffix == '_simbad':
                                            if isinstance(attr_v, dict):
                                                final_val = attr_v.get('value', json.dumps(attr_v, ensure_ascii=False))
                                            elif isinstance(attr_v, list):
                                                final_val = ", ".join([str(x) for x in attr_v])
                                        
                                        structured_data[entity_name][attr_k] = final_val
                                else:
                                    # 如果 value 本身不是字典（例如是列表或字符串），将其作为整体存入
                                    attr_key = f"raw_data{suffix}"
                                    if isinstance(value, (list, str, int, float)):
                                        structured_data[entity_name][attr_key] = str(value)

                                processed_count += 1
                    
                    self.logger.info(f"从结构化文件加载并处理了 {processed_count} 条数据")
                    
                except Exception as e:
                    self.logger.error(f"处理结构化数据文件失败: {e}")
                    self.logger.error(traceback.format_exc())
            else:
                self.logger.warning(f"未找到结构化数据文件: {stage1_structured_path}")

            # -----------------------------------------------------------------------------
            # 来源二：文本抽取属性 (Text Extraction) - 优先级低，作为补充
            # -----------------------------------------------------------------------------
            if hasattr(self, 'all_attributes') and self.all_attributes:
                text_attr_count = 0
                for attr in self.all_attributes:
                    try:
                        # 过滤低置信度属性
                        if attr.get('confidence', 0.0) < 0.9: 
                            continue
                        
                        # 解析实体名
                        raw_name_info = attr.get('entity', '')
                        raw_name = ""
                        if isinstance(raw_name_info, str):
                            # 尝试处理可能的JSON字符串
                            if raw_name_info.startswith('{'):
                                try:
                                    
                                    d = json.loads(raw_name_info)
                                    raw_name = d.get('entity_name', str(raw_name_info))
                                except:
                                    raw_name = raw_name_info
                            else:
                                raw_name = raw_name_info
                        else:
                            raw_name = str(raw_name_info)
                        
                        # 应用映射
                        if raw_name in entity_map:
                            entity_name = entity_map[raw_name]
                        else:
                            entity_name = raw_name
                        
                        if entity_name:
                            if entity_name not in structured_data:
                                structured_data[entity_name] = {}
                            
                            k = attr.get('attribute', '')
                            v = attr.get('value', '')
                            
                            # 排除无效属性
                            if k and v and k not in ['text_id', 'source_id']:
                                # 只有当该属性不存在时才写入（避免覆盖结构化文件的高质量数据）
                                if k not in structured_data[entity_name]:
                                    structured_data[entity_name][k] = v
                                    text_attr_count += 1
                    except Exception as e:
                        continue
                self.logger.info(f"从文本抽取中补充了 {text_attr_count} 个属性")

            # =================================================================================
            # 步骤 2: 关键修复 - 为孤立的结构化数据生成“合成三元组”
            # =================================================================================
            # 目的：确保那些只存在于InfoBox/Simbad但没有被文本抽取到关系的实体也能成为图谱节点
            
            synthetic_triples = []
            orphan_entities = []
            
            for entity_name in structured_data.keys():
                if entity_name not in existing_graph_nodes:
                    orphan_entities.append(entity_name)
                    # 构造一个合成三元组
                    # (Entity) -[data_source]-> (Structured Data)
                    synthetic_triple = {
                        'subject': entity_name,
                        'predicate': 'data_source',
                        'object': 'Structured Data',
                        'confidence': 1.0,
                        'source_text': 'Imported from structured data files',
                        'text_id': 'system_import_synthetic'
                    }
                    synthetic_triples.append(synthetic_triple)
            
            if synthetic_triples:
                self.logger.info(f"⚠️ 发现 {len(synthetic_triples)} 个仅存在于结构化数据中的孤立实体")
                self.logger.info(f"🔄 正在生成合成三元组以强制插入节点 (示例: {orphan_entities[:3]}...)")
                
                # 将合成三元组加入到主列表中
                canonicalized_triples.extend(synthetic_triples)
            else:
                self.logger.info("✅ 所有结构化数据实体均已存在于三元组网络中")

            # =================================================================================
            # 步骤 3: 准备事件数据
            # =================================================================================
            events_to_insert = []
            if hasattr(self, 'all_events') and self.all_events:
                for event in self.all_events:
                    if event.get('confidence', 0.0) >= 0.9:
                        # 同样对事件实体应用映射
                        if 'entity' in event and event['entity'] in entity_map:
                            event['entity'] = entity_map[event['entity']]
                        events_to_insert.append(event)
                self.logger.info(f"准备了 {len(events_to_insert)} 个高置信度事件")

            # =================================================================================
            # 步骤 4: 执行插入
            # =================================================================================
            # 为冲突融合补充权威度/时间戳/CS
            canonicalized_triples = self._enrich_triples_for_conflict_resolution(canonicalized_triples)

            self.logger.info(f"开始调用 GraphArchitect 构建图谱...")
            self.logger.info(f"  - 三元组数量: {len(canonicalized_triples)} (含合成)")
            self.logger.info(f"  - 实体属性集: {len(structured_data)} 个实体")
            
            entity_map_result = await self.agents['constructor'].build_and_persist(
                normalized_triples=canonicalized_triples,
                structured_data=structured_data,
                events=events_to_insert
            )
            
            # =================================================================================
            # 步骤 5: 统计与保存
            # =================================================================================
            self.stats['inserted_triples'] = len(canonicalized_triples)
            self.stats['inserted_events'] = len(events_to_insert)
            self.stats['total_entities'] = len(entity_map_result)
            
            final_result = {
                'inserted_triples': canonicalized_triples,
                'entity_map': entity_map_result,
                'stats': self.stats,
                'timestamp': self.timestamp
            }
            
            self.save_intermediate_result('graph_construction', final_result)
            
            # 保存最终插入的三元组（包含合成的）
            stage_dir = self.stage_dirs['graph_construction']
            inserted_file = stage_dir / "inserted_triples.jsonl"
            with open(inserted_file, 'w', encoding='utf-8') as f:
                for triple in canonicalized_triples:
                    f.write(json.dumps(triple, ensure_ascii=False) + '\n')
            
            # 生成报告
            stats = {
                'inserted_triples': len(canonicalized_triples),
                'synthetic_triples_added': len(synthetic_triples),
                'total_entities_in_graph': len(entity_map_result),
                'total_pipeline_stats': self.stats,
                'execution_time': 0
            }
            details = {
                'graph_constructor': 'GraphArchitect',
                'database': 'Neo4j',
                'connection_uri': self.config['neo4j']['uri'],
                'orphan_nodes_handled': True
            }
            self.generate_agent_report('graph_construction', stats, details)
            
            self.logger.info(f"\n✅ 图数据库构建完成:")
            self.logger.info(f"  - 最终插入三元组数: {len(canonicalized_triples)}")
            self.logger.info(f"  - 图谱总实体数: {len(entity_map_result)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"图数据库构建阶段失败: {e}")
            self.logger.error(traceback.format_exc())

            # 默认允许在Neo4j不可用时降级成功，保留前四阶段产物可用性
            # 如需严格失败，可在配置中设置: pipeline.fail_on_graph_error: true
            fail_on_graph_error = self.config.get('pipeline', {}).get('fail_on_graph_error', False)
            if fail_on_graph_error:
                return False

            self.logger.warning("⚠️ Neo4j入库失败，已降级为仅产出文件结果（前四阶段结果可正常使用）")
            self.stats['inserted_triples'] = 0
            self.stats['inserted_events'] = 0
            return True
    
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
            
            # 检查是否可以从中间结果恢复
            recovery_info = {'can_resume': False}
            if self.resume:
                recovery_info = self.check_intermediate_results()
                if recovery_info['can_resume']:
                    self.logger.info(f"🔄 检测到中间结果文件，可以从 {recovery_info['resume_from_stage']} 阶段恢复")
                else:
                    self.logger.info("未检测到有效的中间结果文件，将执行完整流程")
            else:
                self.logger.info("未启用恢复模式(resume=False)，将执行完整流程")
            
            if recovery_info['can_resume']:
                self.logger.info(f"🔄 检测到中间结果文件，可以从 {recovery_info['resume_from_stage']} 阶段恢复")
                
                # 根据最后完成的阶段决定从哪里开始
                resume_stage = recovery_info['resume_from_stage']
                
                if resume_stage == 'stage_3_knowledge_auditing':
                    # 从规范化阶段开始
                    self.logger.info("从规范化阶段开始恢复执行")
                    stage_info = recovery_info['available_data']['stage_3_knowledge_auditing']
                    high_confidence_triples = self.load_intermediate_data('stage_3_knowledge_auditing', stage_info['key_file'])
                    
                    # 尝试加载属性数据（从信息提取阶段）
                    if 'stage_2_information_extraction' in recovery_info['available_data']:
                        extraction_stage_info = recovery_info['available_data']['stage_2_information_extraction']
                        attributes_file = extraction_stage_info['stage_dir'] / 'extracted_attributes.jsonl'
                        if attributes_file.exists():
                            self.all_attributes = self.load_intermediate_data('stage_2_information_extraction', attributes_file)
                            self.logger.info(f"从中间结果加载了 {len(self.all_attributes)} 个属性")
                        else:
                            self.all_attributes = []
                            self.logger.info("未找到属性数据文件，使用空属性列表")
                            
                        # 尝试加载事件数据
                        events_file = extraction_stage_info['stage_dir'] / 'extracted_events.jsonl'
                        if events_file.exists():
                            self.all_events = self.load_intermediate_data('stage_2_information_extraction', events_file)
                            self.logger.info(f"从中间结果加载了 {len(self.all_events)} 个事件")
                        else:
                            self.all_events = []
                            self.logger.info("未找到事件数据文件，使用空事件列表")
                    else:
                        self.all_attributes = []
                        self.all_events = []
                        self.logger.info("未找到信息提取阶段数据，使用空属性列表和空事件列表")
                    
                    if high_confidence_triples:
                        self.stats['high_confidence_triples'] = len(high_confidence_triples)
                        self.logger.info(f"从中间结果加载了 {len(high_confidence_triples)} 个高置信度三元组")
                        
                        # 直接跳转到规范化阶段
                        canonicalized_triples = await self.stage_4_canonicalization(high_confidence_triples)
                        success = await self.stage_5_graph_construction(canonicalized_triples)
                        
                        # 跳转到最终统计部分
                        end_time = time.time()
                        duration = end_time - start_time
                        return await self._finalize_pipeline(success, duration)
                    else:
                        self.logger.warning("无法从中间结果加载数据，将执行完整流程")
                
                elif resume_stage == 'stage_1_data_acquisition':
                    # 从信息提取阶段开始
                    self.logger.info("从信息提取阶段开始恢复执行")
                    stage_info = recovery_info['available_data']['stage_1_data_acquisition']
                    documents = self.load_intermediate_data('stage_1_data_acquisition', stage_info['key_file'])
                    
                    if documents:
                        self.stats['total_documents'] = len(documents)
                        self.logger.info(f"从中间结果加载了 {len(documents)} 个文档")
                        
                        # 执行信息提取阶段
                        triples = await self.stage_2_information_extraction(documents)
                        
                        # 执行知识审核阶段
                        if self.skip_audit:
                            self.logger.info("⏭️ 跳过知识审核阶段")
                            high_confidence_triples = triples
                            low_confidence_triples = []
                            self.stats['high_confidence_triples'] = len(high_confidence_triples)
                            self.stats['low_confidence_triples'] = 0
                        else:
                            high_confidence_triples, low_confidence_triples = await self.stage_3_knowledge_auditing(triples)
                        
                        # 继续执行后续阶段
                        canonicalized_triples = await self.stage_4_canonicalization(high_confidence_triples)
                        success = await self.stage_5_graph_construction(canonicalized_triples)
                        
                        # 跳转到最终统计部分
                        end_time = time.time()
                        duration = end_time - start_time
                        return await self._finalize_pipeline(success, duration)
                    else:
                        self.logger.warning("无法从中间结果加载数据，将执行完整流程")
                
                elif resume_stage == 'stage_2_information_extraction':
                    # 从知识审核阶段开始
                    self.logger.info("从知识审核阶段开始恢复执行")
                    stage_info = recovery_info['available_data']['stage_2_information_extraction']
                    
                    # 检查是否有extracted_relations.jsonl文件（新架构）
                    relations_file = stage_info['stage_dir'] / 'extracted_relations.jsonl'
                    if relations_file.exists():
                        triples = self.load_intermediate_data('stage_2_information_extraction', relations_file)
                    else:
                        # 回退到旧的high_confidence_triples.jsonl
                        triples = self.load_intermediate_data('stage_2_information_extraction', stage_info['key_file'])
                    
                    # 加载属性数据
                    attributes_file = stage_info['stage_dir'] / 'extracted_attributes.jsonl'
                    if attributes_file.exists():
                        self.all_attributes = self.load_intermediate_data('stage_2_information_extraction', attributes_file)
                        self.logger.info(f"从中间结果加载了 {len(self.all_attributes)} 个属性")
                    else:
                        self.all_attributes = []
                        self.logger.info("未找到属性数据文件，使用空属性列表")
                    
                    # 加载事件数据
                    events_file = stage_info['stage_dir'] / 'extracted_events.jsonl'
                    if events_file.exists():
                        self.all_events = self.load_intermediate_data('stage_2_information_extraction', events_file)
                        self.logger.info(f"从中间结果加载了 {len(self.all_events)} 个事件")
                    else:
                        self.all_events = []
                        self.logger.info("未找到事件数据文件，使用空事件列表")
                    
                    if triples:
                        self.stats['total_extracted_triples'] = len(triples)
                        self.logger.info(f"从中间结果加载了 {len(triples)} 个三元组")
                        
                        # 执行知识审核阶段
                        if self.skip_audit:
                            self.logger.info("⏭️ 跳过知识审核阶段")
                            high_confidence_triples = triples
                            low_confidence_triples = []
                            self.stats['high_confidence_triples'] = len(high_confidence_triples)
                            self.stats['low_confidence_triples'] = 0
                        else:
                            high_confidence_triples, low_confidence_triples = await self.stage_3_knowledge_auditing(triples)
                        
                        # 继续执行后续阶段
                        canonicalized_triples = await self.stage_4_canonicalization(high_confidence_triples)
                        success = await self.stage_5_graph_construction(canonicalized_triples)
                        
                        # 跳转到最终统计部分
                        end_time = time.time()
                        duration = end_time - start_time
                        return await self._finalize_pipeline(success, duration)
                    else:
                        self.logger.warning("无法从中间结果加载数据，将执行完整流程")
                
                elif resume_stage == 'stage_4_canonicalization':
                    # 从图谱构建阶段开始
                    self.logger.info("从图谱构建阶段开始恢复执行")
                    stage_info = recovery_info['available_data']['stage_4_canonicalization']
                    canonicalized_triples = self.load_intermediate_data('stage_4_canonicalization', stage_info['key_file'])
                    
                    # 尝试加载属性数据（从信息提取阶段）
                    if 'stage_2_information_extraction' in recovery_info['available_data']:
                        extraction_stage_info = recovery_info['available_data']['stage_2_information_extraction']
                        attributes_file = extraction_stage_info['stage_dir'] / 'extracted_attributes.jsonl'
                        if attributes_file.exists():
                            self.all_attributes = self.load_intermediate_data('stage_2_information_extraction', attributes_file)
                            self.logger.info(f"从中间结果加载了 {len(self.all_attributes)} 个属性")
                        else:
                            self.all_attributes = []
                            self.logger.info("未找到属性数据文件，使用空属性列表")
                            
                        # 尝试加载事件数据
                        events_file = extraction_stage_info['stage_dir'] / 'extracted_events.jsonl'
                        if events_file.exists():
                            self.all_events = self.load_intermediate_data('stage_2_information_extraction', events_file)
                            self.logger.info(f"从中间结果加载了 {len(self.all_events)} 个事件")
                        else:
                            self.all_events = []
                            self.logger.info("未找到事件数据文件，使用空事件列表")
                    else:
                        self.all_attributes = []
                        self.all_events = []
                        self.logger.info("未找到信息提取阶段数据，使用空属性列表和空事件列表")
                    
                    if canonicalized_triples:
                        self.stats['canonicalized_triples'] = len(canonicalized_triples)
                        self.logger.info(f"从中间结果加载了 {len(canonicalized_triples)} 个规范化三元组")
                        
                        # 直接跳转到图谱构建阶段
                        success = await self.stage_5_graph_construction(canonicalized_triples)
                        
                        # 跳转到最终统计部分
                        end_time = time.time()
                        duration = end_time - start_time
                        return await self._finalize_pipeline(success, duration)
                    else:
                        self.logger.warning("无法从中间结果加载数据，将执行完整流程")
            
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
            
            return await self._finalize_pipeline(success, duration)
            
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
    parser.add_argument('--multi-entity', action='store_true', help='启用多实体属性抽取模式')
    parser.add_argument('--resume', action='store_true', help='接续上一轮处理，若不接续则从头运行')
    parser.add_argument('--output-dir', default=None, help='自定义输出文件夹路径，默认为配置文件中指定的路径')
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    try:
        # 创建并运行Pipeline
        pipeline = IntegratedPipeline(
            args.config, 
            skip_audit=args.skip_audit, 
            multi_entity_mode=args.multi_entity,
            resume=args.resume,
            output_dir=args.output_dir
        )
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
