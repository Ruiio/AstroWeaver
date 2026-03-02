import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import asyncio

# 导入增强的组件
from .html_parse_enhanced import EnhancedHTMLParser
from .webSearch_enhanced import EnhancedWebSearcher
from .get_wikipedia_article_enhanced import EnhancedWikipediaExtractor
from .wikipedia_client_enhanced import EnhancedWikipediaClient
from .triple_postprocessor import TriplePostProcessor

logger = logging.getLogger(__name__)

class DataQualityManager:
    """数据质量管理器，整合所有数据质量增强功能"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 初始化增强组件
        self.html_parser = EnhancedHTMLParser()
        self.web_searcher = EnhancedWebSearcher()
        self.wiki_extractor = EnhancedWikipediaExtractor()
        self.wiki_client = EnhancedWikipediaClient()
        self.triple_processor = TriplePostProcessor(config.get('triple_processing', {}))
        
        # 质量指标
        self.quality_metrics = {
            'data_sources': {
                'web_content_quality': 0.0,
                'wikipedia_content_quality': 0.0,
                'search_result_quality': 0.0
            },
            'triple_quality': {
                'validity_rate': 0.0,
                'uniqueness_rate': 0.0,
                'confidence_score': 0.0
            },
            'overall_quality': 0.0
        }
        
        # 处理统计
        self.processing_stats = {
            'web_content_processed': 0,
            'wikipedia_articles_processed': 0,
            'search_queries_processed': 0,
            'triples_processed': 0,
            'quality_improvements': []
        }
    
    async def enhance_web_content(self, url: str) -> Dict[str, Any]:
        """增强网页内容获取"""
        logger.info(f"增强网页内容获取: {url}")
        
        try:
            # 使用增强的HTML解析器
            content_blocks = self.html_parser.get_web_content(url)
            
            if not content_blocks:
                logger.warning(f"未能获取有效内容: {url}")
                return {
                    'url': url,
                    'content_blocks': [],
                    'quality_score': 0.0,
                    'status': 'failed',
                    'reason': 'no_content'
                }
            
            # 计算内容质量分数
            quality_score = self._calculate_content_quality(content_blocks)
            
            self.processing_stats['web_content_processed'] += 1
            self.quality_metrics['data_sources']['web_content_quality'] = (
                (self.quality_metrics['data_sources']['web_content_quality'] * 
                 (self.processing_stats['web_content_processed'] - 1) + quality_score) /
                self.processing_stats['web_content_processed']
            )
            
            return {
                'url': url,
                'content_blocks': content_blocks,
                'quality_score': quality_score,
                'status': 'success',
                'block_count': len(content_blocks)
            }
            
        except Exception as e:
            logger.error(f"增强网页内容获取失败 {url}: {e}")
            return {
                'url': url,
                'content_blocks': [],
                'quality_score': 0.0,
                'status': 'error',
                'error': str(e)
            }
    
    async def enhance_web_search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """增强网络搜索"""
        logger.info(f"增强网络搜索: {query}")
        
        try:
            # 使用增强的网络搜索器
            search_results = await self.web_searcher.enhanced_search(query, num_results)
            
            if not search_results:
                logger.warning(f"搜索无结果: {query}")
                return {
                    'query': query,
                    'results': [],
                    'quality_score': 0.0,
                    'status': 'no_results'
                }
            
            # 计算搜索结果质量
            quality_score = self._calculate_search_quality(search_results)
            
            self.processing_stats['search_queries_processed'] += 1
            self.quality_metrics['data_sources']['search_result_quality'] = (
                (self.quality_metrics['data_sources']['search_result_quality'] * 
                 (self.processing_stats['search_queries_processed'] - 1) + quality_score) /
                self.processing_stats['search_queries_processed']
            )
            
            return {
                'query': query,
                'results': search_results,
                'quality_score': quality_score,
                'status': 'success',
                'result_count': len(search_results)
            }
            
        except Exception as e:
            logger.error(f"增强网络搜索失败 {query}: {e}")
            return {
                'query': query,
                'results': [],
                'quality_score': 0.0,
                'status': 'error',
                'error': str(e)
            }
    
    async def enhance_wikipedia_content(self, title: str) -> Dict[str, Any]:
        """增强Wikipedia内容获取"""
        logger.info(f"增强Wikipedia内容获取: {title}")
        
        try:
            # 使用增强的Wikipedia客户端
            article_data = self.wiki_client.get_enhanced_article_data(title)
            
            if not article_data['sections'] and not article_data['infobox']:
                logger.warning(f"未能获取Wikipedia内容: {title}")
                return {
                    'title': title,
                    'sections': [],
                    'infobox': {},
                    'quality_score': 0.0,
                    'status': 'no_content'
                }
            
            # 计算内容质量分数
            quality_score = self._calculate_wikipedia_quality(article_data)
            
            self.processing_stats['wikipedia_articles_processed'] += 1
            self.quality_metrics['data_sources']['wikipedia_content_quality'] = (
                (self.quality_metrics['data_sources']['wikipedia_content_quality'] * 
                 (self.processing_stats['wikipedia_articles_processed'] - 1) + quality_score) /
                self.processing_stats['wikipedia_articles_processed']
            )
            
            return {
                'title': title,
                'sections': article_data['sections'],
                'infobox': article_data['infobox'],
                'metadata': article_data['metadata'],
                'quality_score': quality_score,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"增强Wikipedia内容获取失败 {title}: {e}")
            return {
                'title': title,
                'sections': [],
                'infobox': {},
                'quality_score': 0.0,
                'status': 'error',
                'error': str(e)
            }
    
    def enhance_triples(self, triples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """增强三元组质量"""
        logger.info(f"增强三元组质量: {len(triples)} 个三元组")
        
        try:
            # 分析原始质量
            original_analysis = self.triple_processor.analyze_triples(triples)
            
            # 处理三元组
            enhanced_triples = self.triple_processor.process_triples(triples)
            
            # 分析处理后质量
            enhanced_analysis = self.triple_processor.analyze_triples(enhanced_triples)
            
            # 计算质量改进
            quality_improvement = self._calculate_triple_quality_improvement(
                original_analysis, enhanced_analysis
            )
            
            # 更新统计
            self.processing_stats['triples_processed'] += len(triples)
            self.processing_stats['quality_improvements'].append(quality_improvement)
            
            # 更新质量指标
            self._update_triple_quality_metrics(enhanced_analysis)
            
            return {
                'original_count': len(triples),
                'enhanced_count': len(enhanced_triples),
                'enhanced_triples': enhanced_triples,
                'original_analysis': original_analysis,
                'enhanced_analysis': enhanced_analysis,
                'quality_improvement': quality_improvement,
                'processing_stats': self.triple_processor.get_statistics(),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"增强三元组质量失败: {e}")
            return {
                'original_count': len(triples),
                'enhanced_count': 0,
                'enhanced_triples': [],
                'status': 'error',
                'error': str(e)
            }
    
    def _calculate_content_quality(self, content_blocks: List[str]) -> float:
        """计算内容质量分数"""
        if not content_blocks:
            return 0.0
        
        total_score = 0.0
        
        for block in content_blocks:
            score = 0.0
            
            # 长度分数 (0.3权重)
            if 50 <= len(block) <= 1000:
                score += 0.3
            elif len(block) > 1000:
                score += 0.2
            elif len(block) > 20:
                score += 0.1
            
            # 内容丰富度 (0.3权重)
            words = block.split()
            if len(words) >= 10:
                score += 0.3
            elif len(words) >= 5:
                score += 0.2
            elif len(words) >= 2:
                score += 0.1
            
            # 语言质量 (0.2权重)
            alpha_ratio = sum(1 for c in block if c.isalpha()) / len(block)
            if alpha_ratio >= 0.7:
                score += 0.2
            elif alpha_ratio >= 0.5:
                score += 0.1
            
            # 结构化程度 (0.2权重)
            if '.' in block and len(block.split('.')) >= 2:
                score += 0.2
            elif ',' in block:
                score += 0.1
            
            total_score += score
        
        return total_score / len(content_blocks)
    
    def _calculate_search_quality(self, search_results: List[Dict]) -> float:
        """计算搜索结果质量分数"""
        if not search_results:
            return 0.0
        
        total_score = 0.0
        
        for result in search_results:
            score = 0.0
            
            # URL质量 (0.3权重)
            url = result.get('url', '')
            if any(domain in url for domain in ['.edu', '.gov', '.org']):
                score += 0.3
            elif 'wikipedia' in url:
                score += 0.25
            elif any(domain in url for domain in ['.com', '.net']):
                score += 0.15
            
            # 标题质量 (0.3权重)
            title = result.get('title', '')
            if len(title) >= 20 and len(title) <= 100:
                score += 0.3
            elif len(title) >= 10:
                score += 0.2
            
            # 摘要质量 (0.2权重)
            snippet = result.get('snippet', '')
            if len(snippet) >= 50:
                score += 0.2
            elif len(snippet) >= 20:
                score += 0.1
            
            # 相关性分数 (0.2权重)
            relevance_score = result.get('relevance_score', 0.5)
            score += relevance_score * 0.2
            
            total_score += score
        
        return total_score / len(search_results)
    
    def _calculate_wikipedia_quality(self, article_data: Dict) -> float:
        """计算Wikipedia内容质量分数"""
        score = 0.0
        
        # 章节质量 (0.5权重)
        sections = article_data.get('sections', [])
        if sections:
            section_score = min(len(sections) / 10, 1.0) * 0.3  # 章节数量
            
            # 章节内容质量
            content_quality = 0.0
            for section in sections:
                if len(section) >= 100:
                    content_quality += 0.1
                elif len(section) >= 50:
                    content_quality += 0.05
            
            section_score += min(content_quality, 0.2)
            score += section_score
        
        # Infobox质量 (0.3权重)
        infobox = article_data.get('infobox', {})
        if infobox:
            infobox_score = min(len(infobox) / 20, 1.0) * 0.3
            score += infobox_score
        
        # 元数据质量 (0.2权重)
        metadata = article_data.get('metadata', {})
        if metadata and metadata.get('exists', False):
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_triple_quality_improvement(self, original: Dict, enhanced: Dict) -> Dict:
        """计算三元组质量改进"""
        improvement = {
            'null_reduction': 0,
            'duplicate_reduction': 0,
            'self_reference_reduction': 0,
            'overall_improvement': 0.0
        }
        
        if original['total_count'] > 0:
            improvement['null_reduction'] = original['null_values'] - enhanced['null_values']
            improvement['duplicate_reduction'] = original['duplicate_signatures'] - enhanced['duplicate_signatures']
            improvement['self_reference_reduction'] = original['self_references'] - enhanced['self_references']
            
            # 计算总体改进率
            original_issues = (original['null_values'] + original['duplicate_signatures'] + 
                             original['self_references'] + original['low_confidence'])
            enhanced_issues = (enhanced['null_values'] + enhanced['duplicate_signatures'] + 
                             enhanced['self_references'] + enhanced['low_confidence'])
            
            if original_issues > 0:
                improvement['overall_improvement'] = (original_issues - enhanced_issues) / original_issues
        
        return improvement
    
    def _update_triple_quality_metrics(self, analysis: Dict):
        """更新三元组质量指标"""
        if analysis['total_count'] > 0:
            # 有效性率
            valid_count = (analysis['total_count'] - analysis['null_values'] - 
                          analysis['self_references'])
            self.quality_metrics['triple_quality']['validity_rate'] = valid_count / analysis['total_count']
            
            # 唯一性率
            unique_count = analysis['total_count'] - analysis['duplicate_signatures']
            self.quality_metrics['triple_quality']['uniqueness_rate'] = unique_count / analysis['total_count']
            
            # 置信度分数
            high_conf = analysis['confidence_distribution']['high']
            medium_conf = analysis['confidence_distribution']['medium']
            low_conf = analysis['confidence_distribution']['low']
            total_conf = high_conf + medium_conf + low_conf
            
            if total_conf > 0:
                self.quality_metrics['triple_quality']['confidence_score'] = (
                    (high_conf * 1.0 + medium_conf * 0.6 + low_conf * 0.3) / total_conf
                )
    
    def calculate_overall_quality(self) -> float:
        """计算总体数据质量分数"""
        data_source_avg = (
            self.quality_metrics['data_sources']['web_content_quality'] +
            self.quality_metrics['data_sources']['wikipedia_content_quality'] +
            self.quality_metrics['data_sources']['search_result_quality']
        ) / 3
        
        triple_avg = (
            self.quality_metrics['triple_quality']['validity_rate'] +
            self.quality_metrics['triple_quality']['uniqueness_rate'] +
            self.quality_metrics['triple_quality']['confidence_score']
        ) / 3
        
        overall_quality = (data_source_avg * 0.4 + triple_avg * 0.6)
        self.quality_metrics['overall_quality'] = overall_quality
        
        return overall_quality
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """生成数据质量报告"""
        overall_quality = self.calculate_overall_quality()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_quality_score': overall_quality,
            'quality_metrics': self.quality_metrics,
            'processing_statistics': self.processing_stats,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """生成质量改进建议"""
        recommendations = []
        
        # 数据源建议
        if self.quality_metrics['data_sources']['web_content_quality'] < 0.6:
            recommendations.append("建议改进网页内容过滤规则，提高内容质量")
        
        if self.quality_metrics['data_sources']['wikipedia_content_quality'] < 0.7:
            recommendations.append("建议优化Wikipedia内容提取策略")
        
        if self.quality_metrics['data_sources']['search_result_quality'] < 0.6:
            recommendations.append("建议改进搜索结果排序和过滤算法")
        
        # 三元组建议
        if self.quality_metrics['triple_quality']['validity_rate'] < 0.8:
            recommendations.append("建议加强三元组验证规则，过滤更多无效数据")
        
        if self.quality_metrics['triple_quality']['uniqueness_rate'] < 0.9:
            recommendations.append("建议改进重复检测算法")
        
        if self.quality_metrics['triple_quality']['confidence_score'] < 0.7:
            recommendations.append("建议调整置信度计算方法或提高提取质量")
        
        if not recommendations:
            recommendations.append("数据质量良好，继续保持当前处理策略")
        
        return recommendations
    
    def save_quality_report(self, output_dir: str):
        """保存质量报告"""
        report = self.generate_quality_report()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"data_quality_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据质量报告已保存: {report_file}")
        return str(report_file)
    
    async def process_pipeline_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理管道数据，应用所有质量增强"""
        logger.info("开始处理管道数据，应用质量增强")
        
        results = {
            'web_content': [],
            'wikipedia_content': [],
            'search_results': [],
            'enhanced_triples': [],
            'quality_report': None
        }
        
        try:
            # 处理网页内容
            if 'urls' in input_data:
                for url in input_data['urls']:
                    result = await self.enhance_web_content(url)
                    results['web_content'].append(result)
            
            # 处理Wikipedia内容
            if 'wikipedia_titles' in input_data:
                for title in input_data['wikipedia_titles']:
                    result = await self.enhance_wikipedia_content(title)
                    results['wikipedia_content'].append(result)
            
            # 处理搜索查询
            if 'search_queries' in input_data:
                for query in input_data['search_queries']:
                    result = await self.enhance_web_search(query)
                    results['search_results'].append(result)
            
            # 处理三元组
            if 'triples' in input_data:
                result = self.enhance_triples(input_data['triples'])
                results['enhanced_triples'] = result['enhanced_triples']
            
            # 生成质量报告
            results['quality_report'] = self.generate_quality_report()
            
            logger.info("管道数据处理完成")
            return results
            
        except Exception as e:
            logger.error(f"管道数据处理失败: {e}")
            results['error'] = str(e)
            return results


# 便捷函数
async def enhance_data_quality(input_file: str, output_dir: str, config: Optional[Dict] = None) -> str:
    """增强数据质量的便捷函数"""
    manager = DataQualityManager(config)
    
    # 加载输入数据
    with open(input_file, 'r', encoding='utf-8') as f:
        if input_file.endswith('.json'):
            input_data = json.load(f)
        else:
            # 假设是JSONL格式的三元组
            triples = []
            for line in f:
                if line.strip():
                    triples.append(json.loads(line.strip()))
            input_data = {'triples': triples}
    
    # 处理数据
    results = await manager.process_pipeline_data(input_data)
    
    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存增强的三元组
    if results['enhanced_triples']:
        triples_file = output_path / f"enhanced_triples_{timestamp}.jsonl"
        with open(triples_file, 'w', encoding='utf-8') as f:
            for triple in results['enhanced_triples']:
                f.write(json.dumps(triple, ensure_ascii=False) + '\n')
    
    # 保存完整结果
    results_file = output_path / f"enhanced_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存质量报告
    report_file = manager.save_quality_report(output_dir)
    
    logger.info(f"数据质量增强完成，结果保存到: {output_dir}")
    return report_file


if __name__ == "__main__":
    # 测试数据质量管理器
    async def test_manager():
        manager = DataQualityManager()
        
        # 测试数据
        test_data = {
            'urls': ['https://en.wikipedia.org/wiki/Mars'],
            'wikipedia_titles': ['Mars', 'Jupiter'],
            'search_queries': ['Mars planet astronomy'],
            'triples': [
                {'subject': 'Mars', 'predicate': 'orbits', 'object': 'Sun', 'confidence': 0.9},
                {'subject': 'null', 'predicate': 'hasProperty', 'object': 'null', 'confidence': 0.5},
                {'subject': 'Mars', 'predicate': 'orbits', 'object': 'Sun', 'confidence': 0.8},
            ]
        }
        
        # 处理数据
        results = await manager.process_pipeline_data(test_data)
        
        # 生成报告
        report = manager.generate_quality_report()
        
        print("数据质量报告:")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # 运行测试
    asyncio.run(test_manager())