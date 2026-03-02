import re
import logging
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
import json
from pathlib import Path
from .config import config

logger = logging.getLogger(__name__)

class TriplePostProcessor:
    """增强的三元组后处理器，用于过滤无效和重复的三元组，提升数据质量"""
    
    def __init__(self, config_param: Optional[Dict] = None):
        from .config import config as global_config
        self.config = config_param or global_config or {}
        
        # 无效值模式
        self.invalid_patterns = {
            r'^\s*null\s*$',
            r'^\s*none\s*$',
            r'^\s*n/?a\s*$',
            r'^\s*unknown\s*$',
            r'^\s*undefined\s*$',
            r'^\s*empty\s*$',
            r'^\s*[-–—]+\s*$',
            r'^\s*[?]+\s*$',
            r'^\s*tbd\s*$',
            r'^\s*to\s+be\s+determined\s*$',
            r'^\s*$',  # 空字符串
        }
        
        # 低质量指标
        self.low_quality_indicators = {
            'disambiguation', 'may refer to', 'stub', 'redirect',
            'see also', 'category:', 'template:', 'file:', 'image:',
            'wikipedia:', 'user:', 'talk:', 'help:', 'portal:'
        }
        
        # 无意义的关系
        self.meaningless_relations = {
            'is', 'has', 'contains', 'includes', 'related to',
            'associated with', 'connected to', 'linked to',
            'similar to', 'comparable to', 'like', 'such as'
        }
        
        # 最小/最大长度限制
        self.min_length = 2
        self.max_length = 200
        
        # 置信度阈值
        self.min_confidence = self.config.get('confidence', {}).get('min_confidence', 0.3)
        
        # 统计信息
        self.stats = {
            'total_input': 0,
            'filtered_null': 0,
            'filtered_invalid': 0,
            'filtered_duplicate': 0,
            'filtered_low_confidence': 0,
            'filtered_meaningless': 0,
            'filtered_self_reference': 0,
            'filtered_low_quality': 0,
            'total_output': 0
        }
    
    def _is_invalid_value(self, value: str) -> bool:
        """检查值是否无效"""
        if not value or not isinstance(value, str):
            return True
        
        value_lower = value.lower().strip()
        
        # 检查无效模式
        for pattern in self.invalid_patterns:
            if re.match(pattern, value_lower, re.IGNORECASE):
                return True
        
        # 检查长度
        if len(value.strip()) < self.min_length or len(value.strip()) > self.max_length:
            return True
        
        # 检查是否只包含特殊字符
        if re.match(r'^[^a-zA-Z0-9\u4e00-\u9fff]+$', value.strip()):
            return True
        
        return False
    
    def _is_low_quality_content(self, text: str) -> bool:
        """检查内容是否为低质量"""
        if not text:
            return True
        
        text_lower = text.lower().strip()
        
        # 检查低质量指标
        for indicator in self.low_quality_indicators:
            if indicator in text_lower:
                return True
        
        # 检查是否包含太多数字（可能是坐标或ID）
        digit_ratio = sum(1 for c in text if c.isdigit()) / len(text)
        if digit_ratio > 0.7:
            return True
        
        # 检查是否包含太多特殊字符
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.5:
            return True
        
        return False
    
    def _is_meaningless_relation(self, predicate: str) -> bool:
        """检查关系是否无意义"""
        if not predicate:
            return True
        
        predicate_lower = predicate.lower().strip()
        
        # 检查无意义关系
        for meaningless in self.meaningless_relations:
            if meaningless in predicate_lower:
                return True
        
        # 检查是否太短或太长
        if len(predicate_lower) < 2 or len(predicate_lower) > 100:
            return True
        
        return False
    
    def _normalize_triple_component(self, component: str) -> str:
        """标准化三元组组件"""
        if not component or not isinstance(component, str):
            return ''
        
        # 基本清理
        normalized = component.strip()
        
        # 移除多余的空白
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # 移除引号
        normalized = re.sub(r'^["\'](.+)["\']$', r'\1', normalized)
        
        # 移除括号中的注释（如果太长）
        def clean_parentheses(match):
            content = match.group(1)
            if len(content) > 50 or 'disambiguation' in content.lower():
                return ''
            return match.group(0)
        
        normalized = re.sub(r'\(([^)]+)\)', clean_parentheses, normalized)
        
        # 标准化大小写（保持专有名词）
        if not re.search(r'[A-Z]{2,}', normalized):  # 如果没有连续大写字母
            words = normalized.split()
            normalized_words = []
            for word in words:
                if len(word) > 3 and word.lower() not in {'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with'}:
                    normalized_words.append(word.capitalize())
                else:
                    normalized_words.append(word.lower())
            normalized = ' '.join(normalized_words)
        
        return normalized.strip()
    
    def _get_triple_signature(self, triple: Dict[str, Any]) -> Tuple[str, str, str]:
        """获取三元组的唯一签名用于去重"""
        subject = self._normalize_triple_component(str(triple.get('subject', '')))
        predicate = self._normalize_triple_component(str(triple.get('predicate', '')))
        obj = self._normalize_triple_component(str(triple.get('object', '')))
        
        return (subject.lower(), predicate.lower(), obj.lower())
    
    def _is_valid_triple(self, triple: Dict[str, Any]) -> Tuple[bool, str]:
        """验证三元组是否有效，返回(是否有效, 失败原因)"""
        # 检查必需字段
        if not all(key in triple for key in ['subject', 'predicate', 'object']):
            return False, 'missing_required_fields'
        
        subject = str(triple.get('subject', ''))
        predicate = str(triple.get('predicate', ''))
        obj = str(triple.get('object', ''))
        
        # 检查null值
        if self._is_invalid_value(subject) or self._is_invalid_value(predicate) or self._is_invalid_value(obj):
            return False, 'invalid_values'
        
        # 检查自引用
        if subject.lower().strip() == obj.lower().strip():
            return False, 'self_reference'
        
        # 检查低质量内容
        if (self._is_low_quality_content(subject) or 
            self._is_low_quality_content(predicate) or 
            self._is_low_quality_content(obj)):
            return False, 'low_quality_content'
        
        # 检查无意义关系
        if self._is_meaningless_relation(predicate):
            return False, 'meaningless_relation'
        
        # 检查置信度
        confidence = triple.get('confidence', 1.0)
        if isinstance(confidence, (int, float)) and confidence < self.min_confidence:
            return False, 'low_confidence'
        
        return True, 'valid'
    
    def process_triples(self, triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理三元组列表，过滤无效和重复项"""
        logger.info(f"开始处理 {len(triples)} 个三元组")
        
        # 重置统计
        self.stats = {key: 0 for key in self.stats.keys()}
        self.stats['total_input'] = len(triples)
        
        valid_triples = []
        seen_signatures = set()
        
        for i, triple in enumerate(triples):
            # 验证三元组
            is_valid, reason = self._is_valid_triple(triple)
            
            if not is_valid:
                # 更新统计
                if reason == 'invalid_values':
                    self.stats['filtered_null'] += 1
                elif reason == 'self_reference':
                    self.stats['filtered_self_reference'] += 1
                elif reason == 'low_quality_content':
                    self.stats['filtered_low_quality'] += 1
                elif reason == 'meaningless_relation':
                    self.stats['filtered_meaningless'] += 1
                elif reason == 'low_confidence':
                    self.stats['filtered_low_confidence'] += 1
                else:
                    self.stats['filtered_invalid'] += 1
                
                logger.debug(f"过滤三元组 {i}: {reason} - {triple}")
                continue
            
            # 检查重复
            signature = self._get_triple_signature(triple)
            if signature in seen_signatures:
                self.stats['filtered_duplicate'] += 1
                logger.debug(f"过滤重复三元组 {i}: {triple}")
                continue
            
            seen_signatures.add(signature)
            
            # 标准化三元组
            processed_triple = self._normalize_triple(triple)
            valid_triples.append(processed_triple)
        
        self.stats['total_output'] = len(valid_triples)
        
        logger.info(f"三元组处理完成: {self.stats['total_input']} -> {self.stats['total_output']}")
        self._log_statistics()
        
        return valid_triples
    
    def _normalize_triple(self, triple: Dict[str, Any]) -> Dict[str, Any]:
        """标准化单个三元组"""
        normalized = triple.copy()
        
        # 标准化核心字段
        normalized['subject'] = self._normalize_triple_component(str(triple.get('subject', '')))
        normalized['predicate'] = self._normalize_triple_component(str(triple.get('predicate', '')))
        normalized['object'] = self._normalize_triple_component(str(triple.get('object', '')))
        
        # 确保置信度是数值
        if 'confidence' in normalized:
            try:
                normalized['confidence'] = float(normalized['confidence'])
            except (ValueError, TypeError):
                normalized['confidence'] = self.config.get('confidence', {}).get('default_normalization', 0.5)  # 默认置信度
        
        # 清理source_text
        if 'source_text' in normalized and normalized['source_text']:
            source_text = str(normalized['source_text'])
            # 限制source_text长度
            if len(source_text) > 500:
                normalized['source_text'] = source_text[:500] + '...'
        
        return normalized
    
    def _log_statistics(self):
        """记录统计信息"""
        logger.info("=== 三元组处理统计 ===")
        logger.info(f"输入总数: {self.stats['total_input']}")
        logger.info(f"输出总数: {self.stats['total_output']}")
        logger.info(f"过滤统计:")
        logger.info(f"  - Null/无效值: {self.stats['filtered_null']}")
        logger.info(f"  - 重复项: {self.stats['filtered_duplicate']}")
        logger.info(f"  - 自引用: {self.stats['filtered_self_reference']}")
        logger.info(f"  - 低质量内容: {self.stats['filtered_low_quality']}")
        logger.info(f"  - 无意义关系: {self.stats['filtered_meaningless']}")
        logger.info(f"  - 低置信度: {self.stats['filtered_low_confidence']}")
        logger.info(f"  - 其他无效: {self.stats['filtered_invalid']}")
        
        if self.stats['total_input'] > 0:
            retention_rate = self.stats['total_output'] / self.stats['total_input']
            logger.info(f"保留率: {retention_rate:.2%}")
    
    def analyze_triples(self, triples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析三元组质量"""
        analysis = {
            'total_count': len(triples),
            'null_values': 0,
            'self_references': 0,
            'duplicate_signatures': 0,
            'low_confidence': 0,
            'subject_distribution': Counter(),
            'predicate_distribution': Counter(),
            'object_distribution': Counter(),
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'quality_issues': []
        }
        
        seen_signatures = set()
        
        for i, triple in enumerate(triples):
            # 检查null值
            subject = str(triple.get('subject', ''))
            predicate = str(triple.get('predicate', ''))
            obj = str(triple.get('object', ''))
            
            if self._is_invalid_value(subject) or self._is_invalid_value(predicate) or self._is_invalid_value(obj):
                analysis['null_values'] += 1
                analysis['quality_issues'].append(f"Triple {i}: Contains null/invalid values")
            
            # 检查自引用
            if subject.lower().strip() == obj.lower().strip():
                analysis['self_references'] += 1
                analysis['quality_issues'].append(f"Triple {i}: Self-reference ({subject})")
            
            # 检查重复
            signature = self._get_triple_signature(triple)
            if signature in seen_signatures:
                analysis['duplicate_signatures'] += 1
                analysis['quality_issues'].append(f"Triple {i}: Duplicate signature")
            else:
                seen_signatures.add(signature)
            
            # 统计分布
            if not self._is_invalid_value(subject):
                analysis['subject_distribution'][subject] += 1
            if not self._is_invalid_value(predicate):
                analysis['predicate_distribution'][predicate] += 1
            if not self._is_invalid_value(obj):
                analysis['object_distribution'][obj] += 1
            
            # 置信度分析
            confidence = triple.get('confidence', self.config.get('confidence', {}).get('default_normalization', 0.5))
            if isinstance(confidence, (int, float)):
                if confidence < 0.3:
                    analysis['low_confidence'] += 1
                    analysis['confidence_distribution']['low'] += 1
                elif confidence < 0.7:
                    analysis['confidence_distribution']['medium'] += 1
                else:
                    analysis['confidence_distribution']['high'] += 1
        
        return analysis
    
    def save_processed_triples(self, triples: List[Dict[str, Any]], output_path: str):
        """保存处理后的三元组"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for triple in triples:
                f.write(json.dumps(triple, ensure_ascii=False) + '\n')
        
        logger.info(f"已保存 {len(triples)} 个处理后的三元组到: {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.stats.copy()


def process_triple_file(input_file: str, output_file: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """处理三元组文件的便捷函数"""
    processor = TriplePostProcessor(config)
    
    # 加载三元组
    triples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    triple = json.loads(line)
                    triples.append(triple)
                except json.JSONDecodeError as e:
                    logger.warning(f"第{line_num}行JSON解析失败: {e}")
    
    logger.info(f"从 {input_file} 加载了 {len(triples)} 个三元组")
    
    # 处理三元组
    processed_triples = processor.process_triples(triples)
    
    # 保存结果
    processor.save_processed_triples(processed_triples, output_file)
    
    return processor.get_statistics()


if __name__ == "__main__":
    # 测试后处理器
    test_triples = [
        {'subject': 'Mars', 'predicate': 'orbits', 'object': 'Sun', 'confidence': 0.9},
        {'subject': 'null', 'predicate': 'hasProperty', 'object': 'null', 'confidence': 0.5},
        {'subject': 'Mars', 'predicate': 'orbits', 'object': 'Sun', 'confidence': 0.8},  # 重复
        {'subject': 'Earth', 'predicate': 'is', 'object': 'Earth', 'confidence': 0.7},  # 自引用
        {'subject': 'Jupiter', 'predicate': 'has natural satellite', 'object': 'Europa', 'confidence': 0.95},
        {'subject': '', 'predicate': 'related to', 'object': 'something', 'confidence': 0.1},  # 无效
    ]
    
    processor = TriplePostProcessor()
    
    print("原始三元组:")
    for i, triple in enumerate(test_triples):
        print(f"  {i}: {triple}")
    
    # 分析质量
    analysis = processor.analyze_triples(test_triples)
    print(f"\n质量分析: {analysis}")
    
    # 处理三元组
    processed = processor.process_triples(test_triples)
    
    print(f"\n处理后的三元组 ({len(processed)} 个):")
    for i, triple in enumerate(processed):
        print(f"  {i}: {triple}")
    
    # 统计信息
    stats = processor.get_statistics()
    print(f"\n统计信息: {stats}")