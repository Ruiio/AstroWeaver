#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新的抽取架构设计
区分属性抽取和三元组抽取，确保属性不会被当作实体处理
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional, TypedDict
from enum import Enum

logger = logging.getLogger(__name__)


class ExtractionType(Enum):
    """抽取类型枚举"""
    ATTRIBUTE = "attribute"  # 属性抽取
    RELATION = "relation"    # 关系抽取


class AttributeExtraction(TypedDict):
    """属性抽取结果结构"""
    entity: str
    attribute: str
    value: str
    confidence: float
    source_text: str
    text_id: str


class RelationExtraction(TypedDict):
    """关系抽取结果结构"""
    subject: str
    predicate: str
    object: str
    confidence: float
    source_text: str
    text_id: str


class ExtractionResult(TypedDict):
    """完整抽取结果结构"""
    attributes: List[AttributeExtraction]
    relations: List[RelationExtraction]
    is_relevant: bool


class AttributeClassifier:
    """属性分类器，用于判断一个对象是否为属性值"""
    
    def __init__(self):
        # 属性值的特征模式
        self.attribute_patterns = [
            # 数值类型
            r'^\d+(\.\d+)?\s*(km|AU|light-years?|years?|billion years?|million years?|°C|°F|K)$',
            r'^\d+(\.\d+)?\s*%$',  # 百分比
            r'^\d+(\.\d+)?\s*(kg|g|tons?|solar masses?)$',  # 质量
            r'^\d+(\.\d+)?\s*(m/s|km/s|km/h)$',  # 速度
            
            # 时间表达式
            r'^(about|approximately|roughly|around)?\s*\d+(\.\d+)?\s*(billion|million)?\s*years?\s*(ago|from now)?$',
            
            # 物理状态描述
            r'^(solid|liquid|gas|plasma)$',
            r'^(hot|cold|warm|cool|frozen)$',
            r'^(bright|dim|dark|luminous)$',
            
            # 颜色描述
            r'^(red|blue|yellow|white|orange|green|purple|black)(-\w+)?$',
            
            # 大小描述
            r'^(large|small|huge|tiny|massive|compact)$',
            
            # 组成成分（化学元素或化合物）
            r'^(hydrogen|helium|carbon|oxygen|nitrogen|iron|silicon|water|methane|ammonia|carbon dioxide)$',
            
            # 布尔值或状态
            r'^(true|false|yes|no|present|absent|active|inactive)$',
            
            # 方向和位置描述
            r'^(north|south|east|west|inner|outer|upper|lower|central)$',
        ]
        
        # 编译正则表达式
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.attribute_patterns]
        
        # 属性关键词
        self.attribute_keywords = {
            'physical_properties': ['temperature', 'mass', 'diameter', 'radius', 'density', 'gravity', 'pressure'],
            'temporal': ['age', 'period', 'duration', 'time', 'date', 'year'],
            'spatial': ['distance', 'size', 'volume', 'area', 'height', 'width', 'length'],
            'composition': ['composition', 'material', 'element', 'compound', 'atmosphere'],
            'state': ['phase', 'state', 'condition', 'status', 'type', 'class', 'category'],
            'optical': ['color', 'brightness', 'magnitude', 'luminosity', 'spectrum']
        }
    
    def is_attribute_value(self, value: str) -> bool:
        """判断一个值是否为属性值"""
        if not value or not isinstance(value, str):
            return False
        
        value = value.strip()
        
        # 检查是否匹配属性值模式
        for pattern in self.compiled_patterns:
            if pattern.match(value):
                return True
        
        # 检查是否包含数值
        if re.search(r'\d+', value):
            return True
        
        # 检查长度（属性值通常较短）
        if len(value.split()) > 5:
            return False
        
        return False
    
    def is_attribute_predicate(self, predicate: str) -> bool:
        """判断一个谓词是否表示属性关系"""
        if not predicate:
            return False
        
        predicate_lower = predicate.lower()
        
        # 检查是否包含属性关键词
        for category, keywords in self.attribute_keywords.items():
            for keyword in keywords:
                if keyword in predicate_lower:
                    return True
        
        # 检查常见的属性谓词模式
        attribute_predicates = [
            'has', 'is', 'measures', 'weighs', 'contains', 'composed of',
            'temperature', 'mass', 'diameter', 'age', 'distance', 'size',
            'color', 'brightness', 'density', 'pressure', 'gravity'
        ]
        
        for attr_pred in attribute_predicates:
            if attr_pred in predicate_lower:
                return True
        
        return False


class EntityClassifier:
    """实体分类器，用于判断一个对象是否为实体"""
    
    def __init__(self):
        # 天文实体类型
        self.entity_types = {
            'celestial_bodies': ['star', 'planet', 'moon', 'asteroid', 'comet', 'galaxy', 'nebula'],
            'constellations': ['constellation', 'asterism'],
            'organizations': ['nasa', 'esa', 'iau', 'observatory', 'institute', 'university'],
            'people': ['astronomer', 'scientist', 'discoverer'],
            'missions': ['mission', 'probe', 'spacecraft', 'telescope', 'satellite'],
            'regions': ['belt', 'cloud', 'zone', 'region', 'system']
        }
    
    def is_entity(self, value: str) -> bool:
        """判断一个值是否为实体"""
        if not value or not isinstance(value, str):
            return False
        
        value = value.strip()
        
        # 专有名词通常以大写字母开头
        if value[0].isupper():
            return True
        
        # 检查是否包含实体类型关键词
        value_lower = value.lower()
        for category, types in self.entity_types.items():
            for entity_type in types:
                if entity_type in value_lower:
                    return True
        
        # 检查是否为复合实体名称（包含多个单词且首字母大写）
        words = value.split()
        if len(words) > 1 and all(word[0].isupper() for word in words if word):
            return True
        
        return False


def get_attribute_extraction_prompt(entity_name: str, text_content: str) -> List[Dict[str, str]]:
    """生成属性抽取的提示词"""
    system_prompt = (
        "You are an expert in extracting astronomical entity attributes. "
        "Your task is to identify and extract specific attributes (properties) of the given entity from the text. "
        "ONLY extract attributes if the text directly describes the specified entity. "
        "If the text does not directly mention or describe the specified entity, return is_relevant: false with empty attributes. "
        "Respond ONLY with the requested JSON object."
    )
    
    user_prompt = f"""
    **Task:** Extract attributes (properties) of the specified entity from the given text.
    
    **Entity:** {entity_name}
    **Text:** {text_content}
    
    **Critical Instructions:**
    1. FIRST determine if the text directly describes or mentions the specified entity
    2. If the text does NOT directly describe the specified entity, set is_relevant to false and return empty attributes array
    3. If the text DOES directly describe the specified entity, extract only its direct attributes/properties
    4. Attributes should be measurable properties, characteristics, or states explicitly mentioned in the text
    5. Do NOT extract relationships to other entities
    6. Do NOT infer or assume attributes not explicitly stated in the text
    7. Include confidence score (0.0-1.0) for each attribute based on how explicitly it's stated
    
    **Examples of when to return is_relevant: false:**
    - Text describes asteroid belt formation but entity is "solar system"
    - Text describes Jupiter's moons but entity is "Mars"
    - Text describes stellar evolution but entity is "Earth"
    
    **Examples of Valid Attributes (only when text directly describes the entity):**
    - Physical properties: mass, diameter, temperature, density
    - Temporal properties: age, orbital period, rotation period
    - Compositional properties: atmosphere composition, surface material
    - Observational properties: magnitude, color, spectral type
    
    **JSON Output Format:**
    {{
      "is_relevant": boolean,
      "attributes": [
        {{
          "attribute": "mass",
          "value": "1.989 × 10^30 kg",
          "confidence": 0.9
        }},
        {{
          "attribute": "surface_temperature",
          "value": "5778 K",
          "confidence": 0.8
        }}
      ]
    }}
    
    **Important:** If is_relevant is false, the attributes array should be empty: []
    """
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def get_multi_entity_attribute_extraction_prompt(text_content: str) -> List[Dict[str, str]]:
    """生成多实体属性抽取的提示词"""
    system_prompt = (
        "You are an expert in extracting astronomical entity attributes. "
        "Your task is to identify ALL entities mentioned in the text and extract their specific attributes (properties). "
        "Extract attributes for every entity that is directly described in the text. "
        "Respond ONLY with the requested JSON object."
    )
    
    user_prompt = f"""
    **Task:** Extract attributes (properties) of ALL entities mentioned in the given text.
    
    **Text:** {text_content}
    
    **Instructions:**
    1. Identify ALL astronomical entities mentioned in the text
    2. For each entity, extract only its direct attributes/properties that are explicitly mentioned
    3. Attributes should be measurable properties, characteristics, or states explicitly stated in the text
    4. Do NOT extract relationships between entities (those are handled separately)
    5. Do NOT infer or assume attributes not explicitly stated in the text
    6. Include confidence score (0.0-1.0) for each attribute based on how explicitly it's stated
    7. Group attributes by entity
    
    **Examples of Valid Attributes:**
    - Physical properties: mass, diameter, temperature, density, radius
    - Temporal properties: age, orbital period, rotation period, formation time
    - Compositional properties: atmosphere composition, surface material, chemical composition
    - Observational properties: magnitude, color, spectral type, brightness
    - Structural properties: number of moons, ring system, surface features
    
    **JSON Output Format:**
    {{
      "entities": [
        {{
          "entity_name": "Sun",
          "attributes": [
            {{
              "attribute": "mass",
              "value": "1.989 × 10^30 kg",
              "confidence": 0.9
            }},
            {{
              "attribute": "surface_temperature",
              "value": "5778 K",
              "confidence": 0.8
            }}
          ]
        }},
        {{
          "entity_name": "Earth",
          "attributes": [
            {{
              "attribute": "diameter",
              "value": "12,742 km",
              "confidence": 0.9
            }}
          ]
        }}
      ]
    }}
    
    **Important:** Only include entities that have attributes explicitly mentioned in the text.
    """
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def get_relation_extraction_prompt(text_content: str, topic: str) -> List[Dict[str, str]]:
    """生成关系抽取的提示词"""
    system_prompt = (
        "You are an expert in extracting relationships between astronomical entities. "
        "Your task is to identify relationships that connect two distinct entities. "
        "Do NOT extract attributes or properties. Respond ONLY with the requested JSON object."
    )
    
    user_prompt = f"""
    **Task:** Extract relationships between distinct astronomical entities from the given text.
    
    **Topic:** {topic}
    **Text:** {text_content}
    
    **Instructions:**
    1. Extract only entity-to-entity relationships
    2. Both subject and object must be distinct entities (not attribute values)
    3. Do NOT extract attributes, properties, or measurements
    4. Include confidence score (0.0-1.0) for each relationship
    
    **Examples of Valid Relationships:**
    - "Mars orbits Sun" (entity-to-entity)
    - "Hubble discovered galaxy" (entity-to-entity)
    - "Jupiter has_satellite Europa" (entity-to-entity)
    
    **Examples of Invalid Relationships (these are attributes):**
    - "Mars has_mass 6.39 × 10^23 kg" (attribute, not relationship)
    - "Sun has_temperature 5778 K" (attribute, not relationship)
    - "Earth has_diameter 12,742 km" (attribute, not relationship)
    
    **JSON Output Format:**
    {{
      "is_relevant": boolean,
      "relations": [
        {{
          "subject": "Mars",
          "predicate": "orbits",
          "object": "Sun",
          "confidence": 0.9
        }},
        {{
          "subject": "Jupiter",
          "predicate": "has_satellite",
          "object": "Europa",
          "confidence": 0.8
        }}
      ]
    }}
    """
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def parse_attribute_response(response_text: str) -> Optional[List[AttributeExtraction]]:
    """解析属性抽取响应"""
    try:
        # 寻找JSON代码块
        match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = response_text
        
        data = json.loads(json_str)
        
        if not data.get("is_relevant"):
            return []
        
        attributes = data.get("attributes", [])
        if not isinstance(attributes, list):
            logger.warning(f"Attributes response is not a list: {response_text[:200]}")
            return None
        
        # 验证属性格式
        validated_attributes = []
        for attr in attributes:
            if all(k in attr for k in ["attribute", "value", "confidence"]):
                validated_attributes.append(attr)
        
        return validated_attributes
    
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Failed to parse attribute response. Error: {e}. Response: {response_text[:300]}")
        return None


def parse_multi_entity_attribute_response(response_text: str) -> Optional[List[Dict[str, Any]]]:
    """解析多实体属性抽取响应"""
    try:
        # 寻找JSON代码块
        match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = response_text
        
        data = json.loads(json_str)
        
        entities = data.get("entities", [])
        if not isinstance(entities, list):
            logger.warning(f"Entities response is not a list: {response_text[:200]}")
            return None
        
        # 验证实体和属性格式
        validated_entities = []
        for entity_data in entities:
            if not isinstance(entity_data, dict) or "entity_name" not in entity_data:
                continue
                
            entity_name = entity_data["entity_name"]
            
            # 处理entity_name可能是字典的情况
            if isinstance(entity_name, dict):
                if "entity_name" in entity_name:
                    entity_name = entity_name["entity_name"]
                else:
                    logger.warning(f"Invalid entity_name format: {entity_name}")
                    continue
            elif not isinstance(entity_name, str):
                logger.warning(f"Entity name is not string or dict: {entity_name}")
                continue
                
            attributes = entity_data.get("attributes", [])
            
            if not isinstance(attributes, list):
                continue
            
            # 验证属性格式
            validated_attributes = []
            for attr in attributes:
                if all(k in attr for k in ["attribute", "value", "confidence"]):
                    validated_attributes.append(attr)
            
            if validated_attributes:  # 只包含有有效属性的实体
                validated_entities.append({
                    "entity_name": entity_name,
                    "attributes": validated_attributes
                })
        
        return validated_entities
    
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Failed to parse multi-entity attribute response. Error: {e}. Response: {response_text[:300]}")
        return None


def parse_relation_response(response_text: str) -> Optional[List[RelationExtraction]]:
    """解析关系抽取响应"""
    try:
        # 寻找JSON代码块
        match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = response_text
        
        data = json.loads(json_str)
        
        if not data.get("is_relevant"):
            return []
        
        relations = data.get("relations", [])
        if not isinstance(relations, list):
            logger.warning(f"Relations response is not a list: {response_text[:200]}")
            return None
        
        # 验证关系格式并过滤属性关系
        classifier = AttributeClassifier()
        entity_classifier = EntityClassifier()
        
        validated_relations = []
        for rel in relations:
            if all(k in rel for k in ["subject", "predicate", "object", "confidence"]):
                # 检查是否为属性关系
                if classifier.is_attribute_predicate(rel["predicate"]) or \
                   classifier.is_attribute_value(rel["object"]):
                    logger.debug(f"Filtered out attribute relation: {rel}")
                    continue
                
                # 检查对象是否为实体
                if not entity_classifier.is_entity(rel["object"]):
                    logger.debug(f"Filtered out non-entity object: {rel}")
                    continue
                
                validated_relations.append(rel)
        
        return validated_relations
    
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Failed to parse relation response. Error: {e}. Response: {response_text[:300]}")
        return None