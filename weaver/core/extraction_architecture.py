#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新的抽取架构设计
区分属性抽取、三元组抽取和事件抽取，确保属性不会被当作实体处理，事件抽取与属性抽取不重复
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
    EVENT = "event"          # 事件抽取


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
    attributes: Dict[str, Any]


class EventArgument(TypedDict):
    """事件参数结构"""
    role: str
    value: str


class EventExtraction(TypedDict):
    """事件抽取结果结构"""
    event_type: str
    anchor_entity: str
    arguments: List[EventArgument]
    confidence: float
    source_text: str
    text_id: str


class ExtractionResult(TypedDict):
    """完整抽取结果结构"""
    attributes: List[AttributeExtraction]
    relations: List[RelationExtraction]
    events: List[EventExtraction]
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


class EventClassifier:
    """事件分类器，用于判断一个谓词是否表示事件"""
    
    def __init__(self):
        # 事件相关谓词关键词
        self.event_predicates = [
            # 时间变化类事件
            'dimmed', 'brightened', 'exploded', 'collapsed', 'formed', 'evolved',
            'discovered', 'observed', 'detected', 'measured', 'recorded',
            
            # 运动类事件
            'moved', 'orbited', 'rotated', 'accelerated', 'ejected', 'collided',
            'impacted', 'merged', 'separated', 'approached', 'receded',
            
            # 状态变化类事件
            'increased', 'decreased', 'changed', 'transformed', 'converted',
            'expanded', 'contracted', 'cooled', 'heated', 'ignited', 'extinguished',
            
            # 观测类事件
            'imaged', 'photographed', 'analyzed', 'studied', 'surveyed',
            'mapped', 'scanned', 'monitored', 'tracked', 'sampled',
            
            # 任务类事件
            'launched', 'deployed', 'landed', 'docked', 'undocked',
            'activated', 'deactivated', 'repaired', 'upgraded', 'decommissioned'
        ]
        
        # 事件类型
        self.event_types = [
            'Dimming', 'Brightening', 'Explosion', 'Collision', 'Formation',
            'Discovery', 'Observation', 'Measurement', 'Movement', 'Orbit',
            'Rotation', 'Acceleration', 'Ejection', 'Impact', 'Merger',
            'Approach', 'Recession', 'Increase', 'Decrease', 'Change',
            'Transformation', 'Expansion', 'Contraction', 'Cooling', 'Heating',
            'Imaging', 'Analysis', 'Study', 'Survey', 'Mapping',
            'Launch', 'Deployment', 'Landing', 'Docking', 'Activation',
            'Deactivation', 'Repair', 'Upgrade', 'Decommission'
        ]
    
    def is_event_predicate(self, predicate: str) -> bool:
        """判断一个谓词是否表示事件"""
        if not predicate or not isinstance(predicate, str):
            return False
        
        predicate_lower = predicate.lower()
        
        # 检查是否包含事件谓词关键词
        for event_pred in self.event_predicates:
            if event_pred in predicate_lower:
                return True
        
        # 检查是否为事件类型
        for event_type in self.event_types:
            if event_type.lower() in predicate_lower:
                return True
        
        return False
    
    def is_event_type(self, event_type: str) -> bool:
        """判断一个字符串是否为事件类型"""
        if not event_type or not isinstance(event_type, str):
            return False
        
        event_type_lower = event_type.lower()
        
        # 检查是否为事件类型
        for et in self.event_types:
            if et.lower() == event_type_lower or et.lower() in event_type_lower:
                return True
        
        return False



def get_multi_entity_attribute_extraction_prompt(text_content: str) -> List[Dict[str, str]]:
    """生成多实体属性抽取的提示词"""
    system_prompt = (
        "You are an expert in extracting astronomical entity attributes. "
        "Your task is to identify ALL entities mentioned in the text and extract their specific attributes (properties). "
        "Extract attributes for every entity that is directly described in the text. "
        "Respond ONLY with the requested JSON object."
    )
    
    user_prompt = f"""
    Task: You are a forensic data analyst. Your single task is to extract static, intrinsic attributes of entities with extreme precision. Your guiding principle is: if there is any ambiguity, do not extract. It is better to return an empty result than an incorrect one.
    **Text:** {text_content}
    
    **Instructions:**
    1.  **The "Fact Sheet" Principle:** An attribute must be a stable, factual property you would find on a specification sheet or in an encyclopedia (e.g., `cost`, `location`, `mass`). It describes what the entity **IS**, not what it **DOES**.

    2.  **The Rule of Direct Association (CRITICAL):**
        *   An attribute must be **unambiguously and directly linked** to its entity within the same sentence or clause.
        *   **Do not infer attributes from proximity.** Just because a location is mentioned near an entity's name does not mean it belongs to that entity.
        *   **Verification Test:** Can you draw a direct grammatical line from the attribute to the entity?
        *   **Generic Example:** In the sentence, "The project, managed by Acme Inc., used a server located in Virginia," the location "Virginia" belongs to the "server," NOT to "Acme Inc." Extracting Virginia as Acme Inc.'s location would be a critical error.

    3.  **Forbidden Information (Strictly Enforced):**
        *   You **MUST NOT** extract roles, functions, actions, responsibilities, or relationships.
        *   **FORBIDDEN:** `principal investigator`, `lead on the team` (Roles).
        *   **FORBIDDEN:** `spacecraft construction`, `planning adjustments` (Actions/Functions).
        *   **FORBIDDEN:** `built by`, `affiliated with`, `provides data` (Relationships).

    4.  **Entity Identification:**
        *   Identify specific, named entities (organizations, locations, projects).
        *   Do not treat general activities (e.g., "data analysis") as entities.

    5.  **Group Attribution:**
        *   If a value (like a cost) applies to an overarching project, assign it to that project, not its individual components.

    6.  **Output Requirements:**
        *   The final output must be sparse and contain only attributes that are verifiably correct according to the rules above.
        *   Provide a confidence score (0.0-1.0).
        *   Use the specified JSON format.
    **JSON Output Format Example (Generic):**
    ```json
    {{
    "entities": [
        {{
        "entity_name": "Webb Telescope",
        "attributes": [
            {{
            "attribute": "primary_mirror_diameter",
            "value": "6.5 meters",
            "confidence": 1.0
            }}
        ]
        }}
    ]
    }}
    ```
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
        "Do NOT extract attributes, properties, or events. Respond ONLY with the requested JSON object."
    )
    
    user_prompt = f"""
    Task: Extract fundamental relationships between distinct astronomical entities from the given text.
    topic: astronomy

    **Text:** {text_content}

    **Instructions:**
    1.  **Extract Entity-to-Entity Relationships Only:** The subject and object of a relationship must both be distinct entities.
    2.  **Strict Definition of an Entity:** An entity must be a specific, named astronomical object, spacecraft, or instrument.
        *   **Valid Entity Types:** Planets (e.g., Jupiter), Moons (e.g., Titan), Stars (e.g., Proxima Centauri), Galaxies (e.g., Andromeda Galaxy), Spacecraft (e.g., Voyager 1), Instruments (e.g., NIRCam).
        *   **Invalid as Entities:** Do NOT extract abstract concepts, regions, paths, or properties like 'orbit', 'atmosphere', 'heliopause', 'halfway point'. These describe an entity but are not entities themselves.
        *   *Handling Context:* If the text mentions "Jupiter's orbit", extract "Jupiter" as the entity, but capture the "orbit" context in the relationship attributes.
    3.  **Normalize Entity Names:** Always extract the simplest, most common name for an entity, removing descriptive text.
        *   Example: From "the powerful Near-Infrared Camera (NIRCam)", extract the entity "NIRCam".
        *   Example: When the text mentions "the atmosphere of Venus", the core entity is "Venus".
    4.  **Use Specific Names (Co-reference Resolution):** If the text mentions a specific name (e.g., "James Webb Space Telescope") and later uses a general term (e.g., "the telescope"), always use the specific, proper name in your extracted relationships.
    5.  **Extract Fundamental & Comparative Relationships:** 
        *   Identify structural connections (e.g., "orbits", "has_component").
        *   Identify **comparative relationships** (e.g., "larger_than", "similar_to", "compared_to").
        *   The predicate should be as specific as possible (e.g., prefer "spatially_exceeds" over "compared_to" if applicable).
    6.  **Do NOT extract internal attributes (Node Properties).**
        *   Invalid: "Earth has_diameter 12,742 km" (This is a property of Earth).
        *   **Valid:** "Betelgeuse is larger_than Jupiter" (This is a relationship between two entities).
    7.  **Do NOT extract events or temporal occurrences.** Focus only on stable, persistent relationships or significant scientific comparisons.
    8.  **Include a confidence score (0.0-1.0) for each relationship.**

    **Relation Qualifiers (CRITICAL):** 
    For each relation, you **MUST** use the "attributes" object to capture context. This is vital for comparative relationships.
    *   **comparison_dimension**: What is being compared? (e.g., "radius", "mass", "distance").
    *   **context**: Specific context of the entity (e.g., "orbit of", "atmosphere of").
    *   **value_subject**: The value associated with the subject (e.g., "1.2-8.9 AU").
    *   **value_object**: The value associated with the object (e.g., "5.5 AU").
    *   **condition**: Any assumptions (e.g., "assuming placement at Solar System center").
    *   **time_frame**: If the relationship is specific to a time period.

    **JSON Output Format:**
    {{
    "is_relevant": boolean,
    "relations": [
        {{
        "subject": "string",
        "predicate": "string",
        "object": "string",
        "confidence": float,
        "attributes": {{}}
        }}
    ]
    }}
    """
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def get_event_extraction_prompt(text_content: str, entity_name: str = None) -> List[Dict[str, str]]:
    """生成事件抽取的提示词"""
    system_prompt = (
        "You are an expert in extracting astronomical events from text. "
        "Your task is to identify events related to astronomical entities. "
        "Focus on temporal occurrences, changes, and actions. "
        "Do NOT extract attributes or stable relationships. Respond ONLY with the requested JSON object."
    )
    
    entity_focus = f"Focus on events related to {entity_name}." if entity_name else "Extract all events mentioned in the text."
    
    user_prompt = f"""
    Task: Extract astronomical events from the given text. {entity_focus}

    **Text:** {text_content}

    **Instructions:**
    1.  **Extract Events Only:** Events are temporal occurrences, changes, or actions that happen at a specific time or over a period.
        * Events have a beginning and end, unlike permanent attributes or stable relationships.
        * Events involve change, action, or occurrence.
    
    2.  **Event Structure:**
        * **event_type:** A descriptive name for the event category (e.g., "Dimming", "Explosion", "Discovery", "Measurement")
        * **anchor_entity:** The main astronomical entity involved in the event
        * **arguments:** Key details about the event, including:
            * **participant:** Entities involved in the event
            * **time/time_frame/start_time/end_time:** When the event occurred
            * **location:** Where the event occurred (if applicable)
            * **result:** Outcome or effect of the event
            * **instrument:** Tool or method used (if applicable)
            * Other relevant roles specific to the event type
    
    3.  **Distinguish from Attributes:**
        * Attributes are permanent or semi-permanent properties (e.g., mass, diameter, composition)
        * Events are occurrences that happen at specific times (e.g., dimming, explosion, discovery)
        * If something happened at a specific time or represents a change, it's likely an event
    
    4.  **Distinguish from Relationships:**
        * Relationships describe stable connections between entities (e.g., "orbits", "belongs to")
        * Events describe things that happen (e.g., "dimmed", "exploded", "was discovered")
    
    5.  **Include a confidence score (0.0-1.0) for each event.**

    **Examples of Valid Events:**
    - "Betelgeuse dimmed significantly between October 2019 and February 2020" (Event: Dimming)
    - "The first measurement of Betelgeuse's angular size was made in 1920" (Event: Measurement)
    - "Voyager 1 crossed the heliopause in 2012" (Event: Boundary Crossing)

    **Examples of Invalid Events (these are attributes or relationships, not events):**
    - "Betelgeuse is a red supergiant star" (This is an attribute/classification)
    - "Mars orbits the Sun" (This is a stable relationship)
    - "Jupiter has 79 known moons" (This is an attribute/property)

    **JSON Output Format:**
    {{
    "is_relevant": boolean,
    "events": [
        {{
        "event_type": "string",
        "anchor_entity": "string",
        "arguments": [
            {{
            "role": "string",
            "value": "string"
            }}
        ],
        "confidence": float
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
        event_classifier = EventClassifier()
        
        validated_relations = []
        for rel in relations:
            if all(k in rel for k in ["subject", "predicate", "object", "confidence"]):
                # 检查是否为属性关系
                if classifier.is_attribute_predicate(rel["predicate"]) or \
                   classifier.is_attribute_value(rel["object"]):
                    logger.debug(f"Filtered out attribute relation: {rel}")
                    continue
                
                # 检查是否为事件关系
                if event_classifier.is_event_predicate(rel["predicate"]):
                    logger.debug(f"Filtered out event relation: {rel}")
                    continue
                
                # 检查对象是否为实体
                if not entity_classifier.is_entity(rel["object"]):
                    logger.debug(f"Filtered out non-entity object: {rel}")
                    continue
                
                if "attributes" not in rel or not isinstance(rel.get("attributes"), dict):
                    rel["attributes"] = {}
                validated_relations.append(rel)
        
        return validated_relations
    
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Failed to parse relation response. Error: {e}. Response: {response_text[:300]}")
        return None


def parse_event_response(response_text: str) -> Optional[List[EventExtraction]]:
    """解析事件抽取响应"""
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
        
        events = data.get("events", [])
        if not isinstance(events, list):
            logger.warning(f"Events response is not a list: {response_text[:200]}")
            return None
        
        # 验证事件格式
        validated_events = []
        for event in events:
            if all(k in event for k in ["event_type", "anchor_entity", "arguments", "confidence"]):
                validated_events.append(event)
        
        return validated_events
    
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Failed to parse event response. Error: {e}. Response: {response_text[:300]}")
        return None
