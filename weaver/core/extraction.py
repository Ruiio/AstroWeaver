# astroWeaver/core/extraction.py

import logging
import json
import re
from typing import List, Dict, Any, Optional

from ..models.llm_models import LLMClient  # 假设的LLM客户端封装
from ..utils.config import config
from .extraction_architecture import (
    ExtractionType, AttributeExtraction, RelationExtraction, EventExtraction, ExtractionResult,
    AttributeClassifier, EntityClassifier, EventClassifier,
    get_relation_extraction_prompt, get_event_extraction_prompt,
    parse_relation_response, parse_event_response
)

logger = logging.getLogger(__name__)


def _get_extraction_prompt(entity_name: str, section_content: str) -> List[Dict[str, str]]:
    """
    为关系抽取任务生成结构化的LLM prompt。
    """
    system_prompt = (
        "You are an expert in constructing astronomical knowledge graphs. "
        "Your task is to extract triples where the specified core entity is the subject. "
        "Respond ONLY with the requested JSON object."
    )

    user_prompt = f"""
    Role: You are an expert in constructing astronomical knowledge graphs.

    Task:
    Based on the text content provided below, extract relevant astronomical triples for the specified "core entity". In these triples, the "core entity" must act as the "Subject".

    Core Entity: `{entity_name}`
    Text to Extract From: {section_content}

    Output Format Requirements:
    Please strictly follow the JSON format below for the output. Ensure that the "core entity" is the value of the `entity_name` field and is the subject in all extracted relations.

    ```json
    {{
    "relations": {{
        "RelationName1": ["ObjectEntity1", "ObjectEntity2"],
        "RelationName2": ["ObjectEntity3"]
        // ... more relations
        }}
    }}
    ```
    Where:
    *   `entity_name`: String, the "core entity" around which you are extracting information.
    *   `relations`: List, where each element is a dictionary representing a triple.
        *   The dictionary key is the "Relation Name (Predicate)", a string.
        *   The dictionary value is the "Object Entity (Object)", a string, representing another astronomical entity associated with the core entity through this relation.

    Important Rules for Naming and Selecting Relations:
    1.  Professionalism and Standardization: Relation names must conform to professional astronomical naming conventions and standards (e.g., use "Orbits", "Has Natural Satellite", "Member Of", "Discovered By", etc.).
    2.  Clarity, Avoid Ambiguity: Relation names should clearly express the specific connection between entities. Avoid using overly vague or general relation names (e.g., avoid relations like "Related To", "Comparable To", "Is A").
    3.  Conciseness, Avoid Over-Detailing: Relation names should summarize the main relationship between entities, avoiding the inclusion of specific attributes or states of an entity as part of the relation. For example, do not use a relation like "hasAtmosphereComposition" to link an entity to its atmospheric composition (an attribute value). Relations should connect two distinct astronomical entities.
    4.  Entity-to-Entity Relations: Extracted relations should be between the "core entity" and "another astronomical entity (object)". Do not use attribute values of the core entity (such as temperature, diameter, color, percentage of specific components, etc.) as the object entity. The object entity should be another celestial body, astronomical phenomenon, constellation, galaxy, discoverer (person or institution), etc.
    5.  If no astronomically relevant relations can be extracted, the `relations` list in the JSON output should be empty.

    Reference Example:

    Example Input Text:
    Mars is the fourth planet from the Sun. It is also known as the "Red Planet", because of its orange-red appearance.([22])([23]) Mars is a desert-like rocky planet with a tenuous carbon dioxide (CO(2)) atmosphere. At the average surface level the atmospheric pressure is a few thousandths of Earth's, atmospheric temperature ranges from −153 to 20 °C (−243 to 68 °F)([24]) and cosmic radiation is high. Mars retains some water, in the ground as well as thinly in the atmosphere, forming cirrus clouds, frost, larger polar regions of permafrost and ice caps (with seasonal CO(2) snow), but no liquid surface water. Its surface gravity is roughly a third of Earth's or double that of the Moon. It is half as wide as Earth or twice the Moon, with a diameter of 6,779 km (4,212 mi), and has a surface area the size of all the dry land of Earth. Phobos and Deimos are its natural satellites.

    Core Entity: Mars

    Correct Output:
    ```json
    {{
        "relations": {{
            "Orbits": "Sun",
            "Has Natural Satellite": ["Phobos","Deimos"]
        }}
    }}
    ```
    Interpretation: Mars -[Orbits]-> Sun; Mars -[Has Natural Satellite]-> Phobos; Mars -[Has Natural Satellite]-> Deimos

    Incorrect Output 1 (Relation name too vague):
    ```json
    {{
        "relations": {{
            "Comparable Body": ["Earth","Moon"]
        }}
    }}
    ```
    Reason: "Comparable Body" is too vague and does not reflect a specific astronomical relationship.

    Incorrect Output 2 (Relation name too detailed, and attributes mistaken for entity relations):
    ```json
    {{
        "relations": {{
            "hasAtmosphereComposition": ["CarbonDioxide"]
        }}
    }}
    ```
    Reason: "hasAtmosphereComposition" is more like an attribute description, and "CarbonDioxide" is a component of Mars's atmosphere (an attribute value), not an astronomical entity forming an independent relationship with Mars. We are focused on relationships between entities.
        """

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def _parse_extraction_response(response_text: str) -> Optional[Dict[str, List[str]]]:
    """
    解析LLM返回的JSON字符串，提取关系。
    """
    try:
        # 寻找被```json ... ```包裹的代码块
        match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # 如果没有找到，则假设整个响应就是JSON对象
            json_str = response_text

        data = json.loads(json_str)
        relations = data.get("relations", {})

        if not isinstance(relations, dict):
            logger.warning(f"LLM extraction response 'relations' is not a dict. Response: {response_text[:200]}")
            return None

        # 清理和规范化输出
        clean_relations = {}
        for rel, objs in relations.items():
            if not isinstance(objs, list):
                objs = [str(objs)]  # 兼容返回单个字符串的情况

            valid_objs = [str(obj).strip() for obj in objs if str(obj).strip()]
            if valid_objs:
                clean_relations[rel.strip()] = valid_objs

        return clean_relations if clean_relations else None

    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Failed to parse LLM extraction response. Error: {e}. Response: {response_text[:300]}")
        return None


def extract_multi_entity_attributes_from_sections(
        sections: List,
        llm_client: LLMClient,
        model_name: str
) -> List[AttributeExtraction]:
    """
    使用Batch API从文章章节中抽取所有实体的属性。

    Args:
        sections: 包含文章章节的列表。
        llm_client: LLM客户端实例。
        model_name: 用于抽取的LLM模型名称。

    Returns:
        属性抽取结果列表。
    """
    from .extraction_architecture import get_multi_entity_attribute_extraction_prompt, parse_multi_entity_attribute_response
    
    batch_requests = []
    for i, section in enumerate(sections):
        content = section
        if not content:
            continue

        prompt_messages = get_multi_entity_attribute_extraction_prompt(content)
        request_id = f"multi_attr_section_{i}"
        batch_requests.append(llm_client.prepare_batch_request(request_id, model_name, prompt_messages))

    if not batch_requests:
        logger.info(f"No content found in sections. Skipping multi-entity attribute extraction.")
        return []

    logger.info(f"Submitting {len(batch_requests)} multi-entity attribute extraction requests via Batch API.")
    batch_results = llm_client.submit_batch(batch_requests)

    # 聚合所有章节的属性结果
    all_attributes = []
    for i, result in enumerate(batch_results):
        if result.is_success():
            parsed_entities = parse_multi_entity_attribute_response(result.response_text)
            if parsed_entities:
                for entity_data in parsed_entities:
                    entity_name = entity_data["entity_name"]
                    for attr in entity_data["attributes"]:
                        attr_extraction = AttributeExtraction(
                            entity=entity_name,
                            attribute=attr["attribute"],
                            value=attr["value"],
                            confidence=attr["confidence"],
                            source_text=sections[i][:200] if i < len(sections) else "",
                            text_id=f"section_{i}"
                        )
                        all_attributes.append(attr_extraction)
        else:
            logger.error(f"Multi-entity attribute extraction batch request failed (ID: {result.request_id}). Error: {result.error}")

    logger.info(f"Extracted {len(all_attributes)} attributes from {len(set(attr['entity'] for attr in all_attributes))} entities.")
    return all_attributes


def extract_attributes_from_sections(
        entity_name: str,
        sections: List,
        llm_client: LLMClient,
        model_name: str
) -> List[AttributeExtraction]:
    """
    使用Batch API从文章章节中抽取实体属性。

    Args:
        entity_name: 核心实体的名称。
        sections: 包含文章章节的列表。
        llm_client: LLM客户端实例。
        model_name: 用于抽取的LLM模型名称。

    Returns:
        属性抽取结果列表。
    """
    batch_requests = []
    for i, section in enumerate(sections):
        content = section
        if not content:
            continue

        prompt_messages = get_attribute_extraction_prompt(entity_name, content)
        request_id = f"attr_section_{i}"
        batch_requests.append(llm_client.prepare_batch_request(request_id, model_name, prompt_messages))

    if not batch_requests:
        logger.info(f"No content found in sections for entity '{entity_name}'. Skipping attribute extraction.")
        return []

    logger.info(f"Submitting {len(batch_requests)} attribute extraction requests for '{entity_name}' via Batch API.")
    batch_results = llm_client.submit_batch(batch_requests)

    # 聚合所有章节的属性结果
    all_attributes = []
    for i, result in enumerate(batch_results):
        if result.is_success():
            parsed_attrs = parse_attribute_response(result.response_text)
            if parsed_attrs:
                for attr in parsed_attrs:
                    attr_extraction = AttributeExtraction(
                        entity=entity_name,
                        attribute=attr["attribute"],
                        value=attr["value"],
                        confidence=attr["confidence"],
                        source_text=sections[i][:200] if i < len(sections) else "",
                        text_id=f"section_{i}"
                    )
                    all_attributes.append(attr_extraction)
        else:
            logger.error(f"Attribute extraction batch request failed for '{entity_name}' (ID: {result.request_id}). Error: {result.error}")

    logger.info(f"Extracted {len(all_attributes)} attributes for '{entity_name}'.")
    return all_attributes


def extract_relations_from_sections(
        entity_name: str,
        sections: List,
        llm_client: LLMClient,
        model_name: str
) -> Dict[str, set]:
    """
    使用Batch API从一篇文章的所有章节中抽取初步关系，并聚合结果。
    现在使用新的关系抽取架构，过滤掉属性关系。

    Args:
        entity_name: 核心实体的名称。
        sections: 包含文章章节的列表，每个元素是 {'title': str, 'content': str}。
        llm_client: LLM客户端实例。
        model_name: 用于抽取的LLM模型名称。

    Returns:
        一个聚合后的关系字典，格式为 {relation_name: {object_entity_1, object_entity_2}}.
    """
    batch_requests = []
    for i, section in enumerate(sections):
        content = section
        if not content:
            continue

        prompt_messages = get_relation_extraction_prompt(content, entity_name)
        request_id = f"rel_section_{i}"
        batch_requests.append(llm_client.prepare_batch_request(request_id, model_name, prompt_messages))

    if not batch_requests:
        logger.info(f"No content found in sections for entity '{entity_name}'. Skipping relation extraction.")
        return {}

    logger.info(f"Submitting {len(batch_requests)} relation extraction requests for '{entity_name}' via Batch API.")
    batch_results = llm_client.submit_batch(batch_requests)

    # 聚合所有章节的结果
    aggregated_relations = {}
    attribute_classifier = AttributeClassifier()
    entity_classifier = EntityClassifier()
    
    for result in batch_results:
        if result.is_success():
            parsed_rels = parse_relation_response(result.response_text)
            if parsed_rels:
                for rel in parsed_rels:
                    # 额外过滤：确保不是属性关系
                    if attribute_classifier.is_attribute_predicate(rel["predicate"]) or \
                       attribute_classifier.is_attribute_value(rel["object"]):
                        logger.debug(f"Filtered out attribute relation: {rel}")
                        continue
                    
                    # 确保对象是实体
                    if not entity_classifier.is_entity(rel["object"]):
                        logger.debug(f"Filtered out non-entity object: {rel}")
                        continue
                    
                    predicate = rel["predicate"]
                    obj = rel["object"]
                    
                    if predicate not in aggregated_relations:
                        aggregated_relations[predicate] = set()
                    aggregated_relations[predicate].add(obj)
        else:
            logger.error(f"Relation extraction batch request failed for '{entity_name}' (ID: {result.request_id}). Error: {result.error}")

    logger.info(f"Extracted {len(aggregated_relations)} unique raw relation types for '{entity_name}'.")
    return aggregated_relations


def extract_events_from_sections(
        entity_name: str,
        sections: List,
        llm_client: LLMClient,
        model_name: str
) -> List[EventExtraction]:
    """
    使用Batch API从文章章节中抽取事件。

    Args:
        entity_name: 核心实体的名称。
        sections: 包含文章章节的列表。
        llm_client: LLM客户端实例。
        model_name: 用于抽取的LLM模型名称。

    Returns:
        事件抽取结果列表。
    """
    batch_requests = []
    for i, section in enumerate(sections):
        content = section
        if not content:
            continue

        prompt_messages = get_event_extraction_prompt(content, entity_name)
        request_id = f"event_section_{i}"
        batch_requests.append(llm_client.prepare_batch_request(request_id, model_name, prompt_messages))

    if not batch_requests:
        logger.info(f"No content found in sections for entity '{entity_name}'. Skipping event extraction.")
        return []

    logger.info(f"Submitting {len(batch_requests)} event extraction requests for '{entity_name}' via Batch API.")
    batch_results = llm_client.submit_batch(batch_requests)

    # 聚合所有章节的事件结果
    all_events = []
    for i, result in enumerate(batch_results):
        if result.is_success():
            parsed_events = parse_event_response(result.response_text)
            if parsed_events:
                all_events.extend(parsed_events)
        else:
            logger.error(f"Event extraction batch request failed for '{entity_name}' (ID: {result.request_id}). Error: {result.error}")

    # 过滤重复事件
    event_classifier = EventClassifier()
    attribute_classifier = AttributeClassifier()
    
    # 记录已处理的事件类型和实体组合，避免重复
    processed_events = set()
    filtered_events = []
    
    for event in all_events:
        # 创建事件唯一标识
        event_key = (event["event_type"], event["anchor_entity"])
        
        # 如果已处理过该事件类型和实体组合，则跳过
        if event_key in processed_events:
            continue
            
        # 检查是否为有效事件类型
        if not event_classifier.is_event_type(event["event_type"]):
            logger.debug(f"Filtered out invalid event type: {event['event_type']}")
            continue
            
        # 添加到已处理集合和过滤后的事件列表
        processed_events.add(event_key)
        filtered_events.append(event)

    logger.info(f"Extracted {len(filtered_events)} unique events for '{entity_name}'.")
    return filtered_events


def extract_comprehensive_information(
        entity_name: str,
        sections: List,
        llm_client: LLMClient,
        model_name: str
) -> ExtractionResult:
    """
    综合抽取实体的属性、关系和事件信息。

    Args:
        entity_name: 核心实体的名称。
        sections: 包含文章章节的列表。
        llm_client: LLM客户端实例。
        model_name: 用于抽取的LLM模型名称。

    Returns:
        包含属性、关系和事件的完整抽取结果。
    """
    import concurrent.futures
    import threading
    
    # 并行抽取属性、关系和事件
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # 提交属性抽取任务
        attr_future = executor.submit(
            extract_multi_entity_attributes_from_sections,
            sections, llm_client, model_name
        )
        
        # 提交关系抽取任务
        relations_future = executor.submit(
            extract_relations_from_sections,
            entity_name, sections, llm_client, model_name
        )
        
        # 提交事件抽取任务
        events_future = executor.submit(
            extract_events_from_sections,
            entity_name, sections, llm_client, model_name
        )
        
        # 获取结果
        attributes = attr_future.result()
        relations_dict = relations_future.result()
        events = events_future.result()
    
    # 转换关系格式
    relations = []
    for predicate, objects in relations_dict.items():
        for obj in objects:
            relation = RelationExtraction(
                subject=entity_name,
                predicate=predicate,
                object=obj,
                confidence=config.get('confidence', {}).get('default_extraction', 0.8),  # 默认置信度
                source_text="",
                text_id=""
            )
            relations.append(relation)
    
    return ExtractionResult(
        attributes=attributes,
        relations=relations,
        events=events,
        is_relevant=len(attributes) > 0 or len(relations) > 0 or len(events) > 0
    )