
import logging
import asyncio
from typing import List, Dict, Any, Optional, TypedDict

from weaver.models.llm_models import LLMClient
from weaver.utils.config import config
from weaver.core.extraction_architecture import (
    get_multi_entity_attribute_extraction_prompt,
    parse_multi_entity_attribute_response,
    AttributeExtraction,
    ExtractionResult,
    RelationExtraction,
    EventExtraction,
    get_relation_extraction_prompt,
    parse_relation_response,
    get_event_extraction_prompt,
    parse_event_response,
    EventClassifier
)
from weaver.core.extraction import (
    _get_extraction_prompt,
    _parse_extraction_response,
    extract_comprehensive_information
)

logger = logging.getLogger(__name__)


class ExtractedTriple(TypedDict):
    """定义一个抽取得出的三元组结构，包含置信度和来源文本。"""
    subject: str
    predicate: str
    object: str
    confidence: float
    source_text: str
    text_id: str


# 移除重复的方法定义，使用从weaver.core.extraction导入的方法


class InformationExtractor:
    """
    信息提取员Agent，负责从文本块中抽取带置信度的三元组。
    """

    def __init__(self, llm_client: LLMClient, model_name: str, topic: str):
        self.llm_client = llm_client
        self.model_name = model_name
        self.topic = topic

    async def extract_from_text_blocks(self, text_blocks: List[str]) -> List[ExtractedTriple]:
        """
        使用Batch API从一系列文本块中抽取三元组。
        """
        all_extracted_triples: List[ExtractedTriple] = []

        # 为每个文本块生成唯一ID并创建任务
        tasks = []
        for i, block in enumerate(text_blocks):
            text_id = f"text_{i:06d}"
            tasks.append(self._process_single_block(block, text_id))
        
        results = await asyncio.gather(*tasks)

        for res in results:
            if res:
                all_extracted_triples.extend(res)

        logger.info(f"Extracted {len(all_extracted_triples)} triples from {len(text_blocks)} text blocks.")
        return all_extracted_triples

    async def _process_single_block(self, text_block: str, text_id: str) -> Optional[List[ExtractedTriple]]:
        """处理单个文本块的抽取逻辑。"""
        if not text_block or not text_block.strip():
            return None

        prompt = _get_extraction_prompt(self.topic, text_block)
        try:
            raw_response = await self.llm_client.make_request_async(
                model=self.model_name,
                messages=prompt,
                temperature=0.1
            )
            relations_dict = _parse_extraction_response(raw_response)
            if relations_dict:
                # 将关系字典转换为ExtractedTriple列表
                triples = []
                for predicate, objects in relations_dict.items():
                    for obj in objects:
                        triple = ExtractedTriple(
                            subject=self.topic,
                            predicate=predicate,
                            object=obj,
                            confidence=config.get('confidence', {}).get('default_extraction', 0.8),  # 默认置信度
                            source_text=text_block,
                            text_id=text_id
                        )
                        triples.append(triple)
                return triples
            return None
        except Exception as e:
            logger.error(f"Error processing text block. Error: {e}")
            return None

    async def extract_comprehensive_information(self, text_blocks: List[str], entity_name: str = None, multi_entity_mode: bool = False) -> ExtractionResult:
        """综合抽取信息，包括属性、关系和事件。"""
        try:
            # 如果指定了实体名称，使用extraction.py中的综合抽取方法
            if entity_name and not multi_entity_mode:
                # 将text_blocks转换为sections格式（字符串列表）
                sections = [block for block in text_blocks if block.strip()]
                
                # 调用extraction.py中的extract_comprehensive_information方法
                result = extract_comprehensive_information(
                    entity_name=entity_name,
                    sections=sections,
                    llm_client=self.llm_client,
                    model_name=self.model_name
                )
                return result
            else:
                # 多实体模式或未指定实体名称时，使用原有逻辑
                attributes: List[AttributeExtraction] = []
                relations: List[RelationExtraction] = []
                events: List[EventExtraction] = []
                
                for i, text_block in enumerate(text_blocks):
                    if not text_block.strip():
                        continue
                    
                    # 使用多实体属性抽取
                    if multi_entity_mode:
                        attr_prompt = get_multi_entity_attribute_extraction_prompt(text_block)
                        attr_response_text = await self.llm_client.make_request_async(
                            model=self.model_name,
                            messages=attr_prompt
                        )
                        
                        if attr_response_text:
                            parsed_attrs = parse_multi_entity_attribute_response(attr_response_text)
                            if parsed_attrs:
                                for entity_data in parsed_attrs:
                                    entity_name = entity_data["entity_name"]
                                    for attr in entity_data["attributes"]:
                                        attr_extraction = AttributeExtraction(
                                            entity=entity_name,
                                            attribute=attr["attribute"],
                                            value=attr["value"],
                                            confidence=attr.get("confidence", config.get('confidence', {}).get('default_extraction', 0.8)),
                                            source_text=text_block[:200],
                                            text_id=f"block_{i}"
                                        )
                                        attributes.append(attr_extraction)
                    
                    # 抽取关系
                    if multi_entity_mode:
                        # 在多实体模式下，使用新的关系抽取方法
                        rel_prompt = get_relation_extraction_prompt(text_block, self.topic)
                        rel_response_text = await self.llm_client.make_request_async(
                            model=self.model_name,
                            messages=rel_prompt
                        )
                        
                        if rel_response_text:
                            parsed_rels = parse_relation_response(rel_response_text)
                            if parsed_rels:
                                for rel in parsed_rels:
                                    relation = RelationExtraction(
                                        subject=rel['subject'],
                                        predicate=rel['predicate'],
                                        object=rel['object'],
                                        confidence=rel.get('confidence', config.get('confidence', {}).get('default_extraction', 0.8)),
                                        source_text=text_block[:200],
                                        text_id=f"block_{i}",
                                        attributes=rel.get('attributes', {})
                                    )
                                    relations.append(relation)
                        
                        # 抽取事件
                        event_prompt = get_event_extraction_prompt(text_block)
                        event_response_text = await self.llm_client.make_request_async(
                            model=self.model_name,
                            messages=event_prompt
                        )
                        
                        if event_response_text:
                            parsed_events = parse_event_response(event_response_text)
                            if parsed_events:
                                for evt in parsed_events:
                                    event = EventExtraction(
                                        event_type=evt['event_type'],
                                        anchor_entity=evt['anchor_entity'],
                                        arguments=evt['arguments'],
                                        confidence=evt.get('confidence', config.get('confidence', {}).get('default_extraction', 0.8)),
                                        source_text=text_block[:200],
                                        text_id=f"block_{i}"
                                    )
                                    events.append(event)
                    else:
                        # 单实体模式下使用原有方法
                        extracted_triples = await self._process_single_block(text_block, f"block_{i}")
                        if extracted_triples:
                            for triple in extracted_triples:
                                relation = RelationExtraction(
                                    subject=triple['subject'],
                                    predicate=triple['predicate'],
                                    object=triple['object'],
                                    confidence=triple['confidence'],
                                    source_text=triple['source_text'],
                                    text_id=triple['text_id'],
                                    attributes=triple.get('attributes', {})
                                )
                                relations.append(relation)
            
            return ExtractionResult(
                attributes=attributes,
                relations=relations,
                events=events,
                is_relevant=len(attributes) > 0 or len(relations) > 0 or len(events) > 0
            )
        except Exception as e:
            logger.error(f"Error in extract_comprehensive_information: {e}")
            return ExtractionResult(
                attributes=[],
                relations=[],
                events=[],
                is_relevant=False
            )
