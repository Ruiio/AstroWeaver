
import logging
import json
import re
import asyncio
from typing import List, Dict, Any, Optional, TypedDict

from weaver.models.llm_models import LLMClient
from weaver.core.extraction_architecture import (
    ExtractionType, AttributeExtraction, RelationExtraction, ExtractionResult,
    AttributeClassifier, EntityClassifier,
    get_attribute_extraction_prompt, get_relation_extraction_prompt,
    parse_attribute_response, parse_relation_response,
    get_multi_entity_attribute_extraction_prompt, parse_multi_entity_attribute_response
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


def _get_legacy_extraction_prompt(text_block: str, topic: str) -> List[Dict[str, str]]:
    """
    为关系抽取和置信度评估任务生成LLM prompt（保留旧版本以兼容性）。
    """
    system_prompt = (
        "You are an expert in knowledge graph construction. Your task is to analyze a text block, "
        "determine its relevance to a given topic, and if relevant, extract knowledge triples "
        "with a confidence score for each. Respond ONLY with the requested JSON object."
    )

    user_prompt = f"""
    **Task:**
    1.  **Relevance Check:** First, determine if the following "Text Block" is relevant to the main topic: **"{topic}"**.
    2.  **Extraction:** If and only if the text is relevant, extract all meaningful relationships as knowledge triples (Subject, Predicate, Object).
    3.  **Confidence Score:** For each extracted triple, provide a confidence score between 0.0 and 1.0, representing how certain you are that the triple is accurate and explicitly supported by the text.

    **Main Topic:** "{topic}"
    **Text Block:**
    ---
    {text_block}
    ---

    **Output Format Rules:**
    - Your response MUST be a single JSON object.
    - If the text is NOT relevant to the topic, or no triples can be extracted, return an empty list for the "triples" key.
    - Predicate names should be standardized and verb-like (e.g., "orbits", "hasMember", "discoveredBy").
    - The confidence score should be a float. 1.0 means absolute certainty based on the text. 0.5 means it's plausible but not explicit.
    - Do not use phrase as subject or object!
    **JSON Output Structure:**
    ```json
    {{
      "is_relevant": boolean,
      "triples": [
        {{
          "subject": "Entity1",
          "predicate": "hasProperty",
          "object": "Entity2",
          "confidence": 0.8
        }},
        {{
          "subject": "Entity3",
          "predicate": "isPartOf",
          "object": "Entity4",
          "confidence": 0.5
        }}
      ]
    }}
    ```
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def _parse_legacy_extraction_response(response_text: str) -> Optional[List[ExtractedTriple]]:
    """
    解析LLM返回的JSON字符串，提取三元组列表（保留旧版本以兼容性）。
    """
    try:
        # 寻找被```json ... ```包裹的代码块
        match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = response_text

        data = json.loads(json_str)

        if not data.get("is_relevant"):
            return []

        triples = data.get("triples", [])
        if not isinstance(triples, list):
            logger.warning(f"LLM response 'triples' is not a list. Response: {response_text[:200]}")
            return None

        # 验证并返回符合格式的三元组，同时过滤属性关系
        attribute_classifier = AttributeClassifier()
        entity_classifier = EntityClassifier()
        
        validated_triples = []
        for t in triples:
            if all(k in t for k in ["subject", "predicate", "object", "confidence"]):
                # 过滤属性关系
                if attribute_classifier.is_attribute_predicate(t["predicate"]) or \
                   attribute_classifier.is_attribute_value(t["object"]):
                    logger.debug(f"Filtered out attribute relation: {t}")
                    continue
                
                # 确保对象是实体
                if not entity_classifier.is_entity(t["object"]):
                    logger.debug(f"Filtered out non-entity object: {t}")
                    continue
                
                validated_triples.append(t)
        return validated_triples

    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Failed to parse LLM extraction response. Error: {e}. Response: {response_text[:300]}")
        return None


class InformationExtractor:
    """
    信息提取员Agent，负责从文本块中抽取带置信度的三元组和属性。
    现在支持区分属性抽取和关系抽取。
    """

    def __init__(self, llm_client: LLMClient, model_name: str, topic: str):
        self.llm_client = llm_client
        self.model_name = model_name
        self.topic = topic
        self.attribute_classifier = AttributeClassifier()
        self.entity_classifier = EntityClassifier()

    async def extract_from_text_blocks(self, text_blocks: List[str]) -> List[ExtractedTriple]:
        """
        使用Batch API从一系列文本块中抽取三元组（仅关系，不包含属性）。
        """
        all_extracted_triples: List[ExtractedTriple] = []

        # 为每个文本块生成唯一ID并创建任务
        tasks = []
        for i, block in enumerate(text_blocks):
            text_id = f"text_{i:06d}"
            tasks.append(self._process_single_block_relations(block, text_id))
        
        results = await asyncio.gather(*tasks)

        for res in results:
            if res:
                all_extracted_triples.extend(res)

        logger.info(f"Extracted {len(all_extracted_triples)} relation triples from {len(text_blocks)} text blocks.")
        return all_extracted_triples

    async def extract_attributes_from_text_blocks(self, text_blocks: List[str], entity_name: str) -> List[AttributeExtraction]:
        """
        使用Batch API从一系列文本块中抽取属性。
        """
        all_extracted_attributes: List[AttributeExtraction] = []

        # 为每个文本块生成唯一ID并创建任务
        tasks = []
        for i, block in enumerate(text_blocks):
            text_id = f"text_{i:06d}"
            tasks.append(self._process_single_block_attributes(block, text_id, entity_name))
        
        results = await asyncio.gather(*tasks)

        for res in results:
            if res:
                all_extracted_attributes.extend(res)

        logger.info(f"Extracted {len(all_extracted_attributes)} attributes from {len(text_blocks)} text blocks.")
        return all_extracted_attributes

    async def extract_multi_entity_attributes_from_text_blocks(self, text_blocks: List[str]) -> List[AttributeExtraction]:
        """
        从文本块中抽取所有实体的属性。
        """
        all_attributes = []
        
        # 并行处理所有文本块
        tasks = []
        for i, text_block in enumerate(text_blocks):
            text_id = f"block_{i}"
            task = self._process_single_block_multi_entity_attributes(text_block, text_id)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in multi-entity attribute extraction: {result}")
                continue
            if result:
                all_attributes.extend(result)
        
        logger.info(f"Multi-entity attribute extraction completed. Total attributes: {len(all_attributes)}")
        return all_attributes

    async def extract_comprehensive_information(self, text_blocks: List[str], entity_name: str = None, multi_entity_mode: bool = False) -> ExtractionResult:
        """
        综合抽取文本块中的属性和关系信息。
        
        Args:
            text_blocks: 文本块列表
            entity_name: 指定实体名称（单实体模式）
            multi_entity_mode: 是否启用多实体属性抽取模式
        """
        if multi_entity_mode:
            # 多实体模式：抽取所有实体的属性
            attributes_task = self.extract_multi_entity_attributes_from_text_blocks(text_blocks)
            relations_task = self.extract_from_text_blocks(text_blocks)
            
            attributes, relation_triples = await asyncio.gather(attributes_task, relations_task)
        else:
            # 单实体模式：只抽取指定实体的属性
            if not entity_name:
                raise ValueError("单实体模式需要指定entity_name")
            attributes_task = self.extract_attributes_from_text_blocks(text_blocks, entity_name)
            relations_task = self.extract_from_text_blocks(text_blocks)
            
            attributes, relation_triples = await asyncio.gather(attributes_task, relations_task)
        
        # 转换关系格式
        relations = []
        for triple in relation_triples:
            relation = RelationExtraction(
                subject=triple["subject"],
                predicate=triple["predicate"],
                object=triple["object"],
                confidence=triple["confidence"],
                source_text=triple["source_text"],
                text_id=triple["text_id"]
            )
            relations.append(relation)
        
        return ExtractionResult(
            attributes=attributes,
            relations=relations,
            is_relevant=len(attributes) > 0 or len(relations) > 0
        )

    async def _process_single_block_relations(self, text_block: str, text_id: str) -> Optional[List[ExtractedTriple]]:
        """处理单个文本块的关系抽取逻辑。"""
        if not text_block or not text_block.strip():
            return None

        prompt = get_relation_extraction_prompt(text_block, self.topic)
        try:
            raw_response = await self.llm_client.make_request_async(
                model=self.model_name,
                messages=prompt,
                temperature=0.1
            )
            relations = parse_relation_response(raw_response)
            if relations:
                # 转换为ExtractedTriple格式并添加source_text和text_id
                triples = []
                for rel in relations:
                    # 额外过滤：确保不是属性关系
                    if self.attribute_classifier.is_attribute_predicate(rel["predicate"]) or \
                       self.attribute_classifier.is_attribute_value(rel["object"]):
                        logger.debug(f"Filtered out attribute relation: {rel}")
                        continue
                    
                    # 确保对象是实体
                    if not self.entity_classifier.is_entity(rel["object"]):
                        logger.debug(f"Filtered out non-entity object: {rel}")
                        continue
                    
                    triple = ExtractedTriple(
                        subject=rel["subject"],
                        predicate=rel["predicate"],
                        object=rel["object"],
                        confidence=rel["confidence"],
                        source_text=text_block,
                        text_id=text_id
                    )
                    triples.append(triple)
                return triples
            return None
        except Exception as e:
            logger.error(f"Error processing text block for relations. Error: {e}")
            return None

    async def _process_single_block_attributes(self, text_block: str, text_id: str, entity_name: str) -> Optional[List[AttributeExtraction]]:
        """处理单个文本块的属性抽取逻辑。"""
        if not text_block or not text_block.strip():
            return None

        prompt = get_attribute_extraction_prompt(entity_name, text_block)
        try:
            raw_response = await self.llm_client.make_request_async(
                model=self.model_name,
                messages=prompt,
                temperature=0.1
            )
            attributes = parse_attribute_response(raw_response)
            if attributes:
                # 添加source_text和text_id
                attr_extractions = []
                for attr in attributes:
                    attr_extraction = AttributeExtraction(
                        entity=entity_name,
                        attribute=attr["attribute"],
                        value=attr["value"],
                        confidence=attr["confidence"],
                        source_text=text_block,
                        text_id=text_id
                    )
                    attr_extractions.append(attr_extraction)
                return attr_extractions
            return None
        except Exception as e:
            logger.error(f"Error processing text block for attributes. Error: {e}")
            return None

    async def _process_single_block_multi_entity_attributes(self, text_block: str, text_id: str) -> Optional[List[AttributeExtraction]]:
        """处理单个文本块的多实体属性抽取逻辑。"""
        if not text_block or not text_block.strip():
            return None

        prompt = get_multi_entity_attribute_extraction_prompt(text_block)
        try:
            raw_response = await self.llm_client.make_request_async(
                model=self.model_name,
                messages=prompt,
                temperature=0.1
            )
            entities_data = parse_multi_entity_attribute_response(raw_response)
            if entities_data:
                # 添加source_text和text_id
                attr_extractions = []
                for entity_data in entities_data:
                    entity_name = entity_data["entity_name"]
                    
                    # 额外保护：确保entity_name是字符串
                    if isinstance(entity_name, dict):
                        if "entity_name" in entity_name:
                            entity_name = entity_name["entity_name"]
                        else:
                            logger.warning(f"Invalid entity_name format in extractor: {entity_name}")
                            continue
                    elif not isinstance(entity_name, str):
                        logger.warning(f"Entity name is not string in extractor: {entity_name}")
                        entity_name = str(entity_name)
                    
                    for attr in entity_data["attributes"]:
                        attr_extraction = AttributeExtraction(
                            entity=entity_name,
                            attribute=attr["attribute"],
                            value=attr["value"],
                            confidence=attr["confidence"],
                            source_text=text_block,
                            text_id=text_id
                        )
                        attr_extractions.append(attr_extraction)
                return attr_extractions
            return None
        except Exception as e:
            logger.error(f"Error processing text block for multi-entity attributes. Error: {e}")
            return None

    async def _process_single_block_legacy(self, text_block: str, text_id: str) -> Optional[List[ExtractedTriple]]:
        """处理单个文本块的抽取逻辑（保留旧版本以兼容性）。"""
        if not text_block or not text_block.strip():
            return None

        prompt = _get_legacy_extraction_prompt(text_block, self.topic)
        try:
            raw_response = await self.llm_client.make_request_async(
                model=self.model_name,
                messages=prompt,
                temperature=0.1
            )
            triples = _parse_legacy_extraction_response(raw_response)
            if triples:
                # 为每个三元组添加source_text和text_id
                for triple in triples:
                    triple['source_text'] = text_block
                    triple['text_id'] = text_id
            return triples
        except Exception as e:
            logger.error(f"Error processing text block. Error: {e}")
            return None