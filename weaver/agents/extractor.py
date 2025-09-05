
import logging
import json
import re
import asyncio
from typing import List, Dict, Any, Optional, TypedDict

from weaver.models.llm_models import LLMClient

logger = logging.getLogger(__name__)


class ExtractedTriple(TypedDict):
    """定义一个抽取得出的三元组结构，包含置信度和来源文本。"""
    subject: str
    predicate: str
    object: str
    confidence: float
    source_text: str
    text_id: str


def _get_extraction_prompt(text_block: str, topic: str) -> List[Dict[str, str]]:
    """
    为关系抽取和置信度评估任务生成LLM prompt。
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


def _parse_extraction_response(response_text: str) -> Optional[List[ExtractedTriple]]:
    """
    解析LLM返回的JSON字符串，提取三元组列表。
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

        # 验证并返回符合格式的三元组
        validated_triples = []
        for t in triples:
            if all(k in t for k in ["subject", "predicate", "object", "confidence"]):
                validated_triples.append(t)
        return validated_triples

    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Failed to parse LLM extraction response. Error: {e}. Response: {response_text[:300]}")
        return None


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

        prompt = _get_extraction_prompt(text_block, self.topic)
        try:
            raw_response = await self.llm_client.make_request_async(
                model=self.model_name,
                messages=prompt,
                temperature=0.1
            )
            triples = _parse_extraction_response(raw_response)
            if triples:
                # 为每个三元组添加source_text和text_id
                for triple in triples:
                    triple['source_text'] = text_block
                    triple['text_id'] = text_id
            return triples
        except Exception as e:
            logger.error(f"Error processing text block. Error: {e}")
            return None