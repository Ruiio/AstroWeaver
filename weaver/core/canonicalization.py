# weaver/core/canonicalization.py (Final Fixed Version)

import logging
import json
import re
import asyncio
from typing import List, Dict, Set, Optional, Any, Tuple

from ..models.llm_models import LLMClient
from ..storage.vector_db import VectorDBClient

logger = logging.getLogger(__name__)


def _get_canonicalization_prompt(term_type: str, new_term: str, candidates: List[str]) -> List[Dict[str, str]]:
    # ... (此函数保持不变) ...
    system_prompt = (
        "You are a pragmatic ontologist building a knowledge graph about astronomy. "
        "Your goal is to merge terms that represent the same concept to keep the graph clean and consistent. "
        "You MUST respond with a single, valid JSON object and nothing else."
    )

    user_prompt = f"""
        Task: Analyze the "New Term". Should it be merged with one of the "Candidate Canonical Terms"?
        For our knowledge graph, we merge terms if they represent the same core concept, even if they are not perfect linguistic synonyms.

        Term Type: {term_type}
        New Term: "{new_term}"
        Candidate Canonical Terms:
        {json.dumps(candidates, indent=2)}

        **Guiding Principles for Merging:**
        - **MERGE** if the terms are practically interchangeable for describing an astronomical object's properties or relationships.
        - **DO NOT MERGE** if one term describes an action and the other describes a state of being (e.g., "orbits" vs. "has moon").
        - **DO NOT MERGE** if the terms are merely related but distinct concepts (e.g., "is a star" vs. "emits light").

        **Good Examples of Merging (What we want):**
        - New Term: "has estimated age" -> Candidate: "hasAge" (Correct Choice: "hasAge")
        - New Term: "has diameter" -> Candidate: "hasRadius" (Correct Choice: "hasRadius", as they represent the same physical dimension)
        - New Term: "is also known as" -> Candidate: "isAliasOf" (Correct Choice: "isAliasOf")

        **Bad Examples of Merging (What to avoid):**
        - New Term: "has moon" -> Candidate: "orbits" (Incorrect. One is possession, the other is an action.)
        - New Term: "is visible from" -> Candidate: "isLocatedIn" (Incorrect. Visibility is different from physical location.)

        **Your Response:**
        Return a JSON object with a single key "choice".
        - If the "New Term" should be merged, the value is the name of the chosen "Candidate Canonical Term".
        - If it represents a distinct concept and should NOT be merged, the value must be `null`.
        - Do not include any other information in your response. Just return the JSON object.
        Format: {{"choice": "ChosenCandidateName"}} or {{"choice": null}}
        """
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def to_camel_case(text: str) -> str:
    # ... (此函数保持不变) ...
    if not text or not text.strip(): return ""
    text = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', text)).strip()
    text = re.sub(r'[\s_]+', ' ', text)
    parts = [word.lower() for word in text.split()]
    if not parts: return ""
    return parts[0] + ''.join(word.capitalize() for word in parts[1:])


def to_pascal_case(text: str) -> str:
    """
    (修复后) 将各种格式的字符串转换为大驼峰式 (PascalCase)，更健壮地处理数字和特殊字符。
    """
    if not text or not text.strip():
        return "UnnamedEntity"

    # 移除非字母数字字符，但保留空格以便分割
    term_cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).strip()

    # 按空格分割成单词
    parts = term_cleaned.split()

    # 将每个部分首字母大写后连接
    canonical_name = ''.join(word.capitalize() for word in parts)

    # 如果处理后为空（例如，输入只有标点符号），返回默认值
    return canonical_name if canonical_name else "UnnamedEntity"


async def _llm_judge_synonym_async(
        term_type: str, new_term: str, candidates: List[str], llm_client: LLMClient, model_name: str, timeout: float = 60.0
) -> Optional[str]:
    # ... (此函数保持不变) ...
    if not candidates: return None
    prompt = _get_canonicalization_prompt(term_type, new_term, candidates)
    try:
        raw_response = await llm_client.make_request_async(
            model=model_name, 
            messages=prompt, 
            temperature=0.0,
            timeout=timeout
        )
        if not raw_response or not raw_response.strip(): return None
        match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if not match: return None
        # 移除可能导致输出缓冲问题的print语句
        # print(match.group(0))
        result_data = json.loads(match.group(0))
        choice = result_data.get("choice")
        if isinstance(choice, str) and choice in candidates:
            return choice
        return None
    except asyncio.TimeoutError:
        logger.error(f"LLM judge timeout after {timeout}s for term '{new_term}'")
        return None
    except Exception as e:
        logger.error(f"LLM judge failed for term '{new_term}'. Error: {e}", exc_info=True)
        return None


async def _normalize_term(
        term_type: str,
        term: str,
        collection_name: str,
        vector_db_client: VectorDBClient,
        llm_client: LLMClient,
        llm_model_name: str,
        similarity_threshold: float,
        top_k: int,
        new_items_to_add: Dict[str, List[Dict]],
        # **新增**: 传入一个集合来跟踪本次运行中已经生成的新规范名
        staged_canonical_names: Set[str],
        # **新增**: 传入一个字典来跟踪本次运行中的术语到规范名的映射
        staged_term_mappings: Optional[Dict[str, str]] = None,
        llm_timeout: float = 60.0
) -> str:
    """
    (批内规范化增强版) 对单个术语进行完整的规范化流程，支持批内相似术语合并。
    """
    if staged_term_mappings is None:
        staged_term_mappings = {}
    
    # 1. 向量搜索数据库中的候选项
    try:
        search_results = vector_db_client.search(collection_name, [term], top_k)
        db_candidates = [res['metadata']['canonical_name'] for res in search_results[0] if
                        res['score'] >= similarity_threshold]
    except Exception as e:
        logger.warning(f"VectorDB search failed for term '{term}' in '{collection_name}': {e}. Assuming no candidates.")
        db_candidates = []

    # 2. 检查批内是否有相似术语已被规范化
    batch_candidates = []
    if staged_term_mappings:
        # 获取当前批次中已规范化的术语列表
        staged_canonical_list = list(set(staged_term_mappings.values()))
        if staged_canonical_list:
            # 使用LLM判断当前术语是否与批内已有的规范名相似
            batch_canonical = await _llm_judge_synonym_async(
                term_type, term, staged_canonical_list, llm_client, llm_model_name, timeout=llm_timeout
            )
            if batch_canonical:
                logger.info(f"Term '{term}' merged with batch canonical '{batch_canonical}'")
                return batch_canonical

    # 3. 合并数据库候选项和批内候选项
    all_candidates = db_candidates + batch_candidates
    
    # 4. LLM 判断数据库中的候选项
    canonical_name = None
    if all_candidates:
        canonical_name = await _llm_judge_synonym_async(term_type, term, all_candidates, llm_client, llm_model_name, timeout=llm_timeout)

    # 5. 创建或确认规范名称
    if canonical_name:
        logger.debug(f"Term '{term}' mapped to existing canonical '{canonical_name}'.")
    else:
        # 生成一个新的候选规范名
        if term_type == 'relation':
            new_canonical_candidate = to_camel_case(term)
        else:
            new_canonical_candidate = to_pascal_case(term)

        # **关键修复**: 检查这个新生成的规范名是否已经被暂存
        if new_canonical_candidate in staged_canonical_names:
            # 如果已经暂存，直接重用它，不再创建新条目
            logger.debug(f"Term '{term}' mapped to recently staged canonical '{new_canonical_candidate}'.")
            canonical_name = new_canonical_candidate
        else:
            # 如果是全新的，则暂存它
            canonical_name = new_canonical_candidate
            logger.info(f"Staging new canonical {term_type}: '{canonical_name}' from term '{term}'.")

            # 将其添加到本次运行的已暂存集合中
            staged_canonical_names.add(canonical_name)

            # 暂存到待写入数据库的列表中
            if collection_name not in new_items_to_add:
                new_items_to_add[collection_name] = []

            new_items_to_add[collection_name].append({
                "document": term,
                "metadata": {"canonical_name": canonical_name, "original_term": term},
                "id": canonical_name
            })
    
    # 6. 更新批内术语映射
    if staged_term_mappings is not None:
        staged_term_mappings[term] = canonical_name

    return canonical_name