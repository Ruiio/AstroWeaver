# astroWeaver/core/canonicalization.py

import logging
import json
import re
from typing import List, Dict, Set, Optional, Any, Tuple

from ..models.llm_models import LLMClient
from ..storage.vector_db import VectorDBClient

logger = logging.getLogger(__name__)


def _get_canonicalization_prompt(term_type: str, new_term: str, candidates: List[str]) -> List[Dict[str, str]]:
    """
    为规范化判断任务生成LLM prompt (严格版本)。
    """
    system_prompt = (
        "You are a strict ontologist for a knowledge graph. Your task is to perform precise synonym matching. "
        "Respond ONLY with the requested JSON object."
    )

    user_prompt = f"""
    Task: Evaluate if the "New Term" is a **direct and precise synonym** for ONE of the "Candidate Canonical Terms". Do not merge terms that are merely related or conceptually close. The meaning must be almost identical.

    Term Type: {term_type}
    New Term: "{new_term}"
    Candidate Canonical Terms:
    {json.dumps(candidates, indent=2)}

    Instructions:
    1.  If the new term means **exactly the same thing** as one of the candidates, return that candidate's name.
    2.  If it has a different nuance, is more general, more specific, or simply related but not identical, you MUST return `null`.

    Your response MUST be a valid JSON object containing either the chosen candidate string or `null`.

    Example 1 (Good Match):
    - New Term: "revolves around"
    - Candidates: ["Orbits"]
    - Correct JSON Response: {{"choice": "Orbits"}}

    Example 2 (Bad Match - Related, not Synonym):
    - New Term: "has moon"
    - Candidates: ["Orbits"]
    - Correct JSON Response: {{"choice": null}}
    """

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def _llm_judge_synonym(
        term_type: str,
        new_term: str,
        candidates: List[str],
        llm_client,  # 你的LLMClient实例
        model_name: str
) -> Optional[str]:
    """
    Asks an LLM to choose the best synonym for a new term from a list of candidates.
    """
    # --- 步骤2：优化Prompt ---
    # (见下文的Prompt优化部分)
    prompt_system = "You are an expert in knowledge graph ontology. Your task is to determine if a new term is a synonym of existing canonical terms. Respond ONLY with the requested JSON value."
    # --- 核心修改：增强Prompt ---
    prompt_system = "You are a precise ontologist for a knowledge graph. Your task is to identify strict synonyms. A strict synonym means two terms can be interchanged in a sentence without changing the core meaning of the relationship. Respond ONLY with the requested JSON value."

    prompt_user = f"""
    Analyze if the "New {term_type.capitalize()}" is a **strict synonym** for ONE of the "Existing Canonical Candidates".

    New {term_type.capitalize()}: "{new_term}"

    Existing Canonical Candidates:
    {json.dumps(candidates, indent=2)}

    **Definition of Strict Synonym:**
    The new term must represent the exact same action or concept as a candidate, not just be related. For example, "revolves around" is a strict synonym for "Orbits".

    **Crucial Rule:** Do not confuse related concepts with synonyms.
    - **Correct:** "revolves around" -> "Orbits" (They describe the same physical action).
    - **Incorrect:** "has moon" -> "Orbits" (Having a moon IMPLIES orbiting, but "has moon" describes possession of an object, while "Orbits" describes an action. They are related, but NOT synonyms).

    **Your Task:**
    Return a single JSON value.
    - If the new term is a strict synonym for one of the candidates, return that candidate's name as a JSON string.
    - If it is not a strict synonym for any candidate, return JSON `null`.

    Example 1 (Strict Synonym):
    New Relation: "is a moon of"
    Candidates: ["Has Natural Satellite", "Orbits"]
    Correct JSON Response: "Orbits"

    Example 2 (Not a Synonym):
    New Relation: "has moon"
    Candidates: ["Orbits"]
    Correct JSON Response: null
    """

    try:
        raw_response = llm_client.make_request(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0.0,
            is_json=True  # 保持请求JSON格式
        )

        # --- 步骤1：增强解析的健壮性 ---
        if not raw_response or not raw_response.strip():
            logger.warning(f"LLM returned an empty response for term '{new_term}'. Treating as no match.")
            return None

        # 尝试从可能的markdown代码块或解释性文本中提取JSON
        # 这个正则表达式会寻找 "..." 或 null
        match = re.search(r'(".*?"|null)', raw_response)
        if not match:
            logger.warning(
                f"Could not find a valid JSON string or null in LLM response for term '{new_term}'. Raw response: '{raw_response}'")
            return None

        json_str = match.group(0)

        # 现在解析提取出的、更干净的JSON字符串
        result = json.loads(json_str)

        if isinstance(result, str) and result in candidates:
            return result

        # 如果result是null或者不是一个有效的候选者字符串，都视为没有匹配
        return None

    except json.JSONDecodeError as e:
        # 捕获JSON解析错误，这是你遇到的核心问题
        logger.error(
            f"LLM judge for term '{new_term}' returned invalid JSON. Error: {e}. Raw response: '{raw_response if 'raw_response' in locals() else 'N/A'}'")
        return None
    except Exception as e:
        # 捕获其他所有可能的异常，如API调用失败
        logger.error(f"LLM judge failed for term '{new_term}'. Unexpected error: {e}")
        return None


async def _normalize_term(
        term_type: str,
        term: str,
        collection_name: str,
        vector_db_client: VectorDBClient,
        llm_client: LLMClient,
        llm_model_name: str,
        similarity_threshold: float,
        top_k: int
) -> str:
    """
    对单个术语进行完整的规范化流程：搜索 -> 判断 -> 更新。
    """
    # 直接使用文本进行搜索
    search_results = vector_db_client.search(collection_name, [term], top_k)

    candidates = [
        res['metadata']['canonical_name'] for res in search_results
        if res['score'] >= similarity_threshold
    ]

    canonical_name = _llm_judge_synonym(term_type, term, candidates, llm_client, llm_model_name)

    if canonical_name:
        logger.debug(f"Term '{term}' mapped to existing canonical '{canonical_name}'.")
    else:
        canonical_name = term.replace("_", " ").title()
        logger.info(f"Creating new canonical {term_type}: '{canonical_name}' from term '{term}'.")
        # 只使用文本进行添加
        vector_db_client.add(
            collection_name=collection_name,
            documents=[term],
            metadata=[{"canonical_name": canonical_name, "original_term": term}],
            ids=[canonical_name]
        )

    return canonical_name


async def canonicalize_graph(
        raw_relations: Dict[str, Set[str]],
        vector_db_client: VectorDBClient,
        llm_client: LLMClient,
        config: Dict[str, Any],
        relation_collection: str = "canonical_relations",
        entity_collection: str = "canonical_entities"
) -> Tuple[Dict[str, List[str]], Dict, Dict]:
    """
    对从一篇文章中抽取的整个原始图谱进行规范化。
    返回最终图谱和映射关系。
    """
    import asyncio

    all_raw_relations = set(raw_relations.keys())
    all_raw_entities = set()
    for entities in raw_relations.values():
        all_raw_entities.update(entities)

    relation_tasks = [
        _normalize_term(
            "relation", rel, relation_collection, vector_db_client, llm_client,
            config['llm']['judge_model'], config['vector_db']['similarity_threshold'], config['vector_db']['top_k']
        )
        for rel in all_raw_relations
    ]
    entity_tasks = [
        _normalize_term(
            "entity", ent, entity_collection, vector_db_client, llm_client,
            config['llm']['judge_model'], config['vector_db']['similarity_threshold'], config['vector_db']['top_k']
        )
        for ent in all_raw_entities
    ]

    canonical_relations = await asyncio.gather(*relation_tasks)
    canonical_entities = await asyncio.gather(*entity_tasks)

    relation_map = dict(zip(all_raw_relations, canonical_relations))
    entity_map = dict(zip(all_raw_entities, canonical_entities))

    # 重构图谱
    final_graph = {}
    for raw_rel, raw_objs in raw_relations.items():
        canonical_rel = relation_map.get(raw_rel)
        if not canonical_rel: continue

        if canonical_rel not in final_graph:
            final_graph[canonical_rel] = set()

        for raw_obj in raw_objs:
            canonical_obj = entity_map.get(raw_obj)
            if canonical_obj:
                final_graph[canonical_rel].add(canonical_obj)

    final_graph_list_values = {k: sorted(list(v)) for k, v in final_graph.items()}

    return final_graph_list_values, relation_map, entity_map