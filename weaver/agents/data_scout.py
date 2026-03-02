
import json
import os
import logging
from typing import Dict, Any, List, Union, Optional

from data_sources.mineru_client import MinerUClient
from data_sources.wikipedia_client import WikipediaClient
from weaver.models.llm_models import LLMClient
from weaver.utils.MinerU import parse_file
from weaver.utils.config import config
from weaver.utils.get_simbads import get_simbad_data
from weaver.utils.webSearch_enhanced import EnhancedWebSearcher
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==============================================================================
# 0. 日志和配置 (与之前相同)
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




# ==============================================================================
# 1. DataScout
# 负责对外部数据源进行检索与预处理，向后续抽取阶段统一输出 text_blocks。
# ==============================================================================

class DataScout:
    def __init__(self,wikiClient: WikipediaClient,minerUClient:MinerUClient):
        self.wikiClient = wikiClient
        self.minerUClient=minerUClient
        self.web_searcher = EnhancedWebSearcher()

    def execute_simbad_search(self, query: str):
        return get_simbad_data(query)

    def execute_wikipedia_search(self, query: str):
        return self.wikiClient.get_article_sections(query)

    def execute_web_search(self, query: str):
        return self.web_searcher.execute_web_query(query)

    def execute_info_box(self,query:str):
        return  self.wikiClient.get_infobox(query)

    def process_pdf(self, file_path: str):
        # 使用MinerU解析PDF文件
        try:
            result = parse_file(file_path)
            if result and isinstance(result, dict):
                # MinerU返回的是 {filename: md_content} 格式
                text_blocks = []
                
                # 初始化文本分割器
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config['chunk']['chunk_size'],
                    chunk_overlap=config['chunk']['chunk_overlap'],
                    length_function=len,
                )
                
                for filename, content in result.items():
                    if content and content.strip():
                        # 对长文本进行分块处理
                        if len(content) > config['chunk']['chunk_size']:
                            # 使用文本分割器分割长文本
                            chunks = text_splitter.create_documents([content])
                            for chunk in chunks:
                                chunk_content = chunk.page_content.strip()
                                if chunk_content:  # 确保分块内容不为空
                                    text_blocks.append({
                                        "text": chunk_content,
                                        "source_type": "pdf",
                                        "source_path": file_path
                                    })
                        else:
                            text_blocks.append({
                                "text": content,
                                "source_type": "pdf",
                                "source_path": file_path
                            })
                            
                if text_blocks:
                    logger.info(f"PDF解析成功: {file_path}, 提取到 {len(text_blocks)} 个文本块")
                    return text_blocks
                else:
                    logger.warning(f"PDF解析结果为空: {file_path}")
                    return []
            else:
                logger.warning(f"PDF解析结果为空或格式不正确: {file_path}")
                return []
        except Exception as e:
            logger.error(f"PDF解析失败: {file_path}, 错误: {e}")
            return []

    def process_text(self, text: str) -> List[Dict[str, str]]:
        """将原始文本切分为统一 text_blocks 结构。"""
        try:
            if not text or not str(text).strip():
                return []

            raw_text = str(text).strip()
            text_blocks: List[Dict[str, str]] = []

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config['chunk']['chunk_size'],
                chunk_overlap=config['chunk']['chunk_overlap'],
                length_function=len,
            )

            if len(raw_text) > config['chunk']['chunk_size']:
                chunks = text_splitter.create_documents([raw_text])
                for chunk in chunks:
                    chunk_content = chunk.page_content.strip()
                    if chunk_content:
                        text_blocks.append({
                            "text": chunk_content,
                            "source_type": "text",
                            "source_path": "inline_text"
                        })
            else:
                text_blocks.append({
                    "text": raw_text,
                    "source_type": "text",
                    "source_path": "inline_text"
                })

            return text_blocks
        except Exception as e:
            logger.error(f"文本处理失败: {e}")
            return []


# ==============================================================================
# 4. 中枢协调者 (Orchestrator Agent) - 全新 "Brainstorm" 逻辑
# ==============================================================================
class Orchestrator:
    """
    使用LLM作为大脑，能够对宽泛话题进行头脑风暴，分解成具体问题，
    然后为这些问题创建并执行工具调用计划。
    """

    def __init__(self, llm_client: LLMClient, data_scout: DataScout):
        self.llm_client = llm_client
        self.data_scout = data_scout
        # 获取LLM配置
        llm_config = config.get('llm', {})
        provider = llm_config.get('provider', 'zhipu')
        
        # 根据provider获取对应的模型配置
        if provider in llm_config:
            self.llm_model = llm_config[provider].get('base_model')
        else:
            # 向后兼容：使用默认配置
            self.llm_model = llm_config.get('base_model', 'GLM-4-Flash-250414')
        logger.info("Orchestrator Agent (Brainstorming Model) 已初始化。")



    def _brainstorm_sub_queries(self, topic: str) -> List[str]:
        """对于一个宽泛的话题，使用LLM进行头脑风暴，生成一系列具体的研究问题。"""
        prompt = f"""
        As an astronomical researcher, I need to explore the topic "{topic}" comprehensively. Help me generate 3-5 focused research questions that will uncover key information about this subject.

        Each question should:
        - Target specific entities, objects, or phenomena related to {topic}
        - Include relevant astronomical terms, catalog names, or proper nouns when applicable
        - Cover different aspects like physical properties, observational data, historical context, or recent findings
        - Be specific enough to yield detailed, factual information

        For example, if exploring "Betelgeuse", good questions would mention "Betelgeuse supernova predictions", "Orion constellation", "red supergiant characteristics", "Hipparcos parallax measurements", etc.

        Return your response as a JSON object with key "research_questions" containing the list of questions.

        Topic to explore: {topic}
        """
        messages = [{"role": "system", "content": "You are a helpful assistant that only outputs valid JSON."},
                    {"role": "user", "content": prompt}]
        logger.info(f"正在为主题 '{topic}' 进行头脑风暴...")
        try:
            response_str = self.llm_client.make_request(self.llm_model, messages)
            questions = json.loads(response_str.replace("```json", "").replace("```", "")).get("research_questions", [])
            logger.info(f"头脑风暴生成的问题: {questions}")
            return questions
        except Exception as e:
            logger.error(f"头脑风暴失败: {e}. 将使用原始主题作为唯一查询。")
            return [topic]

    def _create_tool_plan_for_query(self, query: str) -> List[Dict[str, str]]:
        """为单个具体问题生成工具调用计划。"""
        # (此函数与版本3中的工具选择prompt完全相同)
        prompt = f"""
        # Role: Expert AI Tool Orchestrator

Your task is to analyze the user's query and select the most appropriate tool(s) to answer it comprehensively. Your goal is to gather all necessary information efficiently.

## Available Tools

Here is a list of tools you can use. Each tool has a specific purpose.

1.  **`simbad_search`**
    *   **Purpose**: Retrieves professional, structured astronomical data from the SIMBAD database.
    *   **Use When**: The user needs precise, scientific data about a specific celestial object (e.g., coordinates, spectral type, parallax, identifiers). This is for expert-level data.
    *   **Example Query**: "Betelgeuse", "M31", "NGC 1976"

2.  **`wikipedia_search`**
    *   **Purpose**: Fetches general, encyclopedic information (narrative text).
    *   **Use When**: The user asks about a well-known entity, concept, or object and needs a descriptive overview, history, or cultural context.
    *   **Example Query**: "Andromeda Galaxy", "Orion Nebula", "Black hole", "Supernova"

3.  **`web_search`**
    *   **Purpose**: Performs a general web search for up-to-date information, complex questions, or topics not covered by other tools.
    *   **Use When**:
        *   The query is a complex question (e.g., "how do neutron stars form?").
        *   The query is about recent news or discoveries (e.g., "latest James Webb telescope findings").
        *   The query is a descriptive phrase rather than a specific entity name (e.g., "stellar formation in molecular clouds").
        *   You are unsure if other tools will find a result.

4.  **`info_box`**
    *   **Purpose**: Extracts a structured summary of key facts (like a Wikipedia infobox) for any named entity.
    *   **Use When**: The user needs a quick, structured summary of an entity's main attributes (e.g., for a star: type, mass, distance; for a person: birth date, nationality). This provides high-level structured data.
    *   **Example Query**: "Betelgeuse", "Starlink", "Zhang Heng"

## Decision-Making Logic & Rules

1.  **Analyze User Intent**: First, understand what the user is asking for. Do they need scientific data, a general description, a quick summary, or an answer to a complex question?

2.  **Prioritize Specificity**:
    *   For **celestial objects**, `simbad_search` is the top priority for *scientific data*.
    *   For any **named entity**, `wikipedia_search` is for *narrative text*, and `info_box` is for a *structured summary*.

3.  **Combine Tools for Comprehensive Answers (IMPORTANT)**: You can and should select multiple tools if they serve different, complementary purposes for the same query.
    *   **Example 1**: User query is "告诉我关于参宿四的详细信息" (Tell me detailed information about Betelgeuse).
        *   A comprehensive answer requires both scientific data and general knowledge.
        *   Therefore, you should call:
            *   `simbad_search` (for precise astronomical data).
            *   `wikipedia_search` (for its history, discovery, and cultural significance).
            *   `info_box` (for a quick, structured summary of its key properties).
    *   **Example 2**: User query is "超新星是如何形成的？" (How do supernovae form?).
        *   This is a complex "how-to" question.
        *   Only one tool is appropriate: `web_search`.

4.  **Default to `web_search`**: If a query is complex, phrased as a question, or you are uncertain, `web_search` is the safest and most versatile option.

## Your Task

User's query: "{query}"

Respond with a JSON object containing a "tool_calls" list. Each item in the list must be an object with "tool" and "query". If no tool is appropriate, return an empty list.Only outputs valid JSON! Do not include any other text.
        """
        messages = [{"role": "system", "content": "You are a helpful assistant that only outputs valid JSON."},
                    {"role": "user", "content": prompt}]
        try:
            response_str = self.llm_client.make_request(self.llm_model, messages)
            # 兼容不含 reasoning 的简化版 prompt
            plan = json.loads(response_str.replace("```json", "").replace("```", ""))
            if isinstance(plan, list):
                return plan
            return plan.get("tool_calls", [])
        except Exception as e:
            logger.error(f"为查询 '{query}' 创建工具计划失败: {e}")
            return []

    def _execute_plan(self, plan: List[Dict[str, str]]) -> Dict[str, Any]:
        """执行一个包含多个工具调用的列表。"""
        final_result = {"structured_data": {}, "text_chunks": []}
        for call in plan:
            tool_name, query = call.get("tool"), call.get("query")
            if not tool_name or not query: continue

            if tool_name == "simbad_search":
                data = self.data_scout.execute_simbad_search(query)
                if data: final_result["structured_data"][query+"_simbad"] = data
            elif tool_name == "wikipedia_search":
                final_result["text_chunks"].extend(self.data_scout.execute_wikipedia_search(query))
            elif tool_name == "web_search":
                final_result["text_chunks"].extend(self.data_scout.execute_web_search(query))
            elif tool_name == "info_box":
                data = self.data_scout.execute_info_box(query)
                if data: final_result["structured_data"][query+"_info_box"] = data
        return final_result

    def run(self, input_data: Union[str, Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Agent的主入口点，处理所有类型的输入。"""
        # 步骤 0: 处理空输入
        if not input_data: 
            return None
            
        # 步骤 1: 根据输入格式确定处理方式
        if isinstance(input_data, dict):
            # 处理字典格式输入 {"topic": "...", "entity": "...", "direct_question": "...", "pdf": "..."}
            if "topic" in input_data:
                topic_value = input_data["topic"]
                logger.info(f"处理topic类型输入: {topic_value}")
                queries_to_process = self._brainstorm_sub_queries(topic_value)
                input_type = "research_topic"
                source_identifier = topic_value
                
            elif "entity" in input_data:
                entity_value = input_data["entity"]
                logger.info(f"处理entity类型输入: {entity_value}")
                # 对于实体类型，直接使用实体名称作为查询，不进行头脑风暴
                queries_to_process = [entity_value]
                input_type = "entity"
                source_identifier = entity_value
                
            elif "direct_question" in input_data:
                question_value = input_data["direct_question"]
                logger.info(f"处理direct_question类型输入: {question_value}")
                # 对于直接问题，不进行头脑风暴，直接生成工具调用计划
                queries_to_process = [question_value]
                input_type = "direct_question"
                source_identifier = question_value
                
            elif "pdf" in input_data:
                pdf_path = input_data["pdf"]
                logger.info(f"处理pdf类型输入: {pdf_path}")
                if os.path.exists(pdf_path):
                    chunks = self.data_scout.process_pdf(pdf_path)
                    return {"source_type": "pdf", "source_identifier": os.path.basename(pdf_path), 
                           "structured_data": {}, "text_chunks": chunks}
                else:
                    logger.error(f"PDF文件不存在: {pdf_path}")
                    return None
            else:
                logger.error(f"不支持的输入格式: {input_data}")
                return None
        else:
            # 向后兼容：处理字符串输入（保留原有逻辑用于测试）
            if not isinstance(input_data, str) or not input_data.strip(): 
                return None
            if input_data.lower().endswith('.pdf') and os.path.exists(input_data):
                logger.info("检测到PDF文件，直接执行。")
                chunks = self.data_scout.process_pdf(input_data)
                return {"source_type": "pdf", "source_identifier": os.path.basename(input_data), 
                       "structured_data": {}, "text_chunks": chunks}
            
            # 默认作为研究主题处理
            logger.info(f"处理字符串输入作为研究主题: {input_data}")
            queries_to_process = self._brainstorm_sub_queries(input_data)
            input_type = "research_topic"
            source_identifier = input_data

        if not queries_to_process:
            logger.warning("没有生成任何可处理的查询。")
            return None

        # 步骤 2: 为所有待处理的查询创建工具计划
        full_tool_plan = []
        for query in queries_to_process:
            tool_calls = self._create_tool_plan_for_query(query)
            full_tool_plan.extend(tool_calls)

        # --- 新增步骤 2.5: 对工具计划进行去重 ---
        unique_tool_plan = []
        seen_calls = set()
        for call in full_tool_plan:
            # 将字典转换为可哈希的元组 (tool_name, query_string)
            # 我们对字典的items进行排序，以确保 {'tool':'a','query':'b'} 和 {'query':'b','tool':'a'} 被视为相同
            call_representation = tuple(sorted(call.items()))

            if call_representation not in seen_calls:
                seen_calls.add(call_representation)
                unique_tool_plan.append(call)

        # 对 research_topic 强制补充 wikipedia_search 兜底，避免 web_search 全失败时无文本
        if input_type == "research_topic" and source_identifier:
            has_wiki_fallback = any(
                call.get("tool") == "wikipedia_search" and str(call.get("query", "")).strip() == str(source_identifier).strip()
                for call in unique_tool_plan
            )
            if not has_wiki_fallback:
                unique_tool_plan.append({"tool": "wikipedia_search", "query": str(source_identifier).strip()})
                logger.info(f"已为主题补充 wikipedia_search 兜底: {source_identifier}")

        logger.info(f"生成了 {len(full_tool_plan)} 个工具调用。去重后剩余 {len(unique_tool_plan)} 个唯一调用。")
        logger.info(f"最终生成的唯一执行计划: {json.dumps(unique_tool_plan, indent=2)}")
        # 步骤 3: 执行去重后的计划
        final_result = self._execute_plan(unique_tool_plan)
        final_result['source_type'] = input_type
        final_result['source_identifier'] = source_identifier
        processed_chunks = []
        for chunk in final_result.get('text_chunks', []):
            if isinstance(chunk, str):
                processed_chunks.append({
                    'text': chunk,
                    'source_type': 'unknown',
                    'origin_type': input_type,
                    'origin_value': source_identifier
                })
            elif isinstance(chunk, dict):
                c = dict(chunk)
                if 'origin_type' not in c:
                    c['origin_type'] = input_type
                if 'origin_value' not in c:
                    c['origin_value'] = source_identifier
                processed_chunks.append(c)
        final_result['text_chunks'] = processed_chunks
        return final_result


# ==============================================================================
# 5. 使用示例 (Example Usage)
# ==============================================================================
if __name__ == '__main__':
    try:
        # 1. 初始化所有组件
        # 获取LLM配置
        llm_config = config.get('llm', {})
        provider = llm_config.get('provider', 'zhipu')
        
        # 根据provider获取对应的base_url配置
        if provider in llm_config:
            base_url = llm_config[provider].get('base_url')
        else:
            # 向后兼容：使用默认配置
            base_url = llm_config.get('base_url', 'https://open.bigmodel.cn/api/paas/v4')
        
        llm_client = LLMClient(api_key=config['api_keys']['llm'], base_url=base_url)
        # 确保DataScoutAgent被正确实例化（此处为示意）
        data_scout = DataScout(wikiClient=WikipediaClient(),minerUClient=MinerUClient())
        orchestrator = Orchestrator(llm_client, data_scout)

        # --- 示例1: 一个需要头脑风暴的宽泛研究主题 ---
        topic = "M7"
        print("\n" + "=" * 30 + f"\n🚀 EXECUTING RESEARCH TOPIC: '{topic}'\n" + "=" * 30)
        result = orchestrator.run(topic)

        if result:
            print("\n✅ 主题研究成功！最终打包结果:")
            print(f"  - 原始输入: {result.get('source_identifier')}")
            print(f"  - 输入类型: {result.get('source_type')}")

            print(f"  - 结构化数据条目数: {len(result.get('structured_data', {}))}")
            print(f"  - 结构化数据: {(result.get('structured_data', {}))}")
            print(f"  - 文本片段总数: {len(result.get('text_chunks', []))}")
            # 打印一些来源以展示多样性
            if result.get('text_chunks'):
                sources = set(chunk for chunk in result['text_chunks'])
                print(f"  - 收集到的数据来源: {list(sources)[:5]}")
        else:
            print("\n❌ 主题研究失败。")

        # --- 示例2: 一个可以直接回答的直接问题 ---
    #     question = "What is the Chandrasekhar Limit?"
    #     print("\n" + "=" * 30 + f"\n🚀 EXECUTING DIRECT QUESTION: '{question}'\n" + "=" * 30)
    #     result = orchestrator.run(question)
    #
    #     if result:
    #         print("\n✅ 直接问题处理成功！最终打包结果:")
    #         print(f"  - 原始输入: {result.get('source_identifier')}")
    #         print(f"  - 输入类型: {result.get('source_type')}")
    #         print(f"  - 文本片段总数: {len(result.get('text_chunks', []))}")
    #     else:
    #         print("\n❌ 直接问题处理失败。")
    #
    #
    # except ValueError as e:
    #     print(f"\n❌ 初始化失败: {e}")
    #     print("请确保在 config 字典或环境变量中正确填写了 API 密钥。")
    # except Exception as e:
    #     print(f"\n❌ 运行中发生意外错误: {e}")
        # --- 示例3: 一个pdf文件---
        # file_path = r"C:\Users\Administrator\Desktop\astroWeaver\data\input\test_data\RAKCR.pdf"
        # print("\n" + "=" * 30 + f"\n🚀 EXECUTING PDF FILE: '{file_path}'\n" + "=" * 30)
        # result = orchestrator.run(file_path)
        #
        # if result:
        #     print("\n✅ PDF处理成功！最终打包结果:")
        #     print(f"  - 原始输入: {result.get('source_identifier')}")
        #     print(f"  - 输入类型: {result.get('source_type')}")
        #     print(f"  - 文本片段总数: {len(result.get('text_chunks', []))}")
        #     print(f"  - 文本片段示例: {result.get('text_chunks', [])[:5]}")
        # else:
        #     print("\n❌ 直接问题处理失败。")


    except ValueError as e:
        print(f"\n❌ 初始化失败: {e}")
        print("请确保在 config 字典或环境变量中正确填写了 API 密钥。")
    except Exception as e:
        print(f"\n❌ 运行中发生意外错误: {e}")
