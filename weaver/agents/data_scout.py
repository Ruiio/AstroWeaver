
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
from weaver.utils.webSearch import execute_web_query

# ==============================================================================
# 0. 日志和配置 (与之前相同)
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




# ==============================================================================
# 1. 工具函数 & 2. LLMClient & 3. DataScoutAgent
# (这些模块与版本3完全相同，此处省略以保持简洁，实际使用时请完整粘贴)
# ==============================================================================

class DataScout:
    def __init__(self,wikiClient: WikipediaClient,minerUClient:MinerUClient):
        self.wikiClient = wikiClient
        self.minerUClient=minerUClient

    def execute_simbad_search(self, query: str):
        return get_simbad_data(query)

    def execute_wikipedia_search(self, query: str):
        return self.wikiClient.get_article_sections(query)

    def execute_web_search(self, query: str):
        return execute_web_query(query)

    def execute_info_box(self,query:str):
        return  self.wikiClient.get_infobox(query)

    def process_pdf(self, file_path: str):
        return self.minerUClient.parseFile(file_path)

    def process_text(self, text: str):
        return

# NOTE: The above classes are placeholders. You must use the full code from the previous answer.


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
        self.llm_model = config['llm']['base_model']
        logger.info("Orchestrator Agent (Brainstorming Model) 已初始化。")

    def _classify_input(self, user_input: str) -> str:
        """使用LLM将用户输入分类为'direct_question'或'research_topic'。"""
        prompt = f"""
        Analyze the user's input and classify it into one of two categories: "direct_question" or "research_topic".
        - "direct_question": A specific question that seeks a direct answer (e.g., "What is a neutron star?", "Who discovered the first pulsar?").
        - "research_topic": A broad subject, entity, or concept that requires exploration from multiple angles (e.g., "Galaxy formation", "The Andromeda Galaxy", "Stellar Nucleosynthesis").

        User input: "{user_input}"

        Respond with a JSON object containing a single key "input_type" with the value "direct_question" or "research_topic".
        """
        messages = [{"role": "system", "content": "You are a helpful assistant that only outputs valid JSON."},
                    {"role": "user", "content": prompt}]
        try:
            response_str = self.llm_client.make_request(self.llm_model, messages)
            return json.loads(response_str.replace("```json", "").replace("```", "")).get("input_type", "direct_question")
        except Exception as e:
            logger.warning(f"输入分类失败: {e}. 默认视为直接问题。")
            return "direct_question"

    def _brainstorm_sub_queries(self, topic: str) -> List[str]:
        """对于一个宽泛的话题，使用LLM进行头脑风暴，生成一系列具体的研究问题。"""
        prompt = f"""
        You are an expert astronomical researcher. Your task is to break down a broad research topic into a set of 3-5 specific, insightful sub-queries. These queries should cover different facets of the topic, such as its definition, history, key components, physical processes, and recent discoveries.

        Broad research topic: "{topic}"

        Generate a JSON object with a single key "research_questions", which contains a list of these specific query strings.

        Example for "Supernovae":
        {{
          "research_questions": [
            "What are the different types of supernovae (e.g., Type Ia, Type II)?",
            "Describe the process of core-collapse in a massive star leading to a supernova.",
            "What is the role of supernovae in galactic chemical enrichment?",
            "Famous historical supernovae observations",
            "Recent discoveries or news about supernovae"
          ]
        }}
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
        You are an expert AI orchestrator. Given a user's query, decide which tool(s) to use. You have access to:
        1. `simbad_search`: For detailed data on a specific celestial object (e.g., "Betelgeuse", "M31").
        2. `wikipedia_search`: For encyclopedic info on a concept, or object (e.g., "Galaxy formation", "Supernova"), do not use topic as a query(eg:M78 discovery and historical significance).
        3. `web_search`: For recent news, discussions, or general questions (e.g., "latest news on Betelgeuse", "what is a neutron star?").
        4. `info_box`: For a json data on a specific celestial object (e.g., "Betelgeuse", "M31").

        User's query: "{query}"

        Respond with a JSON object containing a "tool_calls" list. Each item in the list should be an object with "tool" and "query". If no tool is appropriate, return an empty list.
        """
        messages = [{"role": "system", "content": "You are a helpful assistant that only outputs valid JSON."},
                    {"role": "user", "content": prompt}]
        try:
            response_str = self.llm_client.make_request(self.llm_model, messages)
            # 兼容不含 reasoning 的简化版 prompt
            plan = json.loads(response_str.replace("```json", "").replace("```", ""))
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

    def run(self, input_data: str) -> Optional[Dict[str, Any]]:
        """Agent的主入口点，处理所有类型的输入。"""
        # 步骤 0: 处理非文本或简单文本输入
        if not isinstance(input_data, str) or not input_data.strip(): return None
        if input_data.lower().endswith('.pdf') and os.path.exists(input_data):
            logger.info("检测到PDF文件，直接执行。")
            chunks = self.data_scout.process_pdf(input_data)
            return {"source_type": "pdf", "source_identifier": os.path.basename(input_data), "structured_data": {},
                    "text_chunks": chunks}

        # 步骤 1: 使用LLM对输入进行分类
        input_type = self._classify_input(input_data)
        logger.info(f"输入 '{input_data}' 被分类为: {input_type}")

        queries_to_process = []
        if input_type == "research_topic":
            # 步骤 1B: 如果是宽泛话题，进行头脑风暴
            queries_to_process = self._brainstorm_sub_queries(input_data)
        else:  # direct_question
            # 步骤 1A: 如果是直接问题，直接处理
            queries_to_process = [input_data]

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

        logger.info(f"生成了 {len(full_tool_plan)} 个工具调用。去重后剩余 {len(unique_tool_plan)} 个唯一调用。")
        logger.info(f"最终生成的唯一执行计划: {json.dumps(unique_tool_plan, indent=2)}")
        # 步骤 3: 执行完整的计划
        final_result = self._execute_plan(full_tool_plan)
        final_result['source_type'] = input_type
        final_result['source_identifier'] = input_data
        return final_result


# ==============================================================================
# 5. 使用示例 (Example Usage)
# ==============================================================================
if __name__ == '__main__':
    try:
        # 1. 初始化所有组件
        llm_client = LLMClient(api_key=config['api_keys']['llm'], base_url=config['llm']['base_url'])
        # 确保DataScoutAgent被正确实例化（此处为示意）
        data_scout = DataScout(wikiClient=WikipediaClient(),minerUClient=MinerUClient())
        orchestrator = Orchestrator(llm_client, data_scout)

        # --- 示例1: 一个需要头脑风暴的宽泛研究主题 ---
        # topic = "Messier 78"
        # print("\n" + "=" * 30 + f"\n🚀 EXECUTING RESEARCH TOPIC: '{topic}'\n" + "=" * 30)
        # result = orchestrator.run(topic)
        #
        # if result:
        #     print("\n✅ 主题研究成功！最终打包结果:")
        #     print(f"  - 原始输入: {result.get('source_identifier')}")
        #     print(f"  - 输入类型: {result.get('source_type')}")
        #
        #     print(f"  - 结构化数据条目数: {len(result.get('structured_data', {}))}")
        #     print(f"  - 结构化数据: {(result.get('structured_data', {}))}")
        #     print(f"  - 文本片段总数: {len(result.get('text_chunks', []))}")
        #     # 打印一些来源以展示多样性
        #     if result.get('text_chunks'):
        #         sources = set(chunk for chunk in result['text_chunks'])
        #         print(f"  - 收集到的数据来源: {list(sources)[:5]}")
        # else:
        #     print("\n❌ 主题研究失败。")

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
        file_path = r"C:\Users\Administrator\Desktop\astroWeaver\data\input\test_data\RAKCR.pdf"
        print("\n" + "=" * 30 + f"\n🚀 EXECUTING PDF FILE: '{file_path}'\n" + "=" * 30)
        result = orchestrator.run(file_path)

        if result:
            print("\n✅ PDF处理成功！最终打包结果:")
            print(f"  - 原始输入: {result.get('source_identifier')}")
            print(f"  - 输入类型: {result.get('source_type')}")
            print(f"  - 文本片段总数: {len(result.get('text_chunks', []))}")
            print(f"  - 文本片段示例: {result.get('text_chunks', [])[:5]}")
        else:
            print("\n❌ 直接问题处理失败。")


    except ValueError as e:
        print(f"\n❌ 初始化失败: {e}")
        print("请确保在 config 字典或环境变量中正确填写了 API 密钥。")
    except Exception as e:
        print(f"\n❌ 运行中发生意外错误: {e}")