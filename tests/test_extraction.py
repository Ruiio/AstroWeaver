# tests/test_extraction.py

import unittest
import os
import logging

# 导入我们项目中的真实客户端
from weaver.models.llm_models import LLMClient
from weaver.core.extraction import extract_relations_from_sections, _parse_extraction_response

# 配置日志，以便在测试期间看到输出
logging.basicConfig(level=logging.INFO)

# 使用unittest.skipIf来标记需要API密钥的测试
# 如果环境变量不存在，这个测试将被跳过
API_KEY = "sk-aeb0ed24d0c045e78d73fff879da5f07"
BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = os.environ.get("TEST_MODEL_NAME", "qwen-plus")  # 使用一个可配置的模型


@unittest.skipIf(not API_KEY, "DASHSCOPE_API_KEY environment variable not set, skipping integration tests.")
class TestExtractionIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """在所有测试开始前，初始化一个共享的LLM客户端。"""
        cls.llm_client = LLMClient(api_key=API_KEY, base_url=BASE_URL)
        logging.info(f"Running integration tests with model: {MODEL_NAME}")

    def test_parse_extraction_response_real_case(self):
        """
        测试解析一个真实的、可能由LLM生成的JSON字符串。
        这个测试仍然是本地的，不调用API。
        """
        # 这是一个更真实的例子，LLM可能返回这样的格式
        real_response = '''
        Based on the text, here are the extracted astronomical triples for the core entity "Mars":
        ```json
        {
          "relations": {
            "Orbits": ["Sun"],
            "Has Natural Satellite": ["Phobos", "Deimos"],
            "Member Of": "Solar System"
          }
        }
        ```
        '''
        expected = {
            "Orbits": {"Sun"},
            "Has Natural Satellite": {"Phobos", "Deimos"},
            "Member Of": {"Solar System"}
        }
        parsed = _parse_extraction_response(real_response)
        # 将解析结果的值也转为set进行比较
        parsed_set_values = {k: set(v) for k, v in parsed.items()}
        self.assertEqual(parsed_set_values, expected)

    def test_extract_relations_from_single_section_live(self):
        """
        对单个文本片段进行真实的LLM API调用，并验证结果。
        """
        entity_name = "Earth"
        # 使用一个包含明确关系的简单文本
        section_content = "Earth, our home planet, orbits the Sun and has one natural satellite, the Moon. It is the third planet from the Sun."

        # 我们直接调用内部的抽取函数来测试单个请求
        # 注意：这里我们不测试Batch API，而是测试核心的抽取逻辑
        from weaver.core.extraction import _get_extraction_prompt

        prompt = _get_extraction_prompt(entity_name, section_content)

        # 发起真实API调用
        response_text = self.llm_client.make_request(
            model=MODEL_NAME,
            messages=prompt,
            is_json=True  # 确保请求JSON格式
        )

        self.assertIsNotNone(response_text)

        # 解析返回的结果
        parsed_relations = _parse_extraction_response(response_text)

        # 断言：我们期望至少能抽取出关键关系
        self.assertIsNotNone(parsed_relations, "LLM did not return valid relations.")
        self.assertIn("Orbits", parsed_relations)
        self.assertIn("Has Natural Satellite", parsed_relations)

        # 对关系的值进行断言
        self.assertIn("Sun", parsed_relations["Orbits"])
        self.assertIn("Moon", parsed_relations["Has Natural Satellite"])

        logging.info(f"Live extraction result for '{entity_name}': {parsed_relations}")

    def test_extract_relations_from_sections_batch_live(self):
        """
        使用真实的Batch API调用来测试完整的抽取和聚合流程。
        注意：这个测试会比较慢，因为它需要等待批处理作业完成。
        """
        entity_name = "Jupiter"
        sections = [
            {"title": "Overview",
             "content": "Jupiter is the largest planet in the Solar System and it orbits the Sun."},
            {"title": "Moons",
             "content": "The planet is known for its four large Galilean moons: Io, Europa, Ganymede, and Callisto."},
            {"title": "Exploration",
             "content": "The Juno spacecraft is currently in orbit around Jupiter, studying its composition and gravity field."}
        ]

        # 调用被测试的函数，它会执行真实的Batch API调用
        aggregated_relations = extract_relations_from_sections(
            entity_name, sections, self.llm_client, MODEL_NAME
        )

        # 断言：我们期望关键关系被正确聚合
        self.assertIsNotNone(aggregated_relations)
        self.assertIn("Orbits", aggregated_relations)
        self.assertIn("Has Natural Satellite", aggregated_relations)

        self.assertIn("Sun", aggregated_relations["Orbits"])
        # 检查多个卫星是否都被抽取出
        self.assertTrue(
            {"Io", "Europa", "Ganymede", "Callisto"}.issubset(aggregated_relations["Has Natural Satellite"]))

        logging.info(f"Live batch extraction result for '{entity_name}': {aggregated_relations}")


if __name__ == '__main__':
    # 运行测试时，确保已设置环境变量
    # 例如，在Linux/macOS: export DASHSCOPE_API_KEY="your-key"
    # 在Windows: set DASHSCOPE_API_KEY="your-key"
    unittest.main()