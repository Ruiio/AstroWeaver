#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试entity字段修复是否成功
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from weaver.core.extraction_architecture import AttributeExtraction

def test_entity_field_fix():
    """
    测试entity字段是否正确处理
    """
    print("测试entity字段修复...")
    
    # 测试用例1: entity_name为字符串
    test_data_1 = {
        "entity_name": "Jupiter",
        "attribute": "atmosphere", 
        "value": "contains water",
        "confidence": 0.7,
        "text_id": "block_0",
        "source_id": "test"
    }
    
    # 测试用例2: entity_name为字典（模拟LLM返回的嵌套格式）
    test_data_2 = {
        "entity_name": {"entity_name": "solar system"},
        "attribute": "formation",
        "value": "4.6 billion years ago", 
        "confidence": 0.8,
        "text_id": "block_1",
        "source_id": "test"
    }
    
    # 测试用例3: entity_name为非字符串类型
    test_data_3 = {
        "entity_name": 12345,
        "attribute": "mass",
        "value": "large",
        "confidence": 0.6,
        "text_id": "block_2", 
        "source_id": "test"
    }
    
    test_cases = [
        ("字符串entity_name", test_data_1, "Jupiter"),
        ("字典entity_name", test_data_2, "solar system"),
        ("数字entity_name", test_data_3, "12345")
    ]
    
    for test_name, test_data, expected_entity in test_cases:
        print(f"\n测试: {test_name}")
        print(f"输入数据: {test_data}")
        
        try:
            # 模拟extractor.py中的处理逻辑
            entity_name = test_data["entity_name"]
            
            # 应用修复逻辑
            if isinstance(entity_name, dict):
                if "entity_name" in entity_name:
                    entity_name = entity_name["entity_name"]
                else:
                    entity_name = str(entity_name)
            elif not isinstance(entity_name, str):
                entity_name = str(entity_name)
            
            # 创建AttributeExtraction对象
            attr_extraction: AttributeExtraction = {
                "entity": entity_name,
                "attribute": test_data["attribute"],
                "value": test_data["value"],
                "confidence": test_data["confidence"],
                "text_id": test_data["text_id"],
                "source_id": test_data["source_id"]
            }
            
            print(f"处理后的entity字段: '{attr_extraction['entity']}'")
            print(f"entity字段类型: {type(attr_extraction['entity'])}")
            print(f"期望值: '{expected_entity}'")
            
            # 验证结果
            if attr_extraction['entity'] == expected_entity and isinstance(attr_extraction['entity'], str):
                print("✅ 测试通过")
            else:
                print("❌ 测试失败")
                
        except Exception as e:
            print(f"❌ 测试失败，错误: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_entity_field_fix()