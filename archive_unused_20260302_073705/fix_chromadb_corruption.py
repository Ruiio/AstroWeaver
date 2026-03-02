#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复ChromaDB数据库损坏问题的脚本

这个脚本会重置ChromaDB数据库以解决SQLite元数据损坏问题。
错误信息: TypeError: object of type 'int' has no len()

功能:
1. 修复数据库损坏问题
2. 清空数据库内容
"""

import logging
import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from weaver.models.embedding_models import EmbeddingClient
from weaver.storage.vector_db import VectorDBClient
from weaver.utils.config import config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

def clear_chromadb_database():
    """
    清空ChromaDB数据库中的所有数据
    """
    try:
        logger.info("开始清空ChromaDB数据库...")
        
        # 初始化嵌入客户端
        embedding_config = config.get('embedding', {})
        api_url = embedding_config.get('api_url')
        if not api_url:
            logger.error("未找到嵌入API URL配置")
            return False
            
        embedding_client = EmbeddingClient(api_base_url=api_url)
        logger.info("嵌入客户端初始化成功")
        
        # 获取向量数据库路径
        vector_db_path = config.get('vector_db', {}).get('persist_directory', './data/vectordb')
        logger.info(f"向量数据库路径: {vector_db_path}")
        
        # 初始化向量数据库客户端
        vector_db_client = VectorDBClient(
            path=vector_db_path,
            embedding_client=embedding_client
        )
        logger.info("向量数据库客户端初始化成功")
        
        # 获取所有集合
        try:
            collections = vector_db_client.client.list_collections()
            logger.info(f"找到 {len(collections)} 个集合")
            
            # 删除所有集合中的数据
            for collection in collections:
                collection_name = collection.name
                logger.info(f"正在清空集合: {collection_name}")
                try:
                    # 获取集合中的所有文档ID
                    result = collection.get()
                    if result and result['ids']:
                        # 删除所有文档
                        collection.delete(ids=result['ids'])
                        logger.info(f"集合 '{collection_name}' 已清空，删除了 {len(result['ids'])} 个文档")
                    else:
                        logger.info(f"集合 '{collection_name}' 已经是空的")
                except Exception as e:
                    logger.error(f"清空集合 '{collection_name}' 失败: {e}")
            
            logger.info("ChromaDB数据库清空完成!")
            return True
            
        except Exception as e:
            logger.error(f"获取集合列表失败: {e}")
            return False
        
    except Exception as e:
        logger.error(f"清空ChromaDB数据库时发生错误: {e}")
        logger.exception("详细错误信息:")
        return False

def fix_chromadb_corruption():
    """
    修复ChromaDB数据库损坏问题
    """
    try:
        logger.info("开始修复ChromaDB数据库损坏问题...")
        
        # 初始化嵌入客户端
        embedding_config = config.get('embedding', {})
        api_url = embedding_config.get('api_url')
        if not api_url:
            logger.error("未找到嵌入API URL配置")
            return False
            
        embedding_client = EmbeddingClient(api_base_url=api_url)
        logger.info("嵌入客户端初始化成功")
        
        # 获取向量数据库路径
        vector_db_path = config.get('vector_db', {}).get('persist_directory', './data/vectordb')
        logger.info(f"向量数据库路径: {vector_db_path}")
        
        # 初始化向量数据库客户端
        vector_db_client = VectorDBClient(
            path=vector_db_path,
            embedding_client=embedding_client
        )
        logger.info("向量数据库客户端初始化成功")
        
        # 重置数据库以修复损坏
        logger.warning("正在重置ChromaDB数据库以修复损坏...")
        vector_db_client.reset()
        logger.info("ChromaDB数据库重置完成")
        
        # 重新创建必要的集合
        logger.info("重新创建必要的集合...")
        collections_to_create = ['relations', 'entities', 'canonical_relations', 'canonical_entities']
        
        for collection_name in collections_to_create:
            try:
                vector_db_client.create_collection_if_not_exists(collection_name)
                logger.info(f"集合 '{collection_name}' 创建成功")
            except Exception as e:
                logger.error(f"创建集合 '{collection_name}' 失败: {e}")
        
        logger.info("ChromaDB数据库修复完成!")
        logger.info("现在可以重新运行您的管道了。")
        return True
        
    except Exception as e:
        logger.error(f"修复ChromaDB数据库时发生错误: {e}")
        logger.exception("详细错误信息:")
        return False

def main():
    """
    主函数，处理命令行参数
    """
    parser = argparse.ArgumentParser(
        description='ChromaDB数据库维护工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python fix_chromadb_corruption.py --fix     # 修复数据库损坏
  python fix_chromadb_corruption.py --clear   # 清空数据库内容
  python fix_chromadb_corruption.py --help    # 显示帮助信息
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--fix', action='store_true', help='修复ChromaDB数据库损坏问题')
    group.add_argument('--clear', action='store_true', help='清空ChromaDB数据库中的所有数据')
    
    args = parser.parse_args()
    
    if args.fix:
        logger.info("执行数据库修复操作...")
        success = fix_chromadb_corruption()
        if success:
            print("\n✅ ChromaDB数据库修复成功!")
            print("现在可以重新运行您的管道了。")
        else:
            print("\n❌ ChromaDB数据库修复失败!")
            print("请检查日志以获取更多信息。")
            sys.exit(1)
    
    elif args.clear:
        # 确认操作
        print("⚠️  警告: 此操作将清空ChromaDB数据库中的所有数据!")
        print("这将删除所有向量嵌入和相关的元数据。")
        confirm = input("确定要继续吗? (输入 'yes' 确认): ")
        
        if confirm.lower() == 'yes':
            logger.info("执行数据库清空操作...")
            success = clear_chromadb_database()
            if success:
                print("\n✅ ChromaDB数据库清空成功!")
                print("所有数据已被删除。")
            else:
                print("\n❌ ChromaDB数据库清空失败!")
                print("请检查日志以获取更多信息。")
                sys.exit(1)
        else:
            print("操作已取消。")
            sys.exit(0)

if __name__ == "__main__":
    main()