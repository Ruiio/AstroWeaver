# Windows asyncio修复
import sys
import os
if sys.platform == 'win32':
    import asyncio
    # 使用ProactorEventLoop替代默认的SelectorEventLoop
    if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONASYNCIODEBUG'] = '0'
else:
    import asyncio

import json
import logging
import argparse
import pandas as pd
from pathlib import Path

# --- 文件处理和分块库 ---
# pypdf is no longer needed here, as the Orchestrator handles PDF processing internally.
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 导入所有必要的Agent和客户端
from neo4j import GraphDatabase
from weaver.agents.data_scout import DataScout, Orchestrator
from weaver.agents.extractor import InformationExtractor
from weaver.agents.auditor_enhanced import EnhancedKnowledgeAuditor
from weaver.agents.constructor import GraphArchitect
from weaver.models.embedding_models import EmbeddingClient
from weaver.models.llm_models import LLMClient
from weaver.storage.file_handler import FileHandler
from weaver.storage.vector_db import VectorDBClient
from weaver.utils.config import load_config
from data_sources.mineru_client import MinerUClient
from data_sources.wikipedia_client import WikipediaClient

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==========================================================================
# 🚀 更新：文件内容提取模块 (现在只处理DOCX)
# ==========================================================================
def get_text_from_docx(file_path_str: str, config: dict) -> list[str]:
    """
    从DOCX文件中提取文本，并使用RecursiveCharacterTextSplitter进行分块。
    """
    file_path = Path(file_path_str)
    if not file_path.exists():
        logger.error(f"文件不存在: {file_path}")
        return []

    try:
        if file_path.suffix.lower() not in ['.doc', '.docx']:
            logger.warning(f"不支持的文件类型: {file_path.suffix}。此函数仅支持DOCX。")
            return []

        doc = docx.Document(file_path)
        raw_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

        if not raw_text.strip():
            logger.warning(f"文件 {file_path.name} 中未提取到任何有效文本内容。")
            return []

        logger.info("使用 LangChain RecursiveCharacterTextSplitter 进行文本分块...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['chunk']['chunk_size'],
            chunk_overlap=config['chunk']['chunk_overlap'],
            length_function=len,
        )
        chunks = text_splitter.split_text(raw_text)

        logger.info(f"从文件 {file_path.name} 中成功提取并分割成 {len(chunks)} 个文本块。")
        return chunks

    except KeyError as e:
        logger.error(f"配置错误: 无法在配置文件中找到 chunk 配置: {e}。请确保 config 文件包含 'chunk' 部分。")
        return []
    except Exception as e:
        logger.error(f"提取或分割文件 {file_path_str} 内容时出错: {e}", exc_info=True)
        return []


# ==========================================================================
# 🚀 重构：核心处理流程，区分Orchestrator处理和手动处理
# ==========================================================================
async def process_item(item_type: str, item_value: str, clients: dict, config: dict):
    """
    处理单个输入项。利用Orchestrator处理topic和pdf，手动处理doc。
    """
    item_type = item_type.lower().strip()
    logger.info("-" * 50)
    logger.info(f"🚀 开始处理新项目: Type='{item_type}', Value='{item_value}'")
    logger.info("-" * 50)

    structured_data = {}
    text_blocks = []

    if item_type in ['topic', 'pdf']:
        # 对于PDF，使用文件名作为基础名称
        processing_target_name = Path(item_value).stem if item_type == 'pdf' else item_value
    else:
        processing_target_name = Path(item_value).stem

    try:
        # --- 数据获取阶段 ---
        if item_type in ['topic', 'pdf']:
            logger.info(f"交由 Orchestrator 处理输入: '{item_value}'")
            data_scout = DataScout(wikiClient=clients['wiki'], minerUClient=clients['mineru'])
            orchestrator = Orchestrator(clients['llm'], data_scout)
            # Orchestrator.run可以智能处理主题词和PDF路径
            scout_result = orchestrator.run(item_value)
            if not scout_result:
                logger.error("Orchestrator 未能返回任何结果，跳过此项目。")
                return
            structured_data = scout_result.get('structured_data', {})
            text_blocks = scout_result.get('text_chunks', [])

        elif item_type in ['doc', 'docx']:
            logger.info(f"从DOCX文件 '{item_value}' 中手动提取文本。")
            # DOCX文件没有结构化数据，只有文本块
            text_blocks = get_text_from_docx(item_value, config)
            if not text_blocks:
                logger.error(f"未能从文件 '{item_value}' 中提取任何文本块，跳过此项目。")
                return
        else:
            logger.warning(f"未知的输入类型: '{item_type}'。将跳过此项目。")
            return

        logger.info(f"数据获取完成。获取了 {len(text_blocks)} 个文本块和 {len(structured_data)} 个结构化数据条目。")

        # --- 后续的信息提取、审核、构建流程保持不变 ---

        logger.info("=" * 30 + "\n🚀 Step 3: Extracting Information\n" + "=" * 30)
        extractor = InformationExtractor(clients['llm'], config['llm']['extraction_model'],
                                         topic=processing_target_name)
        raw_triples_with_confidence = await extractor.extract_from_text_blocks(text_blocks)
        logger.info(f"信息提取完成。共抽取出 {len(raw_triples_with_confidence)} 个原始三元组。")

        logger.info("=" * 30 + "\n🚀 Step 4: Auditing & Normalizing Knowledge\n" + "=" * 30)
        auditor = KnowledgeAuditor(clients['llm'], clients['vector_db'], config)
        high_confidence_normalized, low_confidence_pending = await auditor.audit_and_normalize_triples(
            raw_triples_with_confidence)
        logger.info(
            f"知识审核完成。{len(high_confidence_normalized)} 个三元组通过审核，{len(low_confidence_pending)} 个待人工审核。")

        if low_confidence_pending:
            output_file = config['paths'][
                              'output_dir'] + f'/{processing_target_name.replace(" ", "_")}_low_confidence.jsonl'
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in low_confidence_pending:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"待人工审核的三元组已写入: {output_file}")

        if not high_confidence_normalized and not structured_data:
            logger.warning("没有高置信度的三元组且没有结构化数据，跳过图谱构建步骤。")
        else:
            logger.info("=" * 30 + "\n🚀 Step 5: Constructing & Persisting Graph\n" + "=" * 30)
            architect = GraphArchitect(
                driver=clients['graph_db'],
                file_handler=clients['file_handler'],
                llm_client=clients['llm']
            )
            architect.config = config
            await architect.build_and_persist(
                normalized_triples=high_confidence_normalized,
                structured_data=structured_data,
                output_filename=f"{processing_target_name.replace(' ', '_')}_graph.json"
            )
            logger.info("图谱构建与持久化完成！")

        print("\n" + "=" * 40)
        print(f"✅ Item '{processing_target_name}' processed successfully!")
        print("=" * 40 + "\n")

    except Exception as e:
        logger.error(f"处理项目 '{processing_target_name}' 时发生严重错误: {e}", exc_info=True)


async def main(input_filepath: str):
    clients = {}
    driver = None
    try:
        logger.info("=" * 30 + "\n🚀 Step 1: Initializing Clients & Config\n" + "=" * 30)
        config = load_config()

        # ... (客户端初始化代码与之前完全相同，此处省略以保持简洁) ...
        clients = {
            'llm': LLMClient(api_key=config['api_keys']['llm'], base_url=config['llm']['base_url']),
            'embedding': EmbeddingClient(api_base_url=config['embedding']['api_url']),
            'file_handler': FileHandler(output_dir=config['paths']['output_dir']),
            'wiki': WikipediaClient(),
            'mineru': MinerUClient()
        }
        clients['vector_db'] = VectorDBClient(
            path=config['paths']['vector_db_path'],
            embedding_client=clients['embedding']
        )
        driver = GraphDatabase.driver(
            config['neo4j']['uri'],
            auth=(config['neo4j']['user'], config['neo4j']['password'])
        )
        driver.verify_connectivity()
        clients['graph_db'] = driver
        logger.info("所有客户端初始化成功，已连接到Neo4j数据库。")

        logger.info(f"正在从 {input_filepath} 读取输入列表...")
        try:
            df = pd.read_excel(input_filepath)
            df.columns = [col.lower() for col in df.columns]

            if 'type' not in df.columns or 'value' not in df.columns:
                logger.error("XLSX 文件必须包含 'type' 和 'value' 两列。请检查文件。")
                return

        except FileNotFoundError:
            logger.error(f"输入文件未找到: {input_filepath}")
            return
        except Exception as e:
            logger.error(f"读取 XLSX 文件时出错: {e}")
            return

        logger.info(f"找到 {len(df)} 个待处理项目。开始批量处理...")

        for index, row in df.iterrows():
            item_type = str(row['type']).lower().strip()  # 确保类型是小写和无空格
            item_value = str(row['value'])

            if not item_type or not item_value or pd.isna(row['type']) or pd.isna(row['value']):
                logger.warning(f"跳过第 {index + 2} 行，因为 'type' 或 'value' 为空。")
                continue

            await process_item(item_type, item_value, clients, config)

        logger.info("所有项目处理完毕！")

    except Exception as e:
        logger.error(f"工作流主程序发生严重错误: {e}", exc_info=True)
    finally:
        if driver:
            driver.close()
            logger.info("Neo4j数据库连接已关闭。")


if __name__ == "__main__":
    input_file=r"C:\Users\Administrator\Desktop\astroWeaver\data\input\astroEntity.xlsx"
    asyncio.run(main(input_file))