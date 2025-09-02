import json
import logging
import os
import asyncio
from typing import List, Dict, Any, TypedDict, Optional

# 导入所有必需的模块
from neo4j import GraphDatabase, Driver

# 假设这些模块存在于您的项目中，并且可以被正确导入
# 如果您的项目结构不同，请相应地调整这些导入路径
from weaver.storage.file_handler import FileHandler
from weaver.models.llm_models import LLMClient
from weaver.utils.config import load_config


# ==============================================================================
# 1. 类型定义 (Type Definitions)
# ==============================================================================
class NormalizedTriple(TypedDict):
    """从知识审核员处接收的、已规范化的三元组结构。"""
    subject: str
    predicate: str
    object: str


class Relationship(TypedDict):
    """用于内部表示实体关系的结构。"""
    predicate: str
    target: str


class EntityData(TypedDict):
    """整合后的、以实体为中心的完整数据结构。"""
    id: str
    attributes: Dict[str, Any]
    relationships: List[Relationship]


# ==============================================================================
# 2. 图谱构建员 (Graph Architect Agent) - v6 (Data Integrity Fix)
# ==============================================================================
logger = logging.getLogger(__name__)


class GraphArchitect:
    """
    图谱构建员 Agent (v6)，增加了对重复和自引用三元组的过滤。
    """

    def __init__(self, driver: Driver, file_handler: FileHandler, llm_client: LLMClient):
        """
        初始化 GraphArchitect。

        Args:
            driver: Neo4j Driver 实例。
            file_handler: 文件处理器实例。
            llm_client: LLM 客户端实例，用于链接判断。
        """
        self.driver = driver
        self.file_handler = file_handler
        self.llm_client = llm_client
        self.config = {}  # 用于存储配置，例如LLM模型名称
        logger.info("Graph Architect Agent (v6 - Data Integrity Fix) 已初始化。")

    def _get_linking_prompt(self, data_block_summary: Dict, canonical_entities: List[str]) -> List[Dict[str, str]]:
        """为LLM生成数据块到实体的链接Prompt。"""
        system_prompt = "You are an expert astronomer and data analyst. Your task is to determine which canonical entity a given block of data refers to. Respond ONLY with the requested JSON object."

        user_prompt = f"""
        **Task:**
        Analyze the "Data Block Summary" below. It contains data extracted from a source like Simbad or Wikipedia. Your goal is to identify which of the "Canonical Entity Names" this data block belongs to.

        **Data Block Summary:**
        {json.dumps(data_block_summary, indent=2)}

        **Canonical Entity Names (already normalized):**
        {json.dumps(canonical_entities, indent=2)}

        **Instructions:**
        1.  Carefully compare the names, identifiers, and type information in the summary with the list of canonical entities.
        2.  The data block might use a common name or alias (e.g., "M7", "Ptolemy's Cluster"), while the canonical list uses a formal name (e.g., "Messier7").
        3.  If you find a clear match, provide the exact name of the matching canonical entity from the list.
        4.  If the data block does not seem to correspond to any of the provided canonical entities, respond with "None".

        **JSON Output Structure:**
        ```json
        {{
          "linked_entity": "The matching canonical entity name from the list OR None"
        }}
        ```
        """
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    async def _consolidate_knowledge(
            self,
            normalized_triples: List[NormalizedTriple],
            structured_data: Dict[str, Any]
    ) -> Dict[str, EntityData]:
        """
        将三元组和结构化数据整合，并过滤掉重复和自引用的关系。
        """
        entity_map: Dict[str, EntityData] = {}
        seen_triples = set()  # 用于跟踪已处理的唯一三元组

        logger.info(f"开始整合知识，收到 {len(normalized_triples)} 个原始三元组。")

        for triple in normalized_triples:
            s, p, o = triple['subject'], triple['predicate'], triple['object']

            # 关键修复 1: 过滤掉自己指向自己的三元组
            if s == o:
                logger.debug(f"过滤掉自引用三元组: ({s}, {p}, {o})")
                continue

            # 关键修复 2: 过滤掉重复的三元组
            # 我们将三元组转换为一个可哈希的元组来进行比较
            triple_repr = (s, p, o)
            if triple_repr in seen_triples:
                logger.debug(f"过滤掉重复三元组: {triple_repr}")
                continue

            # 如果三元组有效且唯一，则处理并记录
            seen_triples.add(triple_repr)

            if s not in entity_map: entity_map[s] = {'id': s, 'attributes': {}, 'relationships': []}
            if o not in entity_map: entity_map[o] = {'id': o, 'attributes': {}, 'relationships': []}
            entity_map[s]['relationships'].append({'predicate': p, 'target': o})

        logger.info(f"过滤后，处理了 {len(seen_triples)} 个唯一且有效的关系。")

        canonical_entity_names = list(entity_map.keys())
        logger.info(f"从三元组中加载了 {len(canonical_entity_names)} 个规范化实体。")

        if not structured_data or not canonical_entity_names:
            return entity_map

        for data_key, attributes in structured_data.items():
            summary = {
                "source_key": data_key,
                "name_in_key": data_key.split('_')[0],
                "type": attributes.get('type'),
                "identifiers": attributes.get('identifiers', [])[:5],
                "other_designations": attributes.get('Other designations', "")
            }
            prompt = self._get_linking_prompt(summary, canonical_entity_names)

            try:
                # 注意：这里假设您的LLMClient有一个异步方法
                response_str = await self.llm_client.make_request_async(
                    model=self.config.get('llm', {}).get('zhipu', {}).get('base_model', 'GLM-4-Flash-250414'),
                    messages=prompt
                )
                # 移除可能的Markdown代码块标记
                response_json = json.loads(response_str.replace("```json", "").replace("```", "").strip())
                linked_entity = response_json.get("linked_entity")

                if linked_entity and linked_entity in entity_map:
                    flattened_attributes = self._flatten_attributes(attributes)
                    logger.info(f"数据整合: 将 '{data_key}' 的扁平化属性附加到实体 '{linked_entity}'.")
                    entity_map[linked_entity]['attributes'].update(flattened_attributes)
                else:
                    logger.warning(
                        f"LLM未能将数据块 '{data_key}' 链接到任何已知实体 (返回: {linked_entity})。此数据块被忽略。")

            except Exception as e:
                logger.error(f"在链接数据块 '{data_key}' 时发生LLM错误: {e}", exc_info=True)

        logger.info("知识整合完成。")
        return entity_map

    def _flatten_attributes(self, data: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        递归地将嵌套字典扁平化为单层字典。
        """
        items = {}
        for k, v in data.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_attributes(v, new_key, sep=sep))
            elif isinstance(v, list):
                items[new_key] = json.dumps(v)
            else:
                items[new_key] = v
        return items

    async def build_and_persist(
            self,
            normalized_triples: List[NormalizedTriple],
            structured_data: Dict[str, Any],
            output_filename: str = "knowledge_graph.json"
    ) -> Dict[str, EntityData]:
        """Agent的主入口点。这是一个异步方法。"""
        consolidated_graph = await self._consolidate_knowledge(normalized_triples, structured_data)
        self._persist_to_graph_db(consolidated_graph)

        try:
            self.file_handler.save_json(output_filename, consolidated_graph)
        except Exception as e:
            logger.error(f"调用FileHandler保存JSON失败: {e}", exc_info=True)

        return consolidated_graph

    def _persist_to_graph_db(self, entity_map: Dict[str, EntityData]):
        """将整合后的实体和关系数据持久化到Neo4j数据库。"""
        logger.info("开始将数据持久化到Neo4j数据库 (优化版)...")
        with self.driver.session(database="astrokg") as session:
            session.execute_write(self._create_nodes_and_relationships_tx, entity_map)
        logger.info(f"数据成功持久化到Neo4j。共处理 {len(entity_map)} 个实体。")

    @staticmethod
    def _create_nodes_and_relationships_tx(tx, entity_map: Dict[str, EntityData]):
        """在一个事务中创建所有节点和关系。"""
        nodes_to_create = [{"id": eid, "props": data["attributes"]} for eid, data in entity_map.items()]
        node_query = """
        UNWIND $nodes as node_data
        MERGE (n:Entity {id: node_data.id})
        SET n += node_data.props, n.name = node_data.id
        """
        tx.run(node_query, nodes=nodes_to_create)

        rels_to_create = []
        for eid, data in entity_map.items():
            for rel in data['relationships']:
                rels_to_create.append({
                    "source_id": eid,
                    "target_id": rel['target'],
                    "predicate": rel['predicate']
                })

        rel_query = """
        UNWIND $rels as rel_data
        MATCH (a:Entity {id: rel_data.source_id})
        MATCH (b:Entity {id: rel_data.target_id})
        MERGE (a)-[r:RELATED_TO {type: rel_data.predicate}]->(b)
        """
        tx.run(rel_query, rels=rels_to_create)

    def query_subgraph(self, entity_id: str, depth: int = 1) -> List[Dict]:
        """查询给定实体的子图。"""
        logger.info(f"查询实体 '{entity_id}' 的 {depth} 度子图...")
        query = f"MATCH path = (n:Entity {{id: $id}})-[*1..{depth}]-(m) RETURN path"
        with self.driver.session(database="astrokg") as session:
            result = session.run(query, id=entity_id)
            # 将路径对象转换为更易于处理的字典
            return [record["path"].__dict__ for record in result]


# ==============================================================================
# 3. 独立运行和测试 (Standalone Execution & Test)
# ==============================================================================
async def main_test():
    """用于测试GraphArchitect功能的独立异步函数。"""
    # 设置日志级别为DEBUG以查看详细的过滤信息
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    driver = None
    try:
        # --- 1. 加载配置并初始化所有客户端 ---
        # 确保您的项目中有 `weaver/utils/config.py` 和对应的配置文件
        config = load_config()

        # Neo4j
        NEO4J_URI = config['neo4j']['uri']
        NEO4J_USER = config['neo4j']['user']
        NEO4J_PASSWORD = config['neo4j']['password']

        # LLM
        llm_client = LLMClient(api_key=config['api_keys']['llm'], base_url=config['llm'].get('base_url'))

        # File Handler
        file_handler = FileHandler(output_dir=config['paths']['output_dir'])

        # --- 2. 初始化组件 ---
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        logger.info("成功连接到Neo4j数据库。")

        architect = GraphArchitect(
            driver=driver,
            file_handler=file_handler,
            llm_client=llm_client
        )
        architect.config = config  # 手动注入config

        # --- 3. 准备包含重复和自引用三元组的示例数据 ---
        mock_normalized_triples: List[NormalizedTriple] = [
            {'subject': 'Messier7', 'predicate': 'hasType', 'object': 'OpenCluster'},
            {'subject': 'Messier7', 'predicate': 'locatedIn', 'object': 'ScorpiusConstellation'},
            # 添加重复的三元组
            {'subject': 'Messier7', 'predicate': 'hasType', 'object': 'OpenCluster'},
            # 添加另一个重复的三元组
            {'subject': 'Messier7', 'predicate': 'locatedIn', 'object': 'ScorpiusConstellation'},
            # 添加自引用的三元组
            {'subject': 'ScorpiusConstellation', 'predicate': 'is', 'object': 'ScorpiusConstellation'},
        ]

        mock_structured_data = {
            'Messier 7_simbad': {'name': 'NGC 6475', 'type': 'OpC', 'identifiers': ['M 7']},
            'M7_info_box': {'Other designations': 'Ptolemy Cluster, M 7, NGC 6475'},
            'SomeOtherStar_simbad': {'name': 'Alpha Centauri', 'type': 'Star'}  # 这个应该被忽略
        }

        # --- 4. 运行构建和持久化流程 ---
        print("\n" + "=" * 30 + "\n🚀 EXECUTING GRAPH BUILD & PERSIST (V6)\n" + "=" * 30)
        final_graph = await architect.build_and_persist(
            normalized_triples=mock_normalized_triples,
            structured_data=mock_structured_data,
            output_filename="messier7_graph_v6.json"
        )

        print("\n✅ 图谱构建和持久化成功！")
        print("\n--- 最终生成的图谱结构 ---")
        print(json.dumps(final_graph, indent=2))

        # --- 5. 验证结果 ---
        print("\n--- 验证 ---")
        messier7_node = final_graph.get("Messier7", {})
        relationships = messier7_node.get("relationships", [])

        assert len(relationships) == 2, f"错误：期望2个关系，但找到了 {len(relationships)}"
        print("✅ 验证成功：关系数量正确，重复项已被移除。")

        scorpius_node = final_graph.get("ScorpiusConstellation", {})
        assert len(scorpius_node.get("relationships", [])) == 0, "错误：自引用关系未被移除。"
        print("✅ 验证成功：自引用关系已被正确过滤。")

        # --- 6. 运行真实查询 ---
        canonical_query_name = "Messier7"
        print(f"\n" + "=" * 30 + f"\n🚀 EXECUTING SUBGRAPH QUERY FOR '{canonical_query_name}'\n" + "=" * 30)
        subgraph_result = architect.query_subgraph(entity_id=canonical_query_name, depth=1)

        print("\n✅ 子图查询成功！")
        print(f"--- 查询到 {len(subgraph_result)} 条路径 ---")
        for i, path in enumerate(subgraph_result):
            node_ids = [node['id'] for node in path['_nodes']]
            print(f"Path {i + 1}: {' -> '.join(node_ids)}")

    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}", exc_info=True)
    finally:
        if driver:
            driver.close()
            logger.info("Neo4j数据库连接已关闭。")


if __name__ == '__main__':
    asyncio.run(main_test())