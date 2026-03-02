import json
import logging
import os
import asyncio
from typing import List, Dict, Any, TypedDict, Optional, Tuple, Set

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
    props: Dict[str, Any]


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
    图谱构建员 Agent (v7)，支持冲突检测、增量更新与版本演化。
    """

    def __init__(self, driver: Driver, file_handler: FileHandler, llm_client: LLMClient, config: Optional[Dict[str, Any]] = None):
        """
        初始化 GraphArchitect。

        Args:
            driver: Neo4j Driver 实例。
            file_handler: 文件处理器实例。
            llm_client: LLM 客户端实例，用于链接判断。
            config: 全局配置字典，用于冲突融合参数。
        """
        self.driver = driver
        self.file_handler = file_handler
        self.llm_client = llm_client
        self.config = config or {}
        logger.info("Graph Architect Agent (v7 - Incremental Conflict Resolution) 已初始化。")



    async def _consolidate_knowledge(
            self,
            normalized_triples: List[NormalizedTriple],
            structured_data: Dict[str, Any],
            events: List[Dict[str, Any]] = None
    ) -> Dict[str, EntityData]:
        """
        将三元组、结构化数据和事件整合，并过滤掉重复和自引用的关系。
        """
        entity_map: Dict[str, EntityData] = {}
        seen_triples = set()  # 用于跟踪已处理的唯一三元组

        logger.info(f"开始整合知识，收到 {len(normalized_triples)} 个原始三元组和 {len(events) if events else 0} 个事件。")

        for triple in normalized_triples:
            s, p, o = triple['subject'], triple['predicate'], triple['object']

            # 关键修复 1: 过滤掉自己指向自己的三元组
            if s == o:
                logger.debug(f"过滤掉自引用三元组: ({s}, {p}, {o})")
                continue
                
            # 增强过滤: 过滤特定实体的自引用关系（如Gaia的has_detection_instrument或has_measurement_instrument指向自己）
            if p in ["has_detection_instrument", "has_measurement_instrument"] and s.lower() == o.lower():
                logger.debug(f"过滤掉特定实体的自引用关系: ({s}, {p}, {o})")
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
            rel_props = {}
            if isinstance(triple, dict):
                if 'source_id' in triple:
                    rel_props['source_id'] = triple.get('source_id')
                if 'text_id' in triple:
                    rel_props['text_id'] = triple.get('text_id')
                if 'confidence' in triple:
                    rel_props['confidence'] = triple.get('confidence')
                if 'source_authority' in triple:
                    rel_props['source_authority'] = triple.get('source_authority')
                if 'authority_score' in triple:
                    rel_props['authority_score'] = triple.get('authority_score')
                if 'timestamp' in triple:
                    rel_props['timestamp'] = triple.get('timestamp')
                if 'confidence_score' in triple:
                    rel_props['confidence_score'] = triple.get('confidence_score')
                attrs = triple.get('attributes')
                if isinstance(attrs, dict):
                    for ak, av in attrs.items():
                        rel_props[ak] = av
            entity_map[s]['relationships'].append({'predicate': p, 'target': o, 'props': rel_props})

        logger.info(f"过滤后，处理了 {len(seen_triples)} 个唯一且有效的关系。")

        canonical_entity_names = list(entity_map.keys())
        logger.info(f"从三元组中加载了 {len(canonical_entity_names)} 个规范化实体。")

        if not structured_data or not canonical_entity_names:
            return entity_map

        for data_key, attributes in structured_data.items():
            # 使用字符完全匹配，直接检查data_key是否与三元组中的实体匹配
            linked_entity = None
            
            # 检查data_key是否与任何canonical entity完全匹配
            if data_key in canonical_entity_names:
                linked_entity = data_key
            else:
                # 检查data_key的第一部分（去掉下划线后缀）是否匹配
                entity_name = data_key.split('_')[0]
                if entity_name in canonical_entity_names:
                    linked_entity = entity_name
            
            if linked_entity:
                flattened_attributes = self._flatten_attributes(attributes)
                logger.info(f"数据整合: 将 '{data_key}' 的扁平化属性附加到实体 '{linked_entity}'.")
                entity_map[linked_entity]['attributes'].update(flattened_attributes)
            else:
                logger.debug(f"数据块 '{data_key}' 未找到匹配的实体，已忽略。")

        # 处理事件数据
        if events:
            logger.info(f"处理 {len(events)} 个事件数据")
            for event_index, event in enumerate(events):
                event_type = event.get('event_type', '')
                anchor_entity = event.get('anchor_entity', '')
                arguments = event.get('arguments', [])
                confidence = event.get('confidence', 0.0)
                text_id = event.get('text_id', '')
                source_id = event.get('source_id', '')
                
                logger.debug(f"处理第 {event_index+1} 个事件: 类型={event_type}, 锚定实体={anchor_entity}, 置信度={confidence}, 文本ID={text_id}, 源ID={source_id}")
                
                # 跳过没有锚定实体或事件类型的事件
                if not anchor_entity or not event_type:
                    logger.warning(f"跳过无效事件: 缺少锚定实体或事件类型, 事件索引={event_index}")
                    continue
                
                # 确保锚定实体存在于实体映射中
                if anchor_entity not in entity_map:
                    entity_map[anchor_entity] = {'id': anchor_entity, 'attributes': {}, 'relationships': []}
                
                # 将事件参数转换为扁平化键值对
                event_data = {}
                
                # 添加事件类型和源ID
                event_data['event_type'] = event_type
                event_data['source_id'] = source_id
                
                # 1) 先把所有参数作为键值对写入事件数据（保留原始值类型）
                for arg in arguments:
                    role = arg.get('role', '')
                    if not role:
                        continue
                    event_data[role] = arg.get('value', None)
                
                # 将事件属性落到锚定实体上
                # 计算该类型事件的序号（为属性键生成不冲突的后缀）
                event_count = 1
                while True:
                    event_attr_key = f"event_{event_type.lower().replace(' ', '_')}_{event_count}"
                    if event_attr_key not in entity_map[anchor_entity]['attributes']:
                        break
                    event_count += 1
                entity_map[anchor_entity]['attributes'][event_attr_key] = event_data
                logger.info(f"添加事件属性: {anchor_entity}.{event_attr_key} = {event_type}事件(参数数量:{len(arguments)})")
                logger.debug(f"事件数据样例: {event_attr_key} = {event_data}")
                
                # 2) 再遍历参数，按需创建关系（仅对字符串值）
                for arg in arguments:
                    role = arg.get('role', '')
                    value = arg.get('value', None)
                    
                    # 跳过空值
                    if not role or value is None or (isinstance(value, str) and not value.strip()):
                        continue
                    
                    # 仅当值为字符串时才尝试将其作为实体创建关系；
                    # 对于数值/布尔/对象等非字符串类型，仅作为事件属性保留，不创建关系
                    if not isinstance(value, str):
                        logger.debug(f"跳过非字符串参数的关系创建: role={role}, value_type={type(value)}")
                        continue
                    
                    value_str = value.strip()
                    if not value_str:
                        continue
                    
                    # 如果参数值看起来像实体，创建关系（简单启发：单词数不超过5个）
                    if len(value_str.split()) <= 5:
                        # 确保目标实体存在
                        if value_str not in entity_map:
                            entity_map[value_str] = {'id': value_str, 'attributes': {}, 'relationships': []}
                        
                        # 创建从锚定实体到参数实体的关系
                        relation_predicate = f"has_{event_type.lower().replace(' ', '_')}_{role}"
                        
                        # 检查是否为自引用关系（锚定实体指向自己）
                        if anchor_entity.lower() == value_str.lower():
                            logger.debug(f"过滤掉事件中的自引用关系: ({anchor_entity}, {relation_predicate}, {value_str})")
                            continue
                        
                        # 检查特定类型的自引用关系
                        if role in ["instrument"] and anchor_entity.lower() == value_str.lower():
                            logger.debug(f"过滤掉事件中的特定类型自引用关系: ({anchor_entity}, {relation_predicate}, {value_str})")
                            continue
                        
                        # 检查关系是否已存在
                        relation_exists = False
                        for rel in entity_map[anchor_entity]['relationships']:
                            if rel['predicate'] == relation_predicate and rel['target'] == value_str:
                                relation_exists = True
                                break
                        
                        if not relation_exists:
                            logger.debug(f"添加事件关系: ({anchor_entity}, {relation_predicate}, {value_str})")
                            entity_map[anchor_entity]['relationships'].append({
                                'predicate': relation_predicate,
                                'target': value_str
                            })
                    else:
                        # 如果参数值不像实体名称，不需要额外处理
                        # 因为我们已经将参数作为键值对添加到事件数据中
                        logger.debug(f"参数 {role}={value_str} 已作为键值对添加到事件数据中")
        
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
            events: List[Dict[str, Any]] = None,
            output_filename: str = "knowledge_graph.json"
    ) -> Dict[str, EntityData]:
        """Agent的主入口点。这是一个异步方法。"""
        if events:
            logger.info(f"收到 {len(events)} 个事件数据用于图谱构建")
        else:
            logger.info("没有收到事件数据")
            events = []
        
        consolidated_graph = await self._consolidate_knowledge(normalized_triples, structured_data, events)
        self._persist_to_graph_db(consolidated_graph)

        try:
            self.file_handler.save_json(output_filename, consolidated_graph)
        except Exception as e:
            logger.error(f"调用FileHandler保存JSON失败: {e}", exc_info=True)

        return consolidated_graph

    def _persist_to_graph_db(self, entity_map: Dict[str, EntityData]):
        """将整合后的实体和关系数据持久化到Neo4j数据库（V1：支持冲突检测与版本演化）。"""
        logger.info("开始将数据持久化到Neo4j数据库 (V1 冲突融合版)...")

        graph_cfg = self.config.get('graph_conflict', {}) if isinstance(self.config, dict) else {}
        coexist_score_delta = float(graph_cfg.get('coexist_score_delta', 0.08))
        numeric_tolerance_ratio = float(graph_cfg.get('numeric_tolerance_ratio', 0.03))

        def _extract_num(v: Any) -> Optional[float]:
            import re
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return float(v)
            m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(v))
            if not m:
                return None
            try:
                return float(m.group(0))
            except Exception:
                return None

        with self.driver.session(database="neo4j") as session:
            # 1) 先批量写入/更新节点
            nodes_to_create = []
            for eid, data in entity_map.items():
                processed_props = {}
                for key, value in data["attributes"].items():
                    if key.startswith("event_") and isinstance(value, dict):
                        processed_props[key] = json.dumps(value)
                    else:
                        processed_props[key] = value
                nodes_to_create.append({"id": eid, "props": processed_props})

            session.run(
                """
                UNWIND $nodes as node_data
                MERGE (n:Entity {id: node_data.id})
                SET n += node_data.props, n.name = node_data.id
                """,
                nodes=nodes_to_create
            )

            # 2) 逐条关系执行冲突检测与版本演化
            for eid, data in entity_map.items():
                for rel in data['relationships']:
                    source_id = eid
                    target_id = rel['target']
                    predicate = rel['predicate']
                    props = dict(rel.get('props', {}) or {})

                    now_ts = float(props.get('timestamp', 0.0) or 0.0)
                    new_cs = float(props.get('confidence_score', 0.0) or 0.0)

                    existing = session.run(
                        """
                        MATCH (a:Entity {id: $source_id})-[r:RELATED_TO {type: $predicate, status: 'Current'}]->(b:Entity)
                        RETURN b.id AS target_id,
                               coalesce(r.confidence_score, 0.0) AS cs,
                               coalesce(r.timestamp, 0.0) AS ts
                        """,
                        source_id=source_id,
                        predicate=predicate
                    ).data()

                    same_target_exists = any(row.get('target_id') == target_id for row in existing)
                    conflict_rows = [row for row in existing if row.get('target_id') != target_id]

                    status = 'Current'
                    is_contested = False
                    retire_old = False

                    if conflict_rows:
                        best_old = max(conflict_rows, key=lambda x: (float(x.get('cs', 0.0) or 0.0), float(x.get('ts', 0.0) or 0.0)))
                        old_cs = float(best_old.get('cs', 0.0) or 0.0)
                        old_ts = float(best_old.get('ts', 0.0) or 0.0)

                        score_delta = abs(new_cs - old_cs)
                        new_num = _extract_num(target_id)
                        old_num = _extract_num(best_old.get('target_id'))
                        ratio = None
                        if new_num is not None and old_num is not None:
                            denom = abs(old_num) if abs(old_num) > 1e-12 else 1.0
                            ratio = abs(new_num - old_num) / denom

                        if (new_cs > old_cs) and (now_ts >= old_ts):
                            # 新值更可靠且更新：旧值转历史，新值当前
                            retire_old = True
                            status = 'Current'
                        elif score_delta <= coexist_score_delta and (ratio is None or ratio <= numeric_tolerance_ratio):
                            # 分数接近且在误差范围内：并存争议观点
                            status = 'Current'
                            is_contested = True
                        else:
                            # 新值不足以替代旧值：作为历史观察记录
                            status = 'History'

                    if retire_old:
                        session.run(
                            """
                            MATCH (a:Entity {id: $source_id})-[r:RELATED_TO {type: $predicate, status: 'Current'}]->(:Entity)
                            SET r.status = 'History',
                                r.valid_to = $now_ts
                            """,
                            source_id=source_id,
                            predicate=predicate,
                            now_ts=now_ts
                        )

                    if is_contested:
                        session.run(
                            """
                            MATCH (a:Entity {id: $source_id})-[r:RELATED_TO {type: $predicate, status: 'Current'}]->(:Entity)
                            SET r.is_contested = true
                            """,
                            source_id=source_id,
                            predicate=predicate,
                        )

                    props['status'] = status
                    props['is_contested'] = bool(is_contested)
                    if status == 'Current' and 'valid_from' not in props:
                        props['valid_from'] = now_ts
                    if status == 'History' and 'valid_to' not in props:
                        props['valid_to'] = now_ts
                    props['target_id'] = target_id

                    session.run(
                        """
                        MATCH (a:Entity {id: $source_id})
                        MATCH (b:Entity {id: $target_id})
                        MERGE (a)-[r:RELATED_TO {type: $predicate, target_id: $target_id}]->(b)
                        SET r += $props
                        """,
                        source_id=source_id,
                        target_id=target_id,
                        predicate=predicate,
                        props=props
                    )

        logger.info(f"数据成功持久化到Neo4j。共处理 {len(entity_map)} 个实体。")

    def query_subgraph(self, entity_id: str, depth: int = 1) -> List[Dict]:
        """查询给定实体的子图。"""
        logger.info(f"查询实体 '{entity_id}' 的 {depth} 度子图...")
        query = f"MATCH path = (n:Entity {{id: $id}})-[*1..{depth}]-(m) RETURN path"
        with self.driver.session(database="neo4j") as session:
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
