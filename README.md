# AstroWeaver

AstroWeaver 是一个面向天文领域的知识图谱构建流水线：从多源文本中抽取实体关系/事件，经过审核与规范化后写入 Neo4j。

---

## 1. 核心能力

- 多源数据获取（Wikipedia / Web / PDF / 文本文件）
- LLM 驱动的信息抽取（属性、关系、事件）
- 置信度审核与向量相似度辅助校验
- 术语规范化与同义归并
- 图数据库构建（Neo4j）
- 中间结果落盘、阶段化目录、可恢复执行

主入口：`integrated_pipeline.py`

---

## 2. 项目结构（主流程相关）

```text
configs/config.yaml                  # 主配置
integrated_pipeline.py               # 主流水线入口

weaver/agents/
  data_scout.py                      # 数据侦察/抓取编排
  extractor.py                       # 信息抽取
  auditor_enhanced.py                # 知识审核
  constructor.py                     # 图构建

weaver/core/
  extraction.py
  extraction_architecture.py
  canonicalization.py

weaver/models/
  llm_models.py
  embedding_models.py

weaver/storage/
  vector_db.py
  file_handler.py

weaver/utils/
  config.py
  logging_setup.py
  canonicalizer_optimized.py
  getWikidata.py
  get_simbads.py
  MinerU.py
  webSearch_enhanced.py

data_sources/
  wikipedia_client.py
  mineru_client.py
```

---

## 3. 运行环境

建议：

- Python 3.10+
- Neo4j (bolt://localhost:7687)
- 向量服务（见 `configs/config.yaml` 中 `embedding.api_url`）
- 可用的 LLM API Key（见配置）

> 注意：当前 `configs/config.yaml` 中存在明文密钥，仅用于本地调试。生产环境请迁移到环境变量。

---

## 4. 输入格式

`integrated_pipeline.py` 支持：

- `.xlsx`
- `.json`
- `.txt`
- `.jsonl`

### 4.1 JSON 示例

```json
[
  {"type": "topic", "value": "Betelgeuse"},
  {"type": "pdf", "value": "docs/lunwen.pdf"}
]
```

也支持简化格式：

```json
[
  {"topic": "Betelgeuse"},
  {"pdf": "docs/lunwen.pdf"}
]
```

### 4.2 JSONL 说明

当输入为 `.jsonl` 时，主流程会跳过前置获取/抽取/审核阶段，直接把 JSONL 作为高置信度三元组继续执行后续规范化与入图。

---

## 5. 快速开始

### 5.1 配置

修改：`configs/config.yaml`

重点检查：

- `llm.provider` 以及对应 provider 的 `base_url/model`
- `api_keys.*`
- `neo4j.uri/user/password`
- `paths.input_file / output_dir / vector_db_path`

### 5.2 执行

```bash
python3 integrated_pipeline.py data/input/astroEntity.xlsx --config configs/config.yaml
```

常用参数：

- `--skip-audit`：跳过审核阶段
- `--multi-entity`：启用多实体抽取模式
- `--resume`：尝试接续中间结果
- `--output-dir <dir>`：指定输出目录

---

## 6. 输出结果

默认输出目录由 `configs/config.yaml` 的 `paths.output_dir` 指定。

中间结果按阶段保存到：

- `intermediate_results/01_data_acquisition`
- `intermediate_results/02_information_extraction`
- `intermediate_results/03_knowledge_auditing`
- `intermediate_results/04_canonicalization`
- `intermediate_results/05_graph_construction`

总览报告：

- `intermediate_results/pipeline_summary.json`
- `intermediate_results/pipeline_report.json`

---

## 7. 本次仓库整理说明（安全清理）

已完成：

- 修复目录名尾随空格问题：`astroWeaver ` → `astroWeaver`
- 对 `integrated_pipeline.py` 做静态依赖追踪
- 将主流程未引用的 Python 文件归档（非删除）到：
  - `archive_unused_YYYYMMDD_HHMMSS/`
- 归档明细见：
  - `cleanup_report.json`

如需恢复，直接将归档目录中的文件移动回原路径即可。

---

## 8. 建议后续优化

1. 把密钥迁移到环境变量（避免明文）
2. 增加 `requirements.txt` / `pyproject.toml`，固化依赖
3. 提供一个最小可运行样例输入，降低新成员上手成本
4. 增加 CI：语法检查、单测、格式化、类型检查

---

## 9. 许可证

当前仓库未声明 License。对外发布前请补充。