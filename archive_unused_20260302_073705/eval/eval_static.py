import pandas as pd
import ast
import pprint
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import torch

# --- CONFIGURATION ---
# 将此项设为 True 以启用语义匹配，设为 False 以使用严格的精确匹配
USE_SEMANTIC_MATCHING = True
# 语义匹配的相似度阈值 (0.0 - 1.0)。越接近1.0越严格。
# 0.95: 非常相似 (e.g., "1918-1924" vs "between 1918 and 1924")
# 0.85: 强相关 (e.g., "a large-scale attempt" vs "the first major effort")
SIMILARITY_THRESHOLD = 0.90

# --- MODEL LOADING ---
# 在全局加载模型，避免在循环中重复加载，以提高效率
# 'all-MiniLM-L6-v2' 是一个非常优秀且快速的通用模型
if USE_SEMANTIC_MATCHING:
    print("Loading sentence transformer model... (This may take a moment on first run)")
    # 检查是否有可用的GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print("Model loaded.")


# --- HELPER FUNCTIONS (与之前类似) ---
def safe_literal_eval(val):
    if pd.isna(val) or not isinstance(val, str) or val.strip() == '':
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        print(f"Warning: Could not parse value: {val}")
        return []


def normalize_string(s):
    return str(s).strip().lower()


# --- NORMALIZATION FUNCTIONS (现在返回元组以进行精确匹配) ---
def normalize_attribute_exact(attr):
    return (
        normalize_string(attr.get('entity')),
        normalize_string(attr.get('attribute')),
        normalize_string(attr.get('value'))
    )


def normalize_relation_exact(rel):
    return (
        normalize_string(rel.get('subject')),
        normalize_string(rel.get('predicate')),
        normalize_string(rel.get('object'))
    )


def normalize_event_exact(evt):
    args = evt.get('arguments', [])
    frozen_args = frozenset(
        tuple(sorted(normalize_string(v) for v in arg.values())) for arg in args
    )
    return (
        normalize_string(evt.get('event_type')),
        normalize_string(evt.get('anchor_entity')),
        frozen_args
    )


# --- METRIC CALCULATION ---
def calculate_exact_metrics(golden_list, predicted_list, normalize_func):
    golden_set = {normalize_func(item) for item in golden_list}
    predicted_set = {normalize_func(item) for item in predicted_list}
    tp = len(golden_set.intersection(predicted_set))
    fp = len(predicted_set - golden_set)
    fn = len(golden_set - predicted_set)
    return tp, fp, fn


def calculate_semantic_metrics(golden_list, predicted_list, key_fields, value_field):
    """
    使用语义相似度计算TP, FP, FN。
    - key_fields: 用于精确匹配的字段 (e.g., entity, attribute)
    - value_field: 用于语义匹配的字段 (e.g., value)
    """
    if not golden_list and not predicted_list:
        return 0, 0, 0
    if not predicted_list:
        return 0, 0, len(golden_list)
    if not golden_list:
        return 0, len(predicted_list), 0

    tp = 0

    # 创建一个黄金标准的副本，用于从中移除已匹配的项
    unmatched_golden = list(golden_list)

    # 提取所有黄金标准和预测值的文本，一次性编码以提高效率
    golden_values = [str(g.get(value_field, '')) for g in unmatched_golden]
    predicted_values = [str(p.get(value_field, '')) for p in predicted_list]

    # 编码
    golden_embeddings = model.encode(golden_values, convert_to_tensor=True, show_progress_bar=False)
    predicted_embeddings = model.encode(predicted_values, convert_to_tensor=True, show_progress_bar=False)

    # 计算所有预测值与所有黄金值之间的余弦相似度
    cosine_scores = util.cos_sim(predicted_embeddings, golden_embeddings)

    # 使用贪心策略进行匹配
    matched_golden_indices = set()

    for i in range(len(predicted_list)):
        pred_item = predicted_list[i]
        pred_keys = tuple(normalize_string(pred_item.get(k)) for k in key_fields)

        best_match_score = -1
        best_match_idx = -1

        for j in range(len(unmatched_golden)):
            # 跳过已经匹配的黄金项
            if j in matched_golden_indices:
                continue

            golden_item = unmatched_golden[j]
            golden_keys = tuple(normalize_string(golden_item.get(k)) for k in key_fields)

            # 1. 首先检查key字段是否完全匹配
            if pred_keys == golden_keys:
                # 2. 如果key匹配，再检查value的语义相似度
                similarity = cosine_scores[i][j]
                if similarity > best_match_score:
                    best_match_score = similarity
                    best_match_idx = j

        # 如果找到了一个最佳匹配且分数超过阈值
        if best_match_score >= SIMILARITY_THRESHOLD:
            tp += 1
            matched_golden_indices.add(best_match_idx)

    fp = len(predicted_list) - tp
    fn = len(golden_list) - tp

    return tp, fp, fn


def evaluate_from_file(filepath):
    df = pd.read_excel(filepath)

    # 定义任务配置
    tasks = {
        'Attributes': {
            'golden_col': 'golden_attributes', 'llm_col': 'LLM_attributes',
            'normalizer': normalize_attribute_exact,
            'semantic_keys': ['entity', 'attribute'], 'semantic_value': 'value'
        },
        'Relations': {
            'golden_col': 'golden_relations', 'llm_col': 'LLM_relations',
            'normalizer': normalize_relation_exact,
            'semantic_keys': ['subject', 'predicate'], 'semantic_value': 'object'
        },
        # 事件结构复杂，语义匹配较难，此处暂时仍使用精确匹配
        # 若要对事件进行语义匹配，需要更复杂的逻辑来处理论元列表
        'Events': {
            'golden_col': 'golden_events', 'llm_col': 'LLM_events',
            'normalizer': normalize_event_exact,
            'semantic_keys': None  # 禁用语义匹配
        }
    }

    total_counts = {task: {'tp': 0, 'fp': 0, 'fn': 0} for task in tasks}

    for index, row in df.iterrows():
        for task_name, config in tasks.items():
            golden_data = safe_literal_eval(row.get(config['golden_col']))
            llm_data = safe_literal_eval(row.get(config['llm_col']))

            if USE_SEMANTIC_MATCHING and config['semantic_keys']:
                tp, fp, fn = calculate_semantic_metrics(
                    golden_data, llm_data, config['semantic_keys'], config['semantic_value']
                )
            else:
                tp, fp, fn = calculate_exact_metrics(
                    golden_data, llm_data, config['normalizer']
                )

            total_counts[task_name]['tp'] += tp
            total_counts[task_name]['fp'] += fp
            total_counts[task_name]['fn'] += fn

    # 报告生成
    print("\n" + "=" * 50)
    print(f"Evaluation Report (Mode: {'Semantic' if USE_SEMANTIC_MATCHING else 'Exact'})")
    if USE_SEMANTIC_MATCHING:
        print(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")
    print("=" * 50)

    results = {}
    for task_name, counts in total_counts.items():
        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results[task_name] = {
            'Precision': precision, 'Recall': recall, 'F1-Score': f1,
            'TP': tp, 'FP': fp, 'FN': fn
        }
        print(f"\n--- {task_name} ---")
        print(f"  Precision: {results[task_name]['Precision']:.4f}")
        print(f"  Recall:    {results[task_name]['Recall']:.4f}")
        print(f"  F1-Score:  {results[task_name]['F1-Score']:.4f}")
        print(f"  (Counts: TP={results[task_name]['TP']}, FP={results[task_name]['FP']}, FN={results[task_name]['FN']})")

    print("\n" + "=" * 50)
    return results


if __name__ == '__main__':
    # Replace 'your_evaluation_file.xlsx' with the actual path to your file
    file_path = 'extracted_data.xlsx'
    evaluate_from_file(file_path)