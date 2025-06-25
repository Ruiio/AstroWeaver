# astroWeaver/models/llm_models.py

import logging
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import concurrent.futures

from openai import OpenAI, APIError

logger = logging.getLogger(__name__)


@dataclass
class BatchResponse:
    """用于封装批处理返回结果的数据类。接口保持不变。"""
    request_id: str
    response_text: Optional[str] = None
    error: Optional[str] = None

    def is_success(self) -> bool:
        return self.error is None


class LLMClient:
    """
    封装与LLM API的交互，支持标准请求和本地模拟的批处理请求。
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None, max_workers: int = 5):
        """
        初始化LLM客户端。

        Args:
            api_key (str): API密钥。
            base_url (Optional[str]): API的基础URL。
            max_workers (int): 本地批处理时使用的最大并发线程数。
        """
        if not api_key:
            raise ValueError("API key cannot be empty.")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_workers = max_workers
        logger.info(f"LLMClient initialized for base URL: {base_url or 'default OpenAI'}")
        logger.info(f"Local batch processing configured with max_workers={self.max_workers}")

    def make_request(
            self,
            model: str,
            messages: List[Dict[str, str]],
            temperature: float = 0.1,
            is_json: bool = True
    ) -> str:
        """
        执行一个标准的、单次的聊天补全请求。
        (此方法保持不变)
        """
        try:
            # 检查供应商是否支持 response_format
            # 如果不支持，需要移除这个参数，并在返回后手动解析JSON
            # 假设Dashscope兼容模式支持
            response_format = {"type": "json_object"} if is_json else None

            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                # 如果 response_format 不被支持，注释掉下面这行
                # response_format=response_format
            )
            return completion.choices[0].message.content
        except APIError as e:
            logger.error(f"LLM API request failed: {e}")
            raise

    def prepare_batch_request(
            self,
            custom_id: str,
            model: str,
            messages: List[Dict[str, str]],
            temperature: float = 0.1,
            is_json: bool = True
    ) -> Dict[str, Any]:
        """
        准备一个用于本地批处理的请求体。
        (此方法保持不变，因为上层代码依赖它来构建请求)
        """
        # 注意：我们不再需要 "method" 和 "url" 字段，但为了保持接口一致性，
        # 我们可以保留它们，或者简化body。这里我们选择简化，只保留必要信息。
        return {
            "custom_id": custom_id,
            "body": {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "is_json": is_json
            }
        }

    def _execute_single_request_for_batch(self, request: Dict[str, Any]) -> BatchResponse:
        """
        一个辅助函数，用于在线程池中执行单个请求。
        """
        custom_id = request["custom_id"]
        body = request["body"]
        try:
            response_text = self.make_request(
                model=body["model"],
                messages=body["messages"],
                temperature=body["temperature"],
                is_json=body["is_json"]
            )
            return BatchResponse(request_id=custom_id, response_text=response_text)
        except Exception as e:
            logger.error(f"Request {custom_id} failed during local batch execution: {e}")
            return BatchResponse(request_id=custom_id, error=str(e))

    def submit_batch(
            self,
            requests: List[Dict[str, Any]],
            # poll_interval 参数不再需要
    ) -> List[BatchResponse]:
        """
        **[已修改]** 使用本地多线程并行处理一批请求，模拟批处理API。

        Args:
            requests (List[Dict[str, Any]]): 由prepare_batch_request生成的请求列表。

        Returns:
            List[BatchResponse]: 包含每个请求结果的列表。
        """
        if not requests:
            return []

        logger.info(f"Starting local batch processing for {len(requests)} requests using {self.max_workers} workers...")

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务到线程池
            future_to_request = {executor.submit(self._execute_single_request_for_batch, req): req for req in requests}

            # 等待所有任务完成并收集结果
            for future in concurrent.futures.as_completed(future_to_request):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    # 这种情况理论上不应发生，因为_execute_single_request_for_batch已捕获异常
                    # 但作为保险措施，我们仍然处理它
                    request_id = future_to_request[future]['custom_id']
                    logger.error(f"Request {request_id} generated an unexpected exception: {exc}")
                    results.append(BatchResponse(request_id=request_id, error=str(exc)))

        logger.info("Local batch processing finished.")

        # 为了与原始返回顺序一致（如果需要），可以对结果进行排序
        request_order = {req['custom_id']: i for i, req in enumerate(requests)}
        results.sort(key=lambda r: request_order.get(r.request_id, float('inf')))

        return results

    # _process_batch_results 方法不再需要，因为结果是直接生成的