# astroWeaver/models/llm_models.py
import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import concurrent.futures

from openai import OpenAI, APIError

try:
    from zhipuai import ZhipuAI
    ZHIPU_AVAILABLE = True
except ImportError:
    ZHIPU_AVAILABLE = False
    ZhipuAI = None

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
    支持阿里云（OpenAI兼容）和智谱AI两种客户端。
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None, max_workers: int = 5, provider: str = "ali"):
        """
        初始化LLM客户端。

        Args:
            api_key (str): API密钥。
            base_url (Optional[str]): API的基础URL。
            max_workers (int): 本地批处理时使用的最大并发线程数。
            provider (str): LLM提供商，"ali" 或 "zhipu"。
        """
        if not api_key:
            raise ValueError("API key cannot be empty.")

        self.provider = provider
        self.max_workers = max_workers
        
        if provider == "zhipu":
            if not ZHIPU_AVAILABLE:
                raise ImportError("ZhipuAI client not available. Please install: pip install zai")
            self.zhipu_client = ZhipuAI(api_key=api_key)
            self.client = None
            logger.info(f"LLMClient initialized for ZhipuAI")
        else:
            # 默认使用OpenAI兼容客户端（阿里云等）
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.zhipu_client = None
            logger.info(f"LLMClient initialized for base URL: {base_url or 'default OpenAI'}")
        
        logger.info(f"Local batch processing configured with max_workers={self.max_workers}")

    async def make_request_async(
            self,
            model: str,
            messages: List[Dict[str, str]],
            temperature: float = 0.1
    ) -> str:
        """
        异步执行聊天补全请求。

        此方法通过在单独的线程中运行同步的 `make_request` 方法，
        来防止阻塞 asyncio 事件循环。
        """
        # asyncio.to_thread 会在一个独立的线程中运行指定的阻塞函数，
        # 并异步地返回结果。这正是我们需要的。
        # 我们将 `self.make_request` 方法本身和它的所有参数传递进去。
        response = await asyncio.to_thread(
            self.make_request,
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response

    def make_request(
            self,
            model: str,
            messages: List[Dict[str, str]],
            temperature: float = 0.1,
            is_json: bool = True
    ) -> str:
        """
        执行一个标准的、单次的聊天补全请求。
        支持阿里云（OpenAI兼容）和智谱AI两种客户端。
        """
        try:
            if self.provider == "zhipu":
                # 使用智谱AI客户端
                response = self.zhipu_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=4096,
                    temperature=temperature
                )
                return response.choices[0].message.content
            else:
                # 使用OpenAI兼容客户端（阿里云等）
                response_format = {"type": "json_object"} if is_json else None
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM API request failed: {e}")
            raise


def create_llm_client(config: Dict[str, Any]) -> LLMClient:
    """
    根据配置创建LLM客户端的工厂函数
    
    Args:
        config: 配置字典
    
    Returns:
        LLMClient实例
    """
    provider = config.get('llm', {}).get('provider', 'ali')
    
    if provider == 'zhipu':
        api_key = config['api_keys']['zhipu_key']
        return LLMClient(api_key=api_key, provider='zhipu')
    else:
        # 默认使用阿里云
        api_key = config['api_keys']['ali_key']
        base_url = config['llm']['ali']['base_url']
        return LLMClient(api_key=api_key, base_url=base_url, provider='ali')


def get_model_name(config: Dict[str, Any], model_type: str = 'base_model') -> str:
    """
    根据配置获取模型名称。
    
    Args:
        config: 配置字典
        model_type: 模型类型，如 'base_model', 'extraction_model', 'judge_model'
        
    Returns:
        str: 模型名称
    """
    provider = config.get('llm', {}).get('provider', 'ali')
    
    if provider == 'zhipu':
        zhipu_config = config.get('llm', {}).get('zhipu', {})
        return zhipu_config.get(model_type, 'GLM-4-Flash-250414')
    else:
        ali_config = config.get('llm', {}).get('ali', {})
        # 向后兼容：如果ali配置中没有，则使用根级别配置
        return (ali_config.get(model_type) or 
                config.get('llm', {}).get(model_type, 'deepseek-v3'))




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