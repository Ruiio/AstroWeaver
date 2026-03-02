# astroWeaver/models/llm_models.py
import asyncio
import logging
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import concurrent.futures

from openai import OpenAI, APIError
import httpx
import os

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
    支持阿里云/DeepSeek（OpenAI兼容）、智谱AI，以及 NVIDIA(OpenAI兼容) 等客户端。
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None, max_workers: int = 5, provider: str = "ali", http_client: Optional[httpx.Client] = None):
        """
        初始化LLM客户端。

        Args:
            api_key (str): API密钥。
            base_url (Optional[str]): API的基础URL。
            max_workers (int): 本地批处理时使用的最大并发线程数。
            provider (str): LLM提供商，"ali"、"zhipu"、"nvidia" 或 "closeai"。
            http_client (Optional[httpx.Client]): 可选自定义HTTP客户端（用于代理设置等）。
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
            self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
            self.zhipu_client = None
            logger.info(f"LLMClient initialized for base URL: {base_url or 'default OpenAI'} (provider={provider})")
        
        logger.info(f"Local batch processing configured with max_workers={self.max_workers}")

    async def make_request_async(
            self,
            model: str,
            messages: List[Dict[str, str]],
            temperature: float = 0.1,
            timeout: float = 600,
            stream: bool = False,
            is_json: bool = True,
    ) -> str:
        """
        异步执行聊天补全请求。
        """
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.make_request,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    stream=stream,
                    is_json=is_json,
                ),
                timeout=timeout
            )
            return response
        except asyncio.TimeoutError:
            logger.error(f"LLM request timeout after {timeout} seconds for model {model}")
            raise

    def make_request(
            self,
            model: str,
            messages: List[Dict[str, str]],
            temperature: float = 0.1,
            is_json: bool = True,
            stream: bool = False,
    ) -> str:
        """
        执行一个标准的、单次的聊天补全请求。
        支持阿里云（OpenAI兼容）、智谱AI 以及 NVIDIA(OpenAI兼容)。
        """
        try:
            if self.provider == "zhipu":
                # 使用智谱AI客户端（不支持 OpenAI 参数）
                response = self.zhipu_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=4096,
                    temperature=temperature
                )
                return response.choices[0].message.content
            else:
                # 使用OpenAI兼容客户端（阿里云、NVIDIA等）
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "stream": stream,
                }
                if self.provider != "closeai":
                    kwargs["temperature"] = temperature
                # 需要 JSON 输出时，传递 response_format
                if is_json:
                    kwargs["response_format"] = {"type": "json_object"}
                # 仅对阿里系/DeepSeek等要求非流式禁用 thinking 的提供商设置 extra_body
                if not stream and self.provider in {"ali", "openai", "deepseek"}:
                    kwargs["extra_body"] = {"enable_thinking": False}
                # NVIDIA 提供商不传递 extra_body，以避免未知参数错误
                completion = self.client.chat.completions.create(**kwargs)

                if stream:
                    # 组装流式增量内容
                    content_chunks: List[str] = []
                    for event in completion:
                        try:
                            delta = event.choices[0].delta
                            if delta and getattr(delta, "content", None):
                                content_chunks.append(delta.content)
                        except Exception:
                            # 避免因为事件格式差异导致中断
                            continue
                    return "".join(content_chunks)
                else:
                    return completion.choices[0].message.content
        except Exception as e:
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
    ) -> List[BatchResponse]:
        """
        使用本地多线程并行处理一批请求，模拟批处理API。
        """
        if not requests:
            return []

        logger.info(f"Starting local batch processing for {len(requests)} requests using {self.max_workers} workers...")

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_request = {executor.submit(self._execute_single_request_for_batch, req): req for req in requests}
            for future in concurrent.futures.as_completed(future_to_request):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    request_id = future_to_request[future]['custom_id']
                    logger.error(f"Request {request_id} generated an unexpected exception: {exc}")
                    results.append(BatchResponse(request_id=request_id, error=str(exc)))

        logger.info("Local batch processing finished.")

        request_order = {req['custom_id']: i for i, req in enumerate(requests)}
        results.sort(key=lambda r: request_order.get(r.request_id, float('inf')))

        return results


def create_llm_client(config: Dict[str, Any]) -> LLMClient:
    """
    根据配置创建LLM客户端的工厂函数
    """
    provider = config.get('llm', {}).get('provider', 'ali')
    
    if provider == 'zhipu':
        api_key = config['api_keys']['zhipu_key']
        return LLMClient(api_key=api_key, provider='zhipu')
    elif provider == 'nvidia':
        api_key = config['api_keys']['nvidia_key']
        base_url = config.get('llm', {}).get('nvidia', {}).get('base_url', 'https://integrate.api.nvidia.com/v1')
        return LLMClient(api_key=api_key, base_url=base_url, provider='nvidia')
    elif provider == 'closeai':
        api_key = config['api_keys']['closeai_key']
        base_url = config.get('llm', {}).get('closeai', {}).get('base_url', 'https://api.openai-proxy.org/v1')
        return LLMClient(api_key=api_key, base_url=base_url, provider='closeai')
    elif provider == 'deepseek':
        api_key = config['api_keys'].get('deepseek_key') or config['api_keys'].get('ali_key')
        base_url = config.get('llm', {}).get('deepseek', {}).get('base_url', 'https://api.deepseek.com')
        return LLMClient(api_key=api_key, base_url=base_url, provider='deepseek')
    else:
        # 默认使用阿里云
        api_key = config['api_keys']['ali_key']
        base_url = config['llm']['ali']['base_url']
        return LLMClient(api_key=api_key, base_url=base_url, provider='ali')


def get_model_name(config: Dict[str, Any], model_type: str = 'base_model') -> str:
    """
    根据配置获取模型名称。
    """
    provider = config.get('llm', {}).get('provider', 'ali')
    
    if provider == 'zhipu':
        zhipu_config = config.get('llm', {}).get('zhipu', {})
        return zhipu_config.get(model_type, 'GLM-4-Flash-250414')
    elif provider == 'nvidia':
        nv_config = config.get('llm', {}).get('nvidia', {})
        return nv_config.get(model_type, 'meta/llama-4-scout-17b-16e-instruct')
    elif provider == 'closeai':
        ca_config = config.get('llm', {}).get('closeai', {})
        return ca_config.get(model_type, 'gpt-4o-mini')
    elif provider == 'deepseek':
        ds_config = config.get('llm', {}).get('deepseek', {})
        return ds_config.get(model_type, 'deepseek-chat')
    else:
        ali_config = config.get('llm', {}).get('ali', {})
        return (ali_config.get(model_type) or 
                config.get('llm', {}).get(model_type, 'deepseek-v3'))