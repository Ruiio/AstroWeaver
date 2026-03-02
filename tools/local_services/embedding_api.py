#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
轻量本地Embedding服务（无大模型依赖）
接口兼容 AstroWeaver EmbeddingClient:
- GET /
- POST /embeddings  {"texts": [...], "prompt_name": "..."}
返回: {"embeddings": [[...], ...]}
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import re
import math

app = FastAPI(title="AstroWeaver Local Embedding API", version="1.1")

DIM = 384
TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class EmbeddingRequest(BaseModel):
    texts: List[str]
    prompt_name: Optional[str] = None


def _hash_token(token: str) -> int:
    # 稳定hash（避免Python内置hash每进程随机）
    h = 2166136261
    for ch in token.lower():
        h ^= ord(ch)
        h *= 16777619
        h &= 0xFFFFFFFF
    return h


def _embed(text: str) -> List[float]:
    vec = [0.0] * DIM
    if not text:
        return vec
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return vec

    for t in tokens:
        h = _hash_token(t)
        idx = h % DIM
        sign = -1.0 if ((h >> 1) & 1) else 1.0
        weight = 1.0 + ((h >> 8) % 100) / 500.0
        vec[idx] += sign * weight

    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


@app.get("/")
def health():
    return {"status": "ok", "model": "hash-embedding-384d"}


@app.post("/embeddings")
def embeddings(req: EmbeddingRequest):
    return {"embeddings": [_embed(t) for t in req.texts]}
