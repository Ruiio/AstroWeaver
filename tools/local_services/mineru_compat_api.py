#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from pypdf import PdfReader

app = FastAPI(title="MinerU-Compatible API", version="1.0")


@app.get("/")
def health():
    return {"status": "ok", "service": "mineru-compat"}


@app.post("/file_parse")
async def file_parse(
    files: List[UploadFile] = File(...),
    lang_list: str = Form(default="['ch']"),
    return_md: str = Form(default="true"),
    table_enable: str = Form(default="true"),
    formula_enable: str = Form(default="true")
):
    results = {}
    for f in files:
        suffix = os.path.splitext(f.filename)[-1].lower()
        data = await f.read()

        if suffix == ".pdf":
            tmp = f"/tmp/{f.filename}"
            with open(tmp, "wb") as wf:
                wf.write(data)
            reader = PdfReader(tmp)
            text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
        else:
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                text = ""

        results[f.filename] = {
            "md_content": text.strip()
        }

    return {"results": results}
