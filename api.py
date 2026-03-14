import re
import io
import sys
import base64
import traceback
from contextlib import asynccontextmanager

import matplotlib
matplotlib.use("Agg")  # must come before pyplot import
import matplotlib.pyplot as plt

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import (
    PROGRAMMER_MODEL_NAME,
    LLM_MODEL_NAME,
    PROGRAMMER_MAX_NEW_TOKENS,
    LLM_MAX_NEW_TOKENS,
    API_HOST,
    API_PORT,
)
from prompts import build_programmer_messages, build_llm_messages

# Module-level state — populated once at startup
_state: dict = {}


# ── Pydantic schemas ───────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    question: str


class AnalyzeResponse(BaseModel):
    answer: str
    chart_base64: str | None = None
    generated_code: str | None = None


# ── Lifespan (model loading) ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Programmer model (Qwen Coder)...")
    _state["prog_tokenizer"] = AutoTokenizer.from_pretrained(PROGRAMMER_MODEL_NAME)
    _state["prog_model"] = AutoModelForCausalLM.from_pretrained(
        PROGRAMMER_MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

    print("Loading LLM model (Qwen3)...")
    _state["llm_tokenizer"] = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    _state["llm_model"] = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

    print("Both models loaded. API ready.")
    yield

    _state.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="Data Analytics Agent", lifespan=lifespan)


# ── Helpers ────────────────────────────────────────────────────────────────

def generate_text(tokenizer, model, messages: list, max_new_tokens: int) -> str:
    """
    Applies chat template, runs model.generate(), slices off input tokens,
    and batch_decodes. Mirrors the pattern from test.ipynb.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )

    # Slice off the prompt tokens
    trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids
        in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0]


def extract_code_block(model_response: str) -> str:
    """
    Extracts the first ```python ... ``` block from the model response.
    Falls back to stripping any generic triple-backtick fences.
    """
    match = re.search(r"```python\s*(.*?)```", model_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: strip generic fences
    fallback = re.sub(r"```[a-z]*\s*", "", model_response)
    fallback = fallback.replace("```", "")
    return fallback.strip()


def execute_code(code: str) -> tuple[str, str | None]:
    """
    Executes code in a restricted namespace.

    Returns:
        (stdout_text, chart_base64_or_None)

    - matplotlib Agg backend is already set globally; plt.show() is a no-op.
    - Captures stdout via io.StringIO redirect.
    - Saves any open matplotlib figure as a base64 PNG.
    - Returns tracebacks as stdout_text on exception.
    """
    import pandas as pd
    import numpy as np

    exec_globals = {
        "__builtins__": __builtins__,
        "pd": pd,
        "np": np,
        "plt": plt,
        "matplotlib": matplotlib,
    }

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    chart_b64 = None

    try:
        exec(code, exec_globals)  # noqa: S102

        if plt.get_fignums():
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            chart_b64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close("all")

        stdout_text = sys.stdout.getvalue()

    except Exception:
        stdout_text = traceback.format_exc()
        plt.close("all")
    finally:
        sys.stdout = old_stdout

    return stdout_text or "(no text output)", chart_b64


# ── Endpoint ───────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Step 1: Qwen Coder → Python code
    prog_messages = build_programmer_messages(question)
    coder_response = generate_text(
        _state["prog_tokenizer"],
        _state["prog_model"],
        prog_messages,
        PROGRAMMER_MAX_NEW_TOKENS,
    )
    code = extract_code_block(coder_response)

    # Step 2: Execute code → stdout + optional chart
    stdout_text, chart_b64 = execute_code(code)

    # Step 3: Qwen3 → natural language answer
    output_for_llm = stdout_text
    if chart_b64:
        output_for_llm += "\n[A matplotlib chart was also generated and will be shown to the user.]"

    llm_messages = build_llm_messages(question, output_for_llm)
    raw_answer = generate_text(
        _state["llm_tokenizer"],
        _state["llm_model"],
        llm_messages,
        LLM_MAX_NEW_TOKENS,
    )

    # Strip Qwen3 <think>...</think> reasoning blocks if present
    answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()

    return AnalyzeResponse(
        answer=answer,
        chart_base64=chart_b64,
        generated_code=code,
    )


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=False)
