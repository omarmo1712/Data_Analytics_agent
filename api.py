import re
from contextlib import asynccontextmanager

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
from utils import (
    build_programmer_messages,
    build_llm_messages,
    generate_text,
    extract_code_block,
    execute_code,
)

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


# ── Endpoint ───────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    question = request.question.strip()
    print(f"Received question: {question}")
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
