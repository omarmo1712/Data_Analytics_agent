import re
import io
import sys
import base64
import traceback

import matplotlib
matplotlib.use("Agg")  # must come before pyplot import
import matplotlib.pyplot as plt

import torch

from prompts import PROGRAMMER_SYSTEM_PROMPT, LLM_SYSTEM_PROMPT


# ── Message builders ───────────────────────────────────────────────────────

def build_programmer_messages(user_question: str) -> list:
    """Return the messages list for Qwen Coder."""
    return [
        {"role": "system", "content": PROGRAMMER_SYSTEM_PROMPT},
        {"role": "user",   "content": user_question},
    ]


def build_fix_messages(user_question: str, broken_code: str, error: str) -> list:
    """Return messages asking Qwen Coder to fix code that raised an error."""
    user_content = (
        f"The following Python code was generated to answer this question:\n"
        f"Question: {user_question}\n\n"
        f"```python\n{broken_code}\n```\n\n"
        f"It failed with this error:\n{error}\n\n"
        f"Please fix the code so it runs correctly. Output only the corrected ```python ... ``` block."
    )
    return [
        {"role": "system", "content": PROGRAMMER_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def build_llm_messages(user_question: str, code_output: str) -> list:
    """Return the messages list for Qwen3 interpretation."""
    user_content = (
        f"User question: {user_question}\n\n"
        f"Code execution output:\n{code_output}"
    )
    return [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


# ── LLM inference ──────────────────────────────────────────────────────────

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


# ── Code extraction ────────────────────────────────────────────────────────

def extract_code_block(model_response: str) -> str:
    """
    Extracts the first ```python ... ``` block from the model response.
    Falls back to stripping any generic triple-backtick fences.
    """
    match = re.search(r"```python\s*(.*?)```", model_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    fallback = re.sub(r"```[a-z]*\s*", "", model_response)
    fallback = fallback.replace("```", "")
    return fallback.strip()


# ── Code execution ─────────────────────────────────────────────────────────

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
