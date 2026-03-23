# Home Credit Analytics Agent

An AI-powered data analysis tool that answers natural language questions about the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset. Ask a question in plain English — the agent writes Python code, executes it, and returns a human-readable answer with charts.

## How It Works

```
User Question
     ↓
Qwen2.5-Coder-7B  →  generates Python/pandas code
     ↓
Code execution (sandboxed)  →  stdout + matplotlib chart
     ↓  (auto-retry up to 2x on error)
Qwen3-4B  →  interprets results into natural language
     ↓
Answer + Chart + Code  →  Streamlit UI
```

## Project Structure

```
├── api.py          # FastAPI backend — loads models, runs the analysis pipeline
├── ui.py           # Streamlit frontend — chat interface
├── utils.py        # Helpers: message builders, LLM inference, code execution
├── prompts.py      # System prompts and dataset schema documentation
├── config.py       # Model names, paths, ports, token limits
└── data/           # Home Credit CSV files (not tracked in git)
    ├── application_train.csv
    ├── bureau.csv
    ├── bureau_balance.csv
    ├── previous_application.csv
    ├── installments_payments.csv
    ├── POS_CASH_balance.csv
    └── credit_card_balance.csv
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended — models total ~11GB)
- The Home Credit dataset CSVs placed inside a `data/` folder

Install dependencies:

```bash
pip install torch transformers fastapi uvicorn streamlit requests pydantic
```

## Running

**Step 1 — Start the API server** (loads models, takes ~1–2 min on first run):

```bash
python api.py
```

The server starts at `http://0.0.0.0:8889`.

**Step 2 — Start the UI** (in a separate terminal):

```bash
streamlit run ui.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## Example Questions

- What is the median credit amount for approved loans?
- Which education level has the highest default rate?
- Do applicants with more bureau credits have higher default rates?
- What percentage of applicants have active credits in other institutions?
- What is the default rate of applicants who previously had overdue bureau loans?

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `PROGRAMMER_MODEL_NAME` | `Qwen/Qwen2.5-Coder-7B-Instruct` | Code generation model |
| `LLM_MODEL_NAME` | `Qwen/Qwen3-4B-Instruct-2507` | Answer interpretation model |
| `PROGRAMMER_MAX_NEW_TOKENS` | `512` | Max tokens for generated code |
| `LLM_MAX_NEW_TOKENS` | `2048` | Max tokens for the answer |
| `MAX_FIX_RETRIES` | `2` | Auto-retry attempts on code error |
| `API_PORT` | `8889` | FastAPI port |
