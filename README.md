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

## Future Work

### Replace Long-Context Schema with Dynamic Discovery
Currently the full dataset schema (250+ column names and descriptions) is hardcoded directly into the system prompt, consuming a large number of tokens on every request. Better alternatives:

- **RAG (Retrieval-Augmented Generation)** — embed column descriptions into a vector database (e.g. ChromaDB, FAISS). At query time, retrieve only the columns relevant to the user's question and inject just those into the prompt. Reduces token usage significantly and scales to any dataset.
- **Tool/function calling** — give the model a `get_schema(table_name)` tool it can call on demand, rather than front-loading the entire schema.
- **Auto-discovery at startup** — read column names and types directly from CSV headers and `HomeCredit_columns_description.csv` at server startup, generating the prompt dynamically instead of maintaining it by hand.

### Slack / Microsoft Teams Integration
Deploy the agent as a bot so analysts can query it directly from their communication tools without opening a browser:

**Slack:**
- Use the Slack Bolt SDK to create a bot that listens for `@mentions` or `/analyze` slash commands
- Forward the message text to the `/analyze` FastAPI endpoint
- Post the answer back as a threaded reply; upload the chart as a file attachment

**Microsoft Teams:**
- Use the Bot Framework SDK or Teams Toolkit to register a bot
- Handle `message` activity events, call the FastAPI backend, and reply with an Adaptive Card containing the answer and chart image

Both integrations would sit in front of the existing FastAPI backend with no changes to the core pipeline.

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
