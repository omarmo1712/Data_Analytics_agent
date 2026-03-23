import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Model identifiers ---
PROGRAMMER_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
LLM_MODEL_NAME        = "Qwen/Qwen3-4B-Instruct-2507" #"Qwen/Qwen3.5-4B" #"Qwen/Qwen3-4B-Instruct-2507"

# --- Dataset ---
DATA_PATH = os.path.join(BASE_DIR, "data", "application_train.csv")

# --- Generation parameters ---
PROGRAMMER_MAX_NEW_TOKENS = 512
LLM_MAX_NEW_TOKENS        = 2048

# --- Code fix retries ---
MAX_FIX_RETRIES = 2  # max times to send broken code back to Qwen Coder for fixing

# --- FastAPI ---
API_HOST = "0.0.0.0"
API_PORT = 8889

# --- Streamlit ---
STREAMLIT_PAGE_TITLE = "Home Credit Analytics Agent"
STREAMLIT_API_URL    = f"http://localhost:{API_PORT}/analyze"
