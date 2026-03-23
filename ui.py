import warnings
warnings.filterwarnings("ignore")

import base64
import requests
import streamlit as st

from config import STREAMLIT_API_URL, STREAMLIT_PAGE_TITLE

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    page_icon="📊",
    layout="wide",
)
st.title(STREAMLIT_PAGE_TITLE)
st.caption("Powered by Qwen2.5-Coder + Qwen3 | Home Credit Default Risk Dataset")

# ── Session state ──────────────────────────────────────────────────────────

# Each message: {"role": "user"|"assistant", "content": str,
#                "chart_base64": str|None, "code": str|None}
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Helpers ────────────────────────────────────────────────────────────────

def display_base64_image(b64_string: str):
    """Decode a base64 PNG and render it in Streamlit."""
    image_bytes = base64.b64decode(b64_string)
    st.image(image_bytes, use_container_width=True)


def call_api(question: str) -> dict:
    """POST to FastAPI /analyze. Returns response dict or a synthetic error dict."""
    try:
        response = requests.post(
            STREAMLIT_API_URL,
            json={"question": question},
            timeout=300,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {
            "answer": (
                "Error: Could not connect to the API server. "
                "Make sure `api.py` is running (`python api.py`)."
            ),
            "chart_base64": None,
            "generated_code": None,
        }
    except requests.exceptions.Timeout:
        return {
            "answer": (
                "Error: The request timed out. "
                "The model may still be loading or the query is too complex."
            ),
            "chart_base64": None,
            "generated_code": None,
        }
    except Exception as e:
        return {
            "answer": f"Unexpected error: {e}",
            "chart_base64": None,
            "generated_code": None,
        }


def render_chat_history():
    """Replay all stored messages in the chat UI."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("chart_base64"):
                display_base64_image(msg["chart_base64"])
            if msg.get("code") and msg["role"] == "assistant":
                with st.expander("View generated code"):
                    st.code(msg["code"], language="python")


# ── Main interaction loop ──────────────────────────────────────────────────

render_chat_history()

if prompt := st.chat_input("Ask a question about the Home Credit dataset..."):

    # 1. Show user message immediately
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "chart_base64": None,
        "code": None,
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call API and render response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing... (this may take 30–90 seconds)"):
            result = call_api(prompt)

        answer    = result.get("answer", "No answer returned.")
        chart_b64 = result.get("chart_base64")
        gen_code  = result.get("generated_code")

        st.markdown(answer)
        if chart_b64:
            display_base64_image(chart_b64)
        if gen_code:
            with st.expander("View generated code"):
                st.code(gen_code, language="python")

    # 3. Persist to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chart_base64": chart_b64,
        "code": gen_code,
    })
