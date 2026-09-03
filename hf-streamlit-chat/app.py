"""
HuggingFace Model with Streamlit Chat Interface (vLLM backend)
Supports optional login and chat history. Inference via OpenAI-compatible vLLM.
"""
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.auth import AuthManager
from shared.database import ChatHistory
from shared.backends import OpenAIChatClient, require_backend_urls
from shared.deploy import ensure_model_runtime
from shared.multinode import BackendPool, ClusterInfo, get_multinode_config
from shared.tools import get_web_access_tool

os.environ.setdefault("LOAD_MODEL_WEIGHTS", "false")

MN_CFG = get_multinode_config(default_port=8501, app_name="hf-streamlit-chat")
CLUSTER = ClusterInfo(MN_CFG, app_name="hf-streamlit-chat")
WEB = get_web_access_tool("hf-streamlit-chat")

MODEL_NAME = os.getenv(
    "HF_MODEL_NAME", os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
)
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
VLLM_API_KEY = os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
REQUEST_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "300"))

_RUNTIME = ensure_model_runtime(
    "vllm",
    app_name="hf-streamlit-chat",
    model_id=MODEL_NAME,
    deploy_mode=MN_CFG.model_deploy_mode,
    replica_count=int(os.getenv("VLLM_REPLICA_COUNT", "1") or "1"),
    tensor_parallel_size=MN_CFG.tensor_parallel_size,
    existing_urls=MN_CFG.backend_urls or CLUSTER.backend_pool.urls,
    api_key=VLLM_API_KEY,
)
backends = require_backend_urls(_RUNTIME.urls or MN_CFG.backend_urls or CLUSTER.backend_pool.urls)
CLUSTER.backend_pool = BackendPool(backends, strategy=MN_CFG.backend_strategy)
MN_CFG.backend_urls = backends

VLLM_CLIENT = OpenAIChatClient(
    next_url=CLUSTER.next_backend,
    model=MODEL_NAME,
    api_key=VLLM_API_KEY,
    timeout=REQUEST_TIMEOUT,
    mark_success=CLUSTER.backend_pool.mark_success,
    mark_failure=CLUSTER.backend_pool.mark_failure,
    default_temperature=DEFAULT_TEMPERATURE,
    default_max_tokens=DEFAULT_MAX_TOKENS,
)

auth_manager = AuthManager()
chat_history = ChatHistory()


def init_session_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_token" not in st.session_state:
        st.session_state.session_token = None


def login_page():
    st.title("Login / Register")
    tab1, tab2, tab3 = st.tabs(["Login", "Register", "Continue as Guest"])

    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            session_token = auth_manager.login(username, password)
            if session_token:
                st.session_state.session_token = session_token
                st.session_state.user_id = auth_manager.validate_session(session_token)
                st.session_state.username = username
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        st.subheader("Register")
        new_username = st.text_input("Username", key="register_username")
        new_password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input(
            "Confirm Password", type="password", key="confirm_password"
        )
        if st.button("Register"):
            if new_password != confirm_password:
                st.error("Passwords don't match")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters")
            else:
                if auth_manager.register_user(new_username, new_password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists")

    with tab3:
        st.subheader("Continue as Guest")
        st.info("You can use the chat without logging in, but your history won't be saved.")
        if st.button("Continue as Guest"):
            st.session_state.authenticated = True
            st.session_state.user_id = None
            st.session_state.username = "Guest"
            st.rerun()


def generate_response(user_prompt: str, history_messages, web_context: str = "") -> str:
    messages = []
    for msg in history_messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in ("user", "assistant", "system") and content:
            messages.append({"role": role, "content": content})
    # current user turn already appended by caller; ensure last is user
    if not messages or messages[-1].get("content") != user_prompt:
        model_prompt = user_prompt
        if web_context:
            model_prompt = (
                f"{web_context}\n\n"
                f"User question (may reference the URL(s) above):\n{user_prompt}"
            )
        messages.append({"role": "user", "content": model_prompt})
    elif web_context and messages[-1].get("role") == "user":
        messages[-1] = {
            "role": "user",
            "content": (
                f"{web_context}\n\n"
                f"User question (may reference the URL(s) above):\n{user_prompt}"
            ),
        }
    try:
        _data, text = VLLM_CLIENT.chat(messages)
        return text or "(empty response)"
    except Exception as e:
        return f"Error generating response: {e}"


def chat_page():
    st.title("HuggingFace Chat Interface (vLLM)")

    with st.sidebar:
        st.write(f"User: {st.session_state.username}")
        st.write(f"Model: {MODEL_NAME}")
        st.write("Backend: vLLM")
        with st.expander("Cluster / node"):
            node = CLUSTER.health(inference_backend="vllm", model=MODEL_NAME)
            st.caption(f"node: `{node['node']['node_id']}`")
            st.caption(f"multi_node: `{node['node']['multi_node']}`")
            st.caption(f"backends: `{node.get('backends', {})}`")
        with st.expander("Web access"):
            ws = WEB.stats()
            st.caption(f"enabled: `{ws['enabled']}`")
            st.caption(f"fetches: `{ws['fetches']}` failures: `{ws['failures']}`")
            st.caption(f"log: `{ws['log_file']}`")
            st.caption("URLs in messages are fetched; full page text is logged.")

        if st.button("Logout"):
            if st.session_state.session_token:
                auth_manager.logout(st.session_state.session_token)
            st.session_state.clear()
            st.rerun()

        st.divider()

        if st.button("New Conversation"):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()

        if st.session_state.user_id:
            st.subheader("Previous Conversations")
            conversations = chat_history.get_user_conversations(
                st.session_state.user_id, "hf-streamlit-chat"
            )
            for conv in conversations[:10]:
                if st.button(
                    f"Conv {conv['id']} - {conv['updated_at'][:16]}",
                    key=f"conv_{conv['id']}",
                ):
                    st.session_state.conversation_id = conv["id"]
                    messages = chat_history.get_conversation_history(conv["id"])
                    st.session_state.messages = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in messages
                    ]
                    st.rerun()

    if st.session_state.conversation_id is None and st.session_state.user_id:
        st.session_state.conversation_id = chat_history.create_conversation(
            st.session_state.user_id, "hf-streamlit-chat"
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.conversation_id:
            chat_history.add_message(st.session_state.conversation_id, "user", prompt)

        session_key = str(
            st.session_state.conversation_id
            or st.session_state.user_id
            or st.session_state.username
            or "guest"
        )
        web_results, web_context = WEB.process_user_text(prompt, session_id=session_key)
        if web_results:
            with st.expander(f"Fetched {len(web_results)} URL(s) (full content logged)"):
                for wr in web_results:
                    st.markdown(
                        f"- `{wr.url}` — "
                        f"{'OK' if wr.ok else 'FAIL'} "
                        f"status={wr.status_code} "
                        f"log=`{wr.log_path}`"
                    )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # history without the just-added user msg for builder; generate_response merges
                prior = st.session_state.messages[:-1]
                response = generate_response(
                    prompt, prior + [{"role": "user", "content": prompt}], web_context=web_context
                )
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

        if st.session_state.conversation_id:
            chat_history.add_message(
                st.session_state.conversation_id, "assistant", response
            )

        st.rerun()


def main():
    st.set_page_config(
        page_title="HuggingFace Chat Interface",
        page_icon="🤗",
        layout="wide",
    )
    init_session_state()
    if not st.session_state.authenticated:
        login_page()
    else:
        chat_page()


if __name__ == "__main__":
    main()
