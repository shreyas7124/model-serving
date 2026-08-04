"""
HuggingFace Model with Streamlit Chat Interface
Supports optional login and chat history
"""
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import sys
import os

# Add parent directory to path for shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.auth import AuthManager
from shared.database import ChatHistory
from shared.multinode import ClusterInfo, get_multinode_config
from shared.tools import get_web_access_tool

# Multi-node configuration
MN_CFG = get_multinode_config(default_port=8501, app_name="hf-streamlit-chat")
CLUSTER = ClusterInfo(MN_CFG, app_name="hf-streamlit-chat")
WEB = get_web_access_tool("hf-streamlit-chat")


# Configuration
MODEL_NAME = os.getenv("HF_MODEL_NAME", "microsoft/DialoGPT-medium")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize managers
auth_manager = AuthManager()
chat_history = ChatHistory()


@st.cache_resource
def load_model():
    """Load and cache the HuggingFace model"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.to(DEVICE)
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def init_session_state():
    """Initialize session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'session_token' not in st.session_state:
        st.session_state.session_token = None
    if 'chat_history_ids' not in st.session_state:
        st.session_state.chat_history_ids = None

def login_page():
    """Display login/register page"""
    st.title("🔐 Login / Register")
    
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
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
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

def generate_response(prompt, tokenizer, model, web_context: str = ""):
    """Generate response using HuggingFace model"""
    try:
        # Optionally prepend fetched web page context (full pages already logged)
        model_prompt = prompt
        if web_context:
            model_prompt = (
                f"{web_context}\n\n"
                f"User question (may reference the URL(s) above):\n{prompt}"
            )
        # Encode the new user input, add the eos_token and return a tensor in Pytorch
        new_input_ids = tokenizer.encode(
            model_prompt + tokenizer.eos_token,
            return_tensors='pt'
        ).to(DEVICE)

        
        # Append the new user input tokens to the chat history
        bot_input_ids = torch.cat(
            [st.session_state.chat_history_ids, new_input_ids], 
            dim=-1
        ) if st.session_state.chat_history_ids is not None else new_input_ids
        
        # Generate a response
        st.session_state.chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        # Decode the response
        response = tokenizer.decode(
            st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

def chat_page():
    """Display chat interface"""
    st.title("💬 HuggingFace Chat Interface")
    
    # Load model
    tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        st.error("Failed to load model. Please check the model name and try again.")
        return
    
    # Sidebar
    with st.sidebar:
        st.write(f"👤 User: {st.session_state.username}")
        st.write(f"🤖 Model: {MODEL_NAME}")
        st.write(f"💻 Device: {DEVICE}")
        with st.expander("Cluster / node"):
            node = CLUSTER.health()
            st.caption(f"node: `{node['node']['node_id']}`")
            st.caption(f"multi_node: `{node['node']['multi_node']}`")
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
            st.session_state.chat_history_ids = None
            st.rerun()
        
        # Show conversation history for logged-in users
        if st.session_state.user_id:
            st.subheader("Previous Conversations")
            conversations = chat_history.get_user_conversations(
                st.session_state.user_id, 
                "hf-streamlit-chat"
            )
            
            for conv in conversations[:10]:  # Show last 10
                if st.button(f"Conv {conv['id']} - {conv['updated_at'][:16]}", key=f"conv_{conv['id']}"):
                    st.session_state.conversation_id = conv['id']
                    messages = chat_history.get_conversation_history(conv['id'])
                    st.session_state.messages = [
                        {"role": msg['role'], "content": msg['content']} 
                        for msg in messages
                    ]
                    st.session_state.chat_history_ids = None  # Reset model history
                    st.rerun()
    
    # Initialize conversation if needed
    if st.session_state.conversation_id is None and st.session_state.user_id:
        st.session_state.conversation_id = chat_history.create_conversation(
            st.session_state.user_id,
            "hf-streamlit-chat"
        )
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Save to history if logged in
        if st.session_state.conversation_id:
            chat_history.add_message(
                st.session_state.conversation_id,
                "user",
                prompt
            )
        
        # Fetch any URLs (full content logged server-side, not only model context)
        session_key = str(
            st.session_state.conversation_id
            or st.session_state.user_id
            or st.session_state.username
            or "guest"
        )
        web_results, web_context = WEB.process_user_text(prompt, session_id=session_key)
        if web_results:
            with st.expander(f"🌐 Fetched {len(web_results)} URL(s) (full content logged)"):
                for wr in web_results:
                    st.markdown(
                        f"- `{wr.url}` — "
                        f"{'OK' if wr.ok else 'FAIL'} "
                        f"status={wr.status_code} "
                        f"log=`{wr.log_path}`"
                    )

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(
                    prompt, tokenizer, model, web_context=web_context
                )
                st.markdown(response)

        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Save to history if logged in
        if st.session_state.conversation_id:
            chat_history.add_message(
                st.session_state.conversation_id,
                "assistant",
                response
            )
        
        st.rerun()

def main():
    st.set_page_config(
        page_title="HuggingFace Chat Interface",
        page_icon="🤗",
        layout="wide"
    )
    
    init_session_state()
    
    if not st.session_state.authenticated:
        login_page()
    else:
        chat_page()

if __name__ == "__main__":
    main()
