"""
NIM Model with Streamlit Chat Interface
Supports optional login and chat history
"""
import streamlit as st
import requests
import sys
import os

# Add parent directory to path for shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.auth import AuthManager
from shared.database import ChatHistory

# Configuration
NIM_API_URL = os.getenv("NIM_API_URL", "http://localhost:8000/v1/chat/completions")
MODEL_NAME = os.getenv("NIM_MODEL_NAME", "meta/llama-3.1-8b-instruct")

# Initialize managers
auth_manager = AuthManager()
chat_history = ChatHistory()

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

def query_nim_model(messages):
    """Query the NIM model API"""
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        response = requests.post(NIM_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

def chat_page():
    """Display chat interface"""
    st.title("💬 NIM Chat Interface")
    
    # Sidebar
    with st.sidebar:
        st.write(f"👤 User: {st.session_state.username}")
        
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
        
        # Show conversation history for logged-in users
        if st.session_state.user_id:
            st.subheader("Previous Conversations")
            conversations = chat_history.get_user_conversations(
                st.session_state.user_id, 
                "nim-streamlit-chat"
            )
            
            for conv in conversations[:10]:  # Show last 10
                if st.button(f"Conv {conv['id']} - {conv['updated_at'][:16]}", key=f"conv_{conv['id']}"):
                    st.session_state.conversation_id = conv['id']
                    messages = chat_history.get_conversation_history(conv['id'])
                    st.session_state.messages = [
                        {"role": msg['role'], "content": msg['content']} 
                        for msg in messages
                    ]
                    st.rerun()
    
    # Initialize conversation if needed
    if st.session_state.conversation_id is None and st.session_state.user_id:
        st.session_state.conversation_id = chat_history.create_conversation(
            st.session_state.user_id,
            "nim-streamlit-chat"
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
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_nim_model(st.session_state.messages)
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
        page_title="NIM Chat Interface",
        page_icon="💬",
        layout="wide"
    )
    
    init_session_state()
    
    if not st.session_state.authenticated:
        login_page()
    else:
        chat_page()

if __name__ == "__main__":
    main()
