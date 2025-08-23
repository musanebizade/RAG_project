import streamlit as st
import requests
import json
import os
from typing import List, Dict
import time


# ----------------------
# Configuration
# ----------------------
# Use environment variable for Docker, fallback to localhost for local development
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
RAG_QUERY_ENDPOINT = f"{BACKEND_URL}/rag/query"
RAG_STREAM_ENDPOINT = f"{BACKEND_URL}/rag/stream"

# ----------------------
# Streamlit Configuration
# ----------------------
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------
# Helper Functions
# ----------------------
def format_messages_for_api(messages: List[Dict[str, str]]) -> Dict:
    """Format chat messages for the API"""
    api_messages = []
    for msg in messages:
        api_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    return {"messages": api_messages}

def send_query(messages: List[Dict[str, str]]) -> str:
    """Send query to the backend and get response"""
    try:
        payload = format_messages_for_api(messages)
        response = requests.post(RAG_QUERY_ENDPOINT, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        if "error" in result:
            return f"Error: {result['error']}"
        return result.get("response", "No response received")
    
    except requests.exceptions.RequestException as e:
        return f"Connection error: {str(e)}"
    except json.JSONDecodeError:
        return "Error: Invalid response format"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def stream_query(messages: List[Dict[str, str]]):
    """Stream query to the backend and yield chunks"""
    try:
        payload = format_messages_for_api(messages)
        response = requests.post(
            RAG_STREAM_ENDPOINT, 
            json=payload, 
            stream=True, 
            timeout=30
        )
        response.raise_for_status()
        
        for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
            if chunk:
                yield chunk
    
    except requests.exceptions.RequestException as e:
        yield f"Connection error: {str(e)}"
    except Exception as e:
        yield f"Unexpected error: {str(e)}"

# ----------------------
# Initialize Session State
# ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "use_streaming" not in st.session_state:
    st.session_state.use_streaming = True

# ----------------------
# Sidebar Configuration
# ----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Backend URL configuration
    backend_url = st.text_input(
        "Backend URL",
        value=BACKEND_URL,
        help="URL of your FastAPI backend"
    )
    
    if backend_url != BACKEND_URL:
        BACKEND_URL = backend_url
        RAG_QUERY_ENDPOINT = f"{BACKEND_URL}/rag/query"
        RAG_STREAM_ENDPOINT = f"{BACKEND_URL}/rag/stream"
    
    # Show current backend URL for debugging
    st.info(f"Current Backend: {BACKEND_URL}")
    
    # Streaming toggle
    st.session_state.use_streaming = st.toggle(
        "Enable Streaming",
        value=st.session_state.use_streaming,
        help="Stream responses word by word"
    )
    
    # Connection test
    st.subheader("🔗 Connection Test")
    if st.button("Test Backend Connection"):
        try:
            test_response = requests.get(f"{backend_url}/docs", timeout=5)
            if test_response.status_code == 200:
                st.success("✅ Backend is reachable!")
            else:
                st.error(f"❌ Backend returned status: {test_response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Cannot reach backend: {str(e)}")
    
    # Clear chat
    st.subheader("🗑️ Chat Management")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Chat statistics
    if st.session_state.messages:
        st.subheader("📊 Chat Stats")
        total_messages = len(st.session_state.messages)
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.write(f"Total messages: {total_messages}")
        st.write(f"User messages: {user_messages}")
        st.write(f"Assistant messages: {total_messages - user_messages}")
    
    # Export conversation
    st.subheader("💾 Export Chat")
    if st.session_state.messages and st.button("Export Conversation"):
        chat_export = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "messages": st.session_state.messages,
            "total_messages": len(st.session_state.messages)
        }
        st.download_button(
            label="Download JSON",
            data=json.dumps(chat_export, indent=2),
            file_name=f"chat_export_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# ----------------------
# Main Chat Interface
# ----------------------
st.title("🤖 RAG Chat Assistant")
st.markdown("Ask questions and get answers from your knowledge base!")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your knowledge base..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        if st.session_state.use_streaming:
            # Streaming response
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Retrieving from knowledge base..."):
                for chunk in stream_query(st.session_state.messages):
                    full_response += chunk
                    # Display with cursor effect and format markdown properly
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)  # Small delay for visual effect
            
            # Final response without cursor
            message_placeholder.markdown(full_response)
            
        else:
            # Non-streaming response
            with st.spinner("Retrieving from knowledge base and generating response..."):
                response = send_query(st.session_state.messages)
            st.markdown(response)
            full_response = response
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <small>
            Powered by FastAPI RAG Backend | 
            Built with Streamlit | 
            <a href="/docs" target="_blank">API Documentation</a>
        </small>
    </div>
    """,
    unsafe_allow_html=True
)