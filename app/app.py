import os
import tempfile
import uuid
import json
import pickle
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import base64
from io import BytesIO

import streamlit as st
from PIL import Image
import pandas as pd

from .config import TOP_K_DEFAULT
from .search import Retriever
from .agents import run_agent
from .pdf_export import generate_research_pdf
from .external_data import search_web_data
from .image_analysis import analyze_uploaded_image
from .file_processors import process_uploaded_file
import re

def _strip_inline_base64_images(text: str) -> str:
    """Remove inline base64 image markdown from assistant text"""
    if not text:
        return text
    # Remove Markdown image with base64 content
    text = re.sub(
        r'!\[[^\]]*\]\(data:image\/(?:png|jpeg|jpg);base64,[^)]+\)',
        '',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    # Clean up extra blank lines
    return re.sub(r'\n{3,}', '\n\n', text).strip()


# Enhanced UI Configuration
st.set_page_config(
    page_title="Socio-GPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Claude-like interface
def load_custom_css():
    st.markdown("""
    <style>
    /* Main container styling */
    .main-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 12px;
        line-height: 1.6;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    
    .assistant-message {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #2d3748;
        margin-right: 20%;
    }
    
    /* Sidebar styling */
    .sidebar-content {
        padding: 1rem 0;
    }
    
    .chat-session {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        background: white;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .chat-session:hover {
        background: #f7fafc;
        border-color: #4299e1;
    }
    
    .chat-session.active {
        background: #ebf8ff;
        border-color: #3182ce;
    }
    
    /* Input area styling */
    # .input-container {
    #     position: sticky;
    #     bottom: 0;
    #     background: white;
    #     padding: 1rem 0;
    #     border-top: 1px solid #e2e8f0;
    # }
    
    /* Give the page extra bottom padding so the last chat doesn't touch the sticky bar */
    .block-container { 
        padding-bottom: 120px !important; 
    }

    /* Add a little gap and a subtle shadow above the white sticky input bar */
    .input-container {
        margin-top: 8px;                 /* <-- creates the visible space */
        box-shadow: 0 -2px 10px rgba(0,0,0,0.08);
        z-index: 10;                     /* keep it above content if needed */
        position: sticky;                /* keep your existing sticky behavior */
        bottom: 0;
        background: white;
    }

    /* Optional: ensure the very last chat bubble also has some breathing room */
    .chat-message:last-child {
        margin-bottom: 1.25rem;
    }
    
    /* Evidence panel */
    .evidence-panel {
        background: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Image gallery */
    .image-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .image-card {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .image-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online { background: #48bb78; }
    .status-processing { background: #ed8936; }
    .status-offline { background: #e53e3e; }
    
    /* File upload area */
    .upload-area {
        border: 2px dashed #cbd5e0;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background: #f7fafc;
        transition: all 0.2s;
    }
    
    .upload-area:hover {
        border-color: #4299e1;
        background: #ebf8ff;
    }
    </style>
    """, unsafe_allow_html=True)

# Session management
SESSIONS_DIR = Path("./sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

@st.cache_resource
def get_retriever() -> Retriever:
    return Retriever()

def create_new_session() -> str:
    """Create a new chat session with unique ID"""
    session_id = str(uuid.uuid4())[:8]
    session_data = {
        "id": session_id,
        "title": f"Chat {session_id}",
        "created": datetime.now().isoformat(),
        "messages": [],
        "metadata": {}
    }
    save_session(session_id, session_data)
    return session_id

def save_session(session_id: str, data: dict):
    """Save session data to disk"""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    with open(session_file, 'w') as f:
        json.dump(data, f, indent=2)

def load_session(session_id: str) -> dict:
    """Load session data from disk"""
    session_file = SESSIONS_DIR / f"{session_id}.json"
    if session_file.exists():
        with open(session_file, 'r') as f:
            return json.load(f)
    return None

def get_all_sessions() -> List[dict]:
    """Get all available sessions"""
    sessions = []
    for session_file in SESSIONS_DIR.glob("*.json"):
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
                sessions.append(session_data)
        except:
            continue
    return sorted(sessions, key=lambda x: x.get('created', ''), reverse=True)

def generate_session_title(messages: List[dict]) -> str:
    """Generate a title based on the first user message"""
    if not messages:
        return "New Chat"
    
    first_user_msg = next((msg for msg in messages if msg.get('role') == 'user'), None)
    if first_user_msg:
        text = first_user_msg.get('text', '')[:50]
        return text if text else "New Chat"
    return "New Chat"

def render_sidebar():
    """Render the enhanced sidebar with session management"""
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        # Header
        st.markdown("### 🤖 Socio-GPT")
        
        # New chat button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            new_session_id = create_new_session()
            st.session_state.current_session = new_session_id
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        
        # Chat sessions
        st.markdown("**Recent Chats**")
        sessions = get_all_sessions()
        
        current_session = st.session_state.get('current_session')
        
        for session in sessions[:10]:  # Show last 10 sessions
            session_id = session['id']
            title = session.get('title', f"Chat {session_id}")
            created = datetime.fromisoformat(session['created']).strftime("%m/%d %H:%M")
            
            is_active = session_id == current_session
            button_type = "primary" if is_active else "secondary"
            
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(
                    f"{title[:25]}{'...' if len(title) > 25 else ''}",
                    key=f"session_{session_id}",
                    help=f"Created: {created}",
                    type=button_type,
                    use_container_width=True
                ):
                    st.session_state.current_session = session_id
                    st.session_state.messages = session.get('messages', [])
                    st.rerun()
            
            with col2:
                if st.button("🗑", key=f"delete_{session_id}", help="Delete"):
                    session_file = SESSIONS_DIR / f"{session_id}.json"
                    session_file.unlink(missing_ok=True)
                    if current_session == session_id:
                        st.session_state.current_session = None
                        st.session_state.messages = []
                    st.rerun()
        
        st.markdown("---")
        
        # Settings
        st.markdown("**Settings**")
        st.session_state.top_k = st.slider("Retrieval Top-K", 5, 30, TOP_K_DEFAULT)
        st.session_state.enable_web_search = st.checkbox("Enable Web Search", value=True)
        st.session_state.enable_external_data = st.checkbox("Include External Data", value=True)
        
        # System status
        st.markdown("---")
        st.markdown("**System Status**")
        retriever = get_retriever()
        
        status_text = "🟢 Online" if retriever.text_index is not None else "🔴 Offline"
        st.markdown(f"**Text Index:** {status_text}")
        
        status_image = "🟢 Online" if retriever.image_index is not None else "🔴 Offline"  
        st.markdown(f"**Image Index:** {status_image}")
        
        st.markdown(f"**Items:** {len(retriever.metas):,}")
        
        # Export current chat
        if st.session_state.get('messages'):
            if st.button("📄 Export to PDF", use_container_width=True):
                export_current_chat()
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_chat_message(message: dict, retriever: Retriever):
    """Render a single chat message with enhanced styling"""
    role = message.get('role', 'user')
    
    if role == 'user':
        with st.container():
            st.markdown(
                f'<div class="chat-message user-message">{message.get("text", "")}</div>',
                unsafe_allow_html=True
            )
            
            # Show uploaded file if any
            if message.get('uploaded_file'):
                file_info = message['uploaded_file']
                st.markdown(f"📎 Uploaded: {file_info.get('name', 'Unknown file')}")
                
                # Show image preview if it's an image
                if file_info.get('type', '').startswith('image/') and file_info.get('path'):
                    try:
                        st.image(file_info['path'], width=300, caption="Uploaded image")
                    except:
                        pass
    
    else:  # assistant
        with st.container():
            assistant_text = _strip_inline_base64_images(message.get("text", ""))
            st.markdown(
                f'<div class="chat-message assistant-message">{assistant_text}</div>',
                unsafe_allow_html=True
            )
            
            # Show images if any
            image_ids = message.get('image_ids', [])
            if image_ids:
                render_image_gallery(image_ids, retriever)
            
            # Show chart if any
            if message.get('chart_data'):
                render_chart(message['chart_data'])
            
            # Show evidence panel
            # evidence_ids = message.get('evidence_ids', [])
            # if evidence_ids:
            #     render_evidence_panel(evidence_ids, retriever)

def render_image_gallery(image_ids: List[str], retriever: Retriever):
    """Render image gallery with enhanced styling"""
    if not image_ids:
        return
    
    st.markdown('<div class="image-gallery">', unsafe_allow_html=True)
    
    # Group images in columns
    cols = st.columns(min(3, len(image_ids)))
    by_id = {m.id: m for m in retriever.metas}
    
    for idx, image_id in enumerate(image_ids[:6]):  # Limit to 6 images
        meta = by_id.get(image_id)
        if not meta:
            continue
        
        with cols[idx % 3]:
            # Use container width instead of deprecated use_column_width
            if meta.image_url:
                st.image(meta.image_url, use_container_width=True)
            elif meta.cache_path and os.path.exists(meta.cache_path):
                st.image(meta.cache_path, use_container_width=True)
            
            if meta.text_snippet:
                st.caption(meta.text_snippet[:100] + "..." if len(meta.text_snippet) > 100 else meta.text_snippet)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_evidence_panel(evidence_ids: List[str], retriever: Retriever):
    """Render evidence panel with source information"""
    if not evidence_ids:
        return
    
    with st.expander("📚 Evidence Sources", expanded=False):
        by_id = {m.id: m for m in retriever.metas}
        
        for evidence_id in evidence_ids[:10]:  # Limit to 10 evidence items
            meta = by_id.get(evidence_id)
            if not meta:
                continue
            
            source_icon = "📝" if meta.modality == "text" else "🖼️"
            source_type = f"{meta.source} ({meta.modality})"
            
            st.markdown(f"{source_icon} **{source_type}** - {evidence_id}")
            if meta.text_snippet:
                st.markdown(f"> {meta.text_snippet[:200]}{'...' if len(meta.text_snippet) > 200 else ''}")
            st.markdown("---")

def render_chart(chart_data: dict):
    """Render chart with enhanced container"""
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    if chart_data.get('png_b64'):
        # Decode and display chart
        img_data = base64.b64decode(chart_data['png_b64'])
        img = Image.open(BytesIO(img_data))
        st.image(img, use_container_width=True)
        
        # Show chart specification
        if chart_data.get('echo_spec'):
            with st.expander("Chart Details"):
                st.json(chart_data['echo_spec'])
    
    st.markdown('</div>', unsafe_allow_html=True)

def handle_file_upload(uploaded_file) -> dict:
    """Process uploaded file and return file info"""
    if not uploaded_file:
        return None
    
    # Save uploaded file temporarily
    temp_dir = Path("./temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    file_id = str(uuid.uuid4())[:8]
    file_path = temp_dir / f"{file_id}_{uploaded_file.name}"
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    file_info = {
        "id": file_id,
        "name": uploaded_file.name,
        "type": uploaded_file.type,
        "size": uploaded_file.size,
        "path": str(file_path)
    }
    
    return file_info

def export_current_chat():
    """Export current chat session to PDF"""
    try:
        messages = st.session_state.get('messages', [])
        if not messages:
            st.warning("No messages to export")
            return
        
        current_session = st.session_state.get('current_session', 'unknown')
        session_title = generate_session_title(messages)
        
        pdf_buffer = generate_research_pdf(
            messages=messages,
            session_title=session_title,
            session_id=current_session
        )
        
        # Offer download
        st.download_button(
            label="⬇️ Download Research Report",
            data=pdf_buffer.getvalue(),
            file_name=f"research_report_{current_session}.pdf",
            mime="application/pdf"
        )
        
        st.success("PDF generated successfully!")
        
    except Exception as e:
        st.error(f"Failed to export PDF: {str(e)}")

def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_session" not in st.session_state:
        # Try to load the most recent session or create new one
        sessions = get_all_sessions()
        if sessions:
            latest_session = sessions[0]
            st.session_state.current_session = latest_session['id']
            st.session_state.messages = latest_session.get('messages', [])
        else:
            st.session_state.current_session = create_new_session()
    
    if "top_k" not in st.session_state:
        st.session_state.top_k = TOP_K_DEFAULT
    
    if "enable_web_search" not in st.session_state:
        st.session_state.enable_web_search = True
    
    if "enable_external_data" not in st.session_state:
        st.session_state.enable_external_data = True

def main():
    """Main application function"""
    load_custom_css()
    init_session_state()
    
    # Initialize retriever
    retriever = get_retriever()
    
    # Render sidebar
    render_sidebar()
    
    # Main chat interface
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header
    st.title("🤖 SocioGPT")
    st.markdown("Your intelligent companion for multimodal research and analysis")
    
    # Chat history display
    messages = st.session_state.get('messages', [])
    
    for message in messages:
        render_chat_message(message, retriever)
    
    # Input area
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    # File upload area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.chat_input(
            "Ask anything... (You can request images, charts, or analysis)",
            key="main_input"
        )
    
    with col2:
        uploaded_file = st.file_uploader(
            "Upload",
            type=['png', 'jpg', 'jpeg', 'pdf', 'txt', 'csv'],
            label_visibility="collapsed"
        )
    
    # Handle user input
    if user_input:
        # Process uploaded file if any
        file_info = handle_file_upload(uploaded_file) if uploaded_file else None
        
        # Add user message to history
        user_message = {
            "role": "user",
            "text": user_input,
            "timestamp": datetime.now().isoformat(),
            "uploaded_file": file_info
        }
        st.session_state.messages.append(user_message)
        
        # Process with AI agent
        with st.spinner("🤔 Thinking..."):
            try:
                # Enhanced processing with file analysis
                enhanced_query = user_input
                query_image_path = None
                
                if file_info:
                    if file_info['type'].startswith('image/'):
                        query_image_path = file_info['path']
                        # Add image analysis
                        image_analysis = analyze_uploaded_image(file_info['path'])
                        enhanced_query += f"\n\nImage Analysis: {image_analysis}"
                    
                    elif file_info['type'] == 'application/pdf' or file_info['name'].endswith('.txt'):
                        # Process text-based files
                        file_content = process_uploaded_file(file_info['path'], file_info['type'])
                        enhanced_query += f"\n\nUploaded Content: {file_content[:1000]}"
                
                # Add external data if enabled
                if st.session_state.get('enable_external_data'):
                    external_data = search_web_data(user_input)
                    if external_data:
                        enhanced_query += f"\n\nExternal Context: {external_data[:500]}"
                
                # Run the agent
                response = run_agent(
                    retriever=retriever,
                    user_query=enhanced_query,
                    query_image_path=query_image_path,
                    top_k=st.session_state.get('top_k', TOP_K_DEFAULT)
                )
                
                # Add assistant response
                assistant_message = {
                    "role": "assistant",
                    "text": response.get('answer', 'I apologize, but I encountered an issue processing your request.'),
                    "timestamp": datetime.now().isoformat(),
                    "evidence_ids": response.get('evidence_ids', []),
                    "image_ids": response.get('image_ids', []),
                    "chart_data": response.get('chart_data')  # Chart data if any
                }
                st.session_state.messages.append(assistant_message)
                
                # Update session title if this is the first exchange
                if len(st.session_state.messages) == 2:
                    session_title = generate_session_title(st.session_state.messages)
                    if st.session_state.current_session:
                        session_data = load_session(st.session_state.current_session)
                        if session_data:
                            session_data['title'] = session_title
                            session_data['messages'] = st.session_state.messages
                            save_session(st.session_state.current_session, session_data)
                
                # Save session
                if st.session_state.current_session:
                    session_data = load_session(st.session_state.current_session) or {}
                    session_data['messages'] = st.session_state.messages
                    session_data['last_updated'] = datetime.now().isoformat()
                    save_session(st.session_state.current_session, session_data)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                # Still add an error message to chat
                error_message = {
                    "role": "assistant", 
                    "text": f"I encountered an error while processing your request: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(error_message)
        
        # Rerun to show new messages
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()