# app/config_enhanced.py
import os
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Core API Configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Optional API keys for enhanced features
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")  # For web search
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")  # Alternative web search

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "CACHE"
INDEX_DIR = PROJECT_ROOT / "INDEX"
FEATURE_DIR = INDEX_DIR / "features"
SESSIONS_DIR = PROJECT_ROOT / "sessions"
TEMP_UPLOADS_DIR = PROJECT_ROOT / "temp_uploads"

# Create directories
for dir_path in [CACHE_DIR, INDEX_DIR, FEATURE_DIR, SESSIONS_DIR, TEMP_UPLOADS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data files
COMMENTS_CSV = DATA_DIR / "comments_youtube.csv"
POSTS_CSV = DATA_DIR / "posts_youtube.csv"

# Index files
FAISS_TEXT = INDEX_DIR / "faiss_text.bin"
FAISS_IMAGE = INDEX_DIR / "faiss_image.bin"
FAISS_VIDEO = INDEX_DIR / "faiss_video.bin"  # Future use
META_JSONL = INDEX_DIR / "meta.jsonl"
MANIFEST_JSON = INDEX_DIR / "manifest.json"

# Feature files for charting
FEATURE_COMMENTS = FEATURE_DIR / "comments.parquet"
FEATURE_POSTS = FEATURE_DIR / "posts.parquet"

# CLIP Configuration
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"

# Retrieval Configuration
TOP_K_DEFAULT = 8
MODALITY_BOOSTS = {"image": 0.08, "video": 0.04, "text": 0.0}

# Enhanced Features Configuration
MAX_IMAGES_TO_SEND = 6
MAX_CHART_RETRIES = 3
MAX_FILE_SIZE_MB = 50
ENABLE_OCR = True
ENABLE_EXTERNAL_DATA = True

# Ingestion Configuration
TEXT_BATCH = int(os.getenv("MM_TEXT_BATCH", "64"))
IMAGE_BATCH = int(os.getenv("MM_IMAGE_BATCH", "16"))
CHUNK_WORDS = 60
OVERLAP_WORDS = 20
TIMEOUT_SECONDS = (10, 20)  # (connect, read) for HTTP requests

# Chart Configuration
ALLOWED_DATASETS = {"comments", "posts"}
ALLOWED_CHARTS = {"bar", "line", "scatter"}
ALLOWED_TIME_BINS = {"D", "W", "M"}  # Daily, Weekly, Monthly
DEFAULT_CHART_DPI = 110
MAX_CHART_ITEMS = 20  # Max items to show in charts

# File Processing Configuration
SUPPORTED_FILE_TYPES = {
    'text': ['.txt', '.md', '.log', '.py', '.js', '.css', '.html', '.xml'],
    'pdf': ['.pdf'],
    'image': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'],
    'spreadsheet': ['.csv', '.xlsx', '.xls'],
    'document': ['.docx'],
    'data': ['.json', '.jsonl']
}

# PDF Export Configuration
PDF_MAX_PAGES_PER_CONVERSATION = 50
PDF_INCLUDE_IMAGES = True
PDF_INCLUDE_CHARTS = True

# External Data Configuration
EXTERNAL_DATA_SOURCES = {
    'duckduckgo': True,
    'wikipedia': True,
    'arxiv': True,
    'news_feeds': True,
    'reddit': False  # Can be enabled if needed
}

# News RSS Feeds for external data
NEWS_RSS_FEEDS = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.cnn.com/rss/edition.rss", 
    "https://feeds.reuters.com/reuters/topNews",
    "https://feeds.npr.org/1001/rss.xml"
]

# Session Management
MAX_SESSIONS_PER_USER = 50
SESSION_CLEANUP_DAYS = 30
AUTO_SAVE_INTERVAL = 30  # seconds

# UI Configuration
STREAMLIT_CONFIG = {
    'page_title': "Multimodal AI Research Assistant",
    'page_icon': "🤖",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}

# Chat Configuration
MAX_CHAT_HISTORY = 50  # Maximum messages to keep in memory
MAX_MESSAGE_LENGTH = 10000
TYPING_DELAY = 0.05  # For streaming effect

# Image Analysis Configuration
IMAGE_ANALYSIS_PROVIDERS = {
    'clip': True,
    'openai_vision': True,
    'opencv': True,
    'basic_properties': True
}

# OCR Configuration (if pytesseract is available)
OCR_LANGUAGES = ['eng']  # Can add more: ['eng', 'fra', 'deu', etc.]
OCR_CONFIG = '--psm 6'  # Page segmentation mode

# Performance Configuration
MAX_CONCURRENT_REQUESTS = 3
REQUEST_TIMEOUT = 30
CACHE_TTL_HOURS = 24

# Security Configuration
ALLOWED_FILE_EXTENSIONS = set()
for ext_list in SUPPORTED_FILE_TYPES.values():
    ALLOWED_FILE_EXTENSIONS.update(ext_list)

BLOCKED_FILE_PATTERNS = [
    '*.exe', '*.bat', '*.sh', '*.cmd', '*.com', '*.scr', '*.pif'
]

# Debug Configuration
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
VERBOSE_LOGGING = DEBUG_MODE

# Feature Flags
FEATURE_FLAGS = {
    'enhanced_image_analysis': True,
    'external_data_integration': ENABLE_EXTERNAL_DATA,
    'advanced_chart_generation': True,
    'multi_file_upload': True,
    'session_management': True,
    'pdf_export': True,
    'real_time_search': False,  # Future feature
    'voice_input': False,  # Future feature
    'collaborative_sessions': False  # Future feature
}

# Error Messages
ERROR_MESSAGES = {
    'file_too_large': f'File size exceeds {MAX_FILE_SIZE_MB}MB limit',
    'unsupported_file': 'File type not supported for processing',
    'processing_failed': 'File processing failed. Please try again.',
    'no_openai_key': 'OpenAI API key not configured',
    'retrieval_failed': 'Failed to retrieve relevant information',
    'chart_generation_failed': 'Chart generation failed. Please check your request.',
    'external_data_failed': 'External data retrieval failed',
    'session_save_failed': 'Failed to save session data'
}

# Success Messages
SUCCESS_MESSAGES = {
    'file_processed': 'File processed successfully',
    'chart_generated': 'Chart generated successfully', 
    'session_saved': 'Session saved successfully',
    'pdf_exported': 'PDF report exported successfully',
    'analysis_complete': 'Analysis completed successfully'
}

def validate_configuration():
    """Validate that required configuration is present"""
    issues = []
    
    if not OPENAI_API_KEY:
        issues.append("OPENAI_API_KEY not set - core functionality will be limited")
    
    if not COMMENTS_CSV.exists():
        issues.append(f"Comments CSV not found: {COMMENTS_CSV}")
    
    if not POSTS_CSV.exists():
        issues.append(f"Posts CSV not found: {POSTS_CSV}")
    
    return issues

def get_feature_status():
    """Get status of optional features based on available dependencies"""
    status = {}
    
    # Check OpenAI API
    status['openai_api'] = bool(OPENAI_API_KEY)
    
    # Check external search APIs
    status['external_search'] = bool(TAVILY_API_KEY or SERPAPI_API_KEY)
    
    # Check optional dependencies
    try:
        import PyPDF2
        import fitz
        status['pdf_processing'] = True
    except ImportError:
        status['pdf_processing'] = False
    
    try:
        import pytesseract
        status['ocr'] = True
    except ImportError:
        status['ocr'] = False
    
    try:
        import cv2
        status['opencv'] = True
    except ImportError:
        status['opencv'] = False
    
    try:
        import docx
        status['docx_processing'] = True
    except ImportError:
        status['docx_processing'] = False
    
    return status

if __name__ == "__main__":
    # Configuration validation
    issues = validate_configuration()
    if issues:
        print("Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration validated successfully")
    
    # Feature status
    print("\nFeature Status:")
    status = get_feature_status()
    for feature, available in status.items():
        status_text = "✅" if available else "❌"
        print(f"  {feature}: {status_text}")