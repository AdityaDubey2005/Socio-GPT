# Socio-GPT 🤖

A powerful AI-powered research tool that combines text, images, and data visualization to help you analyze and explore your YouTube posts and comments dataset. Think of it as your personal research assistant that can understand both text and images, create charts, and even search the web for additional context.

## What Does This Do?

This system helps you:
- **Chat naturally** with your data using AI
- **Upload images** and get detailed analysis of what's in them  
- **Ask for charts** and get automatic visualizations of your data
- **Search through thousands** of posts and comments instantly
- **Get visual results** - see related images from your dataset
- **Export professional reports** as PDF documents
- **Manage multiple conversations** like ChatGPT
- **Get current information** from the web when needed

## Key Features 🌟

### 1. Smart Conversational Interface
- **Claude-like chat experience** - Clean, modern interface
- **Multiple chat sessions** - Start new conversations or continue old ones
- **Session management** - Save, load, and organize your research sessions
- **Real-time responses** - Get immediate answers to your questions

### 2. Multimodal Understanding  
- **Text Analysis** - Search through all your posts and comments
- **Image Analysis** - Upload any image and get detailed descriptions
- **Visual Search** - Find related images in your dataset
- **File Processing** - Upload PDFs, Word docs, Excel files, and more

### 3. Intelligent Data Visualization
- **Automatic Chart Generation** - Just ask "show me trends over time"
- **Multiple Chart Types** - Bar charts, line graphs, scatter plots
- **Smart Data Insights** - Hashtag analysis, sentiment trends, engagement patterns
- **Interactive Visualizations** - Embedded directly in your chat

### 4. Advanced Search Capabilities
- **Semantic Search** - Find content by meaning, not just keywords
- **Image-to-Image Search** - Upload a photo to find similar ones
- **Mixed Results** - Get both text and image results together
- **Evidence-Based** - See exactly which sources support each answer

### 5. External Data Integration
- **Web Search** - Get current information without API keys
- **News Integration** - Latest news from BBC, CNN, Reuters
- **Academic Sources** - Research papers from arXiv
- **Wikipedia** - Background information and definitions

### 6. Professional PDF Reports
- **Research-Style Export** - Academic format with methodology
- **Include Charts** - All visualizations embedded
- **Evidence Citations** - Proper source attribution
- **Executive Summary** - Auto-generated insights

## How It Works (Simple Explanation)

1. **Data Setup**: Your YouTube posts and comments are processed and indexed
2. **AI Understanding**: The system creates "embeddings" (AI representations) of your content
3. **Fast Search**: When you ask questions, it quickly finds relevant information
4. **Smart Responses**: An AI agent decides what tools to use and combines results
5. **Visual Display**: Results are shown as text, images, and charts in a chat interface

## What You Can Ask

### Text Questions
- "What are the most discussed topics in my posts?"
- "Find comments about education"
- "Show me positive sentiment comments"
- "What hashtags perform best?"

### Visual Questions  
- "Show me images related to technology"
- "Find photos similar to this one I uploaded"
- "What images get the most engagement?"

### Chart Questions
- "Plot hashtag trends over time"
- "Show me comment volume by month"
- "Create a chart of sentiment distribution"
- "Visualize engagement patterns"

### Analysis Questions
- "Analyze this uploaded image"
- "What's trending in my niche recently?"
- "Compare posts from different time periods"
- "Find patterns in successful content"

## Technical Architecture

### Core Components
- **CLIP Embeddings** - Converts text and images into searchable vectors
- **FAISS Indexing** - Lightning-fast similarity search
- **GPT-4 Agent** - Intelligent decision making and response generation
- **Streamlit UI** - Modern web interface
- **Multiple Tools** - Text search, image search, chart generation, web search

### Data Flow
```
Your Data → Processing → Embeddings → FAISS Index → Search → AI Agent → Results
```

## File Structure
```
Multimodal-ai/
├── app/                     # Main application code
│   ├── agents.py           # AI agent logic (ENHANCED)
│   ├── app.py             # Streamlit interface (ENHANCED)  
│   ├── charts.py          # Chart generation
│   ├── config.py          # Configuration settings
│   ├── external_data.py   # Web search without API keys (NEW)
│   ├── file_processors.py # Handle PDFs, images, docs (NEW)
│   ├── image_analysis.py  # Advanced image understanding (NEW)
│   ├── ingest.py          # Data processing pipeline
│   ├── pdf_export.py      # Research report generation (NEW)
│   ├── search.py          # Search functionality
│   ├── tools.py           # LangChain tools
│   └── utils.py           # Helper functions
├── data/                   # Your CSV files
├── CACHE/                 # Downloaded images
├── INDEX/                 # Search indexes
├── sessions/              # Saved chat sessions (NEW)
└── temp_uploads/          # Temporary file storage (NEW)
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (required)
- At least 8GB RAM recommended

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configuration
Create a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here

# Optional (for enhanced web search)
TAVILY_API_KEY=your_tavily_key_here
SERPAPI_API_KEY=your_serpapi_key_here
```

### Step 3: Prepare Your Data
Place your CSV files in the `data/` folder:
- `posts_youtube.csv` 
- `comments_youtube.csv`

### Step 4: Process Your Data (One Time)
```bash
python -m app.ingest
```
This creates searchable indexes from your data.

### Step 5: Launch the Application
```bash
streamlit run run.py
```

Visit `http://localhost:8501` in your browser.

## New Features Added

### Enhanced UI
- **Modern Design** - Professional, Claude-like interface
- **Session Management** - Multiple conversations like ChatGPT
- **Better Image Display** - Gallery view with hover effects
- **Status Indicators** - See system health and data availability
- **Improved Navigation** - Intuitive sidebar and controls

### Fixed Issues
- **Image Understanding** - Now properly analyzes uploaded images
- **Chart Generation** - Fixed plotting functionality with better error handling
- **File Upload** - Support for multiple file types (PDF, Word, Excel, etc.)
- **Performance** - Better memory management and faster responses

### Advanced Capabilities
- **Multi-File Processing** - Handle various document formats
- **OCR Support** - Extract text from images
- **External Context** - Web search without requiring API keys
- **Professional Reports** - Academic-style PDF export
- **Enhanced Analysis** - Deeper insights with multiple AI models

## Usage Examples

### Basic Research
```
User: "What topics do people discuss most in my comments?"
AI: Based on analysis of 1,247 comments, the top discussion topics are:
    • Educational content (23% of comments)
    • Technology tutorials (18% of comments)  
    • Career advice requests (15% of comments)
    [Shows evidence sources and related images]
```

### Visual Analysis
```
User: [Uploads screenshot] "Analyze this interface design"
AI: This appears to be a mobile app interface with:
    • Clean, minimalist design using blue/white color scheme
    • Navigation tabs at bottom
    • Card-based content layout
    
    Finding similar designs in your dataset...
    [Shows related UI screenshots from your posts]
```

### Data Visualization  
```
User: "Show me hashtag performance over the last 6 months"
AI: I've created a line chart showing hashtag trends. Key insights:
    • #programming peaked in March (2.3k mentions)
    • #tutorial shows steady growth (+15% monthly)
    • #career advice surged in May during graduation season
    [Displays interactive chart]
```

## Troubleshooting

### Common Issues

**"No OpenAI API key configured"**
- Add your API key to the `.env` file
- Restart the application

**"Failed to load data"**  
- Check that CSV files are in the `data/` folder
- Run the ingestion process: `python -m app.ingest`

**"Chart generation failed"**
- Ensure your data has the required columns
- Try simpler chart requests first

**"Image analysis not working"**
- Install optional dependencies: `pip install pytesseract opencv-python`
- For OCR, install Tesseract: https://tesseract-ocr.github.io/

### Performance Tips
- **Large Datasets** - Increase batch sizes in config for faster processing
- **Memory Issues** - Reduce `TOP_K_DEFAULT` in config  
- **Slow Responses** - Check internet connection for external data features

## Customization

### Adding New Chart Types
Edit `charts.py` to add new visualization options.

### Custom Data Sources  
Modify `external_data.py` to add new web sources.

### UI Theming
Update CSS in `app.py` to customize the appearance.

### Adding File Types
Extend `file_processors.py` to support new document formats.

## Advanced Features

### API Access (Future)
The system is designed to support API access for programmatic use.

### Collaboration (Planned)
Multi-user sessions and shared research projects.

### Voice Input (Roadmap)  
Speech-to-text for hands-free research.

### Real-time Data (Future)
Live social media monitoring and analysis.

## Support & Contributing

### Getting Help
- Check the troubleshooting section above
- Review configuration in `config.py`
- Look at error messages in the terminal

### Contributing
- Report bugs by describing the issue and steps to reproduce
- Suggest features by explaining the use case
- Submit improvements to any component

## License & Credits

Built on top of:
- **OpenAI GPT-4** for language understanding
- **CLIP** for multimodal embeddings  
- **FAISS** for vector search
- **LangChain** for agent orchestration
- **Streamlit** for the web interface

This enhanced version includes significant improvements to usability, functionality, and reliability compared to the original system.
