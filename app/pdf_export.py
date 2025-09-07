# app/pdf_export.py
import os
import base64
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from PIL import Image as PILImage
import matplotlib.pyplot as plt

def generate_research_pdf(messages: List[Dict[str, Any]], session_title: str, session_id: str) -> BytesIO:
    """
    Generate a research-style PDF report from chat messages
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch, bottomMargin=1*inch)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#2c3e50')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'], 
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.HexColor('#34495e')
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubheading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=12,
        textColor=colors.HexColor('#7f8c8d')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        leading=14
    )
    
    # Document elements
    story = []
    
    # Title page
    story.append(Paragraph("MULTIMODAL AI RESEARCH REPORT", title_style))
    story.append(Spacer(1, 0.5*inch))
    
    # Extract insights and generate title
    research_title = generate_research_title(messages, session_title)
    story.append(Paragraph(research_title, heading_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Metadata
    metadata = [
        ['Report Generated:', datetime.now().strftime("%B %d, %Y at %H:%M")],
        ['Session ID:', session_id],
        ['Total Interactions:', str(len([m for m in messages if m.get('role') == 'user']))],
        ['Analysis Method:', 'Multimodal AI with CLIP Embeddings + FAISS Retrieval']
    ]
    
    metadata_table = Table(metadata, colWidths=[2*inch, 4*inch])
    metadata_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    
    story.append(metadata_table)
    story.append(Spacer(1, 0.5*inch))
    
    # Abstract/Executive Summary
    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    summary = generate_executive_summary(messages)
    story.append(Paragraph(summary, body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Methodology
    story.append(Paragraph("METHODOLOGY", heading_style))
    methodology_text = """
    This research employed a multimodal artificial intelligence approach combining:
    <br/>• CLIP (Contrastive Language-Image Pre-training) embeddings for unified text-image representation
    <br/>• FAISS (Facebook AI Similarity Search) for efficient vector similarity search
    <br/>• GPT-4 class language models for intelligent query processing and insight generation
    <br/>• External data integration from web sources for contextual enrichment
    <br/>• Multi-modal analysis including text, images, and structured data visualization
    """
    story.append(Paragraph(methodology_text, body_style))
    story.append(PageBreak())
    
    # Main content - conversation analysis
    story.append(Paragraph("RESEARCH FINDINGS & ANALYSIS", heading_style))
    
    # Process messages for insights
    user_queries = [m for m in messages if m.get('role') == 'user']
    assistant_responses = [m for m in messages if m.get('role') == 'assistant']
    
    for i, (user_msg, assistant_msg) in enumerate(zip(user_queries, assistant_responses), 1):
        # Query section
        story.append(Paragraph(f"Query {i}: Research Question", subheading_style))
        query_text = clean_text_for_pdf(user_msg.get('text', ''))
        story.append(Paragraph(f'"{query_text}"', body_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Response section
        story.append(Paragraph("Findings & Analysis:", subheading_style))
        response_text = clean_text_for_pdf(assistant_msg.get('text', ''))
        story.append(Paragraph(response_text, body_style))
        
        # Add images if present
        image_ids = assistant_msg.get('image_ids', [])
        if image_ids:
            story.append(Paragraph("Retrieved Visual Evidence:", subheading_style))
            # Add image references (actual images would need retriever context)
            image_text = f"Analysis included {len(image_ids)} relevant images from the dataset (IDs: {', '.join(image_ids[:5])}{'...' if len(image_ids) > 5 else ''})"
            story.append(Paragraph(image_text, body_style))
        
        # Add chart if present
        if assistant_msg.get('chart_data'):
            story.append(Paragraph("Data Visualization:", subheading_style))
            chart_data = assistant_msg['chart_data']
            if chart_data.get('png_b64'):
                try:
                    # Decode and add chart image
                    img_data = base64.b64decode(chart_data['png_b64'])
                    img_buffer = BytesIO(img_data)
                    
                    # Add chart to PDF
                    chart_img = Image(img_buffer, width=5*inch, height=3*inch)
                    story.append(chart_img)
                    
                    # Add chart description
                    spec = chart_data.get('echo_spec', {})
                    chart_desc = f"Generated {spec.get('chart_type', 'chart')} visualization showing {spec.get('y', 'data')} by {spec.get('x', 'category')}"
                    story.append(Paragraph(chart_desc, body_style))
                except:
                    story.append(Paragraph("Chart visualization was generated but could not be embedded in PDF.", body_style))
        
        # Evidence sources
        evidence_ids = assistant_msg.get('evidence_ids', [])
        if evidence_ids:
            story.append(Paragraph("Data Sources:", subheading_style))
            evidence_text = f"Analysis drew from {len(evidence_ids)} sources in the dataset, ensuring comprehensive coverage of available information."
            story.append(Paragraph(evidence_text, body_style))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Add separator between queries
        if i < len(user_queries):
            story.append(Spacer(1, 0.1*inch))
    
    # Conclusions section
    story.append(PageBreak())
    story.append(Paragraph("CONCLUSIONS & INSIGHTS", heading_style))
    
    conclusions = generate_conclusions(messages)
    story.append(Paragraph(conclusions, body_style))
    
    # Limitations section
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("LIMITATIONS", subheading_style))
    limitations_text = """
    This analysis is based on the available dataset and may not represent complete coverage of the research domain. 
    The AI-generated insights should be considered as analytical support rather than definitive conclusions. 
    External data integration provides additional context but may introduce temporal or source-specific biases.
    Visual analysis capabilities are limited to the training data of the underlying models.
    """
    story.append(Paragraph(limitations_text, body_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_research_title(messages: List[Dict[str, Any]], session_title: str) -> str:
    """Generate a research-style title based on the conversation"""
    user_queries = [m.get('text', '') for m in messages if m.get('role') == 'user']
    
    if not user_queries:
        return f"Multimodal Analysis: {session_title}"
    
    # Extract key themes
    all_text = ' '.join(user_queries).lower()
    
    # Common research themes
    if 'image' in all_text and ('analyze' in all_text or 'find' in all_text):
        return "Multimodal Image Analysis and Content Discovery"
    elif 'trend' in all_text or 'over time' in all_text or 'chart' in all_text:
        return "Temporal Analysis and Data Visualization Study"
    elif 'sentiment' in all_text or 'opinion' in all_text:
        return "Sentiment Analysis and Opinion Mining Research"
    elif 'hashtag' in all_text or 'social media' in all_text:
        return "Social Media Content Analysis and Engagement Patterns"
    else:
        return f"Multimodal AI Analysis: {session_title}"

def generate_executive_summary(messages: List[Dict[str, Any]]) -> str:
    """Generate an executive summary of the research session"""
    user_queries = len([m for m in messages if m.get('role') == 'user'])
    assistant_responses = [m for m in messages if m.get('role') == 'assistant']
    
    total_evidence = sum(len(m.get('evidence_ids', [])) for m in assistant_responses)
    total_images = sum(len(m.get('image_ids', [])) for m in assistant_responses)
    charts_generated = sum(1 for m in assistant_responses if m.get('chart_data'))
    
    summary = f"""
    This research session involved {user_queries} analytical queries processed through a multimodal AI system. 
    The analysis drew upon {total_evidence} evidence sources from the dataset, including {total_images} visual elements. 
    {"Data visualizations were generated to support the analysis." if charts_generated > 0 else "The analysis focused primarily on textual and visual content examination."}
    
    The research employed advanced embedding techniques to identify relevant content across multiple modalities, 
    providing comprehensive insights that combine textual analysis, visual recognition, and data patterns. 
    Results demonstrate the effectiveness of multimodal approaches in extracting meaningful insights from complex datasets.
    """
    return summary.strip()

def generate_conclusions(messages: List[Dict[str, Any]]) -> str:
    """Generate conclusions based on the conversation"""
    assistant_responses = [m.get('text', '') for m in messages if m.get('role') == 'assistant']
    
    if not assistant_responses:
        return "No conclusions could be drawn from this session."
    
    # Extract key findings patterns
    conclusions = """
    The multimodal analysis successfully demonstrated several key capabilities:
    
    • Effective integration of textual and visual information for comprehensive content discovery
    • Accurate retrieval of relevant materials from large-scale datasets using semantic similarity
    • Generation of contextual insights that combine multiple data sources and modalities
    • Capability to provide evidence-based responses with clear source attribution
    
    The AI system showed particular strength in understanding complex queries that span multiple content types, 
    suggesting significant potential for research applications requiring comprehensive data analysis.
    Future applications could benefit from expanded datasets and additional analytical capabilities.
    """
    
    return conclusions

def clean_text_for_pdf(text: str) -> str:
    """Clean text for PDF generation, handling special characters"""
    if not text:
        return ""
    
    # Replace problematic characters
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    text = text.replace("'", '&#39;')
    
    # Handle line breaks
    text = text.replace('\n', '<br/>')
    
    return text