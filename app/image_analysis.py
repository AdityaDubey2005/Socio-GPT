# app/image_analysis.py
import os
import base64
from typing import Optional, Dict, Any, List
from PIL import Image, ImageStat
import torch
import open_clip
import numpy as np
from pathlib import Path
import cv2

from .config import CLIP_MODEL, CLIP_PRETRAINED, CLIP_DEVICE, OPENAI_API_KEY
from openai import OpenAI

# Initialize OpenAI client for vision API
client = OpenAI(api_key=OPENAI_API_KEY)

def analyze_uploaded_image(image_path: str) -> str:
    """
    Comprehensive analysis of uploaded image using multiple approaches
    """
    try:
        # 1. Basic image properties
        basic_analysis = get_basic_image_properties(image_path)
        
        # 2. CLIP-based analysis
        clip_analysis = analyze_with_clip(image_path)
        
        # 3. OpenAI Vision API analysis (if available)
        vision_analysis = analyze_with_openai_vision(image_path)
        
        # 4. Computer vision analysis
        cv_analysis = analyze_with_opencv(image_path)
        
        # Combine all analyses
        combined_analysis = compile_image_analysis(
            basic_analysis, clip_analysis, vision_analysis, cv_analysis
        )
        
        return combined_analysis
        
    except Exception as e:
        return f"Image analysis failed: {str(e)}"

def get_basic_image_properties(image_path: str) -> Dict[str, Any]:
    """
    Extract basic properties of the image
    """
    try:
        with Image.open(image_path) as img:
            # Basic properties
            properties = {
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height
            }
            
            # Color analysis
            if img.mode in ['RGB', 'RGBA']:
                stat = ImageStat.Stat(img)
                properties['average_color'] = {
                    'r': int(stat.mean[0]),
                    'g': int(stat.mean[1]),
                    'b': int(stat.mean[2])
                }
                
                # Brightness
                brightness = sum(stat.mean[:3]) / 3
                properties['brightness'] = 'bright' if brightness > 127 else 'dark'
                
                # Dominant colors (simplified)
                properties['color_palette'] = extract_dominant_colors(img)
            
            return properties
            
    except Exception as e:
        return {'error': str(e)}

def extract_dominant_colors(img: Image.Image, k: int = 3) -> List[str]:
    """
    Extract dominant colors from image
    """
    try:
        # Convert to RGB and resize for processing
        img_rgb = img.convert('RGB')
        img_small = img_rgb.resize((50, 50))
        
        # Get pixel data
        pixels = np.array(img_small).reshape(-1, 3)
        
        # Simple clustering to find dominant colors
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = []
        for color in kmeans.cluster_centers_:
            r, g, b = [int(c) for c in color]
            colors.append(f"rgb({r},{g},{b})")
        
        return colors
        
    except ImportError:
        # Fallback without sklearn
        return ['rgb(128,128,128)']  # Default gray
    except Exception as e:
        return ['unknown']

def analyze_with_clip(image_path: str) -> Dict[str, Any]:
    """
    Analyze image using CLIP embeddings and predefined categories
    """
    try:
        # Load CLIP model
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=CLIP_DEVICE
        )
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        model.eval()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(CLIP_DEVICE)
        
        # Predefined categories for classification
        categories = [
            "a photo of a person", "a photo of people", "a group photo",
            "a landscape photo", "a nature photo", "a cityscape",
            "a building", "architecture", "a street scene",
            "food", "a meal", "cooking",
            "an animal", "a pet", "wildlife",
            "a vehicle", "a car", "transportation",
            "technology", "a computer", "electronics",
            "art", "a painting", "artwork",
            "text", "a document", "writing",
            "sports", "exercise", "activity",
            "a screenshot", "user interface", "website",
            "a chart", "a graph", "data visualization",
            "indoor scene", "outdoor scene", "interior",
            "a selfie", "a portrait", "closeup",
            "abstract", "pattern", "texture"
        ]
        
        # Compute similarities
        text_tokens = tokenizer(categories)
        
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tokens.to(CLIP_DEVICE))
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarities
            similarities = (image_features @ text_features.T).cpu().numpy()[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:5]
        top_categories = [(categories[i], float(similarities[i])) for i in top_indices]
        
        return {
            'top_categories': top_categories,
            'primary_category': top_categories[0][0] if top_categories else 'unknown',
            'confidence_score': float(top_categories[0][1]) if top_categories else 0.0
        }
        
    except Exception as e:
        return {'error': str(e)}

def analyze_with_openai_vision(image_path: str) -> Dict[str, Any]:
    """
    Analyze image using OpenAI's Vision API (GPT-4V)
    """
    try:
        if not OPENAI_API_KEY:
            return {'error': 'OpenAI API key not available'}
        
        # Encode image to base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Call OpenAI Vision API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this image in detail. Provide:
                            1. A concise description of what you see
                            2. Key objects or elements present
                            3. The setting or context
                            4. Any notable features or characteristics
                            5. Potential relevance or significance
                            
                            Format as a clear, structured analysis."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        analysis_text = response.choices[0].message.content
        
        return {
            'detailed_description': analysis_text,
            'source': 'openai_vision'
        }
        
    except Exception as e:
        return {'error': f'OpenAI Vision API error: {str(e)}'}

def analyze_with_opencv(image_path: str) -> Dict[str, Any]:
    """
    Analyze image using OpenCV for technical features
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return {'error': 'Could not load image with OpenCV'}
        
        # Convert to RGB for consistency
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        analysis = {}
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        analysis['edge_density'] = float(edge_density)
        analysis['complexity'] = 'high' if edge_density > 0.1 else 'medium' if edge_density > 0.05 else 'low'
        
        # Blur detection (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        analysis['blur_score'] = float(laplacian_var)
        analysis['image_quality'] = 'sharp' if laplacian_var > 500 else 'moderate' if laplacian_var > 100 else 'blurry'
        
        # Color histogram analysis
        hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
        
        # Color distribution
        analysis['color_distribution'] = {
            'blue_dominant': bool(np.argmax(hist_b) > 128),
            'green_dominant': bool(np.argmax(hist_g) > 128),
            'red_dominant': bool(np.argmax(hist_r) > 128)
        }
        
        # Contour analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        analysis['object_count'] = len([c for c in contours if cv2.contourArea(c) > 100])
        
        return analysis
        
    except ImportError:
        return {'error': 'OpenCV not available'}
    except Exception as e:
        return {'error': f'OpenCV analysis error: {str(e)}'}

def compile_image_analysis(
    basic: Dict[str, Any],
    clip: Dict[str, Any], 
    vision: Dict[str, Any],
    opencv: Dict[str, Any]
) -> str:
    """
    Compile all analysis results into a coherent summary
    """
    analysis_parts = []
    
    # Start with basic description
    if vision.get('detailed_description'):
        analysis_parts.append("Visual Analysis:")
        analysis_parts.append(vision['detailed_description'])
        analysis_parts.append("")
    
    # Add CLIP classification
    if clip.get('top_categories'):
        analysis_parts.append("Content Classification:")
        primary_category = clip['primary_category'].replace('a photo of ', '').replace('a ', '')
        confidence = clip['confidence_score']
        analysis_parts.append(f"Primary category: {primary_category} (confidence: {confidence:.2f})")
        
        if len(clip['top_categories']) > 1:
            other_categories = [cat.replace('a photo of ', '').replace('a ', '') 
                             for cat, _ in clip['top_categories'][1:3]]
            analysis_parts.append(f"Alternative categories: {', '.join(other_categories)}")
        analysis_parts.append("")
    
    # Add technical properties
    if basic and not basic.get('error'):
        analysis_parts.append("Technical Properties:")
        analysis_parts.append(f"Dimensions: {basic['width']}x{basic['height']} pixels")
        analysis_parts.append(f"Format: {basic.get('format', 'Unknown')}")
        
        if basic.get('brightness'):
            analysis_parts.append(f"Brightness: {basic['brightness']}")
        
        if basic.get('color_palette'):
            analysis_parts.append(f"Dominant colors: {len(basic['color_palette'])} identified")
        
        analysis_parts.append("")
    
    # Add OpenCV technical analysis
    if opencv and not opencv.get('error'):
        analysis_parts.append("Image Quality Analysis:")
        analysis_parts.append(f"Complexity: {opencv.get('complexity', 'unknown')}")
        analysis_parts.append(f"Quality: {opencv.get('image_quality', 'unknown')}")
        
        if opencv.get('object_count'):
            analysis_parts.append(f"Detected objects/regions: {opencv['object_count']}")
        
        analysis_parts.append("")
    
    # Add any errors encountered
    errors = []
    for source, data in [('Basic', basic), ('CLIP', clip), ('Vision', vision), ('OpenCV', opencv)]:
        if data.get('error'):
            errors.append(f"{source}: {data['error']}")
    
    if errors:
        analysis_parts.append("Analysis Limitations:")
        analysis_parts.extend(errors)
    
    return '\n'.join(analysis_parts) if analysis_parts else "Image analysis could not be completed."

def generate_search_keywords_from_image(image_path: str) -> List[str]:
    """
    Generate search keywords based on image analysis for better dataset retrieval
    """
    try:
        # Get CLIP analysis for categories
        clip_analysis = analyze_with_clip(image_path)
        
        keywords = []
        
        if clip_analysis.get('top_categories'):
            for category, confidence in clip_analysis['top_categories'][:3]:
                if confidence > 0.2:  # Only include confident matches
                    # Extract key terms from category descriptions
                    category_clean = category.replace('a photo of ', '').replace('a ', '')
                    keywords.extend(category_clean.split())
        
        # Add generic visual terms
        keywords.extend(['image', 'visual', 'photo'])
        
        # Remove duplicates and return
        return list(dict.fromkeys(keywords))[:10]
        
    except Exception as e:
        return ['image', 'visual', 'photo']

def is_text_heavy_image(image_path: str) -> bool:
    """
    Determine if image contains significant text content
    """
    try:
        # Use OpenCV to detect text-like regions
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations to detect text-like structures
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Find contours that might be text
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_like_contours = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            # Text-like characteristics: reasonable aspect ratio and size
            if 0.2 <= aspect_ratio <= 10 and 100 <= area <= 10000:
                text_like_contours += 1
        
        # If many text-like contours found, likely contains significant text
        return text_like_contours > 10
        
    except Exception:
        return False