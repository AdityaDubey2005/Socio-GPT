# app/external_data.py
import requests
import time
import json
from typing import Optional, Dict, Any, List
from urllib.parse import quote_plus, urljoin
import re
from bs4 import BeautifulSoup
import feedparser
import urllib.robotparser

def search_web_data(query: str, max_results: int = 5) -> Optional[str]:
    """
    Search for external data without requiring API keys
    Uses multiple free sources and web scraping techniques
    """
    results = []
    
    try:
        # 1. Search DuckDuckGo (no API key required)
        ddg_results = search_duckduckgo(query, max_results=3)
        if ddg_results:
            results.extend(ddg_results)
        
        # 2. Search news via RSS feeds
        news_results = search_news_feeds(query, max_results=2)
        if news_results:
            results.extend(news_results)
        
        # 3. Search academic/research sources
        academic_results = search_academic_sources(query, max_results=2)
        if academic_results:
            results.extend(academic_results)
        
        # Compile results into a summary
        if results:
            return compile_external_summary(results, query)
        
    except Exception as e:
        print(f"External data search error: {e}")
    
    return None

def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search DuckDuckGo using their instant answer API
    """
    results = []
    
    try:
        # DuckDuckGo Instant Answer API (free, no key required)
        ddg_url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        response = requests.get(ddg_url, params=params, timeout=10)
        data = response.json()
        
        # Extract abstract/definition
        if data.get('Abstract'):
            results.append({
                'source': 'DuckDuckGo',
                'title': data.get('Heading', query),
                'content': data['Abstract'],
                'url': data.get('AbstractURL', ''),
                'type': 'definition'
            })
        
        # Extract related topics
        for topic in data.get('RelatedTopics', [])[:3]:
            if isinstance(topic, dict) and topic.get('Text'):
                results.append({
                    'source': 'DuckDuckGo',
                    'title': topic.get('Text', '')[:100] + '...',
                    'content': topic.get('Text', ''),
                    'url': topic.get('FirstURL', ''),
                    'type': 'related'
                })
        
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
    
    return results[:max_results]

def search_news_feeds(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search news through RSS feeds (no API keys required)
    """
    results = []
    
    # List of free RSS feeds
    rss_feeds = [
        "http://feeds.bbci.co.uk/news/rss.xml",
        "https://rss.cnn.com/rss/edition.rss",
        "https://feeds.reuters.com/reuters/topNews",
        "https://feeds.npr.org/1001/rss.xml",
        "https://feeds.washingtonpost.com/rss/world"
    ]
    
    query_words = set(query.lower().split())
    
    try:
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:10]:  # Check first 10 entries per feed
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    content = f"{title} {summary}".lower()
                    
                    # Simple relevance check
                    if any(word in content for word in query_words):
                        results.append({
                            'source': feed.feed.get('title', 'News'),
                            'title': title,
                            'content': summary,
                            'url': entry.get('link', ''),
                            'type': 'news',
                            'published': entry.get('published', '')
                        })
                        
                        if len(results) >= max_results:
                            break
                
                if len(results) >= max_results:
                    break
                    
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"RSS feed error for {feed_url}: {e}")
                continue
                
    except Exception as e:
        print(f"News feed search error: {e}")
    
    return results[:max_results]

def search_academic_sources(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Search academic/research sources (arXiv, SSRN, etc.)
    """
    results = []
    
    try:
        # arXiv API (free, no key required)
        arxiv_results = search_arxiv(query, max_results=2)
        if arxiv_results:
            results.extend(arxiv_results)
        
        # Wikipedia API for general knowledge
        wiki_results = search_wikipedia(query, max_results=1)
        if wiki_results:
            results.extend(wiki_results)
            
    except Exception as e:
        print(f"Academic search error: {e}")
    
    return results[:max_results]

def search_arxiv(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Search arXiv for academic papers
    """
    results = []
    
    try:
        arxiv_url = "http://export.arxiv.org/api/query"
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        response = requests.get(arxiv_url, params=params, timeout=15)
        
        if response.status_code == 200:
            # Parse XML response
            from xml.etree import ElementTree as ET
            root = ET.fromstring(response.content)
            
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns)
                summary = entry.find('atom:summary', ns)
                link = entry.find('atom:id', ns)
                
                if title is not None and summary is not None:
                    results.append({
                        'source': 'arXiv',
                        'title': title.text.strip(),
                        'content': summary.text.strip()[:500] + '...',
                        'url': link.text.strip() if link is not None else '',
                        'type': 'academic'
                    })
    
    except Exception as e:
        print(f"arXiv search error: {e}")
    
    return results

def search_wikipedia(query: str, max_results: int = 1) -> List[Dict[str, Any]]:
    """
    Search Wikipedia for general knowledge
    """
    results = []
    
    try:
        # Wikipedia API
        wiki_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        encoded_query = quote_plus(query.replace(' ', '_'))
        
        response = requests.get(f"{wiki_url}{encoded_query}", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('extract'):
                results.append({
                    'source': 'Wikipedia',
                    'title': data.get('title', query),
                    'content': data['extract'],
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    'type': 'encyclopedia'
                })
    
    except Exception as e:
        print(f"Wikipedia search error: {e}")
    
    return results

def compile_external_summary(results: List[Dict[str, Any]], query: str) -> str:
    """
    Compile external search results into a coherent summary
    """
    if not results:
        return None
    
    summary_parts = []
    summary_parts.append(f"External context for '{query}':")
    
    # Group by type
    by_type = {}
    for result in results:
        result_type = result.get('type', 'general')
        if result_type not in by_type:
            by_type[result_type] = []
        by_type[result_type].append(result)
    
    # Add each type
    for result_type, type_results in by_type.items():
        if result_type == 'news':
            summary_parts.append(f"\nRecent News: {len(type_results)} relevant articles found.")
            for result in type_results[:2]:
                summary_parts.append(f"- {result['title']} ({result['source']})")
        
        elif result_type == 'academic':
            summary_parts.append(f"\nAcademic Sources: {len(type_results)} papers found.")
            for result in type_results[:1]:
                summary_parts.append(f"- {result['title']}")
        
        elif result_type == 'definition':
            summary_parts.append(f"\nDefinition: {type_results[0]['content'][:200]}...")
        
        elif result_type == 'encyclopedia':
            summary_parts.append(f"\nBackground: {type_results[0]['content'][:200]}...")
    
    return '\n'.join(summary_parts)

def get_trending_topics() -> List[str]:
    """
    Get trending topics from various sources (no API keys)
    """
    trending = []
    
    try:
        # Get trending from Google Trends (RSS)
        trends_url = "https://trends.google.com/trends/trendingsearches/daily/rss"
        try:
            feed = feedparser.parse(trends_url)
            for entry in feed.entries[:5]:
                trending.append(entry.title)
        except:
            pass
        
        # Get trending from Reddit (RSS)
        reddit_url = "https://www.reddit.com/hot/.rss"
        try:
            feed = feedparser.parse(reddit_url)
            for entry in feed.entries[:3]:
                trending.append(entry.title)
        except:
            pass
            
    except Exception as e:
        print(f"Trending topics error: {e}")
    
    return trending[:10]

def enrich_query_with_context(query: str) -> str:
    """
    Enrich user query with contextual information
    """
    try:
        # Add current date context
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Add trending context if relevant
        if any(word in query.lower() for word in ['trend', 'current', 'latest', 'recent', 'now']):
            trending = get_trending_topics()
            if trending:
                context = f"Current trending topics include: {', '.join(trending[:3])}"
                return f"{query}\n\nAdditional context (as of {current_date}): {context}"
        
        return query
        
    except Exception as e:
        print(f"Query enrichment error: {e}")
        return query