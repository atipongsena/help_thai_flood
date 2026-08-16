"""
Web Scraper for Thai Flood Relief Posts

This module scrapes flood-related posts from various sources:
- Generic HTML pages (news sites, flood relief pages, etc.)
- X (Twitter) API (if token available)
- Facebook Graph API (if token available)

The scraped data is saved to SQLite database and JSONL file.
"""
import os
import re
import time
import json
import sqlite3
from typing import List, Dict, Optional, Any
from datetime import datetime

import requests
from bs4 import BeautifulSoup

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    FLOOD_HASHTAGS, USER_AGENT, SCRAPE_DELAY_SECONDS,
    DB_PATH, JSONL_PATH, DATA_DIR, URLS_FILE,
    FB_ACCESS_TOKEN
)
from utils.preprocessing import (
    extract_phone_numbers, extract_coordinates, 
    extract_location_line, has_flood_hashtags
)


# =============================================================================
# Database Initialization
# =============================================================================
def init_database(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Initialize SQLite database with posts table
    
    Returns:
        SQLite connection object
    """
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            url TEXT UNIQUE,
            text TEXT,
            hashtags TEXT,
            phones TEXT,
            lat REAL,
            lng REAL,
            location_line TEXT,
            created_at TEXT,
            scraped_at TEXT,
            extra TEXT
        )
    """)
    
    # Create index on url for faster duplicate checking
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_posts_url ON posts(url)
    """)
    
    conn.commit()
    return conn


def save_record(
    conn: sqlite3.Connection,
    record: Dict[str, Any],
    jsonl_path: str = JSONL_PATH
) -> bool:
    """
    Save a record to database and JSONL file
    
    Args:
        conn: SQLite connection
        record: Dict with post data
        jsonl_path: Path to JSONL backup file
    
    Returns:
        True if saved successfully, False if duplicate
    """
    cursor = conn.cursor()
    
    # Check for duplicate URL
    cursor.execute("SELECT id FROM posts WHERE url = ?", (record.get("url"),))
    if cursor.fetchone():
        return False  # Duplicate
    
    # Insert record
    cursor.execute("""
        INSERT INTO posts 
        (source, url, text, hashtags, phones, lat, lng, location_line, created_at, scraped_at, extra)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record.get("source"),
        record.get("url"),
        record.get("text"),
        json.dumps(record.get("hashtags", []), ensure_ascii=False),
        json.dumps(record.get("phones", []), ensure_ascii=False),
        record.get("lat"),
        record.get("lng"),
        record.get("location_line"),
        record.get("created_at"),
        datetime.now().isoformat(),
        json.dumps(record.get("extra", {}), ensure_ascii=False),
    ))
    conn.commit()
    
    # Also save to JSONL for backup
    os.makedirs(os.path.dirname(jsonl_path) if os.path.dirname(jsonl_path) else '.', exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    return True


# =============================================================================
# HTML Scraping (Generic)
# =============================================================================
def fetch_html(url: str, timeout: int = 15) -> str:
    """
    Fetch HTML content from a URL
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
    
    Returns:
        HTML content as string
    """
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


def extract_text_from_html(html: str) -> str:
    """
    Extract main text content from HTML
    
    Args:
        html: Raw HTML string
    
    Returns:
        Cleaned text content
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove script and style elements
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    
    # Get text with line separators
    text = soup.get_text(separator="\n")
    
    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def process_scraped_text(
    text: str,
    url: str,
    source: str = "web",
    hashtags_to_check: List[str] = None
) -> Dict[str, Any]:
    """
    Process scraped text and extract relevant information
    
    Args:
        text: Raw text content
        url: Source URL
        source: Source name (e.g., "facebook", "twitter", "web")
        hashtags_to_check: List of hashtags to look for
    
    Returns:
        Dict with extracted information
    """
    if hashtags_to_check is None:
        hashtags_to_check = FLOOD_HASHTAGS
    
    # Extract information
    phones = extract_phone_numbers(text)
    coordinates = extract_coordinates(text)
    location_line = extract_location_line(text)
    found_hashtags = has_flood_hashtags(text, hashtags_to_check)
    
    record = {
        "source": source,
        "url": url,
        "text": text,
        "hashtags": found_hashtags,
        "phones": phones,
        "lat": coordinates["lat"] if coordinates else None,
        "lng": coordinates["lng"] if coordinates else None,
        "location_line": location_line,
        "created_at": None,
        "extra": {},
    }
    
    return record


def scrape_url(url: str, source: str = "web") -> Optional[Dict[str, Any]]:
    """
    Scrape a single URL and extract flood relief information
    
    Args:
        url: URL to scrape
        source: Source name
    
    Returns:
        Dict with extracted data, or None if failed
    """
    try:
        html = fetch_html(url)
        text = extract_text_from_html(html)
        
        if not text.strip():
            return None
        
        record = process_scraped_text(text, url, source)
        return record
        
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


def scrape_urls_from_file(
    urls_file: str = URLS_FILE,
    source: str = "web",
    delay: float = SCRAPE_DELAY_SECONDS
) -> int:
    """
    Scrape multiple URLs from a file
    
    Args:
        urls_file: Path to file with URLs (one per line)
        source: Source name
        delay: Delay between requests
    
    Returns:
        Number of successfully scraped posts
    """
    # Load URLs
    if not os.path.exists(urls_file):
        print(f"URLs file not found: {urls_file}")
        print(f"Create a file with one URL per line.")
        return 0
    
    with open(urls_file, encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    if not urls:
        print("No URLs found in file.")
        return 0
    
    # Initialize database
    conn = init_database()
    saved_count = 0
    
    for i, url in enumerate(urls, start=1):
        print(f"[{i}/{len(urls)}] Scraping: {url}")
        
        record = scrape_url(url, source)
        
        if record:
            if save_record(conn, record):
                print(f"  → Saved: hashtags={record['hashtags']}, phones={record['phones']}")
                saved_count += 1
            else:
                print(f"  → Duplicate, skipped")
        else:
            print(f"  → Failed to scrape")
        
        # Rate limiting
        if i < len(urls):
            time.sleep(delay)
    
    conn.close()
    print(f"\nTotal saved: {saved_count}/{len(urls)} posts")
    return saved_count


# =============================================================================
# Facebook Graph API Scraping
# =============================================================================
def fetch_facebook_page_posts(
    page_id: str,
    limit: int = 50,
    access_token: str = None
) -> List[Dict[str, Any]]:
    """
    Fetch posts from a Facebook page using Graph API
    
    Args:
        page_id: Facebook page ID
        limit: Maximum number of posts
        access_token: Facebook access token
    
    Returns:
        List of post records
    """
    if access_token is None:
        access_token = FB_ACCESS_TOKEN
    
    if not access_token:
        print("Warning: FB_ACCESS_TOKEN not set. Skipping Facebook API.")
        return []
    
    url = f"https://graph.facebook.com/v19.0/{page_id}/feed"
    params = {
        "access_token": access_token,
        "limit": limit,
        "fields": "message,created_time,permalink_url"
    }
    
    all_records = []
    
    try:
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            print(f"Facebook API error: {response.status_code}")
            return []
        
        data = response.json()
        
        for post in data.get("data", []):
            message = post.get("message", "")
            if not message:
                continue
            
            record = process_scraped_text(
                message,
                post.get("permalink_url", f"fb://{post['id']}"),
                source="facebook"
            )
            record["created_at"] = post.get("created_time")
            record["extra"] = {"fb_post_id": post.get("id")}
            all_records.append(record)
        
        print(f"Fetched {len(all_records)} Facebook posts")
        
    except Exception as e:
        print(f"Facebook API error: {e}")
    
    return all_records


# =============================================================================
# Collect All Posts
# =============================================================================
def collect_all_posts(
    urls_file: str = None,
    use_fb_api: bool = False,
    fb_page_ids: List[str] = None
) -> int:
    """
    Collect posts from all configured sources
    
    Args:
        urls_file: Path to file with URLs to scrape
        use_fb_api: Whether to use Facebook API
        fb_page_ids: List of Facebook page IDs to scrape
    
    Returns:
        Total number of posts saved
    """
    conn = init_database()
    total_saved = 0
    
    # Scrape from URLs file
    if urls_file and os.path.exists(urls_file):
        print("\n=== Scraping from URLs file ===")
        with open(urls_file, encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        
        for i, url in enumerate(urls, start=1):
            print(f"[{i}/{len(urls)}] {url}")
            record = scrape_url(url)
            if record and save_record(conn, record):
                total_saved += 1
            time.sleep(SCRAPE_DELAY_SECONDS)
    
    # Fetch from Facebook API
    if use_fb_api and FB_ACCESS_TOKEN and fb_page_ids:
        print("\n=== Fetching from Facebook API ===")
        for page_id in fb_page_ids:
            fb_records = fetch_facebook_page_posts(page_id)
            for record in fb_records:
                if save_record(conn, record):
                    total_saved += 1
    
    conn.close()
    print(f"\n=== Total posts saved: {total_saved} ===")
    return total_saved


# =============================================================================
# Manual Post Entry
# =============================================================================
def add_manual_post(text: str, source: str = "manual") -> bool:
    """
    Manually add a post to the database
    
    Args:
        text: Post text content
        source: Source identifier
    
    Returns:
        True if saved successfully
    """
    conn = init_database()
    
    record = process_scraped_text(text, f"manual://{datetime.now().isoformat()}", source)
    saved = save_record(conn, record)
    
    conn.close()
    return saved


def add_posts_from_text_file(file_path: str, source: str = "manual") -> int:
    """
    Add posts from a text file (one post per line or separated by empty lines)
    
    Args:
        file_path: Path to text file
        source: Source identifier
    
    Returns:
        Number of posts saved
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return 0
    
    conn = init_database()
    saved_count = 0
    
    with open(file_path, encoding="utf-8") as f:
        content = f.read()
    
    # Split by empty lines or process each non-empty line
    posts = [p.strip() for p in content.split("\n\n") if p.strip()]
    
    if not posts:
        # Try splitting by single newlines
        posts = [p.strip() for p in content.split("\n") if p.strip()]
    
    for i, text in enumerate(posts, start=1):
        record = process_scraped_text(text, f"manual://{i}_{datetime.now().isoformat()}", source)
        if save_record(conn, record):
            saved_count += 1
            print(f"Saved post {i}")
    
    conn.close()
    print(f"Total saved: {saved_count}/{len(posts)}")
    return saved_count


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape flood relief posts")
    parser.add_argument("--urls", "-u", type=str, default=URLS_FILE,
                       help="Path to file with URLs to scrape")
    parser.add_argument("--posts", "-p", type=str,
                       help="Path to text file with posts to add manually")
    parser.add_argument("--fb-pages", type=str, nargs="*",
                       help="Facebook page IDs to scrape")
    
    args = parser.parse_args()
    
    if args.posts:
        add_posts_from_text_file(args.posts)
    else:
        collect_all_posts(
            urls_file=args.urls,
            use_fb_api=bool(args.fb_pages),
            fb_page_ids=args.fb_pages
        )

