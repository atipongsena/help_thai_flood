# Install required packages
# !pip install pandas numpy scikit-learn pythainlp python-dotenv gensim transformers torch datasets joblib uvicorn fastapi

import os
import sys
import json
import re
import html
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
import joblib
import numpy as np
import pandas as pd
import sqlite3

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)


"""
Configuration settings for Thai Flood Relief NLP Pipeline
"""
import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file if exists
load_dotenv()

# =============================================================================
# Hashtags & Keywords for Flood Relief Posts
# =============================================================================
FLOOD_HASHTAGS: List[str] = [
    "น้ำท่วม68",
    "ขอความช่วยเหลือ",
    "น้ำท่วม",
    "อุทกภัย",
    "สงขลา",
    "น้ำท่วมหาดใหญ่",
    "ภัยพิบัติภาคใต้",
    "น้ำท่วม2568",
    "ช่วยด้วย",
    "ขอความช่วยเหลือด่วน",
]

# Keywords that indicate urgency
URGENCY_KEYWORDS: List[str] = [
    "ด่วน",
    "ช่วยด้วย",
    "SOS",
    "ขออพยพ",
    "ติดอยู่",
    "หลังคา",
    "จมน้ำ",
    "ช็อก",
    "เสียชีวิต",
    "บาดเจ็บ",
    "พิการ",
    "ท้อง",
    "ผู้สูงอายุ",
    "เด็ก",
    "ทารก",
]

# =============================================================================
# API Tokens (set via environment variables)
# =============================================================================
FB_ACCESS_TOKEN = os.environ.get("FB_ACCESS_TOKEN", "")

# =============================================================================
# File Paths
# =============================================================================
DATA_DIR = "data"
MODELS_DIR = "models"
DB_PATH = os.path.join(DATA_DIR, "flood_posts.db")
JSONL_PATH = os.path.join(DATA_DIR, "raw_posts.jsonl")
URLS_FILE = os.path.join(DATA_DIR, "urls.txt")

# Dataset files
RAW_CSV_PATH = os.path.join(DATA_DIR, "all_posts_raw.csv")
LABELED_CSV_PATH = os.path.join(DATA_DIR, "all_posts_labeled.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "Train.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "Test.csv")
SOS_JSON_PATH = os.path.join(DATA_DIR, "sos.json")

# Model paths
BERT_MODEL_DIR = os.path.join(MODELS_DIR, "bert_flood_model")
TFIDF_VEC_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm_tfidf.joblib")
W2V_MODEL_PATH = os.path.join(MODELS_DIR, "word2vec.model")

# =============================================================================
# Model Configuration
# =============================================================================
# Thai BERT Model Names (from HuggingFace)
THAI_BERT_MODELS = {
    "wangchanberta": "airesearch/wangchanberta-base-wiki-newmm",
    "wangchanberta_att": "airesearch/wangchanberta-base-att-spm-uncased",
    "phayathaibert": "clicknext/phayathaibert",
}

# Default model to use
DEFAULT_BERT_MODEL = "airesearch/wangchanberta-base-att-spm-uncased"

# =============================================================================
# Preprocessing Configuration
# =============================================================================
@dataclass
class PreprocessConfig:
    """Configuration for text preprocessing pipeline"""
    clean: bool = True
    normalize: bool = True
    remove_stopwords: bool = True
    stemming: bool = False
    lemmatize: bool = False
    
    def __str__(self):
        flags = []
        if self.clean:
            flags.append("clean")
        if self.normalize:
            flags.append("norm")
        if self.remove_stopwords:
            flags.append("stop")
        if self.stemming:
            flags.append("stem")
        if self.lemmatize:
            flags.append("lemma")
        return "_".join(flags) if flags else "raw"


# Define 6 preprocessing pipelines as per project requirements
PREPROCESSING_PIPELINES = {
    "pipeline_1": PreprocessConfig(clean=False, normalize=False, remove_stopwords=False),
    "pipeline_2": PreprocessConfig(clean=True, normalize=False, remove_stopwords=False),
    "pipeline_3": PreprocessConfig(clean=True, normalize=True, remove_stopwords=False),
    "pipeline_4": PreprocessConfig(clean=True, normalize=True, remove_stopwords=True),
    "pipeline_5": PreprocessConfig(clean=True, normalize=True, remove_stopwords=True, stemming=True),
    "pipeline_6": PreprocessConfig(clean=True, normalize=True, remove_stopwords=True, lemmatize=True),
}

# =============================================================================
# Training Configuration
# =============================================================================
@dataclass
class TrainingConfig:
    """Configuration for model training"""
    # General
    test_size: float = 0.2
    random_seed: int = 42
    
    # BERT training
    bert_model_name: str = DEFAULT_BERT_MODEL
    max_length: int = 256  # SOS ข้อความไม่ยาวมาก ปรับลดเพื่อเพิ่ม throughput
    learning_rate: float = 1.5e-5
    batch_size: int = 16
    num_epochs: int = 4
    weight_decay: float = 0.05
    gradient_accumulation_steps: int = 2  # effective batch ~32
    warmup_ratio: float = 0.05
    logging_steps: int = 50
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
    fp16: bool = True
    bf16: bool = False
    dataloader_num_workers: int = 4
    
    # Word2Vec
    w2v_vector_size: int = 200
    w2v_window: int = 5
    w2v_min_count: int = 2


# =============================================================================
# Scraper Configuration
# =============================================================================
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0 Safari/537.36"
)

SCRAPE_DELAY_SECONDS = 2  # Delay between requests to avoid rate limiting


# =============================================================================
# Label Definitions
# =============================================================================
LABELS = {
    0: "not_urgent",
    1: "urgent",
}

LABEL_DESCRIPTIONS = {
    "urgent": "เคสเร่งด่วน - ต้องการความช่วยเหลือทันที",
    "not_urgent": "เคสทั่วไป - ข้อมูลน้ำท่วมแต่ไม่เร่งด่วน",
}


"""
Thai Text Preprocessing Module for Flood Relief NLP Pipeline

Provides text cleaning, normalization, tokenization, and stopword removal
specifically designed for Thai social media text.
"""
import re
from typing import List, Optional, Callable
from dataclasses import dataclass

try:
    from pythainlp import word_tokenize
    from pythainlp.corpus import thai_stopwords
    from pythainlp.util import normalize as thai_normalize
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False
    print("Warning: pythainlp not installed. Thai tokenization will use basic splitting.")



# =============================================================================
# Thai Stopwords
# =============================================================================
def get_thai_stopwords() -> set:
    """Get Thai stopwords from pythainlp or use a basic set"""
    if PYTHAINLP_AVAILABLE:
        return set(thai_stopwords())
    else:
        # Basic Thai stopwords if pythainlp is not available
        return {
            "และ", "หรือ", "แต่", "ที่", "ของ", "ใน", "เป็น", "มี", "ได้",
            "ไม่", "ก็", "จะ", "กับ", "ให้", "ว่า", "แล้ว", "นี้", "นั้น",
            "คือ", "จาก", "เมื่อ", "ถ้า", "อยู่", "กัน", "ครับ", "ค่ะ",
            "นะ", "คะ", "หน่อย", "ด้วย", "เลย", "มาก", "น่า", "ๆ",
        }


THAI_STOPWORDS = get_thai_stopwords()


# =============================================================================
# Text Cleaning Functions
# =============================================================================
def remove_urls(text: str) -> str:
    """Remove URLs from text"""
    return re.sub(r'http[s]?://\S+', ' ', text)


def remove_emojis(text: str) -> str:
    """Remove emojis and special symbols (except Thai and basic ASCII)"""
    # Keep Thai characters, ASCII letters/digits, basic punctuation, spaces
    return re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s.,:;!?/-]', ' ', text)


def remove_hashtag_symbols(text: str) -> str:
    """Remove # and @ symbols but keep the text"""
    return re.sub(r'[#@]', '', text)


def normalize_whitespace(text: str) -> str:
    """Normalize multiple whitespace to single space"""
    return re.sub(r'\s+', ' ', text).strip()


def remove_repeated_chars(text: str, max_repeat: int = 2) -> str:
    """
    Reduce repeated characters to max_repeat occurrences
    e.g., "มากกกกก" -> "มากก" (with max_repeat=2)
    """
    # Match any character repeated more than max_repeat times
    pattern = r'(.)\1{' + str(max_repeat) + r',}'
    replacement = r'\1' * max_repeat
    return re.sub(pattern, replacement, text)


def normalize_thai_digits(text: str) -> str:
    """Convert Thai digits (๐-๙) to Arabic digits (0-9)"""
    thai_digits = '๐๑๒๓๔๕๖๗๘๙'
    arabic_digits = '0123456789'
    trans_table = str.maketrans(thai_digits, arabic_digits)
    return text.translate(trans_table)


def basic_clean(text: str) -> str:
    """
    Basic text cleaning for Thai social media text
    """
    if not isinstance(text, str):
        return ""
    
    text = remove_urls(text)
    text = remove_emojis(text)
    text = remove_hashtag_symbols(text)
    text = normalize_whitespace(text)
    
    return text


# =============================================================================
# Thai Tokenization
# =============================================================================
def tokenize_thai(text: str, engine: str = "newmm") -> List[str]:
    """
    Tokenize Thai text using pythainlp
    
    Args:
        text: Input Thai text
        engine: Tokenization engine (newmm, longest, attacut, etc.)
    
    Returns:
        List of tokens
    """
    if PYTHAINLP_AVAILABLE:
        return word_tokenize(text, engine=engine)
    else:
        # Fallback: split by whitespace (not ideal for Thai)
        return text.split()


def normalize_tokens(tokens: List[str]) -> List[str]:
    """
    Normalize Thai tokens
    """
    normalized = []
    for tok in tokens:
        # Remove repeated characters
        tok = remove_repeated_chars(tok)
        # Normalize Thai digits
        tok = normalize_thai_digits(tok)
        # Remove commas from numbers
        tok = tok.replace(',', '')
        # Skip empty tokens
        if tok.strip():
            normalized.append(tok)
    return normalized


# =============================================================================
# Stopword Removal
# =============================================================================
def remove_stopwords(tokens: List[str], stopwords: set = None) -> List[str]:
    """
    Remove stopwords from token list
    """
    if stopwords is None:
        stopwords = THAI_STOPWORDS
    return [t for t in tokens if t not in stopwords]


# =============================================================================
# Stemming / Lemmatization (Thai-specific)
# =============================================================================
def thai_stem(tokens: List[str]) -> List[str]:
    """
    Simple Thai stemming
    
    Note: Thai doesn't have extensive inflection like English,
    so this mainly handles common suffixes and repetition markers.
    """
    stemmed = []
    for tok in tokens:
        # Remove repetition marker ๆ
        tok = tok.replace('ๆ', '')
        # Remove common verbal suffixes (very basic)
        if tok.endswith('การ'):
            tok = tok[:-3] if len(tok) > 3 else tok
        stemmed.append(tok)
    return [t for t in stemmed if t.strip()]


def thai_lemmatize(tokens: List[str]) -> List[str]:
    """
    Thai lemmatization using pythainlp if available
    
    Note: Thai lemmatization is less common than in English
    since Thai words don't conjugate for tense/number.
    This is a simple implementation.
    """
    if PYTHAINLP_AVAILABLE:
        try:
            from pythainlp.corpus import wordnet
            lemmatized = []
            for tok in tokens:
                # Try to get lemma from WordNet
                synsets = wordnet.synsets(tok)
                if synsets:
                    lemma = synsets[0].lemma_names()[0]
                    lemmatized.append(lemma)
                else:
                    lemmatized.append(tok)
            return lemmatized
        except:
            return tokens
    return tokens


# =============================================================================
# Main Preprocessing Function
# =============================================================================
def preprocess_text(text: str, config: PreprocessConfig) -> List[str]:
    """
    Main preprocessing function that applies the specified pipeline steps.
    
    Args:
        text: Input raw text
        config: PreprocessConfig specifying which steps to apply
    
    Returns:
        List of preprocessed tokens
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    # Step 1: Cleaning
    if config.clean:
        text = basic_clean(text)
    
    # Step 2: Tokenization
    if config.normalize:
        tokens = tokenize_thai(text)
        tokens = normalize_tokens(tokens)
    else:
        tokens = text.split()
    
    # Step 3: Stopword removal
    if config.remove_stopwords:
        tokens = remove_stopwords(tokens)
    
    # Step 4/5: Morphological normalization
    if config.stemming:
        tokens = thai_stem(tokens)
    elif config.lemmatize:
        tokens = thai_lemmatize(tokens)
    
    # Filter out empty tokens
    tokens = [t for t in tokens if t.strip()]
    
    return tokens


def preprocess_text_to_string(text: str, config: PreprocessConfig) -> str:
    """
    Preprocess text and return as space-joined string
    """
    tokens = preprocess_text(text, config)
    return ' '.join(tokens)


# =============================================================================
# Sklearn Analyzer Wrapper
# =============================================================================
class SklearnPreprocessAnalyzer:
    """Pickle-friendly callable wrapper for sklearn analyzer hooks."""

    def __init__(self, config: PreprocessConfig):
        self.config = config

    def __call__(self, text: str) -> List[str]:
        return preprocess_text(text, self.config)


def make_sklearn_analyzer(config: PreprocessConfig) -> Callable[[str], List[str]]:
    """
    Create an analyzer compatible with sklearn vectorizers that can be pickled.
    """
    return SklearnPreprocessAnalyzer(config)


# =============================================================================
# Batch Processing
# =============================================================================
def preprocess_batch(texts: List[str], config: PreprocessConfig) -> List[List[str]]:
    """
    Preprocess a batch of texts
    
    Args:
        texts: List of input texts
        config: PreprocessConfig
    
    Returns:
        List of token lists
    """
    return [preprocess_text(t, config) for t in texts]


def preprocess_batch_to_strings(texts: List[str], config: PreprocessConfig) -> List[str]:
    """
    Preprocess a batch of texts and return as strings
    """
    return [preprocess_text_to_string(t, config) for t in texts]


# =============================================================================
# Named Entity Extraction (Pattern-based)
# =============================================================================
def extract_phone_numbers(text: str) -> List[str]:
    """
    Extract Thai phone numbers from text
    Supports formats: 08X-XXX-XXXX, 08XXXXXXXX, etc.
    """
    # Thai mobile numbers start with 06, 08, 09
    patterns = [
        r'0[689]\d{8}',  # No separator
        r'0[689]\d-\d{3}-\d{4}',  # With dashes
        r'0[689]\d[\s-]\d{3}[\s-]\d{4}',  # With spaces or dashes
    ]
    
    phones = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            # Normalize: remove spaces and dashes
            normalized = re.sub(r'[\s-]', '', m)
            if normalized not in phones:
                phones.append(normalized)
    
    return phones


def extract_coordinates(text: str) -> Optional[dict]:
    """
    Extract GPS coordinates from text
    Format: lat, lng (e.g., "7.0074758, 100.4407940")
    """
    pattern = r'(-?\d{1,3}\.\d{3,})\s*,\s*(-?\d{1,3}\.\d{3,})'
    match = re.search(pattern, text)
    
    if match:
        lat, lng = float(match.group(1)), float(match.group(2))
        # Validate Thailand coordinates roughly
        if 5 <= lat <= 21 and 97 <= lng <= 106:
            return {"lat": lat, "lng": lng}
    
    return None


def extract_location_line(text: str) -> Optional[str]:
    """
    Extract location information line from text
    Looks for lines containing Thai location keywords
    """
    location_keywords = [
        "พิกัด", "บ้านเลขที่", "ที่อยู่", "ตำบล", "ต.", 
        "อำเภอ", "อ.", "จังหวัด", "จ.", "หมู่บ้าน", "หมู่ที่",
        "ซอย", "ถนน", "ถ.", "แขวง", "เขต"
    ]
    
    candidates = []
    for line in text.splitlines():
        line = line.strip()
        if any(kw in line for kw in location_keywords):
            candidates.append(line)
    
    if not candidates:
        return None
    
    # Return the longest candidate (likely most complete address)
    return max(candidates, key=len)


def has_flood_hashtags(text: str, hashtags: List[str]) -> List[str]:
    """
    Check if text contains any of the target flood-related hashtags
    
    Args:
        text: Input text
        hashtags: List of hashtags to check (without # symbol)
    
    Returns:
        List of found hashtags
    """
    found = []
    for tag in hashtags:
        plain_tag = tag.replace("#", "")
        if plain_tag in text or f"#{plain_tag}" in text:
            found.append(plain_tag)
    return list(set(found))


def calculate_urgency_score(text: str, urgency_keywords: List[str]) -> float:
    """
    Calculate a simple urgency score based on keyword presence
    
    Returns:
        Score between 0 and 1
    """
    if not text:
        return 0.0
    
    text_lower = text.lower()
    found_count = sum(1 for kw in urgency_keywords if kw in text_lower)
    
    # Normalize to 0-1 range (cap at 5 keywords = 1.0)
    return min(found_count / 5.0, 1.0)


"""
Shared keyword heuristics for risk, priority, and resource tagging.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

RISK_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "has_children": (
        "เด็ก", "เด็กเล็ก", "เด็กน้อย", "เด็กหญิง", "เด็กชาย",
        "เด็กๆ", "ลูก", "หลาน", "เบบี๋", "baby", "child", "children",
    ),
    "has_infants": (
        "ทารก", "ทารกแรกเกิด", "เด็กแรกเกิด", "แรกเกิด",
        "5 เดือน", "6 เดือน", "3 เดือน", "เบบี๋", "infant",
    ),
    "has_elderly": (
        "ผู้สูงอายุ", "คนแก่", "ผู้สูงวัย", "คนชรา", "คุณตา", "คุณยาย",
        "ตายาย", "อาม่า", "อากง", "ลุง", "ป้า", "คนแก่พิการ",
    ),
    "has_pregnant": (
        "คนท้อง", "ท้องแก่", "ตั้งครรภ์", "แม่ท้อง", "หญิงตั้งครรภ์",
        "ใกล้คลอด", "pregnant",
    ),
    "has_bedridden": (
        "ติดเตียง", "ผู้ป่วยติดเตียง", "ให้อาหารทางสาย", "ให้อาหารผ่านท้อง",
        "ให้อาหารทางสายยาง", "ให้อาหารทางท่อ", "ให้อาหารผ่านสาย", "นอนติดเตียง",
        "สายให้อาหาร", "สายยางให้อาหาร", "สายให้น้ำเกลือ",
    ),
    "has_disabled": (
        "พิการ", "คนพิการ", "นั่งรถเข็น", "วีลแชร์", "ตาบอด",
        "หูหนวก", "down syndrome", "อัมพาต", "เดินไม่ได้",
    ),
    "has_medical": (
        "ฟอกไต", "ไตวาย", "ล้างไต", "โรคหัวใจ", "หัวใจ", "โรคไต",
        "โรคปอด", "เบาหวาน", "ความดัน", "สโตรค", "stroke",
        "dialysis", "oxygen", "ออกซิเจน", "หอบหืด", "asthma",
        "ต้องกินยา", "ยาประจำ", "ยารักษาโรค",
    ),
    "needs_medication": (
        "ยาหมด", "ยาไม่พอ", "ยาขาด", "ยาใกล้หมด", "ต้องการยา",
        "ยาความดัน", "ยาโรคหัวใจ", "ยาโรคไต", "ยาประจำตัว", "medicine",
    ),
    "needs_medical_devices": (
        "ออกซิเจน", "เครื่องออกซิเจน", "ถังออกซิเจน", "เครื่องช่วยหายใจ",
        "oxygen", "ventilator", "เครื่องผลิตออกซิเจน", "เครื่องพ่นยา",
    ),
    "has_animals": (
        "หมา", "สุนัข", "น้องหมา", "น้องแมว", "แมว", "สัตว์เลี้ยง",
        "หมู", "วัว", "ควาย", "ไก่", "เป็ด",
    ),
    "has_large_group": (
        "หลายคน", "หลายสิบคน", "หลายครอบครัว", "หลายหลังคาเรือน",
        "ทั้งซอย", "ทั้งหมู่บ้าน", "จำนวนมาก", "ร่วมร้อยคน",
        "ทั้งตึก", "ทั้งชุมชน",
    ),
    "needs_transport": (
        "ต้องการเรือ", "เรือด่วน", "เจ็ทสกี", "เจสกี", "เจ็ตสกี",
        "เรือเร็ว", "รถยกสูง", "เรือกู้ภัย", "ห้องแถวเรือ",
    ),
}

TRAP_KEYWORDS = (
    "ติดอยู่", "ติดอยู่บนหลังคา", "ติดอยู่ชั้น2", "อยู่บนหลังคา", "อยู่ดาดฟ้า",
    "ออกมาไม่ได้", "บนดาดฟ้า", "อยู่ชั้นสอง", "ชั้น2", "ชั้น 2", "ชั้นลอย",
    "ติดอยู่ในบ้าน", "ไม่มีทางออก", "ออกทางหน้าบ้านไม่ได้",
)

SUPPLY_KEYWORDS = (
    "อาหาร", "ไม่มีอาหาร", "ข้าว", "ข้าวสาร", "เสบียง",
    "น้ำ", "น้ำดื่ม", "ไม่มีน้ำ", "นม", "นมผง", "แพมเพิส",
    "ผ้าอ้อม", "อาหารสัตว์", "ของกิน", "ของใช้", "ยารักษาโรค",
    "ขาดเสบียง", "ของยังชีพ",
)

FATALITY_KEYWORDS = (
    "เสียชีวิต", "ศพ", "ผู้เสียชีวิต", "ร่าง", "จมน้ำ", "ดับ",
    "เสียชีวิตแล้ว", "ศพอยู่", "นำศพออก",
)

POWER_KEYWORDS = (
    "ไฟดับ", "ไม่มีไฟ", "ไฟฟ้าดับ", "ไฟไม่มา", "ไฟถูกตัด",
    "ไฟโดนตัด", "แบตหมด", "แบตเหลือ", "แบตจะหมด", "ชาร์จไม่ได้",
    "powerbank", "พาวเวอร์แบงก์", "เพาเวอร์แบงก์", "ไม่มีไฟชาร์จ", "ชาร์จไม่ติด",
    "เครื่องปั่นไฟ", "ไม่มีไฟส่องสว่าง",
)

COMMUNICATION_KEYWORDS = (
    "สัญญาณไม่มี", "ไม่มีสัญญาณ", "สัญญาณโทรศัพท์ไม่มี", "โทรไม่ติด",
    "ติดต่อไม่ได้", "ขาดการติดต่อ", "ไม่มีเครือข่าย", "สัญญาณไม่ดี",
    "สัญญาณขาด", "โทรศัพท์ไม่มีสัญญาณ", "เน็ตล่ม", "wifi ล่ม",
)

RISK_FLAG_NAMES = sorted(RISK_KEYWORDS.keys()) + [
    "needs_evac",
    "needs_supplies",
    "mentions_fatality",
    "needs_power",
    "needs_comms",
    "mentions_water_level",
]

HIGH_RISK_FLAG_NAMES = {
    "has_pregnant",
    "has_bedridden",
    "has_infants",
    "has_disabled",
    "needs_medical_devices",
    "needs_medication",
    "mentions_fatality",
}

VULNERABLE_FLAG_NAMES = {
    "has_children",
    "has_infants",
    "has_elderly",
    "has_medical",
    "has_disabled",
    "has_animals",
    "has_large_group",
}

RESOURCE_KEYWORDS = {
    "medical_evac": (
        "ฟอกไต", "ไตวาย", "ล้างไต", "ผู้ป่วย", "โรคหัวใจ", "ต้องไปโรงพยาบาล",
        "หายใจไม่ออก", "oxygen", "เครื่องช่วยหายใจ", "ยาหมด",
    ),
    "food_drop": (
        "อาหารหมด", "ไม่มีอาหาร", "ไม่มีน้ำ", "เสบียง", "นมผง", "แพมเพิส",
    ),
    "rescue_boat": (
        "ขอเรือ", "ขอเจ็ทสกี", "ติดหลังคา", "อพยพด่วน", "น้ำถึงชั้นสอง",
        "น้ำท่วมสูง", "เรือกู้ภัย", "เฮลิคอปเตอร์", "เรือเข้าไม่ได้",
        "น้ำไหลแรง", "รอเรือ", "ขนย้ายทางเรือ",
    ),
    "body_recovery": (
        "ศพ", "ผู้เสียชีวิต", "เก็บศพ", "รับศพ",
    ),
    "power_supply": (
        "ไฟดับ", "ไฟฟ้าถูกตัด", "ไม่มีไฟ", "powerbank", "แบตหมด",
    ),
}

RESOURCE_TAGS = tuple(RESOURCE_KEYWORDS.keys())


def _contains(text: str, keyword: str, text_lower: str) -> bool:
    if keyword.isascii():
        return keyword.lower() in text_lower
    return keyword in text


def infer_risk_flags(text: str) -> Dict[str, bool]:
    text = text or ""
    text_lower = text.lower()
    flags = {
        name: any(_contains(text, kw, text_lower) for kw in keywords)
        for name, keywords in RISK_KEYWORDS.items()
    }
    flags["needs_evac"] = any(_contains(text, kw, text_lower) for kw in TRAP_KEYWORDS)
    flags["needs_supplies"] = any(_contains(text, kw, text_lower) for kw in SUPPLY_KEYWORDS)
    flags["mentions_fatality"] = any(_contains(text, kw, text_lower) for kw in FATALITY_KEYWORDS)
    flags["needs_power"] = any(_contains(text, kw, text_lower) for kw in POWER_KEYWORDS)
    flags["needs_comms"] = any(_contains(text, kw, text_lower) for kw in COMMUNICATION_KEYWORDS)
    flags["mentions_water_level"] = (
        "น้ำท่วม" in text or "ระดับน้ำ" in text_lower or "น้ำขึ้น" in text
    )
    # Detect explicit numbers of people (>=10) to boost large group
    if not flags.get("has_large_group"):
        match = re.search(r"\b(\d{2,})\s*คน\b", text.replace(",", ""))
        if match and int(match.group(1)) >= 10:
            flags["has_large_group"] = True
    return flags


def decide_priority(urgency_score: float, flags: Dict[str, bool]) -> Tuple[str, int]:
    score = urgency_score or 0.0
    high_risk = any(flags.get(name, False) for name in HIGH_RISK_FLAG_NAMES)
    vulnerable = any(flags.get(name, False) for name in VULNERABLE_FLAG_NAMES)
    trapped = flags.get("needs_evac", False)
    infra_needs = flags.get("needs_power", False) or flags.get("needs_comms", False)

    if high_risk or (score >= 0.6 and (vulnerable or trapped)) or (trapped and vulnerable):
        return "P1", 1
    if (
        score >= 0.4
        or trapped
        or flags.get("needs_supplies", False)
        or infra_needs
        or flags.get("needs_medication", False)
        or (flags.get("has_large_group", False) and (trapped or flags.get("needs_supplies", False)))
    ):
        return "P2", 1
    return "P3", 0


def infer_resource_tags(text: str) -> List[str]:
    text = text or ""
    text_lower = text.lower()
    tags = []
    for tag, keywords in RESOURCE_KEYWORDS.items():
        if any(_contains(text, kw, text_lower) for kw in keywords):
            tags.append(tag)
    return tags


def summarize_context_reason(text: str, flags: Dict[str, bool]) -> str:
    reasons = []
    if flags.get("needs_evac"):
        reasons.append("ติดอยู่ในพื้นที่น้ำสูง/ออกไม่ได้")
    if flags.get("mentions_water_level"):
        reasons.append("น้ำท่วมลึกถึงชั้นบนหรือไหลแรง")
    if flags.get("needs_supplies"):
        reasons.append("เสบียง/น้ำอาหารหมด")
    if flags.get("needs_medication") or flags.get("has_medical"):
        reasons.append("มีผู้ป่วยต้องใช้ยาหรือรักษาต่อเนื่อง")
    if flags.get("has_pregnant") or flags.get("has_infants"):
        reasons.append("มีแม่ท้องหรือเด็กเล็กในพื้นที่เสี่ยง")
    if flags.get("needs_power"):
        reasons.append("ไฟฟ้าถูกตัด/แบตหมด")
    if flags.get("mentions_fatality"):
        reasons.append("พบผู้เสียชีวิต ต้องการการจัดการโดยด่วน")
    if not reasons:
        reasons.append("สถานการณ์ต้องติดตามเพิ่มเติม")
    return "; ".join(reasons)


PEOPLE_KEYWORDS = {
    "children": ("เด็ก", "ลูก", "หลาน", "ทารก", "baby"),
    "elderly": ("ผู้สูงอายุ", "คนแก่", "ยาย", "ตา", "อาม่า", "อากง"),
    "adults": ("ผู้ใหญ่", "ผู้ชาย", "ผู้หญิง", "ชาวบ้าน"),
}

PEOPLE_PATTERN = re.compile(r"(เด็กเล็ก|เด็ก|ลูก|หลาน|ผู้ใหญ่|ผู้ชาย|ผู้หญิง|คนแก่|ผู้สูงอายุ|คน|ครอบครัว)\s*(?:จำนวน)?\s*(\d+)(?:\s*คน)?")
GENERIC_PEOPLE_PATTERN = re.compile(r"(?:จำนวน)?\s*(\d+)\s*คน")
DURATION_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(วัน|ชั่วโมง|ชม\.|ช.ม.|hrs?|hours?)")


def extract_people_counts(text: str) -> Dict[str, int]:
    counts = {"children": 0, "elderly": 0, "adults": 0, "unknown": 0}
    if not text:
        return counts
    for match in PEOPLE_PATTERN.finditer(text):
        keyword = match.group(1)
        value = int(match.group(2))
        if value > 500:
            continue
        assigned = False
        for key, kw_list in PEOPLE_KEYWORDS.items():
            if keyword in kw_list:
                counts[key] += value
                assigned = True
                break
        if not assigned:
            counts["unknown"] += value
    if counts["unknown"] == 0:
        for match in GENERIC_PEOPLE_PATTERN.finditer(text):
            value = int(match.group(1))
            if value <= 500:
                counts["unknown"] += value
    return counts


def extract_duration_hours(text: str) -> float:
    if not text:
        return 0.0
    longest = 0.0
    for match in DURATION_PATTERN.finditer(text):
        value = float(match.group(1))
        unit = match.group(2)
        if "วัน" in unit:
            hours = value * 24.0
        else:
            hours = value
        longest = max(longest, hours)
    return longest


def serialize_flags(flags: Dict[str, bool]) -> str:
    active = [name for name, value in flags.items() if value]
    return "|".join(active)


def serialize_tags(tags: List[str]) -> str:
    return "|".join(sorted(set(tags)))


def parse_multi_label_field(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, dict):
        return [k for k, v in value.items() if v]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            data = json.loads(value)
            return parse_multi_label_field(data)
        except json.JSONDecodeError:
            return [part.strip() for part in value.split("|") if part.strip()]
    return []



"""
Feature Extraction Module for Thai Flood Relief NLP Pipeline

Provides multiple text representation methods:
1. Bag-of-Words (BoW)
2. TF-IDF
3. Word2Vec embeddings
4. BERT embeddings (via transformers)
"""
import os
import numpy as np
from typing import List, Optional, Tuple, Any
import joblib

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False



# =============================================================================
# Bag-of-Words (BoW) Vectorizer
# =============================================================================
def build_bow_vectorizer(
    config: PreprocessConfig,
    min_df: int = 2,
    max_df: float = 0.95,
    max_features: Optional[int] = None,
    ngram_range: Tuple[int, int] = (1, 1)
) -> CountVectorizer:
    """
    Build a Bag-of-Words vectorizer with Thai preprocessing
    
    Args:
        config: PreprocessConfig for text preprocessing
        min_df: Minimum document frequency
        max_df: Maximum document frequency (fraction)
        max_features: Maximum number of features
        ngram_range: N-gram range (default unigrams only)
    
    Returns:
        CountVectorizer instance
    """
    analyzer = make_sklearn_analyzer(config)
    
    return CountVectorizer(
        analyzer=analyzer,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        ngram_range=ngram_range,
    )


# =============================================================================
# TF-IDF Vectorizer
# =============================================================================
def build_tfidf_vectorizer(
    config: PreprocessConfig,
    min_df: int = 2,
    max_df: float = 0.95,
    max_features: Optional[int] = None,
    ngram_range: Tuple[int, int] = (1, 1),
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """
    Build a TF-IDF vectorizer with Thai preprocessing
    
    Args:
        config: PreprocessConfig for text preprocessing
        min_df: Minimum document frequency
        max_df: Maximum document frequency (fraction)
        max_features: Maximum number of features
        ngram_range: N-gram range
        sublinear_tf: Use sublinear tf scaling (log tf)
    
    Returns:
        TfidfVectorizer instance
    """
    analyzer = make_sklearn_analyzer(config)
    
    return TfidfVectorizer(
        analyzer=analyzer,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
    )


# =============================================================================
# Word2Vec Embeddings
# =============================================================================
class Word2VecFeatures:
    """
    Word2Vec-based feature extraction for Thai text
    """
    
    def __init__(
        self,
        vector_size: int = 200,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        config: PreprocessConfig = None
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.config = config or PreprocessConfig()
        self.model: Optional[Word2Vec] = None
    
    def fit(self, texts: List[str]) -> 'Word2VecFeatures':
        """
        Train Word2Vec model on the corpus
        
        Args:
            texts: List of raw text documents
        
        Returns:
            self
        """
        if not GENSIM_AVAILABLE:
#             raise ImportError("gensim is required for Word2Vec. Install with: pip install gensim")
        
        # Tokenize all texts
        tokenized_corpus = [preprocess_text(t, self.config) for t in texts]
        
        # Train Word2Vec
        self.model = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
        )
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to Word2Vec feature vectors (averaged)
        
        Args:
            texts: List of raw text documents
        
        Returns:
            numpy array of shape (n_samples, vector_size)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        features = []
        for text in texts:
            tokens = preprocess_text(text, self.config)
            vec = self._sentence_to_vector(tokens)
            features.append(vec)
        
        return np.vstack(features)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(texts)
        return self.transform(texts)
    
    def _sentence_to_vector(self, tokens: List[str]) -> np.ndarray:
        """
        Convert a list of tokens to a single vector by averaging
        """
        vectors = []
        for tok in tokens:
            if tok in self.model.wv:
                vectors.append(self.model.wv[tok])
        
        if not vectors:
            return np.zeros(self.vector_size)
        
        return np.mean(vectors, axis=0)
    
    def save(self, path: str) -> None:
        """Save the Word2Vec model"""
        if self.model is not None:
            self.model.save(path)
    
    def load(self, path: str) -> 'Word2VecFeatures':
        """Load a saved Word2Vec model"""
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim is required")
        self.model = Word2Vec.load(path)
        return self


# =============================================================================
# BERT Embeddings
# =============================================================================
class BERTFeatures:
    """
    BERT-based feature extraction for Thai text
    
    Uses pre-trained Thai BERT models (e.g., WangchanBERTa)
    to generate contextual embeddings.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_BERT_MODEL,
        max_length: int = 256,
        pooling: str = "cls",  # "cls", "mean", "max"
        device: str = None
    ):
        if not TRANSFORMERS_AVAILABLE:
#             raise ImportError("transformers and torch are required. Install with: pip install transformers torch")
        
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def transform(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Transform texts to BERT embedding vectors
        
        Args:
            texts: List of raw text documents
            batch_size: Batch size for inference
        
        Returns:
            numpy array of shape (n_samples, hidden_size)
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
                
                # Apply pooling
                if self.pooling == "cls":
                    embeddings = hidden_states[:, 0, :]  # [CLS] token
                elif self.pooling == "mean":
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    masked_hidden = hidden_states * attention_mask
                    embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)
                elif self.pooling == "max":
                    embeddings = hidden_states.max(dim=1)[0]
                else:
                    raise ValueError(f"Unknown pooling method: {self.pooling}")
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of output embeddings"""
        return self.model.config.hidden_size


# =============================================================================
# Factory Functions
# =============================================================================
def build_vectorizer(
    method: str,
    config: PreprocessConfig,
    **kwargs
) -> Any:
    """
    Build a vectorizer based on the specified method
    
    Args:
        method: "bow", "tfidf", "word2vec", or "bert"
        config: PreprocessConfig for preprocessing
        **kwargs: Additional arguments for the specific vectorizer
    
    Returns:
        Vectorizer instance
    """
    method = method.lower()
    
    if method == "bow":
        return build_bow_vectorizer(config, **kwargs)
    elif method == "tfidf":
        return build_tfidf_vectorizer(config, **kwargs)
    elif method == "word2vec":
        return Word2VecFeatures(config=config, **kwargs)
    elif method == "bert":
        return BERTFeatures(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: bow, tfidf, word2vec, bert")


# =============================================================================
# Model Saving/Loading Utilities
# =============================================================================
def save_sklearn_vectorizer(vectorizer, path: str) -> None:
    """Save a sklearn vectorizer (BoW or TF-IDF)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)


def load_sklearn_vectorizer(path: str):
    """Load a sklearn vectorizer"""
    return joblib.load(path)


def save_word2vec_model(w2v_features: Word2VecFeatures, path: str) -> None:
    """Save Word2Vec features"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w2v_features.save(path)


def load_word2vec_model(path: str, config: PreprocessConfig = None) -> Word2VecFeatures:
    """Load Word2Vec features"""
    w2v = Word2VecFeatures(config=config)
    w2v.load(path)
    return w2v


"""
Dataset Preparation Module for Thai Flood Relief NLP Pipeline

This module handles:
1. Exporting data from SQLite to CSV
2. Splitting data into Train/Test sets
3. Data augmentation (optional)
4. Label statistics and analysis
"""
import os
import sys
import sqlite3
import json
import html
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split

# Add parent directory to path

#     DB_PATH, DATA_DIR,
#     RAW_CSV_PATH, LABELED_CSV_PATH,
#     TRAIN_CSV_PATH, TEST_CSV_PATH,
#     SOS_JSON_PATH,
#     URGENCY_KEYWORDS, TrainingConfig
# )
#     infer_risk_flags,
#     infer_resource_tags,
#     summarize_context_reason,
#     decide_priority,
#     serialize_flags,
#     serialize_tags,
#     extract_people_counts,
#     extract_duration_hours,
# )


def _ensure_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []
    return []


def annotate_posts_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features (risk flags, resource tags, priority, counts)."""
    if df.empty:
        return df
    df = df.copy()
    df['text'] = df['text'].fillna('').astype(str)
    if 'hashtags' in df.columns:
        df['hashtags'] = df['hashtags'].apply(_ensure_list)
    else:
        df['hashtags'] = [[] for _ in range(len(df))]
    if 'phones' in df.columns:
        df['phones'] = df['phones'].apply(_ensure_list)
    else:
        df['phones'] = [[] for _ in range(len(df))]
    if 'location_line' not in df.columns:
        df['location_line'] = None
    if 'lat' not in df.columns:
        df['lat'] = None
    if 'lng' not in df.columns:
        df['lng'] = None
    df['text_length'] = df['text'].str.len()
    df['has_phone'] = df['phones'].apply(lambda x: len(x) > 0)
    df['has_location'] = df['location_line'].fillna('').astype(str).str.strip().ne('')
    df['has_coordinates'] = df['lat'].notna() & df['lng'].notna()
    df['urgency_score'] = df['text'].apply(
        lambda x: calculate_urgency_score(str(x), URGENCY_KEYWORDS)
    )
    df['risk_flags_dict'] = df['text'].apply(infer_risk_flags)
    df['risk_flags'] = df['risk_flags_dict'].apply(lambda d: json.dumps(d, ensure_ascii=False))
    df['risk_flags_active'] = df['risk_flags_dict'].apply(serialize_flags)
    df['resource_tags_list'] = df['text'].apply(infer_resource_tags)
    df['resource_tags'] = df['resource_tags_list'].apply(serialize_tags)
    df['context_reason'] = df.apply(
        lambda row: summarize_context_reason(row['text'], row['risk_flags_dict']),
        axis=1
    )
    df['priority_label'] = df.apply(
        lambda row: decide_priority(row['urgency_score'], row['risk_flags_dict'])[0],
        axis=1
    )
    df['priority_numeric'] = df['priority_label'].map({"P1": 2, "P2": 1, "P3": 0})
    people_counts = df['text'].apply(extract_people_counts)
    df['num_children'] = people_counts.apply(lambda c: c['children'])
    df['num_elderly'] = people_counts.apply(lambda c: c['elderly'])
    df['num_adults'] = people_counts.apply(lambda c: c['adults'])
    df['num_unknown_people'] = people_counts.apply(lambda c: c['unknown'])
    df['num_people_total'] = (
        df['num_children'] + df['num_elderly'] + df['num_adults'] + df['num_unknown_people']
    )
    df['duration_hours'] = df['text'].apply(extract_duration_hours)
    return df


def _decode_text(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return value
    value = html.unescape(value)
    try:
        # handle strings that were decoded with wrong codec
        return value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


TYPE_RESOURCE_HINTS = [
    ("ป่วย", "medical_evac"),
    ("หมอ", "medical_evac"),
    ("แพทย์", "medical_evac"),
    ("อาหาร", "food_drop"),
    ("น้ำ", "food_drop"),
    ("เรือ", "rescue_boat"),
    ("อพยพ", "rescue_boat"),
    ("ไฟฟ้า", "power_supply"),
    ("ไฟ", "power_supply"),
]


# =============================================================================
# Export from Database
# =============================================================================
def export_db_to_csv(
    db_path: str = DB_PATH,
    output_path: str = RAW_CSV_PATH
) -> pd.DataFrame:
    """
    Export all posts from SQLite database to CSV
    
    Args:
        db_path: Path to SQLite database
        output_path: Path for output CSV
    
    Returns:
        DataFrame with exported data
    """
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        print("Please run the scraper first to collect posts.")
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    
    df = pd.read_sql_query("""
        SELECT 
            id,
            source,
            url,
            text,
            hashtags,
            phones,
            lat,
            lng,
            location_line,
            created_at,
            scraped_at
        FROM posts
        ORDER BY id
    """, conn)
    
    conn.close()
    
    # Parse JSON columns
    df['hashtags'] = df['hashtags'].apply(lambda x: json.loads(x) if x else [])
    df['phones'] = df['phones'].apply(lambda x: json.loads(x) if x else [])
    
    df = annotate_posts_dataframe(df)
    
    # Add empty label column for manual annotation
    df['label'] = None
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.drop(
        columns=['risk_flags_dict', 'resource_tags_list'],
        inplace=True,
        errors='ignore'
    )
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"Exported {len(df)} posts to: {output_path}")
    print(f"\nStatistics:")
    print(f"  - With location: {df['has_location'].sum()}")
    print(f"  - With phone: {df['has_phone'].sum()}")
    print(f"  - With coordinates: {df['has_coordinates'].sum()}")
    print(f"  - Avg text length: {df['text_length'].mean():.0f} chars")
    print(f"\nPlease add labels to the 'label' column:")
    print("  0 = not urgent")
    print("  1 = urgent")
    print(f"Then save as: {LABELED_CSV_PATH}")
    
    return df


def export_sos_to_csv(
    sos_path: str = SOS_JSON_PATH,
    output_path: str = RAW_CSV_PATH
) -> pd.DataFrame:
    """
    Export SOS API JSON data into CSV compatible with the pipeline.
    """
    if not os.path.exists(sos_path):
        print(f"SOS data file not found: {sos_path}")
        print("Please provide data/sos.json first.")
        return pd.DataFrame()
    
    with open(sos_path, encoding='utf-8') as f:
        payload = json.load(f)
    
    entries = payload.get("data", {}).get("data", [])
    if not entries:
        print("No SOS records found in the JSON payload.")
        return pd.DataFrame()
    
    fetched_at = payload.get("fetched_at")
    rows = []
    for entry in entries:
        location = entry.get("location") or {}
        props = location.get("properties") or {}
        geometry = location.get("geometry") or {}
        coordinates = geometry.get("coordinates") or [None, None]
        lng, lat = None, None
        if isinstance(coordinates, (list, tuple)) and len(coordinates) >= 2:
            lng, lat = coordinates[0], coordinates[1]
        
        text_parts: List[str] = []
        other_text = props.get("other")
        if other_text:
            text_parts.append(str(html.unescape(other_text)).strip())
        fallback_bits: List[str] = []
        for key in ("type_name", "status_text"):
            val = props.get(key)
            if val:
                fallback_bits.append(str(val))
        running_number = entry.get("running_number")
        if running_number:
            fallback_bits.append(str(running_number))
        if not text_parts and fallback_bits:
            text_parts.append(" | ".join(fallback_bits))
        text = " ".join([part for part in text_parts if part]).strip()
        if not text:
            text = props.get("type_name") or props.get("status_text") or "SOS Report"
        
        hashtags = []
        raw_type_name = props.get("type_name")
        type_name = _decode_text(raw_type_name)
        if type_name:
            hashtags.append(type_name)
        status_text = _decode_text(props.get("status_text"))
        status_color = props.get("status_color")
        
        rows.append({
            "id": entry.get("_id"),
            "source": "sos_api",
            "url": f"sos://{running_number or entry.get('_id')}",
            "text": text,
            "hashtags": hashtags,
            "phones": extract_phone_numbers(text),
            "lat": lat,
            "lng": lng,
            "location_line": props.get("address") or props.get("name"),
            "created_at": entry.get("created_at"),
            "scraped_at": fetched_at or entry.get("updated_at"),
            "updated_at": entry.get("updated_at"),
            "status_text": status_text,
            "status_color": status_color,
            "status_code": props.get("status"),
            "type_name": type_name,
            "sick_level_summary": props.get("sick_level_summary"),
            "running_number": running_number,
            "raw_other": other_text,
        })
    
    df = pd.DataFrame(rows)
    df = annotate_posts_dataframe(df)
    
    # Override priority using sick level when available
    if 'sick_level_summary' in df.columns:
        sick = pd.to_numeric(df['sick_level_summary'], errors='coerce')
        df.loc[sick >= 4, 'priority_label'] = 'P1'
        df.loc[(sick >= 3) & (df['priority_label'] == 'P3'), 'priority_label'] = 'P2'
    df['priority_numeric'] = df['priority_label'].map({"P1": 2, "P2": 1, "P3": 0})
    # Enrich resource tags from type hints
    if 'resource_tags_list' in df.columns and 'type_name' in df.columns:
        def add_type_tags(row):
            tags = set(row.get('resource_tags_list') or [])
            type_text = str(row.get('type_name') or '').lower()
            for hint, tag in TYPE_RESOURCE_HINTS:
                if hint in type_text:
                    tags.add(tag)
            return sorted(tags)
        df['resource_tags_list'] = df.apply(add_type_tags, axis=1)
        df['resource_tags'] = df['resource_tags_list'].apply(serialize_tags)
    
    df['label'] = None
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.drop(
        columns=['risk_flags_dict', 'resource_tags_list'],
        inplace=True,
        errors='ignore'
    )
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"Exported {len(df)} SOS posts to: {output_path}")
    if 'type_name' in df.columns:
        type_counts = df['type_name'].fillna('unknown').value_counts().to_dict()
        print(f"Type distribution: {type_counts}")
    if 'priority_label' in df.columns:
        print(f"Priority distribution: {df['priority_label'].value_counts().to_dict()}")
    
    return df


def auto_label_posts(
    df: pd.DataFrame,
    urgency_threshold: float = 0.4
) -> pd.DataFrame:
    """
    Automatically label posts based on heuristics
    
    This is for initial labeling - should be verified manually!
    
    Args:
        df: DataFrame with posts
        urgency_threshold: Score threshold for urgent label
    
    Returns:
        DataFrame with auto-generated labels
    """
    df = df.copy()
    
    # Ensure derived columns exist
    if 'priority_label' not in df.columns or 'urgency_score' not in df.columns:
        df = annotate_posts_dataframe(df)
    
    status_text = df.get('status_text', pd.Series([""] * len(df))).fillna("").astype(str)
    sick = pd.to_numeric(df.get('sick_level_summary'), errors='coerce')
    
    df['auto_label'] = 0
    df.loc[sick >= 3, 'auto_label'] = 1
    df.loc[df['priority_label'] == 'P1', 'auto_label'] = 1
    df.loc[
        status_text.str.contains("รอการช่วยเหลือ|ขอความช่วยเหลือ|ด่วน", regex=True),
        'auto_label'
    ] = 1
    df.loc[df['urgency_score'] >= urgency_threshold, 'auto_label'] = 1
    
    # Boost urgency if location/phone present and urgency moderate
    has_contact_info = df['has_phone'] | df['has_coordinates']
    df.loc[has_contact_info & (df['urgency_score'] > 0.2), 'auto_label'] = 1
    
    print(f"\nAuto-labeling results:")
    print(df['auto_label'].value_counts())
    print(f"\nNote: These are auto-generated labels!")
    print("Please verify and adjust manually.")
    
    return df


# =============================================================================
# Train/Test Split
# =============================================================================
def split_train_test(
    labeled_path: str = LABELED_CSV_PATH,
    train_path: str = TRAIN_CSV_PATH,
    test_path: str = TEST_CSV_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    use_auto_label: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split labeled data into train and test sets
    
    Args:
        labeled_path: Path to labeled CSV
        train_path: Output path for training set
        test_path: Output path for test set
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        use_auto_label: Use 'auto_label' column instead of 'label'
    
    Returns:
        Tuple of (train_df, test_df)
    """
    # Auto-label workflow takes precedence when requested
    if use_auto_label:
        if os.path.exists(RAW_CSV_PATH):
            print(f"\nAuto-labeling from raw file: {RAW_CSV_PATH}")
            df = pd.read_csv(RAW_CSV_PATH)
            df = auto_label_posts(df)
            df['label'] = df['auto_label']
            df.to_csv(labeled_path, index=False, encoding='utf-8-sig')
            print(f"Saved auto-labeled file to: {labeled_path}")
        elif not os.path.exists(labeled_path):
            print("Raw file not found for auto-labeling.")
            return pd.DataFrame(), pd.DataFrame()
    else:
        if not os.path.exists(labeled_path):
            print(f"Labeled file not found: {labeled_path}")
            print("Please label the data first or use --auto-label.")
            return pd.DataFrame(), pd.DataFrame()
    
    # Load labeled data
    df = pd.read_csv(labeled_path)
    
    # Determine label column
    label_col = 'auto_label' if use_auto_label and 'auto_label' in df.columns else 'label'
    
    # Filter rows with valid labels
    df_valid = df.dropna(subset=['text', label_col])
    df_valid[label_col] = df_valid[label_col].astype(int)
    
    if len(df_valid) == 0:
        print("No labeled data found. Please add labels to the 'label' column.")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"\nDataset size: {len(df_valid)} posts")
    print(f"Label distribution:\n{df_valid[label_col].value_counts()}")
    
    # Stratified split
    try:
        train_df, test_df = train_test_split(
            df_valid,
            test_size=test_size,
            random_state=random_state,
            stratify=df_valid[label_col]
        )
    except ValueError as e:
        print(f"Warning: Could not stratify: {e}")
        train_df, test_df = train_test_split(
            df_valid,
            test_size=test_size,
            random_state=random_state
        )
    
    # Ensure label column is named 'label' in output
    if label_col != 'label':
        train_df['label'] = train_df[label_col]
        test_df['label'] = test_df[label_col]
    
    # Save splits
    os.makedirs(os.path.dirname(train_path) if os.path.dirname(train_path) else '.', exist_ok=True)
    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')
    
    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_df)} samples)")
    print(f"  Test:  {test_path} ({len(test_df)} samples)")
    
    return train_df, test_df


# =============================================================================
# Data Analysis
# =============================================================================
def analyze_dataset(df: pd.DataFrame) -> dict:
    """
    Analyze dataset and return statistics
    
    Args:
        df: DataFrame with posts
    
    Returns:
        Dict with statistics
    """
    stats = {
        'total_posts': len(df),
        'text_length_mean': df['text'].str.len().mean(),
        'text_length_std': df['text'].str.len().std(),
        'posts_with_location': df['location_line'].notna().sum() if 'location_line' in df.columns else 0,
        'posts_with_phone': df['phones'].apply(lambda x: len(json.loads(x) if isinstance(x, str) else x) > 0).sum() if 'phones' in df.columns else 0,
        'posts_with_coords': (df['lat'].notna() & df['lng'].notna()).sum() if 'lat' in df.columns else 0,
    }
    
    if 'label' in df.columns:
        label_counts = df['label'].value_counts().to_dict()
        stats['label_distribution'] = label_counts
    
    return stats


def print_dataset_stats(train_path: str = TRAIN_CSV_PATH, test_path: str = TEST_CSV_PATH):
    """Print statistics for train and test datasets"""
    
    for name, path in [("Train", train_path), ("Test", test_path)]:
        if not os.path.exists(path):
            print(f"{name} file not found: {path}")
            continue
        
        df = pd.read_csv(path)
        print(f"\n=== {name} Dataset ===")
        print(f"Total samples: {len(df)}")
        
        if 'label' in df.columns:
            print(f"Label distribution:")
            print(df['label'].value_counts())
        
        print(f"Avg text length: {df['text'].str.len().mean():.0f} chars")


# =============================================================================
# Sample Data Generation (for testing)
# =============================================================================
def create_sample_data(output_path: str = None, n_samples: int = 20):
    """
    Create sample flood relief posts for testing
    
    Args:
        output_path: Path to save sample data
        n_samples: Number of samples to create
    """
    sample_posts = [
        # Urgent posts
        {
            "text": "🆘 ขออพยพด่วน!!! ผู้ประสบภัย 3 คน คนท้องแก่ ผู้สูงอายุพิการ 80 ปี อยู่บนหลังคา น้ำท่วมมิดบ้าน 2 วันแล้ว พิกัด 7.0074, 100.4407 โทร 0819797123",
            "label": 1
        },
        {
            "text": "ช่วยด้วยครับ น้ำเข้าบ้านแล้ว มีเด็กเล็ก 2 คน ที่อยู่ หมู่บ้านพฤกษา ซอย 5 ต.หาดใหญ่ อ.หาดใหญ่ จ.สงขลา ติดต่อ 0891234567",
            "label": 1
        },
        {
            "text": "SOS ครอบครัวติดอยู่ชั้น 2 น้ำเริ่มขึ้นสูง ไฟดับ แบตโทรศัพท์ใกล้หมด บ้านเลขที่ 168 ถนนสุทธิสมิทธิ์ โทรด่วน 0867891234",
            "label": 1
        },
        {
            "text": "ต้องการความช่วยเหลือเร่งด่วน ผู้ป่วยติดเตียง น้ำท่วมถึงระดับเอว ไม่สามารถเคลื่อนย้ายเองได้ ต.ควนลัง อ.หาดใหญ่ 0845678901",
            "label": 1
        },
        {
            "text": "#ขอความช่วยเหลือ น้ำท่วมบ้าน 5 ครอบครัว รวม 15 คน มีผู้สูงอายุ 4 คน ต้องการอพยพ หมู่ 3 ต.คลองแห อ.หาดใหญ่ 0823456789",
            "label": 1
        },
        {
            "text": "ด่วนมาก! คนจมน้ำ รอความช่วยเหลือ พิกัด 7.0234, 100.4567 โทร 0812345678 มาเร็วที่สุด!",
            "label": 1
        },
        {
            "text": "ขอความช่วยเหลือด่วน มีคนป่วยต้องล้างไต น้ำท่วมไม่สามารถไปโรงพยาบาลได้ ต้องการเรือ ที่อยู่ ซอยพัฒนา 3 ต.บ่อยาง",
            "label": 1
        },
        {
            "text": "#น้ำท่วม68 บ้านติดอยู่ 3 วันแล้ว อาหารหมด น้ำดื่มหมด มีเด็กทารก 1 คน ช่วยด้วยค่ะ 0876543210",
            "label": 1
        },
        {
            "text": "ขอเรือด่วน! คนแก่ไม่สามารถเดินได้ น้ำสูงมาก ต้องอพยพ บ้านเลขที่ 45 หมู่ 7 ต.พะตง อ.หาดใหญ่ โทร 0854321098",
            "label": 1
        },
        {
            "text": "🆘 ติดอยู่บนหลังคา 2 คน รอกู้ภัยมาช่วย น้ำไหลแรงมาก หมู่บ้านการเคหะ จ.สงขลา 0898765432",
            "label": 1
        },
        
        # Non-urgent posts
        {
            "text": "น้ำท่วมหาดใหญ่หนักมาก ทุกคนระวังตัวด้วยนะครับ #น้ำท่วม68",
            "label": 0
        },
        {
            "text": "รายงานสถานการณ์น้ำท่วมภาคใต้ ฝนยังตกต่อเนื่อง คาดว่าจะดีขึ้นในอีก 2-3 วัน",
            "label": 0
        },
        {
            "text": "ขอบคุณทีมกู้ภัยที่มาช่วยเหลือครับ ปลอดภัยแล้ว #ขอความช่วยเหลือ #น้ำท่วม",
            "label": 0
        },
        {
            "text": "แชร์ให้ด้วยนะครับ ใครต้องการความช่วยเหลือ โทรสายด่วน 1784 หรือ 199",
            "label": 0
        },
        {
            "text": "ประกาศ: ศูนย์พักพิงผู้ประสบภัยน้ำท่วม เปิดรับผู้อพยพที่โรงเรียนหาดใหญ่วิทยาลัย",
            "label": 0
        },
        {
            "text": "ระดับน้ำในคลองอู่ตะเภาเริ่มลดลงแล้ว คาดว่าอีก 6 ชั่วโมงจะกลับสู่ปกติ",
            "label": 0
        },
        {
            "text": "รวมเบอร์โทรหน่วยกู้ภัย: กู้ภัยหาดใหญ่ 074-xxxxxx, มูลนิธิร่วมกตัญญู 1669",
            "label": 0
        },
        {
            "text": "#น้ำท่วมหาดใหญ่ ถนนเพชรเกษมผ่านได้แล้ว รถเล็กระวังน้ำรอระบาย",
            "label": 0
        },
        {
            "text": "บริจาคสิ่งของช่วยผู้ประสบภัยได้ที่ศาลากลางจังหวัดสงขลา เปิดรับทุกวัน 8:00-18:00",
            "label": 0
        },
        {
            "text": "สถานการณ์น้ำท่วมในพื้นที่อำเภอเมืองสงขลาเริ่มคลี่คลาย ชาวบ้านเริ่มทำความสะอาดบ้านเรือน",
            "label": 0
        },
    ]
    
    # Create DataFrame
    df = pd.DataFrame(sample_posts[:n_samples])
    
    # Add synthetic metadata
    df['id'] = range(1, len(df) + 1)
    df['source'] = 'sample'
    df['url'] = df['id'].apply(lambda x: f'sample://{x}')
    df['hashtags'] = df['text'].apply(
        lambda x: [tag for tag in ['น้ำท่วม68', 'ขอความช่วยเหลือ'] if tag in x]
    )

    def random_phone_list():
        return ['0812345678'] if np.random.rand() < 0.5 else []

    df['phones'] = df['text'].apply(lambda _: random_phone_list())
    df['lat'] = None
    df['lng'] = None
    df['location_line'] = None
    df['has_location'] = False
    df['has_phone'] = df['phones'].apply(lambda x: len(x) > 0)
    df['has_coordinates'] = False
    df['urgency_score'] = df['text'].apply(
        lambda x: calculate_urgency_score(x, URGENCY_KEYWORDS)
    )
    
    if output_path is None:
        output_path = LABELED_CSV_PATH
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"Created {len(df)} sample posts at: {output_path}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df


# =============================================================================
# Main Entry Point
# =============================================================================

"""
Classical Model Training for Thai Flood Relief NLP Pipeline

Trains and evaluates classical ML models:
- Naive Bayes with BoW
- SVM with TF-IDF
- SVM with Word2Vec

Compares all 24 pipeline-representation combinations.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime
import joblib

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score

# Add parent directory to path

# #     TRAIN_CSV_PATH, TEST_CSV_PATH, MODELS_DIR,
# #     TFIDF_VEC_PATH, SVM_MODEL_PATH, W2V_MODEL_PATH,
# #     PREPROCESSING_PIPELINES, PreprocessConfig, TrainingConfig
# # )
#     build_bow_vectorizer, build_tfidf_vectorizer,
#     Word2VecFeatures, save_sklearn_vectorizer, save_word2vec_model
# )


# =============================================================================
# Data Loading
# =============================================================================
def load_train_test_data(
    train_path: str = TRAIN_CSV_PATH,
    test_path: str = TEST_CSV_PATH
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """
    Load training and test data
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Extract text and labels
    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['label'].astype(int).values
    
    X_test = test_df['text'].astype(str).tolist()
    y_test = test_df['label'].astype(int).values
    
    print(f"Loaded: {len(X_train)} train, {len(X_test)} test samples")
    print(f"Train labels: {np.bincount(y_train)}")
    print(f"Test labels: {np.bincount(y_test)}")
    
    return X_train, y_train, X_test, y_test


# =============================================================================
# Model Training Functions
# =============================================================================
def train_bow_nb(
    X_train: List[str],
    y_train: np.ndarray,
    X_test: List[str],
    y_test: np.ndarray,
    config: PreprocessConfig
) -> Dict[str, Any]:
    """
    Train Naive Bayes with Bag-of-Words
    """
    # Build vectorizer and transform
    vectorizer = build_bow_vectorizer(config)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test_vec)
    
    results = {
        'method': 'bow_nb',
        'pipeline': str(config),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
        'model': model,
        'vectorizer': vectorizer,
        'vocab_size': len(vectorizer.vocabulary_),
    }
    
    return results


def train_tfidf_svm(
    X_train: List[str],
    y_train: np.ndarray,
    X_test: List[str],
    y_test: np.ndarray,
    config: PreprocessConfig
) -> Dict[str, Any]:
    """
    Train Linear SVM with TF-IDF
    """
    # Build vectorizer and transform
    vectorizer = build_tfidf_vectorizer(config)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    model = LinearSVC(max_iter=10000)
    model.fit(X_train_vec, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test_vec)
    
    results = {
        'method': 'tfidf_svm',
        'pipeline': str(config),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
        'model': model,
        'vectorizer': vectorizer,
        'vocab_size': len(vectorizer.vocabulary_),
    }
    
    return results


def train_tfidf_lr(
    X_train: List[str],
    y_train: np.ndarray,
    X_test: List[str],
    y_test: np.ndarray,
    config: PreprocessConfig
) -> Dict[str, Any]:
    """
    Train Logistic Regression with TF-IDF
    """
    # Build vectorizer and transform
    vectorizer = build_tfidf_vectorizer(config)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test_vec)
    
    results = {
        'method': 'tfidf_lr',
        'pipeline': str(config),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
        'model': model,
        'vectorizer': vectorizer,
        'vocab_size': len(vectorizer.vocabulary_),
    }
    
    return results


def train_w2v_svm(
    X_train: List[str],
    y_train: np.ndarray,
    X_test: List[str],
    y_test: np.ndarray,
    config: PreprocessConfig,
    vector_size: int = 200
) -> Dict[str, Any]:
    """
    Train Linear SVM with Word2Vec embeddings
    """
    # Build Word2Vec and transform
    w2v = Word2VecFeatures(vector_size=vector_size, config=config)
    X_train_vec = w2v.fit_transform(X_train)
    X_test_vec = w2v.transform(X_test)
    
    # Train model
    model = LinearSVC(max_iter=10000)
    model.fit(X_train_vec, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test_vec)
    
    results = {
        'method': 'w2v_svm',
        'pipeline': str(config),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
        'model': model,
        'w2v_model': w2v,
        'vector_size': vector_size,
    }
    
    return results


# =============================================================================
# Run All Pipeline Combinations
# =============================================================================
def run_all_experiments(
    X_train: List[str],
    y_train: np.ndarray,
    X_test: List[str],
    y_test: np.ndarray,
    save_best: bool = True
) -> pd.DataFrame:
    """
    Run all 24 pipeline-representation combinations
    
    6 pipelines × 4 representations = 24 experiments
    """
    all_results = []
    best_f1 = 0
    best_result = None
    
    representations = ['bow_nb', 'tfidf_svm', 'tfidf_lr', 'w2v_svm']
    
    print("\n" + "="*60)
    print("Running All Pipeline-Representation Combinations")
    print("="*60)
    
    for pipe_name, config in PREPROCESSING_PIPELINES.items():
        print(f"\n--- {pipe_name}: {config} ---")
        
        for rep in representations:
            try:
                print(f"  Training {rep}...", end=" ")
                
                if rep == 'bow_nb':
                    result = train_bow_nb(X_train, y_train, X_test, y_test, config)
                elif rep == 'tfidf_svm':
                    result = train_tfidf_svm(X_train, y_train, X_test, y_test, config)
                elif rep == 'tfidf_lr':
                    result = train_tfidf_lr(X_train, y_train, X_test, y_test, config)
                elif rep == 'w2v_svm':
                    result = train_w2v_svm(X_train, y_train, X_test, y_test, config)
                
                result['pipeline_name'] = pipe_name
                
                print(f"F1={result['f1']:.4f}, Acc={result['accuracy']:.4f}")
                
                # Track best result
                if result['f1'] > best_f1:
                    best_f1 = result['f1']
                    best_result = result.copy()
                
                # Store results (without model objects for DataFrame)
                all_results.append({
                    'pipeline': pipe_name,
                    'config': str(config),
                    'method': result['method'],
                    'accuracy': result['accuracy'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'f1': result['f1'],
                })
                
            except Exception as e:
                print(f"Error: {e}")
                all_results.append({
                    'pipeline': pipe_name,
                    'config': str(config),
                    'method': rep,
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1': 0,
                    'error': str(e),
                })
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    os.makedirs(MODELS_DIR, exist_ok=True)
    results_path = os.path.join(MODELS_DIR, 'classical_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(results_df.sort_values('f1', ascending=False).head(10).to_string())
    
    # Save best model
    if save_best and best_result:
        print(f"\n--- Best Model: {best_result['method']} ({best_result['pipeline_name']}) ---")
        print(f"F1: {best_result['f1']:.4f}")
        print(f"Accuracy: {best_result['accuracy']:.4f}")
        print(f"Precision: {best_result['precision']:.4f}")
        print(f"Recall: {best_result['recall']:.4f}")
        
        # Save best vectorizer and model
        if 'vectorizer' in best_result:
            save_sklearn_vectorizer(best_result['vectorizer'], TFIDF_VEC_PATH)
            print(f"Saved vectorizer to: {TFIDF_VEC_PATH}")
        
        if 'model' in best_result:
            joblib.dump(best_result['model'], SVM_MODEL_PATH)
            print(f"Saved model to: {SVM_MODEL_PATH}")
        
        if 'w2v_model' in best_result:
            save_word2vec_model(best_result['w2v_model'], W2V_MODEL_PATH)
            print(f"Saved Word2Vec to: {W2V_MODEL_PATH}")
    
    return results_df


# =============================================================================
# Quick Training (Single Best Config)
# =============================================================================
def train_best_classical_model(
    X_train: List[str],
    y_train: np.ndarray,
    X_test: List[str],
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Train the recommended classical model (TF-IDF + SVM with Pipeline 4)
    """
    # Use Pipeline 4: clean + normalize + stopwords
    config = PREPROCESSING_PIPELINES['pipeline_4']
    
    print("Training TF-IDF + SVM with Pipeline 4...")
    result = train_tfidf_svm(X_train, y_train, X_test, y_test, config)
    
    print(f"\n--- Results ---")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1 Score: {result['f1']:.4f}")
    print(f"Vocabulary Size: {result['vocab_size']}")
    
    # Save models
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_sklearn_vectorizer(result['vectorizer'], TFIDF_VEC_PATH)
    joblib.dump(result['model'], SVM_MODEL_PATH)
    
    print(f"\nSaved:")
    print(f"  Vectorizer: {TFIDF_VEC_PATH}")
    print(f"  Model: {SVM_MODEL_PATH}")
    
    return result


# =============================================================================
# Prediction Function
# =============================================================================
def predict_with_classical(
    texts: List[str],
    vectorizer_path: str = TFIDF_VEC_PATH,
    model_path: str = SVM_MODEL_PATH
) -> np.ndarray:
    """
    Make predictions using saved classical model
    """
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    
    X = vectorizer.transform(texts)
    predictions = model.predict(X)
    
    return predictions


# =============================================================================
# Main Entry Point
# =============================================================================

"""
BERT Fine-tuning for Thai Flood Relief NLP Pipeline

Fine-tunes WangchanBERTa or other Thai BERT models for
binary classification of urgent vs non-urgent flood relief posts.
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import json

# Add parent directory to path

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback
    )
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers/torch not installed. BERT training unavailable.")

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report
)

# #     TRAIN_CSV_PATH, TEST_CSV_PATH, BERT_MODEL_DIR,
# #     DEFAULT_BERT_MODEL, TrainingConfig, MODELS_DIR
# # )


# =============================================================================
# Data Loading and Preparation
# =============================================================================
def load_data_for_bert(
    train_path: str = TRAIN_CSV_PATH,
    test_path: str = TEST_CSV_PATH,
    label_column: str = "label",
    multi_label: bool = False,
    label_list: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load and prepare data for BERT training
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df = train_df[['text', label_column]].dropna(subset=['text'])
    test_df = test_df[['text', label_column]].dropna(subset=['text'])
    
    label_names: List[str] = []
    
    if multi_label:
        train_labels = train_df[label_column].apply(parse_multi_label_field)
        test_labels = test_df[label_column].apply(parse_multi_label_field)
        if label_list:
            label_names = label_list
        else:
            label_set = set()
            for labels in pd.concat([train_labels, test_labels]):
                label_set.update(labels)
            label_names = sorted(label_set)
        if not label_names:
            raise ValueError("No labels found for multi-label training")
        
        def to_multi_hot(labels: List[str]) -> List[float]:
            return [1.0 if name in labels else 0.0 for name in label_names]
        
        train_df = pd.DataFrame({
            'text': train_df['text'],
            'labels': train_labels.apply(to_multi_hot)
        })
        test_df = pd.DataFrame({
            'text': test_df['text'],
            'labels': test_labels.apply(to_multi_hot)
        })
    else:
        train_df = train_df[['text', label_column]].dropna()
        test_df = test_df[['text', label_column]].dropna()
        if pd.api.types.is_numeric_dtype(train_df[label_column]):
            train_labels = train_df[label_column].astype(int)
            test_labels = test_df[label_column].astype(int)
            label_names = sorted(train_labels.unique().tolist())
        else:
            train_labels = train_df[label_column].astype(str)
            test_labels = test_df[label_column].astype(str)
            label_names = sorted(train_labels.unique().tolist())
            label_map = {name: idx for idx, name in enumerate(label_names)}
            train_labels = train_labels.map(label_map)
            test_labels = test_labels.map(label_map)
        train_df = pd.DataFrame({'text': train_df['text'], 'labels': train_labels})
        test_df = pd.DataFrame({'text': test_df['text'], 'labels': test_labels})
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    if multi_label:
        label_counter = {name: 0 for name in label_names}
        for label_vector in train_df['labels']:
            for idx, value in enumerate(label_vector):
                if value:
                    label_counter[label_names[idx]] += 1
        print(f"Label activation (train): {label_counter}")
    else:
        print(f"Label distribution (train): {pd.Series(train_df['labels']).value_counts().to_dict()}")
    
    return train_df, test_df, label_names


def create_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    max_length: int = 256
) -> Tuple[Dataset, Dataset]:
    """
    Create HuggingFace datasets for training
    """
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df[['text', 'labels']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'labels']])
    
    def tokenize_function(examples):
        tokens = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
        )
        tokens['labels'] = examples['labels']
        return tokens
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    keep_cols = {'input_ids', 'attention_mask', 'labels'}
    drop_train = [col for col in train_dataset.column_names if col not in keep_cols]
    drop_test = [col for col in test_dataset.column_names if col not in keep_cols]
    if drop_train:
        train_dataset = train_dataset.remove_columns(drop_train)
    if drop_test:
        test_dataset = test_dataset.remove_columns(drop_test)
    
    train_dataset.set_format(type='torch')
    test_dataset.set_format(type='torch')
    
    return train_dataset, test_dataset


# =============================================================================
# Metrics Computation
# =============================================================================
def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation
    """
    logits, labels = eval_pred
    labels = np.array(labels)
    logits = np.array(logits)
    
    if labels.ndim > 1 and labels.shape[-1] > 1:
        probs = 1 / (1 + np.exp(-logits))
        predictions = (probs >= 0.5).astype(int)
        accuracy = (predictions == labels).mean()
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='micro', zero_division=0
        )
    elif len(np.unique(labels)) > 2:
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
    else:
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# =============================================================================
# Training Function
# =============================================================================
def train_bert_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = DEFAULT_BERT_MODEL,
    output_dir: str = BERT_MODEL_DIR,
    config: TrainingConfig = None,
    multi_label: bool = False,
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Fine-tune BERT model on flood relief classification
    
    Args:
        train_df: Training DataFrame with 'text' and 'label' columns
        test_df: Test DataFrame
        model_name: HuggingFace model name
        output_dir: Directory to save model
        config: Training configuration
    
    Returns:
        Dict with training results and metrics
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers and torch are required for BERT training")
    
    if config is None:
        config = TrainingConfig()
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning: {model_name}")
    print(f"{'='*60}")
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Using device: {device_name} ({device})")
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    num_labels = len(label_names) if multi_label else len(set(train_df['labels'].tolist()))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    if multi_label:
        model.config.problem_type = "multi_label_classification"
    model.to(device)
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, test_dataset = create_datasets(
        train_df, test_df, tokenizer, config.max_length
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, 'checkpoints'),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        logging_strategy=config.eval_strategy,
        logging_steps=config.logging_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        save_total_limit=2,
        report_to='none',  # Disable wandb/tensorboard
        fp16=config.fp16 and torch.cuda.is_available(),
        bf16=config.bf16 and torch.cuda.is_available(),
        dataloader_num_workers=config.dataloader_num_workers,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    # Train
    print("\nStarting training...")
    train_result = trainer.train()
    
    # Evaluate
    print("\nEvaluating...")
    eval_result = trainer.evaluate()
    
    print(f"\n--- Final Results ---")
    print(f"Accuracy: {eval_result['eval_accuracy']:.4f}")
    print(f"Precision: {eval_result['eval_precision']:.4f}")
    print(f"Recall: {eval_result['eval_recall']:.4f}")
    print(f"F1 Score: {eval_result['eval_f1']:.4f}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nModel saved to: {output_dir}")
    
    # Save results
    results = {
        'model_name': model_name,
        'accuracy': eval_result['eval_accuracy'],
        'precision': eval_result['eval_precision'],
        'recall': eval_result['eval_recall'],
        'f1': eval_result['eval_f1'],
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'epochs': config.num_epochs,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'label_names': label_names,
    }
    
    results_path = os.path.join(output_dir, 'training_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results


# =============================================================================
# Prediction Function
# =============================================================================
class BERTPredictor:
    """
    Class for making predictions with trained BERT model
    """
    
    def __init__(self, model_dir: str = BERT_MODEL_DIR):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch are required")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(
        self,
        texts: list,
        batch_size: int = 16,
        return_probs: bool = True
    ) -> Dict[str, Any]:
        """
        Make predictions on a list of texts
        
        Returns:
            Dict with 'labels' and optionally 'probabilities'
        """
        all_preds = []
        all_probs = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())
        
        result = {'labels': all_preds}
        if return_probs:
            result['probabilities'] = all_probs
        
        return result
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Make prediction on a single text
        """
        result = self.predict([text])
        return {
            'label': result['labels'][0],
            'probability': result['probabilities'][0] if 'probabilities' in result else None
        }


# =============================================================================
# Detailed Evaluation
# =============================================================================
def evaluate_bert_detailed(
    model_dir: str = BERT_MODEL_DIR,
    test_path: str = TEST_CSV_PATH,
    label_column: str = "label",
    multi_label: bool = False,
):
    """
    Run detailed evaluation with classification report
    """
    # Load test data
    if multi_label:
        print("Detailed evaluation for multi-label tasks is not implemented yet.")
        return
    
    test_df = pd.read_csv(test_path)
    if label_column not in test_df.columns:
        raise ValueError(f"Label column '{label_column}' not found in {test_path}")
    texts = test_df['text'].astype(str).tolist()
    labels = pd.to_numeric(test_df[label_column], errors='ignore')
    if not np.issubdtype(labels.dtype, np.number):
        label_names = sorted(pd.Series(labels).astype(str).unique().tolist())
        label_map = {name: idx for idx, name in enumerate(label_names)}
        labels = labels.map(label_map).values
    else:
        labels = labels.astype(int).values
        label_names = None
    
    results_path = os.path.join(model_dir, 'training_results.json')
    if os.path.exists(results_path):
        with open(results_path, encoding='utf-8') as f:
            training_meta = json.load(f)
        saved_names = training_meta.get('label_names')
        if saved_names:
            label_names = saved_names
    
    # Load model and predict
    predictor = BERTPredictor(model_dir)
    results = predictor.predict(texts)
    predictions = results['labels']
    
    # Print detailed report
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    unique_labels = sorted(np.unique(labels))
    if label_names and len(label_names) == len(unique_labels):
        target_names = [str(name) for name in label_names]
    else:
        target_names = [f"class_{idx}" for idx in unique_labels]
    
    print(classification_report(
        labels, predictions,
        labels=unique_labels,
        target_names=target_names,
        zero_division=0
    ))
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, predictions, labels=unique_labels)
    print("\nConfusion Matrix:")
    header = "Predicted " + " ".join(f"{name:^8}" for name in target_names)
    print(header)
    for idx, row in enumerate(cm):
        row_str = " ".join(f"{val:^8}" for val in row)
        print(f"{target_names[idx]:<10} {row_str}")


# =============================================================================
# Main Entry Point
# =============================================================================

#!/usr/bin/env python
"""
Main Pipeline Runner for Thai Flood Relief NLP Pipeline

This script runs the complete pipeline:
1. Create sample data or scrape from URLs
2. Prepare and split dataset
3. Train classical models (BoW, TF-IDF, Word2Vec + SVM/NB)
4. Train BERT model
5. Evaluate and compare all models
6. Start API server
"""
import os
import sys
import argparse
from datetime import datetime

# Add project root to path

# #     DATA_DIR, MODELS_DIR, DB_PATH,
# #     TRAIN_CSV_PATH, TEST_CSV_PATH, LABELED_CSV_PATH
# # )


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def step_1_prepare_data(use_sample: bool = True):
    """Step 1: Prepare data"""
    print_header("Step 1: Preparing Data")
    
        create_sample_data, export_db_to_csv, split_train_test
    )
    
    if use_sample:
        print("Creating sample data for testing...")
        create_sample_data()
    elif os.path.exists(DB_PATH):
        print("Exporting data from database...")
        export_db_to_csv()
    
    # Split into train/test
    print("Splitting into Train/Test...")
    split_train_test(use_auto_label=True)
    
    print("[OK] Data preparation complete!")


def step_2_train_classical(run_all: bool = False):
    """Step 2: Train classical models"""
    print_header("Step 2: Training Classical Models")
    
        load_train_test_data, run_all_experiments, train_best_classical_model
    )
    
    # Check if data exists
    if not os.path.exists(TRAIN_CSV_PATH):
        print("Error: Training data not found. Run step 1 first.")
        return False
    
    X_train, y_train, X_test, y_test = load_train_test_data()
    
    if run_all:
        print("Running all 24 pipeline experiments...")
        run_all_experiments(X_train, y_train, X_test, y_test)
    else:
        print("Training best classical model (TF-IDF + SVM)...")
        train_best_classical_model(X_train, y_train, X_test, y_test)
    
    print("[OK] Classical model training complete!")
    return True


def step_3_train_bert():
    """Step 3: Train BERT model"""
    print_header("Step 3: Training BERT Model")
    
    try:
        import torch
        
        # Check if data exists
        if not os.path.exists(TRAIN_CSV_PATH):
            print("Error: Training data not found. Run step 1 first.")
            return False
        
        train_df, test_df, label_names = load_data_for_bert()
        
        print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        train_bert_model(train_df, test_df, label_names=label_names)
        
        print("[OK] BERT training complete!")
        return True
        
    except ImportError:
        print("Warning: torch/transformers not installed. Skipping BERT training.")
#         print("Install with: pip install torch transformers")
        return False


def step_4_evaluate():
    """Step 4: Evaluate all models"""
    print_header("Step 4: Evaluating Models")
    
    import pandas as pd
    
    # Check for classical results
    classical_results_path = os.path.join(MODELS_DIR, 'classical_results.csv')
    if os.path.exists(classical_results_path):
        print("\n--- Classical Model Results ---")
        df = pd.read_csv(classical_results_path)
        print(df.sort_values('f1', ascending=False).head(10).to_string())
    
    # Check for BERT results
    bert_results_path = os.path.join(MODELS_DIR, 'bert_flood_model', 'training_results.json')
    if os.path.exists(bert_results_path):
        import json
        print("\n--- BERT Model Results ---")
        with open(bert_results_path) as f:
            bert_results = json.load(f)
        for k, v in bert_results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    
    print("\n[OK] Evaluation complete!")


def step_5_start_api():
    """Step 5: Start API server"""
    print_header("Step 5: Starting API Server")
    
    print("Starting FastAPI server on http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Press Ctrl+C to stop\n")
    
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)


def main():
    parser = argparse.ArgumentParser(
        description="Thai Flood Relief NLP Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --all              # Run complete pipeline
  python run_pipeline.py --prepare          # Only prepare data
  python run_pipeline.py --train-classical  # Train classical models
  python run_pipeline.py --train-bert       # Train BERT model
  python run_pipeline.py --api              # Start API server
  python run_pipeline.py --all-experiments  # Run all 24 pipeline combinations
        """
    )
    
    parser.add_argument("--all", action="store_true",
                       help="Run complete pipeline (prepare → train → evaluate)")
    parser.add_argument("--prepare", action="store_true",
                       help="Step 1: Prepare and split data")
    parser.add_argument("--train-classical", action="store_true",
                       help="Step 2: Train classical models")
    parser.add_argument("--train-bert", action="store_true",
                       help="Step 3: Train BERT model")
    parser.add_argument("--evaluate", action="store_true",
                       help="Step 4: Evaluate models")
    parser.add_argument("--api", action="store_true",
                       help="Step 5: Start API server")
    parser.add_argument("--all-experiments", action="store_true",
                       help="Run all 24 pipeline-representation combinations")
    parser.add_argument("--use-real-data", action="store_true",
                       help="Use scraped data instead of sample data")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print(" Thai Flood Relief NLP Pipeline")
    print(" ระบบวิเคราะห์ข้อความขอความช่วยเหลือน้ำท่วม")
    print("=" * 60)
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # If no arguments, show help
    if not any([args.all, args.prepare, args.train_classical, 
                args.train_bert, args.evaluate, args.api]):
        parser.print_help()
        print("\n\nQuick start with sample data:")
        print("  python run_pipeline.py --all")
        return
    
    # Run selected steps
    if args.all or args.prepare:
        step_1_prepare_data(use_sample=not args.use_real_data)
    
    if args.all or args.train_classical:
        step_2_train_classical(run_all=args.all_experiments)
    
    if args.all or args.train_bert:
        step_3_train_bert()
    
    if args.all or args.evaluate:
        step_4_evaluate()
    
    if args.api:
        step_5_start_api()
    
    print("\n" + "=" * 60)
    print(" Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
