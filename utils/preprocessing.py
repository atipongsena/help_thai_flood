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

from utils.config import PreprocessConfig


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

