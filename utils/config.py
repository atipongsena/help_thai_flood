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

