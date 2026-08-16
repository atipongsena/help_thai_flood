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

from utils.config import PreprocessConfig, DEFAULT_BERT_MODEL
from utils.preprocessing import preprocess_text, make_sklearn_analyzer


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
            raise ImportError("gensim is required for Word2Vec. Install with: pip install gensim")
        
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
            raise ImportError("transformers and torch are required. Install with: pip install transformers torch")
        
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

