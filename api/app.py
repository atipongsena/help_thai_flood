"""
FastAPI Backend for Thai Flood Relief NLP Pipeline

Provides REST API endpoints for:
- Text classification (urgent vs not urgent)
- Information extraction (phone, location, coordinates)
- Batch processing
"""
import os
import sys
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import utilities
from utils.config import (
    BERT_MODEL_DIR, TFIDF_VEC_PATH, SVM_MODEL_PATH,
    LABELS, URGENCY_KEYWORDS
)
from utils.preprocessing import (
    extract_phone_numbers, extract_coordinates,
    extract_location_line, calculate_urgency_score,
    basic_clean
)

# Try to import model-specific modules
try:
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import numpy as np
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False


# =============================================================================
# Pydantic Models for API
# =============================================================================
class PredictRequest(BaseModel):
    """Request model for single text prediction"""
    text: str = Field(..., description="Text to classify", min_length=1)
    extract_info: bool = Field(True, description="Whether to extract phone/location info")


class PredictResponse(BaseModel):
    """Response model for prediction"""
    label: int = Field(..., description="Predicted label (0=not urgent, 1=urgent)")
    label_name: str = Field(..., description="Label name")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    urgency_score: float = Field(..., description="Heuristic urgency score")
    extracted_info: Optional[Dict[str, Any]] = Field(None, description="Extracted information")


class BatchPredictRequest(BaseModel):
    """Request model for batch prediction"""
    texts: List[str] = Field(..., description="List of texts to classify")
    extract_info: bool = Field(True, description="Whether to extract info")


class BatchPredictResponse(BaseModel):
    """Response model for batch prediction"""
    predictions: List[PredictResponse]
    total: int


class ExtractRequest(BaseModel):
    """Request for information extraction only"""
    text: str


class ExtractResponse(BaseModel):
    """Response for information extraction"""
    phones: List[str]
    coordinates: Optional[Dict[str, float]]
    location_line: Optional[str]
    urgency_score: float
    cleaned_text: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_type: str
    timestamp: str


# =============================================================================
# Model Loading
# =============================================================================
class ModelManager:
    """Manages model loading and inference"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.vectorizer = None
        self.model_type = "none"
        self.device = "cpu"
        
        # Try to load BERT model first, fallback to classical
        self._load_model()
    
    def _load_model(self):
        """Load the best available model"""
        # Try BERT first
        if BERT_AVAILABLE and os.path.exists(BERT_MODEL_DIR):
            try:
                self._load_bert_model()
                return
            except Exception as e:
                print(f"Failed to load BERT model: {e}")
        
        # Fallback to classical model
        if SKLEARN_AVAILABLE and os.path.exists(TFIDF_VEC_PATH) and os.path.exists(SVM_MODEL_PATH):
            try:
                self._load_classical_model()
                return
            except Exception as e:
                print(f"Failed to load classical model: {e}")
        
        print("Warning: No model loaded. Only extraction features available.")
    
    def _load_bert_model(self):
        """Load BERT model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading BERT model from {BERT_MODEL_DIR} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
        self.model.to(self.device)
        self.model.eval()
        self.model_type = "bert"
        
        print("BERT model loaded successfully!")
    
    def _load_classical_model(self):
        """Load classical (TF-IDF + SVM) model"""
        print(f"Loading classical model...")
        
        self.vectorizer = joblib.load(TFIDF_VEC_PATH)
        self.model = joblib.load(SVM_MODEL_PATH)
        self.model_type = "classical"
        
        print("Classical model loaded successfully!")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Make a prediction"""
        if self.model_type == "bert":
            return self._predict_bert(text)
        elif self.model_type == "classical":
            return self._predict_classical(text)
        else:
            # No model loaded, use heuristics
            return self._predict_heuristic(text)
    
    def _predict_bert(self, text: str) -> Dict[str, Any]:
        """BERT prediction"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_label = int(np.argmax(probs))
            confidence = float(probs[pred_label])
        
        return {
            "label": pred_label,
            "confidence": confidence
        }
    
    def _predict_classical(self, text: str) -> Dict[str, Any]:
        """Classical model prediction"""
        X = self.vectorizer.transform([text])
        pred_label = int(self.model.predict(X)[0])
        
        # SVM doesn't have predict_proba by default
        confidence = None
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)[0]
            confidence = float(probs[pred_label])
        
        return {
            "label": pred_label,
            "confidence": confidence
        }
    
    def _predict_heuristic(self, text: str) -> Dict[str, Any]:
        """Heuristic-based prediction when no model is loaded"""
        score = calculate_urgency_score(text, URGENCY_KEYWORDS)
        label = 1 if score >= 0.4 else 0
        
        return {
            "label": label,
            "confidence": score
        }
    
    @property
    def is_loaded(self) -> bool:
        return self.model is not None or self.model_type == "heuristic"


# =============================================================================
# FastAPI Application
# =============================================================================
app = FastAPI(
    title="Thai Flood Relief NLP API",
    description="API for classifying and extracting information from Thai flood relief posts",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
model_manager = ModelManager()


# =============================================================================
# API Endpoints
# =============================================================================
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.is_loaded,
        model_type=model_manager.model_type,
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.is_loaded,
        model_type=model_manager.model_type,
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Classify a single text as urgent or not urgent
    
    Also extracts phone numbers, coordinates, and location information.
    """
    text = request.text
    
    # Get prediction
    prediction = model_manager.predict(text)
    
    # Calculate urgency score
    urgency_score = calculate_urgency_score(text, URGENCY_KEYWORDS)
    
    # Extract information if requested
    extracted_info = None
    if request.extract_info:
        extracted_info = {
            "phones": extract_phone_numbers(text),
            "coordinates": extract_coordinates(text),
            "location_line": extract_location_line(text),
        }
    
    return PredictResponse(
        label=prediction["label"],
        label_name=LABELS.get(prediction["label"], "unknown"),
        confidence=prediction.get("confidence"),
        urgency_score=urgency_score,
        extracted_info=extracted_info
    )


@app.post("/api/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """
    Classify multiple texts at once
    """
    predictions = []
    
    for text in request.texts:
        prediction = model_manager.predict(text)
        urgency_score = calculate_urgency_score(text, URGENCY_KEYWORDS)
        
        extracted_info = None
        if request.extract_info:
            extracted_info = {
                "phones": extract_phone_numbers(text),
                "coordinates": extract_coordinates(text),
                "location_line": extract_location_line(text),
            }
        
        predictions.append(PredictResponse(
            label=prediction["label"],
            label_name=LABELS.get(prediction["label"], "unknown"),
            confidence=prediction.get("confidence"),
            urgency_score=urgency_score,
            extracted_info=extracted_info
        ))
    
    return BatchPredictResponse(
        predictions=predictions,
        total=len(predictions)
    )


@app.post("/api/extract", response_model=ExtractResponse)
async def extract_info(request: ExtractRequest):
    """
    Extract information from text without classification
    """
    text = request.text
    
    return ExtractResponse(
        phones=extract_phone_numbers(text),
        coordinates=extract_coordinates(text),
        location_line=extract_location_line(text),
        urgency_score=calculate_urgency_score(text, URGENCY_KEYWORDS),
        cleaned_text=basic_clean(text)
    )


@app.get("/api/model/info")
async def model_info():
    """
    Get information about the loaded model
    """
    return {
        "model_type": model_manager.model_type,
        "is_loaded": model_manager.is_loaded,
        "device": model_manager.device if hasattr(model_manager, 'device') else "cpu",
        "labels": LABELS,
    }


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """Run the API server"""
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()

