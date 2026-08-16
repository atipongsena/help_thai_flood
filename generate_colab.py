import json
import os
import re

# Define paths
PROJECT_ROOT = r"d:\help_thai_flood"
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "nlp_pipeline_colab.ipynb")

# Files to include in order
FILES_TO_INCLUDE = [
    ("utils/config.py", "Configuration"),
    ("utils/preprocessing.py", "Preprocessing Utilities"),
    ("utils/risk_tags.py", "Risk & Tagging Logic"),
    ("utils/features.py", "Feature Extraction"),
    ("training/prepare_dataset.py", "Dataset Preparation"),
    ("training/train_classical.py", "Classical Model Training"),
    ("training/train_bert.py", "BERT Model Training"),
    ("run_pipeline.py", "Main Pipeline Runner"),
]

def read_file(filepath):
    full_path = os.path.join(PROJECT_ROOT, filepath)
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()

def clean_code(code, filename):
    """
    Remove local imports and adjust code for notebook environment.
    """
    lines = code.splitlines()
    cleaned_lines = []
    
    # Imports to remove (local modules)
    local_modules = [
        "utils.config", "utils.preprocessing", "utils.risk_tags", "utils.features",
        "training.prepare_dataset", "training.train_classical", "training.train_bert",
        "api.app"
    ]
    
    for line in lines:
        # Skip local imports
        if any(f"from {mod}" in line for mod in local_modules):
            continue
        if any(f"import {mod}" in line for mod in local_modules):
            continue
        
        # Skip sys.path modifications
        if "sys.path" in line or "PROJECT_ROOT =" in line:
            continue
            
        # Skip if __name__ == "__main__" blocks in modules (except main runner)
        if 'if __name__ == "__main__":' in line and "run_pipeline.py" not in filename:
            break
            
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)

def create_notebook():
    cells = []
    
    # 1. Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Thai Flood Relief NLP Pipeline\n",
            "\n",
            "This notebook contains the complete NLP pipeline for the Thai Flood Relief project.\n",
            "It includes data preparation, preprocessing, model training (Classical & BERT), and evaluation."
        ]
    })
    
    # 2. Install Dependencies
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install required packages\n",
            "!pip install pandas numpy scikit-learn pythainlp python-dotenv gensim transformers torch datasets joblib uvicorn fastapi"
        ]
    })
    
    # 3. Global Imports
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import json\n",
            "import re\n",
            "import html\n",
            "import argparse\n",
            "from datetime import datetime\n",
            "from dataclasses import dataclass, field\n",
            "from typing import List, Optional, Tuple, Dict, Any, Callable\n",
            "import joblib\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import sqlite3\n",
            "\n",
            "# Create directories\n",
            "os.makedirs('data', exist_ok=True)\n",
            "os.makedirs('models', exist_ok=True)\n"
        ]
    })
    
    # 4. Add content from files
    for filepath, title in FILES_TO_INCLUDE:
        code = read_file(filepath)
        cleaned_code = clean_code(code, filepath)
        
        # Add Markdown Header
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## {title}"]
        })
        
        # Add Code
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cleaned_code.splitlines(keepends=True)
        })

    # 5. Notebook Structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    # Write to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
        
    print(f"Notebook generated successfully at: {OUTPUT_FILE}")

if __name__ == "__main__":
    create_notebook()
