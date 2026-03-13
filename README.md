## Overview

A production-grade jersey authentication API that combines CLIP image embeddings, 
Supabase pgvector similarity search, and OpenAI GPT-4o Vision to determine whether 
a submitted jersey is authentic or counterfeit.

**How it works:**
1. Uploaded jersey images are converted to 512-dimensional CLIP embeddings
2. Cosine similarity search runs against a labeled reference database of authentic 
   and counterfeit jerseys stored in Supabase
3. Optionally, GPT-4o performs paired visual comparison between the user's image 
   and its closest reference match, returning a structured verdict

**Three endpoints with graduated speed/accuracy tradeoffs:**
- `/authenticate-multiple` — vector search only (~1–3s, fast)
- `/authenticate-multiple-vlm` — vector search + single GPT-4o synthesis (~10s)
- `/authenticate-multiple-vlm-pairs` — per-image paired VLM comparison + final 
  comprehensive analysis (~15–30s, most detailed)

**Built with:** FastAPI · OpenAI GPT-4o · CLIP (sentence-transformers) · 
Supabase (pgvector) · asyncio · Pydantic

## Quick Start

### System Requirements
- **Python**: 3.9+
- **OpenAI API Key**: Required for VLM analysis
- **GPU**: Optional (CLIP model runs on CPU)

### Installation

1. **Create Virtual Environment**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. **Install Dependencies**
```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install all requirements (quick install, no large models)
pip install -r requirements.txt
```

3. **Configure Environment**
Create `.env` file:
```
SUPABASE_URL=your-supabase-url
SUPABASE_SERVICE_KEY=your-service-key
OpenAI_API=your-openai-api-key
```

4. **Upload Reference Images**
```powershell
python upload_references_supabase.py
```

5. **Start API Server**
```powershell
uvicorn app.main:app --reload
```

Server will start at: http://127.0.0.1:8000

## Key Features

- **Fast Vector Search**: 1-2 seconds per image with parallel processing
- **Cloud-Based VLM**: OpenAI - no local GPU required
- **Three Analysis Modes**: Vector-only (fast, free), Comprehensive VLM (all images analyzed together), Paired VLM (side-by-side comparisons + final analysis) NEW
- **Batch Processing**: Handle up to 10 images at once
- **Side-by-Side Comparison**: Compare each user image to best reference match
- **Human-Like Analysis**: Natural language explanations without technical disclaimers
- **Brief + Comprehensive**: Get both per-image feedback and overall synthesis
- **Reference URLs**: Direct links to matched reference images
- **Weighted Verdict**: Combines VLM and vector similarity intelligently 
