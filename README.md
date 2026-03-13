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
