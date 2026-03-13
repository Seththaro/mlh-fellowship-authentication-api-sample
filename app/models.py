# Data Models - Pydantic models for API requests/responses

from pydantic import BaseModel, Field
from typing import List, Optional


class NearestMatch(BaseModel):
    # Single matching reference image from vector database
    image_path: str = Field(..., description="Reference image path in storage")
    similarity_score: float = Field(..., description="Similarity score (0-1, higher is better)")
    model: str = Field(..., description="Model type (e.g., 'adidas:authentic')")
    feature_type: str = Field(..., description="Feature analyzed (front/back/crest/tag)")


class AuthenticationResponse(BaseModel):
    # Authentication result for a single image (vector search only)
    identified_model: str = Field(..., description="Matched model from vector search")
    auth_confidence: float = Field(..., description="Confidence score (0-1)")
    nearest_matches: List[NearestMatch] = Field(..., description="Top similar reference images")
    summary: str = Field(..., description="Human-readable explanation")
    verdict: str = Field(..., description="Final verdict: authentic/counterfeit/uncertain")


class ImageResult(BaseModel):
    # Result for one uploaded image in multi-image request
    filename: str = Field(..., description="Original filename of uploaded image")
    result: AuthenticationResponse = Field(..., description="Authentication result for this image")


class MultiImageAuthenticationResponse(BaseModel):
    # Combined results for multiple images (vector search only)
    image_results: List[ImageResult] = Field(..., description="Per-image authentication results")
    overall_verdict: str = Field(..., description="Overall verdict across all images")
    overall_confidence: float = Field(..., description="Overall confidence score (0-1)")
    summary: str = Field(..., description="Aggregated summary explanation")


class BestMatch(BaseModel):
    # Single best-matching reference image for paired VLM analysis
    image_path: str = Field(..., description="URL to reference image in Supabase Storage")
    reference_image_base64: Optional[str] = Field(None, description="Base64 encoded reference image (optional)")
    similarity_score: float = Field(..., description="Vector similarity score (0-1)")
    model: str = Field(..., description="Model type (e.g., 'adidas:authentic')")
    feature_type: str = Field(..., description="Feature type (front/back/crest/tag)")
    authenticity: str = Field(..., description="authentic or counterfeit")


class VLMPairAnalysis(BaseModel):
    # VLM analysis result for one user-reference pair
    comparison_text: str = Field(..., description="Detailed comparison from VLM (~100 words)")
    authenticity_verdict: str = Field(..., description="VLM verdict: authentic/counterfeit/uncertain")
    confidence_level: str = Field(..., description="high/medium/low")
    key_observations: List[str] = Field(..., description="List of specific observations")
    processing_time: float = Field(..., description="VLM processing time in seconds")


class PairedImageResult(BaseModel):
    # Result for one user image with paired VLM analysis
    filename: str = Field(..., description="Original filename")
    user_image_base64: Optional[str] = Field(None, description="Base64 of user image (optional)")
    vector_search_results: AuthenticationResponse = Field(..., description="All vector search data")
    best_match: BestMatch = Field(..., description="Highest-scoring reference image")
    vlm_pair_analysis: VLMPairAnalysis = Field(..., description="VLM comparison of the pair")
    processing_time: float = Field(..., description="Total processing time for this image")


class PairedVLMResponse(BaseModel):
    # Response for /authenticate-multiple-vlm-pairs endpoint
    image_results: List[PairedImageResult] = Field(..., description="Per-image results with pairs")
    overall_verdict: str = Field(..., description="Overall verdict (VLM-weighted)")
    overall_confidence: float = Field(..., description="Overall confidence score (0-1)")
    summary: str = Field(..., description="Aggregated summary")
    final_vlm_analysis: str = Field(..., description="Comprehensive final VLM analysis of all pairs")
    total_vlm_processing_time: float = Field(..., description="Total time for all VLM calls")
    total_images_analyzed: int = Field(..., description="Number of images processed")
