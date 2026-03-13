# Jersey Authentication API - Main Application
# Uses CLIP embeddings, Supabase vector database, and OpenAI Vision API

from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import asyncio
import time

from app.models import MultiImageAuthenticationResponse, PairedVLMResponse
from app.embeddings import embedder
from app.vector_store_supabase import supabase_vector_store
from app.auth_logic import auth_logic
from app.config import config
from app.vlm_analyzer import vlm_analyzer

app = FastAPI(
    title="Jersey Authentication API",
    description="Authenticate jerseys using AI and Supabase vector database with optional VLM analysis",
    version="2.0.0"
)

@app.on_event("startup")
async def startup_event():
    print("Starting Jersey Authentication API...")
    try:
        embedder.load_model()  # Load CLIP model
        supabase_vector_store.connect()  # Connect to Supabase
        print("✓ All services ready")
    except Exception as e:
        print(f"✗ Startup error: {e}")
        raise

@app.post("/authenticate-multiple", response_model=MultiImageAuthenticationResponse)
async def authenticate_multiple_images(files: List[UploadFile] = File(...)):
    # Authenticate multiple images using vector search only (FAST - 1-3 seconds)
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 images allowed")

        for file in files:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail=f"{file.filename} must be an image (JPG/PNG)")

        print(f"Processing {len(files)} images...")

        async def process_single_image(file: UploadFile):
            image_bytes = await file.read()
            query_embedding = embedder.generate_embedding(image_bytes)

            # Get 2x TOP_K to have enough matches for filtering
            all_matches = await asyncio.to_thread(
                supabase_vector_store.query_vectors,
                query_embedding,
                config.TOP_K_MATCHES * 2
            )

            # Separate authentic and counterfeit matches
            authentic_matches = [m for m in all_matches if 'authentic' in m['metadata']['authenticity'].lower()]
            counterfeit_matches = [m for m in all_matches if 'counterfeit' in m['metadata']['authenticity'].lower()]

            # Combine and get top matches
            all_matches = authentic_matches + counterfeit_matches
            all_matches.sort(key=lambda x: x['score'], reverse=True)
            top_matches = all_matches[:config.TOP_K_MATCHES]

            result = auth_logic.process_authentication(top_matches)

            return {'filename': file.filename, 'result': result}

        # Process all images in parallel
        results = await asyncio.gather(*[process_single_image(file) for file in files])

        # Aggregate results
        overall_verdict = auth_logic.aggregate_multi_image_results([r['result'] for r in results])

        print(f"Overall verdict: {overall_verdict['verdict']}")

        return {
            'image_results': results,
            'overall_verdict': overall_verdict['verdict'],
            'overall_confidence': overall_verdict['confidence'],
            'summary': overall_verdict['summary']
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/authenticate-multiple-vlm", response_model=dict)
async def authenticate_multiple_images_with_vlm(files: List[UploadFile] = File(...)):
    # Authenticate with VLM analysis (SLOWER but detailed)
    # Process: 1) Vector search all images, 2) Calculate confidence, 3) ONE VLM analysis at end
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 images allowed")

        for file in files:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail=f"{file.filename} must be an image (JPG/PNG)")

        print(f"\n{'='*70}")
        print(f"PROCESSING {len(files)} IMAGES WITH VLM ANALYSIS")
        print(f"{'='*70}\n")

        # PHASE 1: Vector search for all images
        print("PHASE 1: Vector similarity search for all images...\n")

        async def process_single_image_vector(file: UploadFile, index: int):
            print(f"[{index + 1}/{len(files)}] Processing: {file.filename}")
            start_time = time.time()

            image_bytes = await file.read()
            query_embedding = embedder.generate_embedding(image_bytes)

            matches = await asyncio.to_thread(
                supabase_vector_store.query_vectors,
                query_embedding,
                config.TOP_K_MATCHES
            )

            result = auth_logic.process_authentication(matches)

            # Get metadata for VLM later
            top_match = matches[0] if matches else None
            feature_type = top_match['metadata']['feature_type'] if top_match else 'front'
            authenticity = top_match['metadata']['authenticity'] if top_match else 'unknown'

            processing_time = time.time() - start_time

            print(f"  - Vector confidence: {result.auth_confidence:.2%}")
            print(f"  - Verdict: {result.verdict}")
            print(f"  - Processing time: {processing_time:.2f}s\n")

            return {
                'filename': file.filename,
                'image_bytes': image_bytes,
                'identified_model': result.identified_model,
                'auth_confidence': result.auth_confidence,
                'nearest_matches': [
                    {
                        'image_path': m.image_path,
                        'similarity_score': m.similarity_score,
                        'model': m.model,
                        'feature_type': m.feature_type
                    }
                    for m in result.nearest_matches
                ],
                'summary': result.summary,
                'verdict': result.verdict,
                'feature_type': feature_type,
                'authenticity': authenticity,
                'processing_time': processing_time
            }

        # Process all images in parallel
        results = await asyncio.gather(*[
            process_single_image_vector(file, i) for i, file in enumerate(files)
        ])

        # PHASE 2: Aggregate vector results
        print("PHASE 2: Calculating overall vector-based verdict...\n")

        from app.models import AuthenticationResponse, NearestMatch
        auth_responses = []
        for r in results:
            auth_responses.append(
                AuthenticationResponse(
                    identified_model=r['identified_model'],
                    auth_confidence=r['auth_confidence'],
                    nearest_matches=[NearestMatch(**m) for m in r['nearest_matches']],
                    summary=r['summary'],
                    verdict=r['verdict']
                )
            )

        overall_verdict = auth_logic.aggregate_multi_image_results(auth_responses)

        print(f"Vector-based overall verdict: {overall_verdict['verdict']}")
        print(f"Vector-based confidence: {overall_verdict['confidence']:.2%}\n")

        # PHASE 3: VLM analysis on ALL images together
        print("PHASE 3: Running comprehensive VLM analysis on all images...\n")

        vlm_start_time = time.time()

        # Prepare data for VLM
        images_for_vlm = []
        for r in results:
            images_for_vlm.append({
                'image_bytes': r['image_bytes'],
                'filename': r['filename'],
                'feature_type': r['feature_type'],
                'authenticity': r['authenticity'],
                'confidence': r['auth_confidence']
            })

        # Run VLM analysis ONCE on ALL images
        print(f"Analyzing {len(images_for_vlm)} images together with OpenAI Vision API...")
        vlm_analysis = await asyncio.to_thread(
            vlm_analyzer.analyze_multiple_jerseys,
            images_for_vlm
        )

        vlm_processing_time = time.time() - vlm_start_time
        print(f"[OK] VLM analysis completed in {vlm_processing_time:.2f}s\n")

        # PHASE 4: Prepare final response
        print("PHASE 4: Preparing final response...\n")

        # Remove image_bytes from response (don't send binary data in JSON)
        final_results = []
        for r in results:
            result_copy = {k: v for k, v in r.items() if k != 'image_bytes'}
            final_results.append(result_copy)

        print(f"{'='*70}")
        print(f"FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Overall Verdict: {overall_verdict['verdict'].upper()}")
        print(f"Overall Confidence: {overall_verdict['confidence']:.2%}")
        print(f"Images Analyzed: {len(results)}")
        print(f"VLM Analysis Time: {vlm_processing_time:.2f}s")
        print(f"{'='*70}\n")

        return {
            'image_results': final_results,
            'overall_verdict': overall_verdict['verdict'],
            'overall_confidence': overall_verdict['confidence'],
            'summary': overall_verdict['summary'],
            'vlm_analysis': vlm_analysis,  # Single analysis for ALL images
            'vlm_processing_time': vlm_processing_time,
            'total_images_analyzed': len(results)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def calculate_paired_overall_verdict(vector_results: list, vlm_analyses: list) -> dict:

    # Count verdicts from VLM (higher weight)
    vlm_verdicts = [v['authenticity_verdict'] for v in vlm_analyses]
    vlm_authentic = vlm_verdicts.count('authentic')
    vlm_counterfeit = vlm_verdicts.count('counterfeit')

    # Count verdicts from vector search
    vector_verdicts = [v['vector_result'].verdict for v in vector_results]
    vector_authentic = vector_verdicts.count('authentic')
    vector_counterfeit = vector_verdicts.count('counterfeit')

    # Weighted scoring (VLM: 60%, Vector: 40%)
    authentic_score = (vlm_authentic * 0.6) + (vector_authentic * 0.4)
    counterfeit_score = (vlm_counterfeit * 0.6) + (vector_counterfeit * 0.4)

    # Calculate average confidence
    vector_confidences = [v['vector_result'].auth_confidence for v in vector_results]
    avg_vector_conf = sum(vector_confidences) / len(vector_confidences)

    # VLM confidence mapping
    vlm_conf_map = {'high': 0.9, 'medium': 0.7, 'low': 0.5}
    vlm_confidences = [vlm_conf_map.get(v['confidence_level'], 0.7) for v in vlm_analyses]
    avg_vlm_conf = sum(vlm_confidences) / len(vlm_confidences)

    # Weighted average confidence
    overall_confidence = (avg_vlm_conf * 0.6) + (avg_vector_conf * 0.4)

    # Determine verdict
    total = len(vector_results)
    if authentic_score > counterfeit_score:
        verdict = 'authentic'
        summary = f"Based on {total} images with paired VLM analysis: {vlm_authentic} VLM authentic, {vlm_counterfeit} VLM counterfeit. Assessment: likely authentic."
    elif counterfeit_score > authentic_score:
        verdict = 'counterfeit'
        summary = f"Based on {total} images with paired VLM analysis: {vlm_counterfeit} VLM counterfeit, {vlm_authentic} VLM authentic. Assessment: likely counterfeit."
    else:
        verdict = 'uncertain'
        summary = f"Based on {total} images with paired VLM analysis: mixed signals. Unable to determine with confidence."

    return {
        'verdict': verdict,
        'confidence': overall_confidence,
        'summary': summary
    }


@app.post("/authenticate-multiple-vlm-pairs", response_model=PairedVLMResponse)
async def authenticate_multiple_images_with_paired_vlm(
    files: List[UploadFile] = File(...)
):
    """
    Authenticate with PAIRED VLM analysis (user image + best reference match)

    Process:
    1. Vector search for each image → get TOP_K_MATCHES
    2. Select HIGHEST confidence reference as "best match" for each image
    3. Fetch reference images from Supabase Storage URLs
    4. Perform brief VLM analysis on each (user, reference) PAIR (1-3 sentences)
    5. Perform FINAL comprehensive VLM analysis synthesizing all pairs
    6. Return per-pair brief analysis + final comprehensive analysis + overall verdict
    """
    try:
        # Validation
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 images allowed")
        for file in files:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail=f"{file.filename} must be an image")

        print(f"\n{'='*70}")
        print(f"PAIRED VLM ANALYSIS: {len(files)} images")
        print(f"{'='*70}\n")

        # PHASE 1: Vector search + select best matches
        print("PHASE 1: Vector search and best match selection...\n")

        async def process_single_image_vector(file: UploadFile, index: int):
            print(f"[{index + 1}/{len(files)}] Processing: {file.filename}")
            start_time = time.time()

            image_bytes = await file.read()
            query_embedding = embedder.generate_embedding(image_bytes)

            # Get TOP_K_MATCHES
            matches = await asyncio.to_thread(
                supabase_vector_store.query_vectors,
                query_embedding,
                config.TOP_K_MATCHES
            )

            # Process authentication result (all matches)
            result = auth_logic.process_authentication(matches)

            # Select BEST match (highest score)
            best_match = matches[0] if matches else None
            if not best_match:
                raise HTTPException(status_code=500, detail=f"No matches found for {file.filename}")

            processing_time = time.time() - start_time
            print(f"  - Best match: {best_match['metadata']['image_path']}")
            print(f"  - Similarity: {best_match['score']:.4f}")
            print(f"  - Processing time: {processing_time:.2f}s\n")

            return {
                'filename': file.filename,
                'image_bytes': image_bytes,
                'vector_result': result,
                'best_match': best_match,
                'processing_time': processing_time
            }

        # Process all images in parallel
        vector_results = await asyncio.gather(*[
            process_single_image_vector(file, i) for i, file in enumerate(files)
        ])

        # PHASE 2: Fetch reference images and prepare pairs
        print("PHASE 2: Fetching reference images from Supabase Storage...\n")

        async def fetch_reference_image(url: str) -> bytes:
            def _fetch():
                import requests
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response.content

            return await asyncio.to_thread(_fetch)

        # Fetch all reference images in parallel
        reference_fetch_tasks = [
            fetch_reference_image(result['best_match']['metadata']['image_path'])
            for result in vector_results
        ]

        try:
            reference_images = await asyncio.gather(*reference_fetch_tasks)
            print(f"[OK] Fetched {len(reference_images)} reference images\n")
        except Exception as e:
            print(f"[ERROR] Failed to fetch reference images: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch reference images: {str(e)}")

        # PHASE 3: VLM analysis on pairs (parallel)
        print("PHASE 3: Running VLM analysis on image pairs...\n")

        vlm_start_time = time.time()

        async def analyze_pair(result: dict, ref_image_bytes: bytes, index: int):
            print(f"[{index + 1}/{len(files)}] Analyzing pair: {result['filename']}")
            pair_start_time = time.time()

            user_data = {
                'image_bytes': result['image_bytes'],
                'filename': result['filename'],
                'feature_type': result['best_match']['metadata']['feature_type'],
                'confidence': result['vector_result'].auth_confidence
            }

            ref_data = {
                'image_bytes': ref_image_bytes,
                'image_path': result['best_match']['metadata']['image_path'],
                'model': result['best_match']['metadata']['model'],
                'feature_type': result['best_match']['metadata']['feature_type'],
                'authenticity': result['best_match']['metadata']['authenticity']
            }

            # Call VLM analyzer
            vlm_result = await asyncio.to_thread(
                vlm_analyzer.analyze_jersey_pair,
                user_data,
                ref_data
            )

            pair_processing_time = time.time() - pair_start_time
            vlm_result['processing_time'] = pair_processing_time

            print(f"  - VLM verdict: {vlm_result['authenticity_verdict']}")
            print(f"  - Confidence: {vlm_result['confidence_level']}")
            print(f"  - Processing time: {pair_processing_time:.2f}s\n")

            return vlm_result

        # Analyze all pairs in parallel
        vlm_analyses = await asyncio.gather(*[
            analyze_pair(result, ref_bytes, i)
            for i, (result, ref_bytes) in enumerate(zip(vector_results, reference_images))
        ])

        total_vlm_time = time.time() - vlm_start_time
        print(f"[OK] All pair analyses completed in {total_vlm_time:.2f}s\n")

        # PHASE 4: Final comprehensive VLM analysis
        print("PHASE 4: Running final comprehensive VLM analysis...\n")

        # Prepare data for final analysis
        user_images_data = [{
            'filename': result['filename'],
            'feature_type': result['best_match']['metadata']['feature_type']
        } for result in vector_results]

        final_analysis_start = time.time()
        final_vlm_analysis = await asyncio.to_thread(
            vlm_analyzer.analyze_all_pairs_final,
            vlm_analyses,
            user_images_data
        )
        final_analysis_time = time.time() - final_analysis_start
        total_vlm_time += final_analysis_time

        print(f"[OK] Final analysis completed in {final_analysis_time:.2f}s\n")

        # PHASE 5: Build final response
        print("PHASE 5: Building final response...\n")

        final_results = []
        for result, ref_bytes, vlm_analysis in zip(vector_results, reference_images, vlm_analyses):
            paired_result = {
                'filename': result['filename'],
                'user_image_base64': None,  # Not included - just URLs
                'vector_search_results': result['vector_result'],
                'best_match': {
                    'image_path': result['best_match']['metadata']['image_path'],
                    'reference_image_base64': None,  # Not included - just URLs
                    'similarity_score': result['best_match']['score'],
                    'model': result['best_match']['metadata']['model'],
                    'feature_type': result['best_match']['metadata']['feature_type'],
                    'authenticity': result['best_match']['metadata']['authenticity']
                },
                'vlm_pair_analysis': vlm_analysis,
                'processing_time': result['processing_time'] + vlm_analysis['processing_time']
            }
            final_results.append(paired_result)

        # Calculate overall verdict (weighted by VLM + vector)
        overall_verdict_data = calculate_paired_overall_verdict(vector_results, vlm_analyses)

        print(f"{'='*70}")
        print(f"FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Overall Verdict: {overall_verdict_data['verdict'].upper()}")
        print(f"Overall Confidence: {overall_verdict_data['confidence']:.2%}")
        print(f"VLM Processing Time: {total_vlm_time:.2f}s")
        print(f"{'='*70}\n")

        return {
            'image_results': final_results,
            'overall_verdict': overall_verdict_data['verdict'],
            'overall_confidence': overall_verdict_data['confidence'],
            'summary': overall_verdict_data['summary'],
            'final_vlm_analysis': final_vlm_analysis,
            'total_vlm_processing_time': total_vlm_time,
            'total_images_analyzed': len(files)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))