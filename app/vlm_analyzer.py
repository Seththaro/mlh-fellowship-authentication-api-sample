# VLM Analyzer - Uses OpenAI GPT-4 Vision API to analyze jersey authenticity

import base64
import io
from PIL import Image
from typing import Union
from openai import OpenAI
from app.config import Config


class VLMAnalyzer:

    def __init__(self):
        self.client = None
        self.model_loaded = False

    def load_model(self):
        # Initialize OpenAI client (no actual model download needed)
        if self.model_loaded:
            return

        print("Initializing OpenAI Vision API...")
        try:
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            self.model_loaded = True
            print("[OK] OpenAI Vision API ready!")
        except Exception as e:
            print(f"[ERROR] Error initializing OpenAI API: {e}")
            raise

    def unload_model(self):
        # Clean up client
        if self.client is not None:
            self.client = None
            self.model_loaded = False
            print("[OK] OpenAI client cleared")

    def _load_image(self, image_input: Union[str, bytes, Image.Image]) -> Image.Image:
        # Convert various image formats to PIL Image
        if isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise ValueError("Invalid image format. Use file path, bytes, or PIL Image")

    def _image_to_base64(self, image: Image.Image) -> str:
        # Convert PIL Image to base64 string for OpenAI API
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def analyze_multiple_jerseys(self, images_data: list) -> str:
        # Analyze ALL images together - returns ONE comprehensive ~100 word analysis
        self.load_model()

        # Convert all images to base64
        images_base64 = []
        for img_data in images_data:
            upload_img = self._load_image(img_data['image_bytes'])
            image_base64 = self._image_to_base64(upload_img)
            images_base64.append({
                'base64': image_base64,
                'filename': img_data['filename'],
                'feature_type': img_data['feature_type'],
                'authenticity': img_data['authenticity'],
                'confidence': img_data['confidence']
            })

        # Generate prompt and analyze
        prompt = self._get_multi_image_analysis_prompt(images_base64)
        analysis = self._analyze_multiple_with_openai(images_base64, prompt)
        return analysis

    def _get_multi_image_analysis_prompt(self, images_base64: list) -> str:
        # Generate prompt for analyzing multiple images together
        image_summary = []
        for i, img in enumerate(images_base64, 1):
            image_summary.append(
                f"Image {i} ({img['filename']}): "
                f"{img['feature_type']}, "
                f"vector similarity confidence {img['confidence']:.1%}"
            )

        images_text = "\n".join(image_summary)

        prompt = f"""You are an expert in authenticating hockey jerseys. You have been provided with {len(images_base64)} images of a jersey submission.

IMAGES PROVIDED:
{images_text}

TASK:
Analyze ALL the provided images together and provide a single comprehensive assessment (approximately 100 words) of whether this jersey submission appears AUTHENTIC or COUNTERFEIT.

Consider:
- Overall construction quality across all views
- Consistency of materials and craftsmanship
- Quality of stitching, printing, and finishing
- Alignment with expected authentic characteristics
- Any red flags or concerning details

IMPORTANT: Provide ONE unified verdict for the entire submission, not separate verdicts for each image. Focus on the overall impression from viewing all images together.

Conclude with a clear final verdict: AUTHENTIC or COUNTERFEIT, with key supporting evidence."""

        return prompt

    def _analyze_multiple_with_openai(self, images_base64: list, prompt: str) -> str:
        # Send multiple images + prompt to OpenAI API
        try:
            # Build content with prompt + all images
            content = [{"type": "text", "text": prompt}]

            for img_data in images_base64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_data['base64']}"}
                })

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                max_tokens=300,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] Error calling OpenAI API for multiple images: {e}")
            raise

    def analyze_jersey_pair(self, user_image_data: dict, reference_image_data: dict) -> dict:

        self.load_model()

        # Convert both images to base64
        user_img = self._load_image(user_image_data['image_bytes'])
        user_base64 = self._image_to_base64(user_img)

        ref_img = self._load_image(reference_image_data['image_bytes'])
        ref_base64 = self._image_to_base64(ref_img)

        # Generate pair-specific prompt
        prompt = self._get_pair_comparison_prompt(user_image_data, reference_image_data)

        # Call OpenAI with both images
        analysis = self._analyze_pair_with_openai(user_base64, ref_base64, prompt)

        # Parse response into structured format
        parsed = self._parse_pair_analysis(analysis)

        return parsed

    def _get_pair_comparison_prompt(self, user_data: dict, ref_data: dict) -> str:
        prompt = f"""You are an expert hockey jersey authenticator. Analyze these two images side-by-side:

IMAGE 1 (USER SUBMISSION): {user_data['filename']} - {user_data['feature_type']}
IMAGE 2 (REFERENCE): {ref_data['authenticity'].upper()} {ref_data['model']} - {ref_data['feature_type']}

Compare the images and provide your assessment in this format:

VERDICT: [AUTHENTIC/COUNTERFEIT/UNCERTAIN]
CONFIDENCE: [HIGH/MEDIUM/LOW]
COMPARISON: [Write 1-3 specific sentences about what you observe. Compare stitching quality, logo details, material appearance, color accuracy, and any notable differences. Be direct and specific - describe what you see, not what you can't do. Focus on tangible observations that help the user understand the comparison.]

IMPORTANT: Never mention being unable to view images. You ARE viewing them - describe specific visual details you observe in each image.
"""
        return prompt

    def _analyze_pair_with_openai(self, user_base64: str, ref_base64: str, prompt: str) -> str:
        try:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{user_base64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref_base64}"}}
            ]

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                max_tokens=200,
                temperature=0.5
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] OpenAI API error: {e}")
            raise

    def _parse_pair_analysis(self, analysis_text: str) -> dict:
        import re

        # Extract verdict
        verdict_match = re.search(r'VERDICT:\s*(AUTHENTIC|COUNTERFEIT|UNCERTAIN)', analysis_text, re.IGNORECASE)
        verdict = verdict_match.group(1).lower() if verdict_match else 'uncertain'

        # Extract confidence
        confidence_match = re.search(r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)', analysis_text, re.IGNORECASE)
        confidence = confidence_match.group(1).lower() if confidence_match else 'medium'

        # Extract comparison text (short 1-3 sentences)
        comparison_match = re.search(r'COMPARISON:\s*(.+?)(?:\n\n|\Z)', analysis_text, re.DOTALL | re.IGNORECASE)
        comparison = comparison_match.group(1).strip() if comparison_match else analysis_text

        return {
            'comparison_text': comparison,
            'authenticity_verdict': verdict,
            'confidence_level': confidence,
            'key_observations': []  # Not used in brief comparison
        }

    def analyze_all_pairs_final(self, pair_analyses: list, user_images_data: list) -> str:

        self.load_model()

        # Build summary of pair comparisons
        summary_parts = []
        for i, (analysis, img_data) in enumerate(zip(pair_analyses, user_images_data), 1):
            summary_parts.append(
                f"Pair {i} ({img_data['filename']} - {img_data['feature_type']}): "
                f"Verdict: {analysis['authenticity_verdict'].upper()}, "
                f"Confidence: {analysis['confidence_level'].upper()}"
            )

        pairs_summary = "\n".join(summary_parts)

        prompt = f"""You are an experienced hockey jersey authenticator providing expert analysis. You've examined {len(pair_analyses)} images from this submission.

INDIVIDUAL ASSESSMENTS:
{pairs_summary}

Write a clear, direct final analysis (100-150 words) in a conversational tone that:
- Summarizes what you observed across all images
- Points out any patterns (consistent quality, red flags, or mixed signals)
- Explains your reasoning for the final verdict
- Sounds human and natural (avoid overly formal language)
- Gives actionable insight without being overly cautious

Be specific about what you saw - mention actual details like stitching patterns, logo quality, material appearance, or color accuracy. Avoid phrases like "I cannot determine" or "unable to assess." Instead, say what you DO observe and what it suggests.

If results are mixed or unclear, explain WHY (e.g., "The crest shows authentic stitching but the tag has inconsistencies") rather than just saying you're uncertain. Help the user understand your thought process.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.6
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] Error calling OpenAI API for final analysis: {e}")
            raise


# Global instance
vlm_analyzer = VLMAnalyzer()