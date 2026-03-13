# Authentication Logic - Processes vector matches and determines authenticity

from typing import List, Dict, Tuple
from app.models import NearestMatch, AuthenticationResponse
from app.config import config


class AuthenticationLogic:

    @staticmethod
    def calculate_confidence(matches: List[Dict]) -> Tuple[float, str]:
        # Calculate confidence score from vector matches
        if not matches:
            return 0.0, "unknown"

        # Group matches by model (e.g., "adidas:authentic", "nike:counterfeit")
        model_scores = {}
        for match in matches:
            model = match['metadata'].get('model', 'unknown')
            score = match['score']

            if model not in model_scores:
                model_scores[model] = []
            model_scores[model].append(score)

        # Calculate average score for each model
        model_avg_scores = {}
        for model, scores in model_scores.items():
            model_avg_scores[model] = sum(scores) / len(scores)

        # Find model with highest average score
        identified_model = max(model_avg_scores, key=model_avg_scores.get)
        confidence = model_avg_scores[identified_model]

        return confidence, identified_model

    @staticmethod
    def generate_summary(confidence: float, identified_model: str, matches: List[Dict]) -> str:
        # Generate human-readable summary of authentication result

        # Determine confidence level
        if confidence >= config.CONFIDENCE_HIGH:
            confidence_level = "Strong"
        elif confidence >= config.CONFIDENCE_MEDIUM:
            confidence_level = "Moderate"
        else:
            confidence_level = "Weak"

        # Determine authenticity type
        if "authentic" in identified_model.lower():
            auth_type = "authentic"
            detail = "Features align with genuine jerseys"
        elif "counterfeit" in identified_model.lower():
            auth_type = "counterfeit"
            detail = "Features match known counterfeit patterns"
        else:
            auth_type = "unknown"
            detail = "Unable to determine authenticity"

        # Collect analyzed features (crest, tag, front, etc.)
        features = set()
        for match in matches[:3]:
            feature = match['metadata'].get('feature_type', '')
            if feature:
                features.add(feature)

        feature_text = ", ".join(sorted(features)) if features else "multiple aspects"

        # Build summary text
        summary = f"{confidence_level} visual match to {auth_type} reference images. "
        summary += f"Analysis of {feature_text} shows {detail}."

        return summary

    @staticmethod
    def determine_verdict(confidence: float, identified_model: str) -> str:
        # Determine final verdict: authentic, counterfeit, or uncertain
        if confidence < config.CONFIDENCE_MEDIUM:
            return "uncertain"

        if "authentic" in identified_model.lower():
            return "authentic"
        elif "counterfeit" in identified_model.lower():
            return "counterfeit"
        else:
            return "uncertain"

    @staticmethod
    def process_authentication(matches: List[Dict]) -> AuthenticationResponse:
        # Process vector matches into authentication result
        confidence, identified_model = AuthenticationLogic.calculate_confidence(matches)
        summary = AuthenticationLogic.generate_summary(confidence, identified_model, matches)
        verdict = AuthenticationLogic.determine_verdict(confidence, identified_model)

        # Format matches for response
        nearest_matches = []
        for match in matches:
            nearest_match = NearestMatch(
                image_path=match['metadata'].get('image_path', match['id']),
                similarity_score=round(match['score'], 4),
                model=match['metadata'].get('model', 'unknown'),
                feature_type=match['metadata'].get('feature_type', 'unknown')
            )
            nearest_matches.append(nearest_match)

        return AuthenticationResponse(
            identified_model=identified_model,
            auth_confidence=round(confidence, 4),
            nearest_matches=nearest_matches,
            summary=summary,
            verdict=verdict
        )

    @staticmethod
    def aggregate_multi_image_results(results: List[AuthenticationResponse]) -> Dict:
        # Combine multiple image results into overall verdict
        if not results:
            return {
                'verdict': 'uncertain',
                'confidence': 0.0,
                'summary': 'No images analyzed'
            }

        # Count verdicts
        authentic_count = sum(1 for r in results if r.verdict == 'authentic')
        counterfeit_count = sum(1 for r in results if r.verdict == 'counterfeit')

        # Calculate average confidence
        avg_confidence = sum(r.auth_confidence for r in results) / len(results)

        # Determine overall verdict (majority wins)
        total = len(results)
        if authentic_count > counterfeit_count:
            overall_verdict = 'authentic'
            summary = f"Based on {total} images: {authentic_count} authentic, {counterfeit_count} counterfeit. Assessment: likely authentic."
        elif counterfeit_count > authentic_count:
            overall_verdict = 'counterfeit'
            summary = f"Based on {total} images: {counterfeit_count} counterfeit, {authentic_count} authentic. Assessment: likely counterfeit."
        else:
            overall_verdict = 'uncertain'
            summary = f"Based on {total} images: mixed signals. Unable to determine with confidence."

        return {
            'verdict': overall_verdict,
            'confidence': round(avg_confidence, 4),
            'summary': summary
        }


# Global instance
auth_logic = AuthenticationLogic()
