# Image Embedder - Converts images to 512D vectors using CLIP model

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from typing import Union, List
import io
from app.config import config


class ImageEmbedder:

    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        # Load CLIP model (sentence-transformers)
        if self.model is None:
            print(f"Loading {config.EMBEDDING_MODEL}...")
            self.model = SentenceTransformer(config.EMBEDDING_MODEL)
            self.model.to(self.device)
            print(f"✓ Model loaded on {self.device}")

    def generate_embedding(self, image_input: Union[str, bytes, Image.Image]) -> List[float]:
        # Convert image to 512D vector - accepts file path, bytes, or PIL Image
        self.load_model()

        # Convert to PIL Image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image.convert("RGB")
        else:
            raise ValueError("Image must be path, bytes, or PIL Image")

        # Generate 512D embedding vector
        embedding = self.model.encode(image, convert_to_tensor=True)
        return embedding.cpu().tolist()

    def generate_embeddings_batch(self, image_paths: List[str]) -> List[List[float]]:
        # Process multiple images at once (faster than one-by-one)
        self.load_model()
        images = [Image.open(path).convert("RGB") for path in image_paths]
        embeddings = self.model.encode(images, convert_to_tensor=True, show_progress_bar=True)
        return [emb.cpu().tolist() for emb in embeddings]


# Global instance
embedder = ImageEmbedder()
