# Supabase Vector Store - Handles database operations for vector similarity search

from supabase import create_client, Client
from typing import List, Dict, Any
from app.config import config


class SupabaseVectorStore:

    def __init__(self):
        self.client: Client = None

    def connect(self):
        # Connect to Supabase database
        if self.client is None:
            config.validate()
            self.client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
            print(f"✓ Connected to Supabase")

    def upsert_vectors(self, vectors: List[Dict[str, Any]], table_name: str):
        # Upload image embeddings to Supabase
        # vectors: List of dicts with 'id', 'values', 'metadata'
        # table_name: 'authentic_jerseys' or 'counterfeit_jerseys'
        self.connect()

        # Convert to Supabase format
        records = []
        for vector in vectors:
            records.append({
                'id': vector['id'],
                'embedding': vector['values'],  # pgvector column (512D array)
                'image_path': vector['metadata'].get('image_path'),
                'model': vector['metadata'].get('model'),
                'model_name': vector['metadata'].get('model_name'),
                'feature_type': vector['metadata'].get('feature_type'),
                'authenticity': vector['metadata'].get('authenticity')
            })

        # Insert in batches of 100
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            self.client.table(table_name).upsert(batch).execute()
            print(f"  Uploaded {min(i + batch_size, len(records))}/{len(records)} to {table_name}")

    def query_vectors(self, query_vector: List[float], top_k: int = 10) -> List[Dict]:
        # Search for similar images using vector similarity
        # query_vector: 512D embedding from CLIP
        # top_k: Number of results to return
        self.connect()

        # Call Supabase RPC function for vector similarity search
        result = self.client.rpc(
            'match_reference_jerseys',
            {
                'query_embedding': query_vector,
                'match_count': top_k
            }
        ).execute()

        # Format results
        matches = []
        for row in result.data:
            matches.append({
                'id': str(row['id']),
                'score': float(row['similarity']),  # Cosine similarity (0-1)
                'metadata': {
                    'image_path': row['image_url'],
                    'model': f"{row['brand']}:{row['authenticity']}",
                    'model_name': row['model_name'],
                    'feature_type': row['feature_type'],
                    'authenticity': row['authenticity']
                }
            })

        return matches

    def get_stats(self, table_name: str) -> Dict:
        # Get database statistics (total vectors, dimension, etc.)
        self.connect()
        result = self.client.table(table_name).select('id', count='exact').execute()
        return {
            'total_vectors': result.count,
            'dimension': config.EMBEDDING_DIMENSION,
            'table': table_name
        }

# Global instance
supabase_vector_store = SupabaseVectorStore()
