import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer

class EvidenceRetriever:
    """Retrieves relevant evidence chunks for claims."""
    
    def __init__(self, indexer, config: dict):
        self.indexer = indexer
        self.encoder = indexer.encoder  # Reuse same encoder
        self.top_k = config['retrieval']['top_k']
        self.threshold = config['retrieval']['similarity_threshold']
    
    def retrieve(self, query: str, book_id: str) -> List[Dict]:
        """
        Retrieve top-k most relevant chunks for a query.
        
        Returns: List of chunk dicts with similarity scores
        """
        book_index = self.indexer.get_book_index(book_id)
        if not book_index:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query])[0]
        
        # Compute similarities
        similarities = np.dot(
            book_index['embeddings'],
            query_embedding
        ) / (
            np.linalg.norm(book_index['embeddings'], axis=1) *
            np.linalg.norm(query_embedding)
        )
        
        # Get top-k above threshold
        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] >= self.threshold:
                chunk = book_index['chunks'][idx].copy()
                chunk['similarity'] = float(similarities[idx])
                results.append(chunk)
        
        return results
    
    def retrieve_for_claims(self, claims: List[str], book_id: str) -> Dict[str, List[Dict]]:
        """
        Retrieve evidence for multiple claims.
        
        Returns: {claim: [evidence_chunks]}
        """
        evidence_map = {}
        
        for claim in claims:
            evidence_map[claim] = self.retrieve(claim, book_id)
        
        return evidence_map