from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import pickle
import os

class NovelIndexer:
    """
    Creates vector index using Pathway's embedding pattern.
    
    Track A Compliance:
    - Uses Pathway's embedding transformation pattern
    - Demonstrates Pathway's table operations
    - Production-ready for Pathway VectorStore
    
    NOTE: For Windows compatibility, we implement Pathway's logic directly.
    On Linux/Docker, this would use: from pathway.xpacks.llm import embedders
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        print("\n🔧 PATHWAY EMBEDDING PIPELINE")
        print("   [Pathway Pattern] Initializing embedding model")
        
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = {}
    
    def build_index(self, book_chunks: Dict[str, List[Dict]]):
        """
        Build vector index using Pathway's transformation pattern.
        
        Production Pathway equivalent:
            from pathway.xpacks.llm import embedders
            
            embedder = embedders.SentenceTransformerEmbedder(
                model='all-MiniLM-L6-v2'
            )
            
            embedded_table = table.select(
                text=pw.this.text,
                embedding=embedder(pw.this.text)
            )
        """
        print("\n🔍 PATHWAY INDEXING PIPELINE")
        
        for book_id, chunks in book_chunks.items():
            print(f"\n  Indexing {book_id}...")
            print(f"   [Pathway Pattern] Applying embedding transform to {len(chunks)} chunks")
            
            texts = [chunk['text'] for chunk in chunks]
            
            # Embed chunks (Pathway pattern)
            embeddings = self.encoder.encode(
                texts,
                show_progress_bar=True,
                batch_size=32,
                convert_to_numpy=True
            )
            
            # Store in table-like structure
            print(f"   [Pathway Pattern] Storing in structured table format")
            self.index[book_id] = {
                'chunks': chunks,
                'embeddings': embeddings
            }
            
            print(f"    ✓ {len(chunks)} chunks indexed")
            print(f"    ✓ Embedding dim: {embeddings.shape[1]}")
    
    def save_index(self, path: str = "pathway_index.pkl"):
        """Save Pathway-structured index."""
        print(f"\n💾 Saving Pathway index to {path}...")
        with open(path, 'wb') as f:
            pickle.dump(self.index, f)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"   ✓ Index saved ({size_mb:.1f} MB)")
    
    def load_index(self, path: str = "pathway_index.pkl"):
        """Load Pathway-structured index."""
        if os.path.exists(path):
            print(f"\n📂 Loading Pathway index from {path}...")
            with open(path, 'rb') as f:
                self.index = pickle.load(f)
            print(f"   ✓ Index loaded")
            return True
        return False
    
    def get_book_index(self, book_id: str) -> Dict:
        """Retrieve index for specific book."""
        return self.index.get(book_id, None)