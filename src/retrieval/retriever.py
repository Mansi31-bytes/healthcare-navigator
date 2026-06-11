import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from loguru import logger
from typing import Optional
from src.ingestion.chunker import Chunk

class HybridRetriever:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        faiss_path: str = "./data/indexes/faiss.index",
        bm25_path: str = "./data/indexes/bm25.pkl",
        chunks_path: str = "./data/indexes/chunks.pkl",
        top_k_dense: int = 20,
        top_k_bm25: int = 20,
        top_k_reranked: int = 5,
    ):
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        self.reranker = CrossEncoder(reranker_model)
        self.faiss_path = faiss_path
        self.bm25_path = bm25_path
        self.chunks_path = chunks_path
        self.top_k_dense = top_k_dense
        self.top_k_bm25 = top_k_bm25
        self.top_k_reranked = top_k_reranked
        self.index: Optional[faiss.Index] = None
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: list = []

    def build_index(self, chunks: list):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self.embedder.encode(
            texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
        )
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        logger.info(f"FAISS index built: {self.index.ntotal} vectors")
        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built.")
        self._save_indexes()

    def _save_indexes(self):
        os.makedirs(os.path.dirname(self.faiss_path), exist_ok=True)
        faiss.write_index(self.index, self.faiss_path)
        with open(self.bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info("Indexes saved to disk.")

    def load_index(self):
        self.index = faiss.read_index(self.faiss_path)
        with open(self.bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(self.chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        logger.info(f"Loaded index with {len(self.chunks)} chunks.")

    def retrieve(self, query: str) -> list:
        if not self.index or not self.bm25:
            raise RuntimeError("Index not loaded. Call build_index() or load_index() first.")
        q_emb = self.embedder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        scores, indices = self.index.search(q_emb, self.top_k_dense)
        dense_hits = set(indices[0].tolist())
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top = np.argsort(bm25_scores)[::-1][:self.top_k_bm25]
        sparse_hits = set(bm25_top.tolist())
        candidate_idxs = list(dense_hits | sparse_hits)
        candidates = [self.chunks[i] for i in candidate_idxs if i < len(self.chunks)]
        pairs = [[query, c.text] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)
        scored = sorted(
            zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True
        )[:self.top_k_reranked]
        results = []
        for chunk, score in scored:
            results.append({
                "text": chunk.text,
                "score": float(score),
                "metadata": chunk.metadata,
            })
        logger.info(f"Retrieved {len(results)} chunks for: '{query[:60]}'")
        return results