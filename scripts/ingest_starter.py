import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.ingestion.pubmed_fetcher import PubMedFetcher
from src.ingestion.chunker import MedicalChunker
from src.retrieval.retriever import HybridRetriever
from loguru import logger

SEED_QUERIES = [
    "SGLT2 inhibitors heart failure treatment",
    "hypertension management chronic kidney disease guidelines",
    "atrial fibrillation anticoagulation therapy",
    "type 2 diabetes HbA1c targets treatment",
    "sepsis management bundle treatment protocol",
]

def main():
    fetcher = PubMedFetcher(api_key=os.getenv("PUBMED_API_KEY"))
    chunker = MedicalChunker(chunk_size=400, overlap=50)
    retriever = HybridRetriever(
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        faiss_path="./data/indexes/faiss.index",
        bm25_path="./data/indexes/bm25.pkl",
        chunks_path="./data/indexes/chunks.pkl",
    )

    all_docs = []
    for query in SEED_QUERIES:
        logger.info(f"Fetching: {query}")
        docs = fetcher.fetch_by_query(query, max_results=20)
        all_docs.extend(docs)
        logger.info(f"  -> {len(docs)} articles fetched")

    logger.info(f"Total documents: {len(all_docs)}")
    chunks = chunker.chunk_documents(all_docs)
    logger.info(f"Total chunks: {len(chunks)}")

    logger.info("Building index...")
    retriever.build_index(chunks)
    logger.info("Done! Index saved to ./data/indexes/")

if __name__ == "__main__":
    main()