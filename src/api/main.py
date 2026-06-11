import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
from dotenv import load_dotenv

from src.ingestion.pdf_ingester import PDFIngester
from fastapi import UploadFile, File, Form
import tempfile
from src.ingestion.pubmed_fetcher import PubMedFetcher
from src.ingestion.chunker import MedicalChunker
from src.retrieval.retriever import HybridRetriever
from src.generation.synthesizer import ClinicalSynthesizer

load_dotenv()

retriever = HybridRetriever(
    embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    faiss_path=os.getenv("FAISS_INDEX_PATH", "./data/indexes/faiss.index"),
    bm25_path=os.getenv("BM25_PATH", "./data/indexes/bm25.pkl"),
    chunks_path=os.getenv("CHUNKS_PATH", "./data/indexes/chunks.pkl"),
    top_k_dense=int(os.getenv("TOP_K_DENSE", 20)),
    top_k_bm25=int(os.getenv("TOP_K_BM25", 20)),
    top_k_reranked=int(os.getenv("TOP_K_RERANKED", 5)),
)
synthesizer = ClinicalSynthesizer()
fetcher = PubMedFetcher(api_key=os.getenv("PUBMED_API_KEY"))
chunker = MedicalChunker(
    chunk_size=int(os.getenv("CHUNK_SIZE", 400)),
    overlap=int(os.getenv("CHUNK_OVERLAP", 50)),
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    faiss_path = os.getenv("FAISS_INDEX_PATH", "./data/indexes/faiss.index")
    if os.path.exists(faiss_path):
        logger.info("Loading existing index...")
        retriever.load_index()
    else:
        logger.warning("No index found. Run /ingest first.")
    yield

app = FastAPI(
    title="Healthcare Knowledge Navigator",
    description="RAG-powered clinical query API",
    version="1.0.0",
    lifespan=lifespan,
)

class QueryRequest(BaseModel):
    query: str
    specialty: str = ""

class IngestRequest(BaseModel):
    queries: list
    max_per_query: int = 30

@app.post("/query")
async def query_knowledge_base(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not retriever.index:
        raise HTTPException(status_code=503, detail="Index not loaded. Run /ingest first.")
    chunks = retriever.retrieve(req.query)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant evidence found.")
    answer = synthesizer.synthesize(req.query, chunks, specialty=req.specialty)
    return {"query": req.query, "answer": answer, "chunks_used": len(chunks)}

@app.post("/ingest")
async def ingest_literature(req: IngestRequest):
    all_docs = []
    for q in req.queries:
        docs = fetcher.fetch_by_query(q, max_results=req.max_per_query)
        all_docs.extend(docs)
        logger.info(f"Fetched {len(docs)} docs for '{q}'")
    if not all_docs:
        raise HTTPException(status_code=404, detail="No documents fetched.")
    chunks = chunker.chunk_documents(all_docs)
    retriever.build_index(chunks)
    return {
        "status": "ok",
        "documents_ingested": len(all_docs),
        "chunks_indexed": len(chunks),
    }

@app.post("/ingest/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    title: str = Form(default=""),
    source: str = Form(default=""),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    ingester = PDFIngester()
    new_chunks = ingester.ingest(tmp_path, title=title, source=source)

    if not new_chunks:
        raise HTTPException(status_code=400, detail="No text could be extracted from PDF.")

    all_chunks = retriever.chunks + new_chunks
    retriever.build_index(all_chunks)

    return {
        "status": "ok",
        "filename": file.filename,
        "new_chunks": len(new_chunks),
        "total_chunks": len(all_chunks),
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "index_loaded": retriever.index is not None,
        "chunks": len(retriever.chunks) if retriever.chunks else 0,
    }