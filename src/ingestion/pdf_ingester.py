import os
from pypdf import PdfReader
from loguru import logger
from src.ingestion.chunker import MedicalChunker


class PDFIngester:
    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        self.chunker = MedicalChunker(chunk_size=chunk_size, overlap=overlap)

    def extract_text(self, pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        full_text = "\n".join(pages)
        logger.info(f"Extracted {len(full_text)} chars from {os.path.basename(pdf_path)}")
        return full_text

    def ingest(self, pdf_path: str, title: str = "", source: str = "") -> list:
        text = self.extract_text(pdf_path)
        if not text:
            logger.warning(f"No text extracted from {pdf_path}")
            return []

        filename = os.path.basename(pdf_path)
        doc = {
            "pmid": filename.replace(".pdf", ""),
            "title": title or filename,
            "abstract": text,
            "journal": source or "Uploaded PDF",
            "year": "",
            "authors": [],
            "source_type": "pdf",
            "url": "",
        }
        chunks = self.chunker.chunk_document(doc)
        logger.info(f"Created {len(chunks)} chunks from {filename}")
        return chunks