import re
from dataclasses import dataclass, field

@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""

class MedicalChunker:
    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split_sentences(self, text: str) -> list:
        pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s'
        sentences = re.split(pattern, text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def chunk_document(self, doc: dict) -> list:
        full_text = f"{doc.get('title', '')}. {doc.get('abstract', '')}".strip()
        sentences = self._split_sentences(full_text)
        chunks = []
        current_words = []
        chunk_idx = 0
        for sentence in sentences:
            words = sentence.split()
            if len(current_words) + len(words) > self.chunk_size and current_words:
                chunk_text = " ".join(current_words)
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=f"{doc.get('pmid', 'doc')}_{chunk_idx}",
                    metadata={
                        "pmid": doc.get("pmid", ""),
                        "title": doc.get("title", ""),
                        "journal": doc.get("journal", ""),
                        "year": doc.get("year", ""),
                        "authors": doc.get("authors", []),
                        "url": doc.get("url", ""),
                        "source_type": doc.get("source_type", "pubmed"),
                        "chunk_index": chunk_idx,
                    }
                ))
                current_words = current_words[-self.overlap:] + words
                chunk_idx += 1
            else:
                current_words.extend(words)
        if current_words:
            chunks.append(Chunk(
                text=" ".join(current_words),
                chunk_id=f"{doc.get('pmid', 'doc')}_{chunk_idx}",
                metadata={
                    "pmid": doc.get("pmid", ""),
                    "title": doc.get("title", ""),
                    "journal": doc.get("journal", ""),
                    "year": doc.get("year", ""),
                    "authors": doc.get("authors", []),
                    "url": doc.get("url", ""),
                    "source_type": doc.get("source_type", "pubmed"),
                    "chunk_index": chunk_idx,
                }
            ))
        return chunks

    def chunk_documents(self, docs: list) -> list:
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks