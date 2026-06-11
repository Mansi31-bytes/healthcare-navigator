import json
import os
from groq import Groq
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a clinical knowledge assistant for healthcare professionals.
You will be given a clinical question and relevant excerpts from medical literature.
Synthesize the evidence into a structured, accurate answer.

Rules:
- Only use information from the provided context chunks
- Insert inline citation markers [1], [2] etc. matching the source IDs given
- Be precise with dosages, thresholds, and clinical criteria
- If evidence is insufficient, state it clearly and lower confidence accordingly
- NEVER invent sources or statistics not present in the context

Respond ONLY with valid JSON, no markdown, no backticks, no preamble:
{
  "summary": "2-3 sentence direct clinical answer with inline [1][2] citation markers",
  "confidence": "high|moderate|low",
  "confidence_reason": "one short sentence",
  "key_points": ["point with [1]", "point with [2]"],
  "sources": [{"id":1,"title":"...","org":"...","year":"...","type":"guideline|rct|review|protocol","url":"..."}],
  "evidence_tags": ["RCT-supported", "Level I Evidence"],
  "caution": "one sentence clinical caveat"
}"""

def build_context_block(chunks: list) -> tuple:
    context_lines = []
    sources = []
    seen_pmids = {}
    source_id = 1
    for chunk in chunks:
        meta = chunk["metadata"]
        pmid = meta.get("pmid", "")
        if pmid not in seen_pmids:
            seen_pmids[pmid] = source_id
            sources.append({
                "id": source_id,
                "title": meta.get("title", "Unknown"),
                "org": meta.get("journal", ""),
                "year": meta.get("year", ""),
                "type": _infer_source_type(meta),
                "url": meta.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
            })
            source_id += 1
        sid = seen_pmids[pmid]
        context_lines.append(f"[{sid}] {chunk['text']}")
    context_str = "\n\n".join(context_lines)
    return context_str, sources

def _infer_source_type(meta: dict) -> str:
    title = meta.get("title", "").lower()
    if any(k in title for k in ["guideline", "recommendation", "consensus"]):
        return "guideline"
    if any(k in title for k in ["randomized", "trial", "rct", "placebo"]):
        return "rct"
    if any(k in title for k in ["review", "meta-analysis", "systematic"]):
        return "review"
    return "review"

class ClinicalSynthesizer:
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model

    def synthesize(self, query: str, chunks: list, specialty: str = "") -> dict:
        context_str, sources = build_context_block(chunks)
        spec_note = f"\nSpecialty context: {specialty}" if specialty else ""

        user_message = f"""Clinical question: {query}{spec_note}

Retrieved evidence ({len(chunks)} chunks):
{context_str}

Available sources for citation:
{json.dumps(sources, indent=2)}

Synthesize an evidence-based answer using ONLY the above context."""

        logger.info(f"Calling Groq for: '{query[:60]}'")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            result = json.loads(raw)
            result["sources"] = sources
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return {
                "summary": "Unable to parse response. Please retry.",
                "confidence": "low",
                "confidence_reason": "Generation error",
                "key_points": [],
                "sources": sources,
                "evidence_tags": [],
                "caution": "Response parsing failed.",
            }