import requests
import xmltodict
import time
from typing import Optional
from loguru import logger

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

class PubMedFetcher:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.session = requests.Session()
        self.delay = 0.11 if api_key else 0.34

    def search(self, query: str, max_results: int = 50) -> list:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        resp = self.session.get(f"{BASE_URL}/esearch.fcgi", params=params)
        resp.raise_for_status()
        pmids = resp.json()["esearchresult"]["idlist"]
        logger.info(f"Found {len(pmids)} articles for: '{query}'")
        return pmids

    def fetch_details(self, pmids: list) -> list:
        if not pmids:
            return []
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        resp = self.session.get(f"{BASE_URL}/efetch.fcgi", params=params)
        resp.raise_for_status()
        time.sleep(self.delay)
        data = xmltodict.parse(resp.text)
        articles = data.get("PubmedArticleSet", {}).get("PubmedArticle", [])
        if isinstance(articles, dict):
            articles = [articles]
        results = []
        for article in articles:
            parsed = self._parse_article(article)
            if parsed:
                results.append(parsed)
        return results

    def _parse_article(self, article: dict) -> Optional[dict]:
        try:
            medline = article["MedlineCitation"]
            art = medline["Article"]
            title = art.get("ArticleTitle", "")
            if isinstance(title, dict):
                title = title.get("#text", "")
            abstract_raw = art.get("Abstract", {}).get("AbstractText", "")
            if isinstance(abstract_raw, list):
                abstract = " ".join(
                    (t.get("#text", t) if isinstance(t, dict) else t)
                    for t in abstract_raw
                )
            elif isinstance(abstract_raw, dict):
                abstract = abstract_raw.get("#text", "")
            else:
                abstract = abstract_raw or ""
            if not abstract:
                return None
            journal = art.get("Journal", {})
            journal_name = journal.get("Title", "")
            pub_date = journal.get("JournalIssue", {}).get("PubDate", {})
            year = pub_date.get("Year", pub_date.get("MedlineDate", "")[:4])
            author_list = art.get("AuthorList", {}).get("Author", [])
            if isinstance(author_list, dict):
                author_list = [author_list]
            authors = [
                f"{a.get('LastName', '')} {a.get('Initials', '')}".strip()
                for a in author_list[:3]
            ]
            pmid = str(medline.get("PMID", {}).get("#text", medline.get("PMID", "")))
            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal_name,
                "year": year,
                "authors": authors,
                "source_type": "pubmed",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        except Exception as e:
            logger.warning(f"Failed to parse article: {e}")
            return None

    def fetch_by_query(self, query: str, max_results: int = 50) -> list:
        pmids = self.search(query, max_results)
        return self.fetch_details(pmids)