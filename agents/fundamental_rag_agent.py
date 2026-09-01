import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from config import FILINGS_DIR, CHROMA_PERSIST_DIR
from schemas import FundamentalSignal
from utils.filing_freshness import check_filing_freshness

class FundamentalRAGAgent:
    def __init__(self):
        self.name = "FundamentalRAGAgent"
        self.chroma_client = None
        self.collection = None
        self.documents_index: List[Dict[str, Any]] = []
        self._init_rag_store()

    def _init_rag_store(self):
        """
        Initializes ChromaDB or fallback local semantic document index.
        """
        try:
            import chromadb
            # Use ephemeral or persistent client
            self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
            self.collection = self.chroma_client.get_or_create_collection(
                name="corporate_filings",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            # Fallback to internal in-memory index
            self.chroma_client = None
            self.collection = None

        self.ingest_documents()

    def ingest_documents(self):
        """
        Parses all filing/earnings documents in data/corporate_filings/
        and indexes chunks with metadata and citation tags.
        """
        self.documents_index = []
        files = glob.glob(str(FILINGS_DIR / "*.txt"))
        
        doc_ids = []
        documents = []
        metadatas = []

        for idx, file_path in enumerate(files):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                lines = content.strip().split("\n")
                doc_type = "UNKNOWN"
                ticker = "UNKNOWN"
                doc_id = Path(file_path).stem
                doc_date = "2026-01-01"
                citation_tag = f"[{doc_id}]"

                body_lines = []
                for line in lines:
                    if line.startswith("DOCUMENT_TYPE:"):
                        doc_type = line.replace("DOCUMENT_TYPE:", "").strip()
                    elif line.startswith("TICKER:"):
                        ticker = line.replace("TICKER:", "").strip()
                    elif line.startswith("DOCUMENT_ID:"):
                        doc_id = line.replace("DOCUMENT_ID:", "").strip()
                    elif line.startswith("DATE:"):
                        doc_date = line.replace("DATE:", "").strip()
                    elif line.startswith("CITATION_TAG:"):
                        citation_tag = line.replace("CITATION_TAG:", "").strip()
                    else:
                        body_lines.append(line)

                body_text = "\n".join(body_lines).strip()
                
                # Split body into meaningful paragraphs/chunks
                chunks = [c.strip() for c in body_text.split("\n\n") if c.strip()]
                for c_idx, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{c_idx}"
                    metadata = {
                        "ticker": ticker,
                        "document_type": doc_type,
                        "document_id": doc_id,
                        "date": doc_date,
                        "citation_tag": citation_tag,
                        "source_file": Path(file_path).name
                    }
                    item = {
                        "id": chunk_id,
                        "text": chunk,
                        "metadata": metadata
                    }
                    self.documents_index.append(item)
                    doc_ids.append(chunk_id)
                    documents.append(chunk)
                    metadatas.append(metadata)
            except Exception as e:
                continue

        if self.collection and documents:
            try:
                # Add to chroma (upsert)
                self.collection.upsert(
                    ids=doc_ids,
                    documents=documents,
                    metadatas=metadatas
                )
            except Exception:
                pass

    def retrieve(self, query: str, ticker: str, top_k: int = 3, use_stale: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieves relevant document chunks for a ticker.
        Distinguishes stale vs fresh filings.
        """
        results: List[Dict[str, Any]] = []

        # Filter documents for this ticker
        ticker_docs = [
            d for d in self.documents_index 
            if d["metadata"]["ticker"].upper() == ticker.upper()
        ]

        if use_stale:
            # Prioritize stale filing if requested for scenario D
            ticker_docs = [
                d for d in ticker_docs 
                if "STALE" in d["metadata"].get("document_id", "") or "STALE" in d["metadata"].get("source_file", "")
            ] or ticker_docs
        else:
            # Exclude stale filings by default for normal analysis
            fresh_docs = [
                d for d in ticker_docs 
                if "STALE" not in d["metadata"].get("document_id", "") and "STALE" not in d["metadata"].get("source_file", "")
            ]
            if fresh_docs:
                ticker_docs = fresh_docs

        if not ticker_docs:
            return []

        # Keyword / token overlap relevance ranking
        query_terms = set(query.lower().split())
        scored_docs = []
        for doc in ticker_docs:
            text = doc["text"].lower()
            overlap = sum(1 for term in query_terms if term in text)
            # Higher weight for debt / pledge / margin / growth terms
            if any(k in text for k in ["debt", "pledge", "liabilities", "covenants", "deleveraging", "margins"]):
                overlap += 3
            score = overlap / (len(query_terms) + 1)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_items = scored_docs[:top_k]

        for score, doc in top_items:
            results.append({
                "text": doc["text"],
                "citation_tag": doc["metadata"]["citation_tag"],
                "date": doc["metadata"]["date"],
                "document_id": doc["metadata"]["document_id"],
                "document_type": doc["metadata"]["document_type"],
                "score": round(float(score), 3)
            })

        return results

    def analyze_fundamentals(
        self,
        ticker: str,
        market_data: Dict[str, Any],
        scenario: str = "aligned",
        degraded_override: bool = False
    ) -> Tuple[FundamentalSignal, Optional[str]]:
        """
        Analyzes fundamental strength via RAG citations and financial ratios.
        Returns: (FundamentalSignal, Optional[filing_freshness_warning])
        """
        if degraded_override:
            return FundamentalSignal(
                ticker=ticker,
                rag_citations=[],
                debt_to_equity=0.0,
                earnings_growth=0.0,
                filing_verdict="NEUTRAL",
                confidence=0.25,
                evidence=["Fundamental filings feed unavailable; analysis operating in degraded mode."],
                degraded=True
            ), None

        use_stale = (scenario == "stale_behavioral")
        retrieved_chunks = self.retrieve(
            query="debt to equity balance sheet revenue margin pledge liabilities",
            ticker=ticker,
            top_k=3,
            use_stale=use_stale
        )

        financials = market_data.get("financials", {})
        debt_to_equity = financials.get("debt_to_equity", 1.0)
        earnings_growth = financials.get("earnings_growth", 0.0)

        rag_citations: List[str] = []
        evidence: List[str] = []
        stale_warning: Optional[str] = None
        is_stale_filing = False

        if retrieved_chunks:
            for chunk in retrieved_chunks:
                citation = chunk["citation_tag"]
                if citation not in rag_citations:
                    rag_citations.append(citation)
                evidence.append(f"{citation}: {chunk['text']}")
                
                # Check freshness of each chunk's date
                doc_date = chunk.get("date", "")
                stale_flag, age_months, warning_msg = check_filing_freshness(doc_date)
                if stale_flag:
                    is_stale_filing = True
                    stale_warning = warning_msg

        # Evaluate fundamental verdict
        if debt_to_equity > 2.5:
            verdict = "CRITICAL_RISK"
            confidence = 0.90
            evidence.append(f"Elevated financial leverage: Debt-to-Equity stands at {debt_to_equity:.2f}x.")
        elif debt_to_equity > 1.3:
            verdict = "CONCERNING"
            confidence = 0.78
            evidence.append(f"Moderate balance sheet leverage: Debt-to-Equity is {debt_to_equity:.2f}x.")
        elif earnings_growth > 0.15 and debt_to_equity < 1.0:
            verdict = "POSITIVE"
            confidence = 0.88
            evidence.append(f"Strong solvency & growth: Debt-to-Equity is {debt_to_equity:.2f}x with {earnings_growth*100:.1f}% earnings growth.")
        else:
            verdict = "NEUTRAL"
            confidence = 0.70

        # Handle filing freshness degradation
        is_degraded = is_stale_filing or (len(retrieved_chunks) == 0)
        if is_stale_filing:
            confidence = max(0.40, confidence - 0.25)

        signal = FundamentalSignal(
            ticker=ticker,
            rag_citations=rag_citations,
            debt_to_equity=debt_to_equity,
            earnings_growth=earnings_growth,
            filing_verdict=verdict,
            confidence=round(confidence, 2),
            evidence=evidence,
            degraded=is_degraded
        )

        return signal, stale_warning

fundamental_rag_agent = FundamentalRAGAgent()
