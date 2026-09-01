from typing import List, Dict, Any, Optional
from schemas import EvidenceNode, SynthesizedOutput

class EvidenceGraphEngine:
    def build_graph(self, synthesis: SynthesizedOutput, fundamental_evidence: List[str]) -> List[EvidenceNode]:
        """
        Builds connected tree/graph: DECISION -> CLAIM -> EVIDENCE -> DOCUMENT.
        Guarantees that claims with available citations link to EVIDENCE nodes with citation_tags,
        while uncited claims are marked without citation_tag (Test 20).
        """
        nodes: List[EvidenceNode] = []

        # 1. Root Decision Node
        decision_node_id = f"node_dec_{synthesis.session_id}"
        nodes.append(EvidenceNode(
            node_id=decision_node_id,
            node_type="DECISION",
            label=f"Verdict: {synthesis.synthesized_verdict} ({synthesis.ticker})",
            citation_tag=None,
            parent_node_id=None
        ))

        # 2. Claim Nodes (from reasoning trace / signals)
        raw = synthesis.raw_signals
        tech_claim_id = f"node_claim_tech_{synthesis.session_id}"
        tech_class = raw.get("technical", {}).get("classification", "NEUTRAL")
        nodes.append(EvidenceNode(
            node_id=tech_claim_id,
            node_type="CLAIM",
            label=f"Technical Momentum: {tech_class}",
            citation_tag=None,
            parent_node_id=decision_node_id
        ))

        fund_claim_id = f"node_claim_fund_{synthesis.session_id}"
        fund_verdict = raw.get("fundamental", {}).get("filing_verdict", "NEUTRAL")
        nodes.append(EvidenceNode(
            node_id=fund_claim_id,
            node_type="CLAIM",
            label=f"Fundamental Solvency & Solvency: {fund_verdict}",
            citation_tag=None,
            parent_node_id=decision_node_id
        ))

        sent_claim_id = f"node_claim_sent_{synthesis.session_id}"
        sent_class = raw.get("sentiment", {}).get("classification", "NEUTRAL")
        nodes.append(EvidenceNode(
            node_id=sent_claim_id,
            node_type="CLAIM",
            label=f"Market Sentiment & Flow: {sent_class}",
            citation_tag=None,
            parent_node_id=decision_node_id
        ))

        # 3. Evidence & Document Nodes for Fundamental Claims
        citations = synthesis.source_attributions
        if citations:
            for idx, cit in enumerate(citations):
                ev_id = f"node_ev_{synthesis.session_id}_{idx}"
                doc_id = f"node_doc_{synthesis.session_id}_{idx}"
                
                # Matched evidence text
                ev_text = fundamental_evidence[idx] if idx < len(fundamental_evidence) else f"Disclosed in {cit}"
                nodes.append(EvidenceNode(
                    node_id=ev_id,
                    node_type="EVIDENCE",
                    label=ev_text[:120] + "...",
                    citation_tag=cit,
                    parent_node_id=fund_claim_id
                ))
                nodes.append(EvidenceNode(
                    node_id=doc_id,
                    node_type="DOCUMENT",
                    label=f"Source Document: {cit}",
                    citation_tag=cit,
                    parent_node_id=ev_id
                ))
        else:
            # Explicitly uncited evidence marker
            nodes.append(EvidenceNode(
                node_id=f"node_ev_uncited_{synthesis.session_id}",
                node_type="EVIDENCE",
                label="No verified regulatory filing citations available (Uncited / Degraded feed)",
                citation_tag=None,
                parent_node_id=fund_claim_id
            ))

        return nodes

evidence_graph_engine = EvidenceGraphEngine()
