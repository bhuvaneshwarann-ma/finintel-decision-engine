# SPEC.md — Multi-Agent Autonomous Financial Intelligence System for Retail Investors

**Project:** FinIntelligence AI (working title)
**Event:** HACKVERSE 2026 — VIT Chennai, IEEE Robotics & Automation Society Student Chapter
**Track:** Sprint 1 — Rapid Vibe Coding, PS-01
**Spec version:** 3.0 (adds §18 Investor Decision Intelligence Engine)
**Status:** Approved for build

---

## 1. Overview

A multi-agent AI system that converts market data, regulatory filings, and behavioral signals into explainable, personalized investment intelligence for retail investors. The system must reason about what market events mean for a *specific* user's financial position and justify that reasoning transparently at every step — not just surface raw signals.

This is a **research/education prototype**, not a trading system. No trade execution. All outputs are framed as `BUY CANDIDATE / HOLD-WATCH / REDUCE EXPOSURE / AVOID-HIGH RISK`, never as guarantees.

---

## 2. Goals and Non-Goals

**Goals**
- Demonstrate a working, end-to-end pipeline from raw simulated market data to a personalized, cited, explainable recommendation.
- Run fully offline / API-key-free in a deterministic mock mode; enhance with Gemini when available, never depend on it.
- Make every PS-01 minimum requirement independently verifiable in a live demo.
- Differentiate from published reference architectures (FinRobot, FINCON, P1GPT, AWS multi-agent finance patterns) via four specific new capabilities (Section 9).

**Non-Goals**
- No live brokerage integration, no real trade execution, no live real-time market feed dependency.
- No claim of proven predictive accuracy — calibration tracking (Section 9.4) is explicitly a session-local, demo-only illustration.
- No production-grade auth/multi-tenant user system — two hardcoded demo personas are sufficient.

---

## 3. System Architecture

```
RAW MARKET / FINANCIAL DATA (simulated)
        ↓
DATA VALIDATION
        ↓
PARALLEL SPECIALIZED AGENTS (Technical | Fundamental-RAG | Sentiment)
        ↓
DRAFT SYNTHESIS (Gemini or deterministic fallback)
        ↓
DEVIL'S ADVOCATE CHALLENGE (evidence-constrained adversarial pass)
        ↓
FINAL SYNTHESIS (verdict upheld or revised, with rationale)
        ↓
USER RISK / BEHAVIORAL PROFILE (persona + portfolio + Behavioral Bias Mirror)
        ↓
EXPLAINABLE, PERSONALIZED, CITED INTELLIGENCE
        ↓
TELEMETRY + SESSION LOG + CALIBRATION LEDGER
        ↓
FASTAPI API → DARK-TERMINAL WEB UI
```

**Orchestration diagram:**

```
Technical ───────┐
                 │
Fundamental ─────┼──→ Synthesis (draft) ──→ Devil's Advocate ──→ Synthesis (final)
                 │                              ↑
Sentiment ───────┘                        (may only re-weigh existing
                      ↑                    evidence, no new facts/citations)
                   Risk/User Profile
                   + Behavioral Bias Mirror
```

Technical, Fundamental, and Sentiment agents execute **in parallel** via `asyncio.gather()`. The Risk Agent runs after (it consumes portfolio state). The Devil's Advocate Agent runs after the draft synthesis and before the final synthesis call.

---

## 4. Technology Stack

| Layer | Technology |
|---|---|
| Language / runtime | Python 3.11+ |
| API framework | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Vector store | ChromaDB (local, embedded) |
| LLM synthesis | google-genai (Gemini), deterministic fallback required |
| Numerics | pandas, NumPy, scikit-learn (optional lightweight classifier) |
| Config | python-dotenv |
| Testing | pytest, pytest-asyncio, httpx |
| Frontend | HTML, Tailwind CDN, Lucide Icons, vanilla JS (no framework) |

No additional frameworks beyond this list without explicit justification.

---

## 5. Project Structure

```
project_root/
│
├── requirements.txt
├── .env.example
├── .gitignore
├── config.py
├── schemas.py
├── main.py
├── README.md
│
├── data/
│   ├── sample_market_data.json
│   └── corporate_filings/
│       ├── TATAMOTORS_q3_filing.txt
│       ├── INFOSYS_q3_filing.txt
│       ├── XYZ_CORP_sebi_filing.txt
│       ├── XYZ_CORP_sebi_filing_STALE.txt
│       ├── TATAMOTORS_earnings.txt
│       ├── INFOSYS_earnings.txt
│       └── XYZ_CORP_earnings.txt
│
├── agents/
│   ├── __init__.py
│   ├── technical_agent.py
│   ├── fundamental_rag_agent.py
│   ├── sentiment_agent.py
│   ├── risk_agent.py
│   ├── devils_advocate_agent.py
│   └── orchestrator.py
│
├── profiling/
│   ├── __init__.py
│   ├── user_profile.py
│   └── behavioral_bias_mirror.py
│
├── logs/
│   ├── __init__.py
│   ├── session_logger.py
│   └── calibration_ledger.py
│
├── utils/
│   ├── __init__.py
│   ├── resilience.py
│   ├── metrics.py
│   ├── security.py
│   └── filing_freshness.py
│
├── static/
│   └── index.html
│
├── tests/
│   ├── __init__.py
│   └── test_system.py
│
└── chroma_db/
    └── .gitkeep
```

---

## 6. Configuration

`.env.example`:
```
GEMINI_API_KEY=
AI_MODE=mock
CHROMA_PERSIST_DIR=./chroma_db
APP_ENV=development
LOG_LEVEL=INFO
```

`.gitignore`:
```
.env
__pycache__/
.pytest_cache/
chroma_db/
*.pyc
```

Rules: secrets loaded exclusively from environment variables; never hardcoded; `.env` never committed; `AI_MODE=mock` is the safe default with no API key present.

---

## 7. Data Contracts (Pydantic Models)

### 7.1 Core signal models

**`TechnicalSignal`**: `ticker`, `rsi`, `macd_signal`, `momentum_score`, `volume_anomaly_score`, `classification` (`STRONG_BULLISH…STRONG_BEARISH`), `confidence`, `reasons[]`, `degraded`

**`FundamentalSignal`**: `ticker`, `rag_citations[]`, `debt_to_equity`, `earnings_growth`, `filing_verdict`, `confidence`, `evidence[]`, `degraded`

**`SentimentSignal`**: `ticker`, `news_sentiment`, `fii_flow_trend`, `social_chatter_score`, `classification`, `confidence`, `reasons[]`, `degraded`

**`RiskAssessment`**: `risk_profile`, `portfolio_concentration`, `sector_exposure`, `risk_flags[]`, `risk_score`, `personalized_constraints`, `behavioral_flags: List[BehavioralFlag]`

**`AgentResult`**: `agent_name`, `status`, `latency_ms`, `confidence`, `payload`, `errors[]`

**`SynthesizedOutput`**: `session_id`, `ticker`, `raw_signals`, `synthesized_verdict`, `market_classification`, `confidence`, `reasoning_trace`, `conflict_summary`, `source_attributions[]`, `personalized_advice`, `risk_profile`, `degraded_data`, `disclaimer`, `devils_advocate_challenge: Optional[DevilsAdvocateChallenge]`, `confidence_breakdown: ConfidenceBreakdown`, `filing_freshness_warning: Optional[str]`

### 7.2 New models

**`DevilsAdvocateChallenge`**: `ticker`, `target_verdict`, `counter_argument` (grounded only in already-cited evidence), `weakest_evidence_point`, `severity` (`LOW/MEDIUM/HIGH`)

**`BehavioralFlag`**: `flag_type` (`RECENCY_BIAS | FAMILIARITY_CONCENTRATION | MOMENTUM_CHASING | LOSS_AVERSION_HOLD`), `trigger_reason`, `nudge_message` (plain-language, non-judgmental)

**`ConfidenceBreakdown`**: `data_freshness_score`, `agent_agreement_score`, `evidence_strength_score`, `historical_calibration_score`, `composite_confidence`

**Constraint:** all four component scores are bounded `[0,1]`; `composite_confidence` must equal their weighted combination within a defined tolerance (verified by Test 14, Section 12).

---

## 8. Functional Requirements by Module

### 8.1 Market data
Simulated, explicitly labeled NSE-style data for three tickers:
- **TATAMOTORS** — strong technical breakout, positive momentum, healthy balance sheet, explainable high volume. → Scenario A.
- **INFOSYS** — neutral technical setup, strong cash reserves, moderate sentiment, stable fundamentals.
- **XYZ_CORP** — bullish technical + volume, apparently positive sentiment, **hidden debt risk in filing**. → Scenario B / D (conflict + staleness demo).

Data includes raw historical price/volume series (not precomputed indicators) sufficient to compute RSI and MACD, plus headlines, social sentiment, FII/DII trend, financial ratios, sector info.

### 8.2 Technical Agent
Computes RSI, MACD signal, momentum score, moving-average relationship, and volume-anomaly z-score from raw price/volume data using pandas/numpy. Classification: `STRONG_BULLISH / BULLISH / NEUTRAL / BEARISH / STRONG_BEARISH`. Confidence is calculated, not assigned. Reasons must be machine-readable (structured, not prose-only).

### 8.3 Sentiment Agent
Deterministic scoring over simulated headlines, social chatter, and FII/DII flow. Optional lightweight scikit-learn classifier. Output includes classification, social chatter score, FII trend, confidence, evidence. Must not claim simulated sentiment reflects real-world sentiment.

### 8.4 Fundamental RAG Agent
- Ingests all filing/earnings documents into a ChromaDB collection with metadata: `ticker`, `document_type`, `document_id`, `page`, `citation_tag` (e.g. `[SEBI-Filing-XYZ-Q3: Page 14]`).
- Implements `ingest_documents()`, `retrieve(query, ticker, top_k)`, `analyze_fundamentals()`.
- Retrieval returns chunk text, relevance score, citation tag, source metadata. No invented citations, ever.
- XYZ_CORP must surface an explicit simulated debt warning with citation.
- **Filing freshness check** (`utils/filing_freshness.py`): every retrieved chunk's declared date is checked against a staleness threshold (>12 months = stale). If the only available evidence is stale: evidence is still returned (never withheld), `degraded=true` is set, and `filing_freshness_warning` is populated on the synthesis output. Stale evidence and absent evidence are distinguished in both backend output and UI — they require different user framing.

### 8.5 Risk / Behavioral Profiling
Two fixed personas:

| | Conservative Senior | Aggressive Gen-Z |
|---|---|---|
| Priority | Capital preservation | Tactical opportunity |
| Volatility tolerance | Low | High |
| Debt risk weighting | Strongly penalized | Still surfaced, less penalized |
| Concentration risk | Strongly penalized | More tolerant |
| Portfolio (TATAMOTORS / INFOSYS / XYZ_CORP) | 15% / 20% / 5% | 5% / 10% / 2% |

Personalization occurs strictly **after** independent market analysis; persona never alters raw signal computation. Identical market input must produce demonstrably divergent advice across personas (verified by Test 4).

**Behavioral Bias Mirror** (`profiling/behavioral_bias_mirror.py`): given persona + portfolio + requested ticker/scenario, emits `BehavioralFlag`s (e.g. momentum chasing when requesting analysis right after a sharp spike; familiarity concentration from repeated sector over-allocation; loss-aversion holding). Flags are persona-specific — not generic boilerplate copy — and must produce zero flags in a control case with no triggering pattern (Test 16).

### 8.6 Multi-Agent Orchestrator
`asyncio.gather()` runs Technical, Fundamental-RAG, and Sentiment agents concurrently; each agent's start/end time and `latency_ms` are recorded. Concurrency must be provably real: total elapsed time should track `max(agent times)`, not `sum(agent times)` (Test 3).

Risk Agent runs after, consuming portfolio state. Devil's Advocate Agent runs after draft synthesis, before final synthesis (Section 8.8).

### 8.7 Devil's Advocate Agent
Receives the draft synthesis verdict and the full evidence pool already retrieved by the other agents. Constrained to only re-weigh or re-emphasize existing evidence — **must never introduce a new fact or citation** (verified by Test 13). Outputs a `DevilsAdvocateChallenge` identifying the weakest evidence point behind the draft verdict and a severity rating. This operationalizes "must not simply majority-vote" (Section 8.9) as an explicit adversarial step rather than relying on synthesis-prompt discipline alone.

### 8.8 Gemini Synthesis (two-pass)
- **Pass 1 (draft):** receives only structured, validated agent outputs and retrieved evidence. Identifies agreements/disagreements, produces market classification, verdict, confidence, reasoning trace, conflict summary, source attribution. Never invents facts or citations; states uncertainty explicitly.
- **Devil's Advocate pass** runs on the draft (Section 8.7).
- **Pass 2 (final):** receives original evidence + the challenge; either upholds the draft verdict (with explanation of why the challenge doesn't change it) or revises it (with explanation of what changed). Both passes are recorded in `reasoning_trace` for UI replay.
- Output validated against Pydantic schema. **On Gemini failure at any point, fallback to deterministic synthesis; the API must continue functioning** (Test 7).

### 8.9 Conflict Resolution
Reference case — XYZ_CORP: Technical = BULLISH, Sentiment = BULLISH, Fundamental = BEARISH/debt-risk. Synthesis must not majority-vote to BULLISH. Expected reasoning: *"Technical momentum is strong, but regulatory filing evidence identifies elevated debt risk."* Expected risk-adjusted verdict: `HOLD / AVOID NEW AGGRESSIVE ENTRY`, routed through the Devil's Advocate pass (Test 5).

### 8.10 Calibration Ledger
Logged in memory only, session-scoped. For each of the four demo scenarios, a predefined "reasonable outcome" classification is used as an illustrative reference point (explicitly **not** live-market backtesting). After each analysis run, `(stated_confidence, scenario_id)` pairs are logged. `/api/calibration` exposes a session summary. Feeds `historical_calibration_score` in `ConfidenceBreakdown`. UI and README must state clearly this is a session-local, demo-only signal.

### 8.11 Resilience / Degraded Data
Failure modes to handle without HTTP 500, each returning `degraded_data=true` with `degraded_components`, `fallback_used`, confidence reduction, and explanation:
1. Missing market feed
2. Missing filing
3. Chroma retrieval failure
4. Gemini API failure
5. Agent timeout
6. Stale-but-present filing (distinct from #2 — see 8.4)

### 8.12 Telemetry
Per session: `session_id`, `timestamp`, `ticker`, `persona`, per-agent `latency_ms`, total latency, agent status, confidence, signal classification, `degraded_data` flag, source count, `risk_concentration_score`, `filing_freshness_flag_count`, `devils_advocate_verdict_changed` (bool), `behavioral_flags_triggered_count`.

### 8.13 Logging & Security
- No PII, credentials, API keys, or payment info logged, ever — in-memory session metrics only.
- Ticker, persona, and scenario are whitelisted and Pydantic-validated at the API boundary.
- Bounded RAG `top_k` and bounded input length.
- No arbitrary file/path access, no shell execution via user input, no `GEMINI_API_KEY` exposed to the frontend.

---

## 9. Differentiating Capabilities

These four capabilities are not present, in this combination, in the reference architectures reviewed (FinRobot, FINCON, P1GPT, AWS multi-agent finance patterns) and should be called out explicitly in the judge write-up:

1. **Devil's Advocate Agent** (8.7) — an evidence-constrained adversarial pass between draft and final synthesis, rather than trusting a single-shot LLM call to genuinely reconcile (not just average) conflicting agent signals.
2. **Confidence Breakdown** (7.2, 8.10) — decomposes a single opaque confidence score into checkable components (freshness, agreement, evidence strength, calibration).
3. **Behavioral Bias Mirror** (8.5) — models the *user's own decision pattern* as a risk input, directly addressing the problem statement's framing that 89% of retail F&O participants lose money due to lack of research infrastructure, not just lack of data access.
4. **Filing Freshness Check** (8.4) — distinguishes "no evidence" from "old evidence," which most retrieval-only fundamental agents treat identically.

---

## 10. API Specification

**GET**
```
/                      → serves frontend
/health                → liveness + AI_MODE
/api/stocks            → available tickers
/api/personas          → available personas
/api/demo-scenarios    → scenario metadata (A/B/C/D)
/api/calibration       → session calibration ledger summary
```

**POST**
```
/api/analyze
Request:  { "ticker": "XYZ_CORP", "persona": "conservative", "scenario": "conflict" }
Response: session_id, agent outputs, synthesis (incl. devils_advocate_challenge,
          confidence_breakdown, filing_freshness_warning), reasoning trace,
          citations, risk assessment (incl. behavioral_flags), telemetry, degraded status
```

All request bodies validated via Pydantic; invalid ticker/persona/scenario rejected at the API boundary (Tests 8, 9).

---

## 11. Frontend Requirements

Dark-mode terminal-style dashboard (Tailwind CDN + Lucide + vanilla JS, no framework):

1. Header — system name, status, AI mode
2. Watchlist — TATAMOTORS, INFOSYS, XYZ_CORP
3. Market signal cards — Technical / Sentiment / Fundamental, each with classification, confidence, status
4. Persona switcher — Conservative Senior / Aggressive Gen-Z
5. Main decision panel — market signal, personalized verdict, confidence
6. Agent reasoning trace — per-agent detail
7. Conflict resolution panel — explicit agent disagreement display
8. Citation drawer — retrieved evidence with citation tags
9. Telemetry panel — latencies, risk score, citation count, degraded status
10. Demo scenario buttons — A / B / C / D
11. **Devil's Advocate panel** — draft verdict → challenge → final verdict, shown as connected steps
12. **Confidence breakdown widget** — component bars summing to composite score
13. **Behavioral Bias Mirror card** — persona-specific flags with nudge messages
14. **Reasoning replay scrubber** — step control across Technical → Fundamental → Sentiment → Risk → Draft Synthesis → Devil's Advocate → Final Synthesis

---

## 12. Test Plan

| # | Test |
|---|---|
| 1 | Technical calculations produce a valid signal from raw data |
| 2 | RAG retrieves expected XYZ_CORP debt evidence |
| 3 | Parallel agents execute concurrently (elapsed time ≈ max, not sum, of agent times) |
| 4 | Conservative and Aggressive personas produce divergent advice on identical market input |
| 5 | Conflict resolution detects Technical BUY vs Fundamental BEARISH and does not majority-vote |
| 6 | Missing filing does not produce HTTP 500 |
| 7 | Gemini failure triggers deterministic fallback |
| 8 | Invalid ticker rejected by validation |
| 9 | Invalid persona rejected by validation |
| 10 | No API key required in mock mode |
| 11 | Every fundamental recommendation has a citation when fundamental data is available |
| 12 | Degraded mode reduces confidence or flags uncertainty |
| 13 | Devil's Advocate never introduces a citation absent from the original evidence pool |
| 14 | Confidence breakdown components are bounded [0,1] and sum to composite within tolerance |
| 15 | Stale filing produces `degraded=true` + non-empty freshness warning; fresh filing does not |
| 16 | Behavioral Bias Mirror flags the momentum-chasing case and produces zero flags in the control case |

Target: 16/16 passing, executed via `pytest`, results not asserted without execution.

---

## 13. Demo Scenarios

| Scenario | Ticker | Technical | Sentiment | Fundamental | Expected Verdict |
|---|---|---|---|---|---|
| A — Aligned bullish | TATAMOTORS | BULLISH | POSITIVE | POSITIVE | BULLISH / BUY CANDIDATE, cited |
| B — Conflicting signals | XYZ_CORP | BULLISH | BULLISH | BEARISH (debt) | Risk-adjusted HOLD/AVOID, debt citation shown |
| C — Degraded data | any | — | — | missing feed/filing | Stays operational, `degraded=true`, confidence reduced, no fabricated evidence |
| D — Stale evidence + behavioral nudge | XYZ_CORP (stale filing) | BULLISH (spike) | BULLISH | Degraded, stale | Behavioral flag (`MOMENTUM_CHASING`) on Aggressive persona; Devil's Advocate cites staleness as weakest point |

**Scenario D should lead the judge demo** — it is the only scenario exercising all four differentiating capabilities in one pass.

---

## 14. Build Priority

```
P0:   End-to-end API, 3 core agents, orchestrator, persona logic, synthesis, Tests 1–12
P1:   Resilience, telemetry, frontend (sections 1–10 of Section 11)
P1.5: Devil's Advocate Agent, Confidence Breakdown, Behavioral Bias Mirror,
      Filing Freshness Check, Calibration Ledger, Scenario D, Tests 13–16,
      frontend sections 11–14
P2:   Visual polish, extra metrics, additional demo enhancements
```

A fully working P0+P1 system takes priority over a partially wired P1.5. If time is constrained, ship whichever single P1.5 feature is fully working end-to-end rather than partially wiring several.

---

## 15. Acceptance Criteria

- [ ] All files in Section 5 exist and `requirements.txt` installs cleanly
- [ ] `/health` returns healthy status; ChromaDB initializes and indexes documents
- [ ] RAG returns citation-tagged evidence; technical indicators computed from raw data
- [ ] Three agents execute concurrently; risk personalization produces divergent persona advice
- [ ] Gemini synthesis works with API key; deterministic fallback works without one
- [ ] Scenarios A, B, C, D all pass; both personas verified on identical input
- [ ] Devil's Advocate upholds/revises verdicts using only existing evidence
- [ ] Confidence breakdown is bounded and internally consistent
- [ ] Behavioral flags are persona-specific, not generic
- [ ] Filing freshness distinguished from filing absence, backend and UI
- [ ] 16/16 tests pass; no hardcoded secrets; no unsupported financial claims presented as fact
- [ ] README complete per Section 16

---

## 16. README Requirements

Overview, problem statement, architecture (ASCII + Mermaid), agent responsibilities, data flow, RAG architecture, personalization logic, conflict resolution logic, resilience strategy, security model, API docs, demo scenarios (A–D), install/env/test/run instructions, example requests + expected outputs, limitations, financial disclaimer, judge walkthrough, explanation of the four differentiating capabilities (Section 9) with one sentence each on why they're absent from reference systems, and an explicit note that the calibration ledger is session-local/demo-only.

---

## 17. Financial & Safety Disclaimer

This system is a research/education prototype for a hackathon demonstration. It does not execute trades, does not constitute financial advice, and all simulated data is explicitly labeled as such. Confidence and calibration figures reflect internal consistency on bundled demo scenarios only, not validated real-market predictive performance.

---

## 18. Investor Decision Intelligence Engine (v3 Addition)

### 18.0 Scope note — what this is and isn't

This is a **new analysis layer bolted onto the existing multi-agent pipeline (§3, §8)**, not a rebuild and not a chatbot. It does not replace the Technical/Sentiment/Fundamental agents, the Devil's Advocate pass, the Behavioral Bias Mirror, or the Calibration Ledger from §7–§9 — it consumes their outputs and adds one more question on top: *"Is this decision still justified for this specific investor, right now?"* No conversational interface is introduced; every feature below is a structured panel/endpoint, not a chat turn.

### 18.0.1 Prior-art check (honesty first)

Before treating any of the eight features below as novel, they were checked against real products, not just research papers, because overclaiming novelty in front of judges is a fast way to lose credibility:

| Concept | What already exists | Where this spec differs |
|---|---|---|
| Thesis journaling / "why I bought this" | **Horyzon, ThesisTrack, usethesis.com** — manual thesis journals with reminder dates | These are standalone note-taking tools with no market-signal integration |
| Automated thesis-break monitoring | **Helm Terminal, MyThesis, Tracking Alpha** — actively monitor filings/news against a saved thesis and alert on drift, live products as of 2026 | These operate on US equities via Plaid brokerage sync; none run on a multi-agent evidence pipeline with per-claim citation tags, and none combine thesis monitoring with a parallel agent-debate layer |
| Bull/Base/Bear valuation scenarios | **Alpha Spread, TIKR** — DCF-driven best/base/worst intrinsic value bands | These are valuation-only; they don't tie scenarios to *which specific thesis assumption failed* or to portfolio concentration impact |
| FOMO / momentum-chasing guardrails | TradingView indicators, trading-psychology tools | These are chart-side entry guardrails for active traders, not persona-aware post-hoc behavioral audits tied to a retail investor's stored decision history |
| Conviction/drift tracking research | Academic work (e.g. LLM-based conviction-trajectory tracking with evidence-grounded deltas) exists in the literature | Not shipped as a retail product; this spec is closer to that research direction than to any live consumer app |

**Conclusion:** individual pieces of this idea exist elsewhere. The combination — thesis tracking **fused into** a parallel multi-agent evidence pipeline with citation-level provenance, agent debate, and Indian-retail-specific behavioral framing, all in one decision-fit score — is not offered by any single product found. Frame it that way in the judge write-up: **integration is the innovation, not any single component in isolation.**

### 18.1 Investment Decision Twin

For a given ticker + persona, run a synthesis pass that outputs five *separately scored* dimensions instead of one blended confidence number:

| Dimension | What it measures | Primary inputs |
|---|---|---|
| Market Confidence | How strong/aligned the raw Technical + Sentiment + Fundamental signals are | Existing `TechnicalSignal`, `SentimentSignal`, `FundamentalSignal` (§8.2–8.4) |
| Decision Fit | Whether this specific action suits this specific investor's persona, portfolio, and constraints | `RiskAssessment`, persona table (§8.5) |
| Evidence Quality | Strength/freshness/citation density of the fundamental evidence behind the verdict | `filing_freshness` (§8.4), citation count |
| Thesis Health | Whether the reasons the investor originally gave for this position still hold (§18.2) | `ThesisRecord`, `ThesisBreakEvent` |
| Behavioral Risk | Whether current behavior/request pattern suggests bias-driven rather than evidence-driven action | `BehavioralFlag` (§7.2), §18.5 |

**New Pydantic model — `DecisionTwin`:** `ticker`, `persona`, `market_confidence`, `decision_fit`, `evidence_quality`, `thesis_health`, `behavioral_risk`, `composite_note` (short plain-language synthesis of how the five relate — not a sixth hidden score).

These five scores are shown **side by side, never averaged into one number** — collapsing them back into a single score would recreate the "opaque confidence" problem this system already solves in §9.2.

### 18.2 Thesis Break Detector

**New Pydantic model — `ThesisRecord`:** `ticker`, `user_id`, `created_at`, `stated_reasons[]` (free text the user enters once — "why I'm watching/holding this"), `key_assumptions[]` (structured: e.g. `"debt_to_equity stays below 1.2"`, `"management guidance holds"`), `invalidating_conditions[]`.

**New Pydantic model — `ThesisBreakEvent`**: `ticker`, `thesis_record_id`, `triggered_at`, `broken_assumption`, `evidence_citation`, `severity` (`WATCH / WEAKENED / BROKEN`), `explanation`.

**New module — `agents/thesis_break_agent.py`:** on each analysis run, compares the current `FundamentalSignal`/`TechnicalSignal`/`SentimentSignal` outputs against the stored `ThesisRecord.key_assumptions` for that ticker. If a new filing, earnings figure, or signal classification contradicts a stated assumption, emit a `ThesisBreakEvent` with the specific citation that triggered it — reusing the existing citation-tagged evidence from §8.4, never inventing new evidence. This is the mechanism that answers your framing question directly: *"is this decision still justified"* is operationalized as *"do the assumptions I originally wrote down still hold, with a citation either way."*

This must **not** become conversational — the user enters a thesis once via a structured form (stated reasons + assumptions), not via back-and-forth chat.

### 18.3 Counterfactual Decision Simulator

**New Pydantic model — `ScenarioSimulation`:** `ticker`, `scenario_type` (`BEST / BASE / FAILURE / THESIS_BREAK`), `assumption_changed`, `projected_portfolio_impact_pct`, `projected_concentration_after`, `narrative_explanation`, `simulation_disclaimer` (fixed string, always rendered — "Simulated illustration based on demo data. Not a forecast or guarantee.").

**New module — `agents/counterfactual_simulator.py`:** given the current portfolio state (§8.5) and a ticker's signal bundle, deterministically computes four labeled scenarios:
- **Best** — signals continue in their current favorable direction.
- **Base** — signals hold roughly flat.
- **Failure** — a plausible adverse move (e.g. earnings miss) sized off historical demo volatility, not invented.
- **Thesis-Break** — specifically models the scenario where the `ThesisBreakEvent`'s broken assumption plays out fully (e.g. the debt risk materializes).

Each scenario reports portfolio-level impact and post-scenario concentration, and explicitly names which assumption is being varied. No scenario may claim a specific guaranteed return — outputs are ranges/directional, always paired with the disclaimer field. This is deterministic arithmetic over the existing simulated data, not a new LLM call, keeping it consistent with §1.B (no external dependency as single point of failure).

### 18.4 Two-Track Confidence

This is the UI/API contract for the split introduced in §18.1: every response surfaces **Market Signal Confidence** and **Investor Decision Fit** as two distinct numbers with a short generated explanation of *why* they diverge when they do (e.g. "Market confidence is high because technical and sentiment agree, but decision fit is low because this position would push your portfolio concentration in this sector above your stated limit"). This reuses `DecisionTwin.market_confidence` and `DecisionTwin.decision_fit` — no new scoring logic beyond what §18.1 already defines; this section is the explicit contract that they must never be silently merged into one number anywhere in the API or UI.

### 18.5 Behavioral Drift + FOMO Detection

Extends the existing Behavioral Bias Mirror (§8.5) rather than replacing it. **New Pydantic model — `BehavioralDriftReport`:** `declared_risk_profile`, `observed_behavior_pattern`, `drift_flags: List[BehavioralFlag]`, `drift_severity` (`NONE / MILD / NOTABLE`).

Compares the persona's *declared* risk profile (from the fixed persona table, §8.5) against *simulated in-app behavior signals* bundled with the demo data — e.g. a simulated action log showing "user queried this ticker 3 times in the hour after a 12% spike" or "position size requested is 4x the persona's typical sizing." Detection logic stays rule-based and transparent (thresholds on rate-of-request, position-size ratio, sector-repeat count) — **explicitly not a psychological diagnosis**: output language is always framed as "this pattern is associated with X" with a plain-language nudge, never a label on the person (consistent with the non-judgmental framing already required in §8.5's `nudge_message`).

### 18.6 Agent Debate + Devil's Advocate (extends §8.6–8.7, does not duplicate)

No new agents beyond what §8.6–8.7 already specify (Technical, Fundamental-RAG, Sentiment running in parallel; Devil's Advocate challenging the draft synthesis). What's new here is that the Devil's Advocate Agent's challenge target is now **the thesis**, not just the market verdict: it must explicitly check the draft verdict against `ThesisRecord.key_assumptions` (§18.2) and flag if the synthesis is agreeing with a thesis that a `ThesisBreakEvent` has already contradicted. Conflicting-evidence confidence reduction (already required in §7.2/`ConfidenceBreakdown.agent_agreement_score`) now also factors in thesis-vs-signal conflict, not just agent-vs-agent conflict.

### 18.7 Evidence Graph

**New Pydantic model — `EvidenceNode`:** `node_id`, `node_type` (`DECISION / CLAIM / EVIDENCE / DOCUMENT`), `label`, `citation_tag` (nullable — only `DOCUMENT`/`EVIDENCE` nodes carry one), `parent_node_id`.

**New endpoint — `GET /api/evidence-graph/{session_id}`:** returns the node list needed to render `Final Decision → Agent Claims → Evidence → Source Documents` as a connected graph. Every `CLAIM` node must resolve to at least one `EVIDENCE` node with a `citation_tag` (page/chunk/date) when fundamental evidence exists; claims with no resolvable evidence must be visually distinguishable (e.g. dashed edge), never silently presented as equally sourced. This is a **rendering/traceability layer over data that already exists** (`source_attributions` in `SynthesizedOutput`, §7.1; citation tags in §8.4) — it does not require new evidence collection, only a graph-shaped view of what's already cited.

### 18.8 What Changed Engine

**New Pydantic model — `ChangeDigest`:** `ticker`, `since_session_id`, `changed_items: List[ChangeItem]`, where `ChangeItem` = `{category: THESIS | RISK | MARKET, description, citation_tag (nullable), materiality: LOW/MEDIUM/HIGH}`.

**New endpoint — `GET /api/what-changed/{ticker}?since={session_id}`:** diffs the current analysis run against the investor's previous stored run for the same ticker, and returns only what materially changed — prioritized in this order: (1) `ThesisBreakEvent`s since last run, (2) `BehavioralDriftReport` changes, (3) raw market classification changes. This is a diff over already-computed structured outputs (not a new agent, not an LLM summarization call) — keeps it deterministic and consistent with §1.B.

### 18.9 Project structure additions

```
agents/
    ├── thesis_break_agent.py          [NEW §18.2]
    └── counterfactual_simulator.py    [NEW §18.3]

profiling/
    └── behavioral_drift.py            [NEW §18.5, extends behavioral_bias_mirror.py]

engine/
    ├── decision_twin.py                [NEW §18.1]
    ├── evidence_graph.py               [NEW §18.7]
    └── change_digest.py                [NEW §18.8]

data/
    └── thesis_records.json             [NEW — seeded demo thesis for XYZ_CORP: "bought/watching for
                                          margin recovery; would exit if debt/equity crosses 1.2"]
```

### 18.10 New API endpoints

```
POST /api/thesis                          → create/update a ThesisRecord
GET  /api/thesis/{ticker}                 → retrieve stored thesis + break events
GET  /api/decision-twin/{ticker}          → DecisionTwin (5-dimension breakdown)
POST /api/simulate                        → ScenarioSimulation (best/base/failure/thesis-break)
GET  /api/evidence-graph/{session_id}     → EvidenceNode graph
GET  /api/what-changed/{ticker}?since=... → ChangeDigest
```

`/api/analyze` response (§10) is extended with: `decision_twin`, `thesis_break_events`, `behavioral_drift_report`.

### 18.11 New frontend panels (extends §11)

15. **Decision Twin panel** — five side-by-side scores (§18.1), never merged.
16. **Thesis tracker panel** — structured form to enter/edit a thesis; break-event timeline with citations.
17. **Counterfactual simulator panel** — four labeled scenario cards (Best/Base/Failure/Thesis-Break) with portfolio-impact bars and the fixed disclaimer always visible.
18. **Evidence graph view** — simple node-link rendering (Decision → Claims → Evidence → Documents), dashed edges for unresolved claims.
19. **What Changed digest** — a short prioritized list surfaced at the top of a return visit to a ticker, not a chat message.

### 18.12 New tests

| # | Test |
|---|---|
| 17 | `DecisionTwin`'s five scores are independently computed and never silently averaged into a returned single value |
| 18 | Thesis Break Detector flags a stored assumption as `BROKEN` only when a citation-backed evidence conflict exists — never on absence of new data |
| 19 | Counterfactual Simulator never returns a scenario without `simulation_disclaimer` populated, and Failure/Thesis-Break scenarios cite the specific assumption varied |
| 20 | Evidence Graph: every `CLAIM` node with available fundamental evidence resolves to ≥1 `EVIDENCE` node with a citation tag; claims without evidence are marked, not hidden |
| 21 | What Changed Engine returns an empty digest (not an error) when nothing material changed between two runs |
| 22 | Behavioral Drift Report never assigns a diagnostic/clinical label — output strings pass a banned-term check (no "disorder," "addiction," "pathological," etc.) |

### 18.13 Priority tier

```
P1.75 [NEW §18]: Investment Decision Twin, Thesis Break Detector, Two-Track Confidence
                  (these three are the core of the "is this decision still justified" claim —
                   build first if time allows)
P2.5  [NEW §18]: Counterfactual Simulator, Evidence Graph, What Changed Engine
                  (valuable for the demo but not load-bearing for the core claim)
```

This sits **after** the original P1.5 tier (§14) — the Devil's Advocate / Confidence Breakdown / Behavioral Bias Mirror / Filing Freshness / Calibration Ledger layer must be working first, since §18 is built on top of it, not alongside it.

### 18.14 Acceptance criteria additions

- [ ] Decision Twin's five dimensions are independently visible in both API and UI, never collapsed to one number
- [ ] A stored thesis for XYZ_CORP produces a real `ThesisBreakEvent` when the stale/debt-risk filing is analyzed, with a citation
- [ ] Counterfactual scenarios always carry the fixed disclaimer and never state a guaranteed return
- [ ] Evidence graph correctly distinguishes cited claims from uncited claims
- [ ] Behavioral Drift output contains no diagnostic/clinical language
- [ ] What Changed digest is empty (not erroring) on a repeat run with no material change
- [ ] Tests 17–22 pass alongside the original 16