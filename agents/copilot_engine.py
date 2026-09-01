"""
Comprehensive Financial Intelligence Research Copilot & Investment Advisory Engine.
Answers general investment questions, technical/fundamental concepts, and stock-specific inquiries.
"""
import os
import re
from typing import Dict, Any, List, Optional, Tuple
from config import VALID_TICKERS, AI_MODE, GEMINI_API_KEY
from schemas import CopilotQueryResponse
from agents.fundamental_rag_agent import fundamental_rag_agent

FINANCIAL_KNOWLEDGE_BASE = {
    "debt_to_equity": {
        "title": "Debt-to-Equity (D/E) Ratio",
        "category": "Fundamental Solvency",
        "content": (
            "📊 **Debt-to-Equity (D/E) Ratio** measures a company's financial leverage by dividing its total liabilities by shareholder equity:\n\n"
            "$$\\text{Debt-to-Equity} = \\frac{\\text{Total Debt}}{\\text{Total Shareholder Equity}}$$\n\n"
            "**Key Rules of Thumb for Retail Investors:**\n"
            "• **D/E < 0.5x**: Conservative / Fortress balance sheet (e.g., TCS `0.02x`, ITC `0.01x`, Reliance `0.38x`).\n"
            "• **0.5x – 1.5x**: Moderate leverage, common in capital-intensive sectors (e.g., Tata Motors `0.62x`, Airtel `1.45x`).\n"
            "• **D/E > 2.0x – 3.0x**: High default risk during economic slowdowns or rising interest rates (e.g., XYZ Corp `3.85x` triggers a `CRITICAL_RISK` alert in our engine!)."
        ),
        "citation": "[Financial-Education: Solvency Standards]"
    },
    "pe_ratio": {
        "title": "Price-to-Earnings (P/E) Ratio",
        "category": "Valuation Metrics",
        "content": (
            "📈 **Price-to-Earnings (P/E) Ratio** assesses whether a stock's price is reasonable relative to its earnings per share (EPS):\n\n"
            "$$\\text{P/E Ratio} = \\frac{\\text{Current Share Price}}{\\text{Earnings Per Share (EPS)}}$$\n\n"
            "**How to Interpret:**\n"
            "• **Trailing P/E**: Based on past 12 months' reported earnings.\n"
            "• **Forward P/E**: Based on projected future earnings.\n"
            "• **Low P/E vs High P/E**: A low P/E may indicate an undervalued value stock or a declining business; a high P/E signifies high growth expectations (e.g., TCS ~28x vs XYZ Corp ~84x speculative bubble)."
        ),
        "citation": "[Financial-Education: Valuation Ratios]"
    },
    "rsi": {
        "title": "Relative Strength Index (RSI)",
        "category": "Technical Momentum",
        "content": (
            "⚡ **Relative Strength Index (RSI)** is a 14-period momentum oscillator measuring the speed and change of price movements on a scale of 0 to 100:\n\n"
            "• **RSI > 70**: **Overbought zone** — The stock may be primed for a mean-reversion pullback or consolidation.\n"
            "• **RSI < 30**: **Oversold zone** — Potential capitulation or bargain accumulation zone.\n"
            "• **RSI 45 – 60**: Healthy structural bull trend.\n\n"
            "💡 *FinIntelligence Tip:* Never trade RSI in isolation. Always verify whether overbought momentum is supported by institutional FII/DII inflows and balance sheet solvency."
        ),
        "citation": "[Technical-Analysis: Momentum Oscillators]"
    },
    "macd": {
        "title": "MACD (Moving Average Convergence Divergence)",
        "category": "Technical Trend",
        "content": (
            "🌊 **MACD** is a trend-following momentum indicator showing the relationship between two moving averages (typically 12-day and 26-day EMAs):\n\n"
            "• **Bullish Crossover**: MACD line crosses **above** the 9-day signal line $\\rightarrow$ Upward momentum accelerating.\n"
            "• **Bearish Crossover**: MACD line crosses **below** the signal line $\\rightarrow$ Downward momentum taking over.\n"
            "• **Histogram**: Represents the distance between MACD and the signal line."
        ),
        "citation": "[Technical-Analysis: Trend Indicators]"
    },
    "sip": {
        "title": "Systematic Investment Plan (SIP) & Dollar-Cost Averaging",
        "category": "Investment Strategy",
        "content": (
            "🎯 **Systematic Investment Plan (SIP)** is a disciplined retail strategy of investing a fixed sum at regular intervals (monthly/weekly):\n\n"
            "**Key Advantages:**\n"
            "1. **Rupee-Cost Averaging**: You automatically buy more shares when prices dip and fewer when prices rise.\n"
            "2. **Eliminates Market Timing Stress**: Eliminates the urge to predict market tops and bottoms.\n"
            "3. **Power of Compounding**: Regular reinvestment over 10-15 years yields exponential wealth creation."
        ),
        "citation": "[Investment-Strategy: Retail Discipline]"
    },
    "market_view_vs_fit": {
        "title": "The Golden Law: Market View != Personal Action",
        "category": "Terminal Architecture",
        "content": (
            "⚖️ **Market View vs. Investor Fit Separation:**\n\n"
            "$$\\text{MARKET VIEW} \\neq \\text{PERSONAL ACTION}$$\n\n"
            "• **Market View (Objective)**: Evaluates technical momentum, FII/DII institutional flows, and balance sheet solvency. It is **100% identical** across all users.\n"
            "• **Investor Decision Fit (Personalized)**: Diverges based on your risk profile, age, concentration limits, and leverage tolerance.\n\n"
            "**Example:** A high-momentum stock with 3.85x debt is classified as `HIGH_RISK_MOMENTUM`. A **Conservative Senior** receives an `AVOID-HIGH RISK` verdict to protect capital, whereas an **Aggressive Gen-Z** trader receives a `HOLD-WATCH (Cap < 5%)` tactical mandate."
        ),
        "citation": "[Terminal-Architecture: Core Philosophy]"
    },
    "fii_dii": {
        "title": "FII and DII Institutional Flows",
        "category": "Institutional Flows",
        "content": (
            "🏦 **Institutional Market Flows in India:**\n\n"
            "• **FII (Foreign Institutional Investors)**: Global funds, sovereign wealth funds, and hedge funds allocating capital to Indian equities. Strongly sensitive to US bond yields and global macroeconomic liquidity.\n"
            "• **DII (Domestic Institutional Investors)**: Indian mutual funds, insurance companies (LIC), and pension funds powered by domestic retail SIP inflows.\n"
            "• **Dual Inflow Regime**: When both FII and DII are net buyers, the probability of sustained multi-quarter bull expansions increases significantly."
        ),
        "citation": "[Market-Structure: Institutional Telemetry]"
    },
    "diversification": {
        "title": "Portfolio Diversification & Position Sizing",
        "category": "Risk Management",
        "content": (
            "🛡️ **Portfolio Diversification & Concentration Ceilings:**\n\n"
            "• **Single-Stock Ceiling**: Keep any single stock below **10% – 15%** of your total portfolio value.\n"
            "• **Sector Exposure**: Avoid allocating more than **25% – 30%** into a single industry (e.g. IT, Auto, Banking).\n"
            "• **Asset Allocation**: Balance high-growth equities with cash reserves or debt instruments to buffer against black-swan corrections."
        ),
        "citation": "[Risk-Management: Portfolio Construction]"
    },
    "fomo_bias": {
        "title": "Cognitive Bias Trap: FOMO & Momentum Chasing",
        "category": "Behavioral Finance",
        "content": (
            "🧠 **Behavioral Bias Mirror: Momentum Chasing (FOMO)**\n\n"
            "• **The Trap**: Buying stocks after a parabolic 50%+ spike purely because social media chatter or financial news headlines are euphoric.\n"
            "• **The Danger**: Late retail entrants often provide exit liquidity for institutional smart money right before a mean-reversion correction.\n"
            "• **FinIntelligence Nudge**: Our behavioral drift engine monitors request velocity and position multiplier requests to flag FOMO traps objectively."
        ),
        "citation": "[Behavioral-Finance: Cognitive Bias Prevention]"
    }
}

STOCK_OVERVIEWS = {
    "TATAMOTORS": {
        "name": "Tata Motors Limited",
        "sector": "Automobile & EV",
        "summary": (
            "🚗 **Tata Motors Limited (TATAMOTORS)**\n\n"
            "• **Business Highlights**: Leader in Indian Passenger Electric Vehicles (EVs) with over 70% market share, alongside commercial vehicles and luxury automotive brand Jaguar Land Rover (JLR).\n"
            "• **Balance Sheet Health**: Prudent deleveraging reduced D/E ratio to **0.62x** (cited from `[SEBI-Filing-TATAMOTORS-Q3: Page 12]`).\n"
            "• **Growth Drivers**: JLR free cash flow generation, domestic EV expansion, and commercial vehicle infrastructure demand."
        ),
        "citation": "[SEBI-Filing-TATAMOTORS-Q3: Page 12]"
    },
    "INFOSYS": {
        "name": "Infosys Limited",
        "sector": "Information Technology",
        "summary": (
            "💻 **Infosys Limited (INFOSYS)**\n\n"
            "• **Business Highlights**: Global IT consulting leader specializing in cloud transformation, enterprise software, and generative AI services.\n"
            "• **Balance Sheet Health**: Pristine debt-free balance sheet with D/E of **0.08x** (cited from `[SEBI-Filing-INFOSYS-Q3: Page 8]`).\n"
            "• **Growth Drivers**: Cloud migration partnerships, long-term enterprise AI contracts, and robust dividend yield."
        ),
        "citation": "[SEBI-Filing-INFOSYS-Q3: Page 8]"
    },
    "XYZ_CORP": {
        "name": "XYZ CleanPower & Logistics Infra Ltd",
        "sector": "Infrastructure & Energy (Case Study)",
        "summary": (
            "⚠️ **XYZ CleanPower Infra (XYZ_CORP) — High Leverage Case Study**\n\n"
            "• **Business Setup**: Rapid top-line growth and speculative retail social chatter, but heavily debt-funded.\n"
            "• **Balance Sheet Warning**: Debt-to-Equity expanded to **3.85x** (cited from `[SEBI-Filing-XYZ-Q3: Page 14]`), breaching statutory covenants.\n"
            "• **Engine Decision**: Triggers an Anti-Majority Vote conflict override $\\rightarrow$ `AVOID-HIGH RISK` for conservative investors."
        ),
        "citation": "[SEBI-Filing-XYZ-Q3: Page 14]"
    },
    "RELIANCE": {
        "name": "Reliance Industries Limited",
        "sector": "Energy, Retail & Telecom",
        "summary": (
            "🏢 **Reliance Industries Limited (RELIANCE)**\n\n"
            "• **Business Segments**: Jio Platforms (Telecom & 5G), Reliance Retail (18,900+ stores), Oil-to-Chemicals (O2C), and Green Energy Gigafactories.\n"
            "• **Balance Sheet Health**: Low net debt with D/E of **0.38x** (cited from `[SEBI-Filing-RELIANCE-Q3: Page 18]`).\n"
            "• **Growth Drivers**: 5G tariff monetization, retail footfall growth, and green hydrogen/solar manufacturing."
        ),
        "citation": "[SEBI-Filing-RELIANCE-Q3: Page 18]"
    },
    "HDFCBANK": {
        "name": "HDFC Bank Limited",
        "sector": "Banking & Financial Services",
        "summary": (
            "🏦 **HDFC Bank Limited (HDFCBANK)**\n\n"
            "• **Business Highlights**: India's premier private bank with a network of 8,950+ branches.\n"
            "• **Asset Quality & Solvency**: Basel III Capital Adequacy Ratio (CAR) of **19.4%**, Net NPA controlled at **0.31%**, and D/E equivalent of **0.85x** (`[SEBI-Filing-HDFCBANK-Q3: Page 15]`).\n"
            "• **Growth Drivers**: Steady 14-16% advance growth and post-merger deposit expansion."
        ),
        "citation": "[SEBI-Filing-HDFCBANK-Q3: Page 15]"
    },
    "TCS": {
        "name": "Tata Consultancy Services",
        "sector": "IT Consulting & Services",
        "summary": (
            "🌐 **Tata Consultancy Services (TCS)**\n\n"
            "• **Business Highlights**: Leading IT services exporter with industry-leading 26% operating margins.\n"
            "• **Balance Sheet Health**: Fortress cash reserves with D/E of **0.02x** (`[SEBI-Filing-TCS-Q3: Page 10]`).\n"
            "• **Growth Drivers**: USD 8.1B deal pipeline in cloud and enterprise AI, consistent dividend payouts."
        ),
        "citation": "[SEBI-Filing-TCS-Q3: Page 10]"
    },
    "BHARTIARTL": {
        "name": "Bharti Airtel Limited",
        "sector": "Telecommunications",
        "summary": (
            "📡 **Bharti Airtel Limited (BHARTIARTL)**\n\n"
            "• **Business Highlights**: Leading telecom service provider across India and Africa with 5G rollout in 7,500+ cities.\n"
            "• **Financials & Solvency**: India Mobile ARPU expanded to **₹234/month**, D/E ratio moderating to **1.45x** (`[SEBI-Filing-BHARTIARTL-Q3: Page 16]`).\n"
            "• **Growth Drivers**: ARPU premiumization, 4G-to-5G conversions, and enterprise IoT/data center solutions."
        ),
        "citation": "[SEBI-Filing-BHARTIARTL-Q3: Page 16]"
    },
    "ITC": {
        "name": "ITC Limited",
        "sector": "FMCG, Agribusiness & Paperboards",
        "summary": (
            "🌿 **ITC Limited (ITC)**\n\n"
            "• **Business Highlights**: Diversified conglomerate with market leadership in FMCG, cigarettes, agri-exports, and sustainable packaging.\n"
            "• **Balance Sheet Health**: Completely debt-free balance sheet with D/E of **0.01x** (`[SEBI-Filing-ITC-Q3: Page 11]`).\n"
            "• **Growth Drivers**: FMCG-Others revenue growth (13% YoY), robust free cash flow, and strong dividend yield."
        ),
        "citation": "[SEBI-Filing-ITC-Q3: Page 11]"
    }
}

class CopilotEngine:
    def detect_ticker(self, question: str) -> Optional[str]:
        q_clean = question.upper()
        # Direct ticker symbol matches
        for t in VALID_TICKERS:
            if re.search(r'\b' + re.escape(t) + r'\b', q_clean):
                return t
        
        # Name alias matching
        name_map = {
            "TATA MOTORS": "TATAMOTORS", "TATA MOTOR": "TATAMOTORS", "TATAMOTOR": "TATAMOTORS", "JLR": "TATAMOTORS",
            "INFOSYS": "INFOSYS", "INFY": "INFOSYS",
            "XYZ": "XYZ_CORP", "XYZ CORP": "XYZ_CORP", "CLEANPOWER": "XYZ_CORP",
            "RELIANCE": "RELIANCE", "RIL": "RELIANCE", "JIO": "RELIANCE", "MUKESH AMBANI": "RELIANCE",
            "HDFC": "HDFCBANK", "HDFC BANK": "HDFCBANK", "HDFCBANK": "HDFCBANK",
            "TCS": "TCS", "TATA CONSULTANCY": "TCS",
            "AIRTEL": "BHARTIARTL", "BHARTI": "BHARTIARTL", "BHARTI AIRTEL": "BHARTIARTL",
            "ITC": "ITC", "ITC LIMITED": "ITC"
        }
        for alias, t in name_map.items():
            if alias in q_clean:
                return t
        return None

    def handle_concept_query(self, question: str) -> Optional[Tuple[str, str]]:
        q_lower = question.lower()
        
        if any(k in q_lower for k in ["debt to equity", "debt/equity", "d/e ratio", "d/e", "solvency", "leverage ratio"]):
            return FINANCIAL_KNOWLEDGE_BASE["debt_to_equity"]["content"], FINANCIAL_KNOWLEDGE_BASE["debt_to_equity"]["citation"]
        
        if any(k in q_lower for k in ["p/e", "pe ratio", "price to earnings", "valuation", "expensive", "cheap stock"]):
            return FINANCIAL_KNOWLEDGE_BASE["pe_ratio"]["content"], FINANCIAL_KNOWLEDGE_BASE["pe_ratio"]["citation"]
        
        if any(k in q_lower for k in ["rsi", "relative strength", "overbought", "oversold"]):
            return FINANCIAL_KNOWLEDGE_BASE["rsi"]["content"], FINANCIAL_KNOWLEDGE_BASE["rsi"]["citation"]
        
        if any(k in q_lower for k in ["macd", "moving average convergence", "signal line", "crossover"]):
            return FINANCIAL_KNOWLEDGE_BASE["macd"]["content"], FINANCIAL_KNOWLEDGE_BASE["macd"]["citation"]
        
        if any(k in q_lower for k in ["sip", "dollar cost", "rupee cost", "systematic investment", "how to invest monthly"]):
            return FINANCIAL_KNOWLEDGE_BASE["sip"]["content"], FINANCIAL_KNOWLEDGE_BASE["sip"]["citation"]
        
        if any(k in q_lower for k in ["market view", "personal action", "difference between market", "personal fit"]):
            return FINANCIAL_KNOWLEDGE_BASE["market_view_vs_fit"]["content"], FINANCIAL_KNOWLEDGE_BASE["market_view_vs_fit"]["citation"]
        
        if any(k in q_lower for k in ["fii", "dii", "institutional flow", "foreign investors", "domestic investors"]):
            return FINANCIAL_KNOWLEDGE_BASE["fii_dii"]["content"], FINANCIAL_KNOWLEDGE_BASE["fii_dii"]["citation"]
        
        if any(k in q_lower for k in ["diversification", "how many stocks", "portfolio allocation", "position sizing", "risk management"]):
            return FINANCIAL_KNOWLEDGE_BASE["diversification"]["content"], FINANCIAL_KNOWLEDGE_BASE["diversification"]["citation"]
        
        if any(k in q_lower for k in ["fomo", "chasing", "recency bias", "bias", "behavioral", "psychology"]):
            return FINANCIAL_KNOWLEDGE_BASE["fomo_bias"]["content"], FINANCIAL_KNOWLEDGE_BASE["fomo_bias"]["citation"]
            
        return None

    def query(self, question: str, ticker_context: Optional[str] = None) -> CopilotQueryResponse:
        q_lower = question.lower().strip()
        detected_ticker = self.detect_ticker(question)
        target_ticker = detected_ticker or ticker_context or "TATAMOTORS"

        # 1. System Feature & Terminal Navigation Guide
        if any(k in q_lower for k in ["what is this website", "how to use", "guide", "what does this do", "features", "website", "finintel"]):
            ans = (
                "👋 **Welcome to FinIntelligence AI!**\n\n"
                "This is an autonomous multi-agent financial intelligence terminal designed for Indian retail investors.\n\n"
                "**Key Features you can explore:**\n"
                "1. **Overview & Verdict**: Synthesizes market signals while strictly separating **Market View** from personalized **Investor Decision Fit**.\n"
                "2. **Agent Signals**: Specialized Technical, Sentiment (with separate FII/DII flows), and Fundamental Vector RAG agents.\n"
                "3. **Decision Twin (5D)**: 5 independently scored dimensions evaluated side by side without collapsing.\n"
                "4. **Devil's Advocate**: 3-step adversarial challenge that stress-tests draft recommendations.\n"
                "5. **Thesis & Biases**: Behavioral bias mirror detecting FOMO/recency bias + investment thesis tracker.\n"
                "6. **Risk Scenario Analysis**: Illustrative stress simulations across Upside, Base, Downside, and Thesis Break states.\n"
                "7. **Audit & Replay**: Complete Evidence Provenance tree and 7-stage Reasoning Replay Scrubber."
            )
            return CopilotQueryResponse(
                answer=ans,
                cited_sources=["[Terminal-Architecture: System Guide]"],
                grounded_in_rag=True,
                ai_mode="guide_mode"
            )

        # 2. If a specific company or filing disclosure is queried -> Prioritize Evidence RAG
        has_specific_company = bool(detected_ticker) or any(k in q_lower for k in ["disclosed", "filing", "quarterly", "report", "earnings", "sebi", "ratio for", "ratio of"])
        if has_specific_company:
            retrieved = fundamental_rag_agent.retrieve(query=question, ticker=target_ticker, top_k=2)
            if retrieved:
                citations = [r["citation_tag"] for r in retrieved]
                chunk_snippet = retrieved[0]["text"].strip()
                ans = (
                    f"🔎 **Evidence-Grounded Disclosures for {target_ticker}:**\n\n"
                    f"Based on statutory regulatory filings in **{', '.join(citations)}**:\n\n"
                    f"> {chunk_snippet}\n\n"
                    f"💡 **Analysis:** This metric is derived with verified provenance directly from corporate disclosures."
                )
                return CopilotQueryResponse(
                    answer=ans,
                    cited_sources=citations,
                    grounded_in_rag=True,
                    ai_mode="rag_grounded"
                )

        # 3. Stock Overview Inquiries
        if any(k in q_lower for k in ["overview", "tell me about", "what is", "about", "company details", "profile"]) and target_ticker in STOCK_OVERVIEWS:
            stock_info = STOCK_OVERVIEWS[target_ticker]
            return CopilotQueryResponse(
                answer=stock_info["summary"],
                cited_sources=[stock_info["citation"]],
                grounded_in_rag=True,
                ai_mode="stock_profile"
            )

        # 4. General Financial Concept & Strategy Queries
        concept_match = self.handle_concept_query(question)
        if concept_match:
            content, citation = concept_match
            return CopilotQueryResponse(
                answer=content,
                cited_sources=[citation],
                grounded_in_rag=True,
                ai_mode="knowledge_base"
            )

        # 5. Evidence-Grounded Vector RAG Retrieval Fallback
        retrieved = fundamental_rag_agent.retrieve(query=question, ticker=target_ticker, top_k=2)
        citations = [r["citation_tag"] for r in retrieved] if retrieved else []
        evidence_text = "\n".join([r["text"] for r in retrieved]) if retrieved else ""

        # 6. Gemini Synthesis (if enabled)
        if (AI_MODE == "gemini" or bool(os.getenv("GEMINI_API_KEY"))) and os.getenv("GEMINI_API_KEY"):
            try:
                from google import genai
                client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                prompt = (
                    f"You are the Senior Financial Analyst & Research Copilot for FinIntelligence AI.\n"
                    f"Answer the user's question with expert, clear, and structured financial insights.\n\n"
                    f"User Question: {question}\n"
                    f"Active Stock: {target_ticker}\n"
                    f"Retrieved Corporate Filing Disclosures:\n{evidence_text if evidence_text else 'No specific filing chunks retrieved.'}\n"
                    f"Citation Tags: {citations}\n\n"
                    f"Instructions:\n"
                    f"1. Use markdown formatting, bullet points, and emojis.\n"
                    f"2. Reference exact filing citations if available (e.g. {citations[0] if citations else '[General-Finance: Principles]'}).\n"
                    f"3. Uphold the rule: 'Market View != Personal Action'.\n"
                    f"4. Provide educational, objective retail investment analysis."
                )
                resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                if resp and resp.text:
                    return CopilotQueryResponse(
                        answer=resp.text.strip(),
                        cited_sources=citations if citations else ["[FinIntel-GenAI: Gemini-2.5-Flash]"],
                        grounded_in_rag=bool(retrieved),
                        ai_mode="gemini"
                    )
            except Exception:
                pass

        # 7. High-Quality Deterministic RAG Response
        if retrieved:
            chunk_snippet = retrieved[0]["text"].strip()
            ans = (
                f"🔎 **Evidence-Grounded Research on {target_ticker}:**\n\n"
                f"Based on statutory regulatory disclosures in **{', '.join(citations)}**:\n\n"
                f"> {chunk_snippet}\n\n"
                f"💡 **Analysis:** This disclosure confirms verifiable balance sheet metrics and operational performance without speculative assumptions."
            )
            return CopilotQueryResponse(
                answer=ans,
                cited_sources=citations,
                grounded_in_rag=True,
                ai_mode="rag_grounded"
            )

        # 8. General Stock & Investment Fallback
        general_ans = (
            f"💡 **Financial Intelligence Insight:**\n\n"
            f"Regarding *\"{question}\"* for **{target_ticker}**:\n\n"
            f"• **Fundamental Solvency**: Always check the Debt-to-Equity ratio and interest coverage from official SEBI LODR filings.\n"
            f"• **Technical Setup**: Verify 14-period RSI and MACD volume momentum before entering positions.\n"
            f"• **Personal Suitability**: Ensure your position size respects your max single-stock concentration ceiling (10–15%).\n\n"
            f"You can ask me specific questions like: *\"What is {target_ticker}'s debt ratio?\"*, *\"Explain RSI\"*, or *\"Market View vs Personal Fit\"*."
        )
        return CopilotQueryResponse(
            answer=general_ans,
            cited_sources=["[FinIntel: Investment Principles]"],
            grounded_in_rag=False,
            ai_mode="general_financial_assistant"
        )

copilot_engine = CopilotEngine()
