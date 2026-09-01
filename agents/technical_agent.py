import pandas as pd
import numpy as np
from typing import Dict, Any, List
from schemas import TechnicalSignal

class TechnicalAgent:
    def __init__(self):
        self.name = "TechnicalAgent"

    def analyze(self, ticker: str, market_data: Dict[str, Any], degraded_override: bool = False) -> TechnicalSignal:
        """
        Calculates technical indicators from raw OHLCV price/volume series.
        """
        price_series = market_data.get("price_series", [])
        if degraded_override or not price_series or len(price_series) < 14:
            return TechnicalSignal(
                ticker=ticker,
                rsi=50.0,
                macd_signal="NEUTRAL",
                momentum_score=0.0,
                volume_anomaly_score=0.0,
                classification="NEUTRAL",
                confidence=0.35,
                reasons=["Technical data degraded or insufficient historical series; defaulted to neutral baseline."],
                degraded=True
            )

        df = pd.DataFrame(price_series)
        close = df["close"].values
        volume = df["volume"].values

        # 1. RSI (14 periods)
        deltas = np.diff(close)
        if len(deltas) == 0:
            rsi = 50.0
        else:
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = float(np.mean(gains[-14:])) if len(gains) >= 14 else float(np.mean(gains))
            avg_loss = float(np.mean(losses[-14:])) if len(losses) >= 14 else float(np.mean(losses))
            
            if avg_loss == 0:
                rsi = 100.0 if avg_gain > 0 else 50.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = float(round(rsi, 2))

        # 2. MACD (Fast 12, Slow 26, Signal 9)
        s_close = pd.Series(close)
        ema12 = s_close.ewm(span=12, adjust=False).mean()
        ema26 = s_close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        latest_macd = macd_line.iloc[-1]
        latest_signal = signal_line.iloc[-1]
        
        if latest_macd > latest_signal and (len(macd_line) > 1 and macd_line.iloc[-2] <= signal_line.iloc[-2] or latest_macd > 0):
            macd_status = "BULLISH_CROSSOVER"
        elif latest_macd < latest_signal and (len(macd_line) > 1 and macd_line.iloc[-2] >= signal_line.iloc[-2] or latest_macd < 0):
            macd_status = "BEARISH_CROSSOVER"
        else:
            macd_status = "NEUTRAL"

        # 3. Momentum (Rate of Change over last 10 periods)
        if len(close) >= 10 and close[-10] != 0:
            momentum = float(round(((close[-1] - close[-10]) / close[-10]) * 100, 2))
        elif len(close) > 0 and close[0] != 0:
            momentum = float(round(((close[-1] - close[0]) / close[0]) * 100, 2))
        else:
            momentum = 0.0

        # 4. Volume Anomaly (Z-score of latest volume vs 20-day mean)
        recent_vols = volume[-20:] if len(volume) >= 20 else volume
        vol_mean = float(np.mean(recent_vols)) if len(recent_vols) > 0 else 0.0
        vol_std = float(np.std(recent_vols)) if len(recent_vols) > 0 else 0.0
        if vol_std > 0:
            vol_z = float(round((volume[-1] - vol_mean) / vol_std, 2))
        else:
            vol_z = 0.0

        # 5. Structured reasons & classification logic
        reasons: List[str] = []
        bullish_votes = 0
        bearish_votes = 0

        # RSI logic
        if rsi > 70:
            reasons.append(f"RSI at {rsi} indicates strong upside momentum but approaches overbought territory.")
            bullish_votes += 1
        elif rsi >= 55:
            reasons.append(f"RSI at {rsi} demonstrates healthy bullish momentum above centerline.")
            bullish_votes += 2
        elif rsi <= 30:
            reasons.append(f"RSI at {rsi} is in oversold territory; potential stabilization watch.")
            bearish_votes += 1
        elif rsi <= 45:
            reasons.append(f"RSI at {rsi} reflects defensive consolidation below centerline.")
            bearish_votes += 2
        else:
            reasons.append(f"RSI at {rsi} is balanced near neutral midpoint.")

        # MACD logic
        if macd_status == "BULLISH_CROSSOVER":
            reasons.append("MACD line resides above the 9-period signal line, confirming positive trend acceleration.")
            bullish_votes += 2
        elif macd_status == "BEARISH_CROSSOVER":
            reasons.append("MACD line crossed below signal line, indicating momentum deceleration.")
            bearish_votes += 2
        else:
            reasons.append("MACD histogram reflects balanced momentum without clear divergence.")

        # Momentum & Volume
        if momentum > 10.0:
            reasons.append(f"10-period rate of change (+{momentum}%) indicates strong price expansion.")
            bullish_votes += 1
        elif momentum < -5.0:
            reasons.append(f"10-period rate of change ({momentum}%) indicates downward price drift.")
            bearish_votes += 1

        if vol_z > 1.0:
            reasons.append(f"Volume is {vol_z:.2f} standard deviations above its 20-day mean, indicating elevated trading activity.")
            bullish_votes += 1
        elif vol_z < -1.0:
            reasons.append(f"Volume is {abs(vol_z):.2f} standard deviations below average, reflecting subdued turnover.")
        else:
            reasons.append(f"Trading volume resides within normal bounds ({vol_z:.2f} standard deviations).")

        # Final Classification
        if bullish_votes >= 4 and bearish_votes == 0:
            classification = "STRONG_BULLISH"
            conf = 0.88
        elif bullish_votes >= 2 and bullish_votes > bearish_votes:
            classification = "BULLISH"
            conf = 0.80
        elif bearish_votes >= 4 and bullish_votes == 0:
            classification = "STRONG_BEARISH"
            conf = 0.85
        elif bearish_votes >= 2 and bearish_votes > bullish_votes:
            classification = "BEARISH"
            conf = 0.75
        else:
            classification = "NEUTRAL"
            conf = 0.65

        return TechnicalSignal(
            ticker=ticker,
            rsi=rsi,
            macd_signal=macd_status,
            momentum_score=momentum,
            volume_anomaly_score=vol_z,
            classification=classification,
            confidence=conf,
            reasons=reasons,
            degraded=False
        )

technical_agent = TechnicalAgent()
