import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ============================================================
# STRATEGY V1: Signal 3 + Signal 6 + Signal 8
# ============================================================
# Signal 3 (from teammate code):
#   - MA48 / MA72 trend-following
#   - ATR48 used for risk-scaled position sizing
#   - signal confirmed on close, trade at next bar open
#
# Signal 6:
#   - RSI(14) reversal
#   - compare 25/75 and 30/70
#   - 1.5% stop-loss variant
#
# Signal 8:
#   - past 6h return < -2%
#   - current volume < previous 24h average volume
#   - used ONLY as a confirmation/size boost for a new S6 long
#
# NOTE:
#   This is a first combined backtest, not final competition code.
#   It counts ACTIVE TRADING DAYS so you can check the >= 8-day rule.
# ============================================================

# -----------------------------
# EDIT THESE FILE PATHS
# -----------------------------
# CSV must contain timestamp/open/high/low/close/volume.
# Headerless Binance 6-column files are also supported.
FILES: Dict[str, str] = {
    # Example:
    # "BTC": "data/BTCUSDT_1h_2024-09_to_2026-08.csv",
    # "ETH": "data/ETHUSDT_1h_2024-09_to_2026-08.csv",
}

# -----------------------------
# COST / CAPITAL
# -----------------------------
INITIAL_CASH = 100_000.0
TAKER_FEE = 0.001  # 0.1% per trade side; change if official rule differs

# -----------------------------
# SIGNAL 3 SETTINGS
# -----------------------------
S3_SHORT_MA = 48
S3_LONG_MA = 72
S3_ATR_PERIOD = 48
S3_RISK_PER_TRADE = 0.02
S3_ATR_MULTIPLIER = 3.0
S3_MAX_WEIGHT = 0.15  # combined-strategy cap, NOT teammate's original 100%

# -----------------------------
# SIGNAL 6 SETTINGS
# -----------------------------
RSI_PERIOD = 14
S6_STOP_LOSS = 0.015
S6_NORMAL_WEIGHT = 0.075
S6_STRONG_WEIGHT = 0.15

# -----------------------------
# SIGNAL 8 SETTINGS
# -----------------------------
S8_LOOKBACK_HOURS = 6
S8_SHOCK_THRESHOLD = 0.02
S8_VOLUME_MA = 24
S8_VOLUME_THRESHOLD = 1.0

# -----------------------------
# PORTFOLIO LIMIT
# -----------------------------
TOTAL_GROSS_EXPOSURE_CAP = 0.60


@dataclass
class ModelConfig:
    name: str
    use_s3: bool = True
    use_s6: bool = False
    use_s8: bool = False
    rsi_low: int = 25
    rsi_high: int = 75


MODELS = [
    ModelConfig("A_S3_only", use_s3=True),
    ModelConfig("B_S3_S6_25_75", use_s3=True, use_s6=True, rsi_low=25, rsi_high=75),
    ModelConfig("B_S3_S6_30_70", use_s3=True, use_s6=True, rsi_low=30, rsi_high=70),
    ModelConfig("C_S3_S6_S8_25_75", use_s3=True, use_s6=True, use_s8=True, rsi_low=25, rsi_high=75),
    ModelConfig("C_S3_S6_S8_30_70", use_s3=True, use_s6=True, use_s8=True, rsi_low=30, rsi_high=70),
]


def load_ohlcv(path: str) -> pd.DataFrame:
    """Load a standard OHLCV CSV or a headerless Binance-style first-6-column CSV."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    # First try normal headered CSV
    df = pd.read_csv(p)
    lower_cols = [str(c).strip().lower() for c in df.columns]
    required = {"timestamp", "open", "high", "low", "close", "volume"}

    if not required.issubset(set(lower_cols)):
        # Retry as headerless Binance first 6 columns
        df = pd.read_csv(p, header=None)
        if df.shape[1] < 6:
            raise ValueError(f"{path}: expected at least 6 columns")
        df = df.iloc[:, :6]
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    else:
        df.columns = lower_cols
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    # Parse timestamps robustly
    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        # Binance timestamps are usually milliseconds
        median_abs = pd.to_numeric(ts, errors="coerce").abs().median()
        unit = "ms" if median_abs > 10_000_000_000 else "s"
        df["timestamp"] = pd.to_datetime(ts, unit=unit, utc=True, errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime(ts, utc=True, errors="coerce")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    # Signal 3 moving averages
    x["ma_short"] = x["close"].rolling(S3_SHORT_MA).mean()
    x["ma_long"] = x["close"].rolling(S3_LONG_MA).mean()

    # ATR (Wilder-style EMA, same structure as teammate code)
    prev_close = x["close"].shift(1)
    tr = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - prev_close).abs(),
            (x["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    x["atr"] = tr.ewm(alpha=1 / S3_ATR_PERIOD, adjust=False).mean()

    # Signal 6 RSI(14), matching teammate's simple rolling-average RSI
    delta = x["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(RSI_PERIOD).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = 100 - (100 / (1 + rs))
    x.loc[(loss == 0) & (gain > 0), "rsi"] = 100.0
    x.loc[(gain == 0) & (loss > 0), "rsi"] = 0.0

    # Signal 8: 6h downside shock + low current volume vs PREVIOUS 24h mean
    x["ret_6h"] = x["close"].pct_change(S8_LOOKBACK_HOURS)
    x["volume_ma_24_prev"] = x["volume"].shift(1).rolling(S8_VOLUME_MA).mean()
    x["volume_ratio"] = x["volume"] / x["volume_ma_24_prev"]
    x["s8_event"] = (
        (x["ret_6h"] < -S8_SHOCK_THRESHOLD)
        & (x["volume_ratio"] < S8_VOLUME_THRESHOLD)
    )

    return x


def compute_s3_state(df: pd.DataFrame) -> pd.DataFrame:
    """Create a persistent long trend state: enter on MA48 cross above MA72, exit on cross below."""
    x = df.copy()
    prev_short = x["ma_short"].shift(1)
    prev_long = x["ma_long"].shift(1)

    x["s3_entry_signal"] = (prev_short <= prev_long) & (x["ma_short"] > x["ma_long"])
    x["s3_exit_signal"] = (prev_short >= prev_long) & (x["ma_short"] < x["ma_long"])

    state = 0
    states = []
    for entry, exit_ in zip(x["s3_entry_signal"].fillna(False), x["s3_exit_signal"].fillna(False)):
        if state == 0 and entry:
            state = 1
        elif state == 1 and exit_:
            state = 0
        states.append(state)
    x["s3_state"] = states

    # Risk-scaled S3 weight, capped at 15% for combined strategy.
    # Formula comes from teammate's risk budget / ATR stop distance logic.
    raw_weight = (
        S3_RISK_PER_TRADE
        * x["close"]
        / (x["atr"] * S3_ATR_MULTIPLIER)
    )
    x["s3_weight"] = raw_weight.clip(lower=0, upper=S3_MAX_WEIGHT).fillna(0.0)
    return x


def simulate_asset(df: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """
    Build the target weight for one asset.

    All indicator decisions are made using bar t CLOSE.
    The target is shifted by one bar and executed at bar t+1 OPEN.
    """
    x = compute_s3_state(add_indicators(df))

    # S6 persistent state, with close-based 1.5% stop-loss.
    s6_state = 0
    s6_entry_price: Optional[float] = None
    s6_strong_entry = False

    target_signal_weight = []
    reason = []

    for _, row in x.iterrows():
        close = float(row["close"])
        rsi = row["rsi"]
        s8 = bool(row["s8_event"]) if pd.notna(row["s8_event"]) else False

        # Update S6 state only if this model uses it.
        if cfg.use_s6 and pd.notna(rsi):
            if s6_state == 1:
                stopped = (
                    s6_entry_price is not None
                    and close <= s6_entry_price * (1 - S6_STOP_LOSS)
                )
                if stopped or rsi > cfg.rsi_high:
                    s6_state = 0
                    s6_entry_price = None
                    s6_strong_entry = False
            else:
                if rsi < cfg.rsi_low:
                    s6_state = 1
                    s6_entry_price = close
                    # Signal 8 only boosts a NEW S6 entry.
                    s6_strong_entry = cfg.use_s8 and s8

        # Priority rule:
        # 1) If S3 trend is active -> trend position (do not stack S6 on top)
        # 2) Else if S6 reversal is active -> 7.5% or 15% if S8 confirmed at entry
        # 3) Else cash
        if cfg.use_s3 and int(row["s3_state"]) == 1:
            w = float(row["s3_weight"])
            target_signal_weight.append(w)
            reason.append("S3_TREND")
        elif cfg.use_s6 and s6_state == 1:
            w = S6_STRONG_WEIGHT if s6_strong_entry else S6_NORMAL_WEIGHT
            target_signal_weight.append(w)
            reason.append("S6_S8_REVERSAL" if s6_strong_entry else "S6_REVERSAL")
        else:
            target_signal_weight.append(0.0)
            reason.append("CASH")

    x["raw_target_weight"] = target_signal_weight
    x["reason"] = reason

    # Execute next bar open to avoid using information from the future.
    x["target_weight"] = x["raw_target_weight"].shift(1).fillna(0.0)
    x["exec_reason"] = x["reason"].shift(1).fillna("CASH")
    return x


def align_assets(asset_frames: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    if not asset_frames:
        raise ValueError("No asset data loaded")
    common = None
    for df in asset_frames.values():
        idx = df.index
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) < 100:
        raise ValueError("Not enough overlapping timestamps across assets")
    return common.sort_values()


def backtest_portfolio(files: Dict[str, str], cfg: ModelConfig):
    if not files:
        raise ValueError("FILES is empty. Add your CSV paths at the top of the script.")

    prepared = {}
    for symbol, path in files.items():
        df = load_ohlcv(path)
        prepared[symbol] = simulate_asset(df, cfg)

    idx = align_assets(prepared)
    prepared = {k: v.loc[idx].copy() for k, v in prepared.items()}

    cash = INITIAL_CASH
    units = {s: 0.0 for s in prepared}
    total_fees = 0.0
    trade_rows = []
    equity_rows = []

    # We rebalance at each bar OPEN only when target weights changed enough to create an order.
    prev_targets = {s: 0.0 for s in prepared}

    for ts in idx:
        open_prices = {s: float(prepared[s].at[ts, "open"]) for s in prepared}
        close_prices = {s: float(prepared[s].at[ts, "close"]) for s in prepared}

        # Mark equity using current OPEN just before rebalancing.
        equity_open = cash + sum(units[s] * open_prices[s] for s in prepared)

        raw_targets = {s: float(prepared[s].at[ts, "target_weight"]) for s in prepared}

        # Portfolio gross exposure cap: scale all targets proportionally if needed.
        gross = sum(abs(w) for w in raw_targets.values())
        scale = 1.0 if gross <= TOTAL_GROSS_EXPOSURE_CAP or gross == 0 else TOTAL_GROSS_EXPOSURE_CAP / gross
        targets = {s: raw_targets[s] * scale for s in prepared}

        for s in prepared:
            price = open_prices[s]
            target_value = equity_open * targets[s]
            current_value = units[s] * price
            delta_value = target_value - current_value

            # Ignore tiny rebalance noise (< $1).
            if abs(delta_value) < 1.0:
                prev_targets[s] = targets[s]
                continue

            delta_units = delta_value / price
            fee = abs(delta_value) * TAKER_FEE

            # Long-only safeguard; do not spend more cash than available.
            if delta_units > 0:
                max_buy_value = max(cash / (1 + TAKER_FEE), 0.0)
                buy_value = min(delta_value, max_buy_value)
                if buy_value < 1.0:
                    continue
                delta_units = buy_value / price
                fee = buy_value * TAKER_FEE
                cash -= buy_value + fee
                units[s] += delta_units
                side = "BUY"
                traded_value = buy_value
            else:
                sell_units = min(-delta_units, units[s])
                if sell_units * price < 1.0:
                    continue
                traded_value = sell_units * price
                fee = traded_value * TAKER_FEE
                cash += traded_value - fee
                units[s] -= sell_units
                side = "SELL"

            total_fees += fee
            trade_rows.append(
                {
                    "timestamp": ts,
                    "symbol": s,
                    "side": side,
                    "notional": traded_value,
                    "fee": fee,
                    "target_weight": targets[s],
                    "reason": prepared[s].at[ts, "exec_reason"],
                }
            )
            prev_targets[s] = targets[s]

        equity_close = cash + sum(units[s] * close_prices[s] for s in prepared)
        equity_rows.append({"timestamp": ts, "equity": equity_close})

    trades = pd.DataFrame(trade_rows)
    equity = pd.DataFrame(equity_rows).set_index("timestamp")
    equity["return"] = equity["equity"].pct_change().fillna(0.0)

    metrics = calculate_metrics(equity, trades, total_fees)
    return metrics, equity, trades


def calculate_metrics(equity: pd.DataFrame, trades: pd.DataFrame, total_fees: float) -> dict:
    start = float(equity["equity"].iloc[0])
    end = float(equity["equity"].iloc[-1])
    net_return = end / start - 1

    r = equity["return"]
    if r.std(ddof=0) > 0:
        sharpe = r.mean() / r.std(ddof=0) * math.sqrt(24 * 365)
    else:
        sharpe = np.nan

    downside = r[r < 0]
    if len(downside) > 1 and downside.std(ddof=0) > 0:
        sortino = r.mean() / downside.std(ddof=0) * math.sqrt(24 * 365)
    else:
        sortino = np.nan

    roll_max = equity["equity"].cummax()
    dd = equity["equity"] / roll_max - 1
    max_dd = float(dd.min())

    elapsed_days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86400, 1 / 24)
    annualized_return = (end / start) ** (365 / elapsed_days) - 1 if end > 0 else np.nan
    calmar = annualized_return / abs(max_dd) if max_dd < 0 and pd.notna(annualized_return) else np.nan

    if len(trades) > 0:
        active_dates = pd.to_datetime(trades["timestamp"]).dt.date
        active_days = active_dates.nunique()
        unique_dates = sorted(set(active_dates))
        if len(unique_dates) >= 2:
            gaps = [(unique_dates[i] - unique_dates[i - 1]).days for i in range(1, len(unique_dates))]
            longest_no_trade_gap = max(gaps) - 1
        else:
            longest_no_trade_gap = int(math.floor(elapsed_days))
    else:
        active_days = 0
        longest_no_trade_gap = int(math.floor(elapsed_days))

    return {
        "Net Return %": net_return * 100,
        "Annualized Return %": annualized_return * 100 if pd.notna(annualized_return) else np.nan,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max Drawdown %": max_dd * 100,
        "Total Trades": int(len(trades)),
        "Active Trading Days": int(active_days),
        "Meets 8-Day Rule": bool(active_days >= 8),
        "Longest No-Trade Gap (days)": int(longest_no_trade_gap),
        "Total Fees": total_fees,
        "End Equity": end,
    }


def main():
    if not FILES:
        print("\nFILES is currently empty.")
        print("Edit the FILES dictionary at the top, for example:")
        print('FILES = {"BTC": "data/BTC.csv", "ETH": "data/ETH.csv"}')
        return

    out_dir = Path("combined_results")
    out_dir.mkdir(exist_ok=True)

    rows = []
    for cfg in MODELS:
        print(f"\nRunning {cfg.name} ...")
        metrics, equity, trades = backtest_portfolio(FILES, cfg)
        rows.append({"Model": cfg.name, **metrics})
        equity.to_csv(out_dir / f"{cfg.name}_equity.csv")
        trades.to_csv(out_dir / f"{cfg.name}_trades.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "model_comparison.csv", index=False)

    print("\n" + "=" * 140)
    print("MODEL COMPARISON")
    print("=" * 140)
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
