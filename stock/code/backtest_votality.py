import pandas as pd
import numpy as np
from scipy.stats import spearmanr

coins = ["INTCBUSDT", "SKHYBUSDT", "NVDABUSDT", "SNDKBUSDT", "GOOGLBUSDT", "MUBUSDT", "AMDBUSDT", "METABUSDT"]

dfs = {}

for coin in coins:
    df = pd.read_csv(f"stock/data/{coin}-1h-2026-10-03.csv")
    df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = df[["open_time", "close", "volume"]].rename(columns={"close": coin, "volume": f"{coin}_vol"})
    dfs[coin] = df
    print(f"{coin}: {len(df)} rows")

merged = dfs[coins[0]]
for coin in coins[1:]:
    merged = pd.merge(merged, dfs[coin], on="open_time", how="outer")
merged.sort_values("open_time", inplace=True)

merged["open_time"] = merged["open_time"].astype("int64") // 1000
merged["datetime"] = pd.to_datetime(merged["open_time"], unit="ms")
merged.set_index("datetime", inplace=True)
merged.drop(columns=["open_time"], inplace=True)

price = merged[coins].copy()
volume = merged[[f"{coin}_vol" for coin in coins]].copy()
volume.columns = coins


hourly_ret = price.pct_change()
ret_24h = price.pct_change(periods=24)
vol_24h = price.pct_change().rolling(24).std()
signal_1 = (ret_24h / vol_24h).replace([np.inf, -np.inf], np.nan)

short_vol = hourly_ret.rolling(12).std()
long_vol = hourly_ret.rolling(48).std()
signal_2 = (short_vol / long_vol).replace([np.inf, -np.inf], np.nan)

signal_3 = -vol_24h

vol_12h = hourly_ret.rolling(12).std()
vol_72h = hourly_ret.rolling(72).std()
signal_4 = (vol_24h / vol_72h).replace([np.inf, -np.inf], np.nan)

signals = {
    "Vol-adjusted Momentum": signal_1,
    "Volatility Expansion": signal_2,
    "Inverse Volatility": signal_3,
    "Volatility Ratio": signal_4
}


results = []
rebalance_frequency = 24

for name, signal in signals.items():
    n_valid = signal.notna().sum(axis=1)
    rank_matrix = signal.rank(axis=1, ascending=True)
    long_pos = pd.DataFrame(0, index=signal.index, columns=signal.columns)
    current_holding = None

    for i in range(len(rank_matrix)):
        if i % rebalance_frequency == 0:
            row = rank_matrix.iloc[i]
            n = n_valid.iloc[i]
            if n > 0 and not row.isna().all():
                candidates = row[row == n]
                if len(candidates) > 0:
                    current_holding = candidates.index[0]
        if current_holding is not None:
            long_pos.loc[long_pos.index[i], current_holding] = 1

    strategy_returns = (hourly_ret * long_pos.shift(1)).sum(axis=1).fillna(0)
    taker_fee = 0.001
    maker_fee = 0.0005
    turnover = long_pos.diff().abs().sum(axis=1)
    strategy_returns = (hourly_ret * long_pos.shift(1)).sum(axis=1).fillna(0)
    strategy_returns -= turnover * taker_fee

    
    initial_equity = 10000
    equity = (1 + strategy_returns).cumprod() * initial_equity
    total_return = (equity.iloc[-1] / initial_equity - 1) * 100
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_drawdown = drawdown.min() * 100

    sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(24 * 365)
    win_rate = (strategy_returns > 0).sum() / (strategy_returns != 0).sum() * 100

    results.append({
        "Signal": name,
        "Total Return (%)": round(total_return, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown (%)": round(max_drawdown, 2),
        "Win Rate (%)": round(win_rate, 2)
    })


    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.plot(equity, label="Equity Curve", color='#2E86AB', linewidth=2)
    plt.axhline(y=initial_equity, color='gray', linestyle='--', label='Initial Equity = $10000')
    plt.title(f'Equity Curve: {name}', fontsize=14)
    plt.xlabel("Time")
    plt.ylabel("Equity($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"equity_{name.replace(' ', '_')}.png", dpi=150)
    plt.close()
    print(f"{name}: Return: {total_return:.2f}%, Sharpe: {sharpe:.2f}, Max Drawdown: {max_drawdown:.2f}%, Win Rate: {win_rate:.2f}%")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Total Return (%)", ascending=False)
print("\n" + "+" *60)
print("AVERAGE SIGNAL PERFORMANCE (PORTFOLIO)")
print("=" *60)
print(results_df.to_string(index=False))
results_df.to_csv("signal_performance.csv", index=False)
print("\nSignal performance saved to signal_performance.csv")