import pandas as pd

coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"]

dfs = {}

for coin in coins:
    df = pd.read_csv(f"data/{coin}-1h-2026-09.csv")
    df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = df[["open_time", "close"]].rename(columns={"close": coin})
    dfs[coin] = df
    print(f"{coin}: {len(df)} rows")


merged = dfs[coins[0]]
for coin in coins[1:]:
    merged = pd.merge(merged, dfs[coin], on="open_time", how="outer")
merged.sort_values("open_time", inplace=True)
merged.ffill(inplace=True)

merged["open_time"] = merged["open_time"].astype("int64") // 1000
merged["datetime"] = pd.to_datetime(merged["open_time"].astype("int64"), unit="ms")
merged.set_index("datetime", inplace=True)
merged.drop(columns=["open_time"], inplace=True)

print(merged.head())
print(f"Merged DataFrame: {len(merged)} rows")


returns = merged.pct_change(periods=24)
print(returns.tail())

returns = returns.dropna()

latest = returns.iloc[-1]
best_coin = latest.idxmax()
worst_coin = latest.idxmin()

print(f"Best performing coin in the last 24 hours: {best_coin} with return {latest[best_coin]:.2f}%")
print(f"Worst performing coin in the last 24 hours: {worst_coin} with return {latest[worst_coin]:.2f}%")


hourly_returns = merged.pct_change()
lookback_returns = merged.pct_change(periods=24)
rank_matrix = lookback_returns.rank(axis=1, ascending=True)

position_matrix = (rank_matrix == len(coins)).astype(int)

strategy_returns = (hourly_returns * position_matrix.shift(1)).sum(axis=1)
strategy_returns = strategy_returns.iloc[25:].fillna(0)

initial_equity = 10000
equity = (1 + strategy_returns).cumprod() * initial_equity
peak = equity.cummax()
drawdown = (equity - peak) / peak
max_drawdown = drawdown.min()

print(f"\nInitial Equity: ${initial_equity:.2f}")
print(f"Final Equity: ${equity.iloc[-1]:.2f}")
print(f"Total Return: {(equity.iloc[-1] / initial_equity - 1) * 100:.2f}%")
print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
