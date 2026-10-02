import pandas as pd
import requests
import zipfile
import io
from datetime import datetime, timedelta

coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"]

start_date = datetime(2025, 9, 1)
end_date = datetime(2026, 9, 1)

for coin in coins:
    all_data = []
    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        url = f"https://data.binance.vision/data/spot/daily/klines/{coin}/1h/{coin}-1h-{date_str}.zip"
        print(f"Downloading data for {coin} on {date_str} from {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()

            z = zipfile.ZipFile(io.BytesIO(response.content))
            csv_name = z.namelist()[0]
            cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
            df_day = pd.read_csv(z.open(csv_name), names=cols)
            all_data.append(df_day)
        except Exception as e:
            print(f"Failed to download data for {coin} on {date_str}: {e}")

        current += timedelta(days=1)

        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            output_file = f"{coin}-1h-2026-09.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved data for {coin} to {output_file}")
        else:
            print(f"No data downloaded for {coin} in the specified date range.")