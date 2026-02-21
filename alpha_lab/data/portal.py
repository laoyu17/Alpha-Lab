from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class DataPortal:
    """Unified data access for daily/minute parquet datasets."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def load(
        self,
        frequency: str,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        file_name = "daily.parquet" if frequency == "daily" else "minute.parquet"
        path = self.data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"dataset not found: {path}")

        df = pd.read_parquet(path)
        if "datetime" not in df.columns:
            raise ValueError(f"missing datetime column in {path}")
        if "symbol" not in df.columns:
            raise ValueError(f"missing symbol column in {path}")

        df["datetime"] = pd.to_datetime(df["datetime"])
        if symbols:
            df = df[df["symbol"].isin(symbols)]
        if start is not None:
            df = df[df["datetime"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["datetime"] <= pd.Timestamp(end)]

        df = df.sort_values(["datetime", "symbol"])
        return df.set_index(["datetime", "symbol"]).sort_index()

    def load_daily(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        return self.load("daily", symbols=symbols, start=start, end=end)

    def load_minute(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        return self.load("minute", symbols=symbols, start=start, end=end)


def generate_demo_data(output_dir: str | Path, seed: int = 7) -> None:
    """Generate reproducible daily and minute parquet demo datasets."""

    rng = np.random.default_rng(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    industries = {
        "AAA": "tech",
        "BBB": "finance",
        "CCC": "tech",
        "DDD": "industrial",
        "EEE": "finance",
    }
    sizes = {"AAA": 10.5, "BBB": 10.2, "CCC": 9.8, "DDD": 10.0, "EEE": 9.6}

    daily_dates = pd.bdate_range("2023-01-03", periods=260)
    rows: list[dict[str, object]] = []

    for sym in symbols:
        base = 100 + rng.uniform(-2, 2)
        drift = rng.normal(0.0002, 0.0001)
        vol = rng.uniform(0.012, 0.02)
        prices = [base]
        for _ in range(1, len(daily_dates)):
            ret = drift + rng.normal(0, vol)
            prices.append(prices[-1] * (1 + ret))
        price_arr = np.array(prices)

        for idx, dt in enumerate(daily_dates):
            close = float(price_arr[idx])
            open_price = close * float(1 + rng.normal(0, 0.003))
            high = max(open_price, close) * float(1 + abs(rng.normal(0, 0.004)))
            low = min(open_price, close) * float(1 - abs(rng.normal(0, 0.004)))
            suspended = rng.random() < 0.01
            limit_hit = rng.random() < 0.02
            rows.append(
                {
                    "datetime": dt,
                    "symbol": sym,
                    "open": round(open_price, 4),
                    "high": round(high, 4),
                    "low": round(low, 4),
                    "close": round(close, 4),
                    "volume": float(rng.integers(800_000, 5_000_000)),
                    "amount": round(close * float(rng.integers(800_000, 5_000_000)), 2),
                    "adj_factor": 1.0,
                    "suspended": suspended,
                    "limit_hit": limit_hit,
                    "industry": industries[sym],
                    "size": sizes[sym] + float(rng.normal(0, 0.05)),
                }
            )

    daily_df = pd.DataFrame(rows)
    daily_df.to_parquet(output_path / "daily.parquet", index=False)
    daily_lookup = daily_df.set_index(["datetime", "symbol"])["close"].astype(float).to_dict()

    minute_rows: list[dict[str, object]] = []
    intraday_slots = pd.date_range("09:30", "14:50", freq="40min").time
    for dt in daily_dates[-60:]:
        for sym in symbols:
            day_close = float(daily_lookup[(dt, sym)])
            minute_price = day_close * (1 + rng.normal(0, 0.01))
            for tm in intraday_slots:
                minute_dt = pd.Timestamp.combine(dt.date(), tm)
                shock = rng.normal(0, 0.0015)
                open_price = minute_price
                close = open_price * (1 + shock)
                high = max(open_price, close) * (1 + abs(rng.normal(0, 0.0008)))
                low = min(open_price, close) * (1 - abs(rng.normal(0, 0.0008)))
                volume = float(rng.integers(30_000, 150_000))
                minute_rows.append(
                    {
                        "datetime": minute_dt,
                        "symbol": sym,
                        "open": round(open_price, 4),
                        "high": round(high, 4),
                        "low": round(low, 4),
                        "close": round(close, 4),
                        "volume": volume,
                        "adj_factor": 1.0,
                        "suspended": False,
                        "limit_hit": False,
                        "industry": industries[sym],
                        "size": sizes[sym],
                    }
                )
                minute_price = close

    minute_df = pd.DataFrame(minute_rows)
    minute_df.to_parquet(output_path / "minute.parquet", index=False)
