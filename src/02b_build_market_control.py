from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, ensure_directories


def main() -> None:
    ensure_directories()
    manifest = json.loads((RAW_DIR / "market_manifest.json").read_text(encoding="utf-8"))
    rows: list[dict] = []
    for filename in manifest["raw_files"]:
        payload = json.loads((RAW_DIR / filename).read_text(encoding="utf-8"))
        prices = payload.get("coins", {}).get(manifest["coin_identifier"], {}).get("prices", [])
        if not prices:
            raise ValueError(f"No ETH prices found in {filename}")
        rows.extend(prices)
    for filename in manifest.get("gap_files", []):
        payload = json.loads((RAW_DIR / filename).read_text(encoding="utf-8"))
        observation = payload.get("coins", {}).get(manifest["coin_identifier"], {})
        if "timestamp" not in observation or "price" not in observation:
            raise ValueError(f"No historical ETH price found in {filename}")
        rows.append({"timestamp": observation["timestamp"], "price": observation["price"]})

    prices = pd.DataFrame(rows)
    prices["date"] = (
        pd.to_datetime(prices["timestamp"], unit="s", utc=True).dt.round("D").dt.tz_localize(None)
    )
    prices = prices.sort_values(["date", "timestamp"]).drop_duplicates("date", keep="last")
    prices = prices[prices["date"] <= pd.Timestamp(manifest["sample_cutoff"])].reset_index(drop=True)
    full_calendar = pd.date_range(prices["date"].min(), prices["date"].max(), freq="D")
    if len(prices) != len(full_calendar) or not prices["date"].equals(pd.Series(full_calendar)):
        missing = full_calendar.difference(prices["date"])
        raise ValueError(f"ETH control has calendar gaps: {list(missing[:5])}")

    prices = prices.rename(columns={"price": "eth_price_usd"})
    prices["eth_log_return"] = np.log(prices["eth_price_usd"] / prices["eth_price_usd"].shift(1))
    prices["source"] = "DefiLlama Coins API / coingecko:ethereum"
    prices[["date", "timestamp", "eth_price_usd", "eth_log_return", "source"]].to_csv(
        PROCESSED_DIR / "eth_market_control.csv", index=False
    )
    print(
        f"Built ETH market control with {len(prices):,} calendar days "
        f"from {prices['date'].min().date()} to {prices['date'].max().date()}."
    )


if __name__ == "__main__":
    main()
