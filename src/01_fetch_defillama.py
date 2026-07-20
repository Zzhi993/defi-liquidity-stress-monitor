from __future__ import annotations

import gzip
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import RAW_DIR, SELECTED_PROTOCOLS, SOURCE_URLS, STAGING_DIR, ensure_directories


def clear_proxy_environment() -> None:
    for key in list(os.environ):
        if "proxy" in key.lower():
            os.environ.pop(key, None)


def fetch_json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "web3-capstone/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def write_json(path: Path, payload: object) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: Path) -> object:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def existing_raw_path(directory: Path, stem: str) -> Path:
    compressed = directory / f"{stem}.json.gz"
    uncompressed = directory / f"{stem}.json"
    if compressed.exists():
        return compressed
    if uncompressed.exists():
        return uncompressed
    raise FileNotFoundError(f"Missing raw snapshot file: {compressed} or {uncompressed}")


def parse_protocol_tvl(slug: str, configured_name: str, category: str, payload: dict) -> list[dict]:
    rows: list[dict] = []
    display_name = payload.get("name") or configured_name
    for item in payload.get("tvl", []):
        date_value = item.get("date")
        tvl_value = item.get("totalLiquidityUSD")
        if date_value is None or tvl_value is None:
            continue
        dt = datetime.fromtimestamp(int(date_value), tz=timezone.utc).date().isoformat()
        rows.append(
            {
                "date": dt,
                "source_timestamp_unix": int(date_value),
                "protocol_slug": slug,
                "protocol_name": display_name,
                "category": category,
                "tvl_usd": float(tvl_value),
            }
        )
    return rows


def parse_stablecoin_supply(payload: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for item in payload:
        date_value = item.get("date")
        supply = item.get("totalCirculatingUSD", {}).get("peggedUSD")
        if date_value is None or supply is None:
            continue
        dt = datetime.fromtimestamp(int(date_value), tz=timezone.utc).date().isoformat()
        rows.append(
            {
                "date": dt,
                "source_timestamp_unix": int(date_value),
                "stablecoin_supply_usd": float(supply),
            }
        )
    return rows


def main() -> None:
    ensure_directories()
    use_existing_raw = os.environ.get("USE_EXISTING_RAW", "").lower() in {"1", "true", "yes"}
    snapshot_time = datetime.now(timezone.utc)
    snapshot_id = snapshot_time.strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = RAW_DIR if use_existing_raw else RAW_DIR / "snapshots" / snapshot_id

    if not use_existing_raw:
        clear_proxy_environment()
        snapshot_dir.mkdir(parents=True, exist_ok=False)
        (RAW_DIR / "manifests").mkdir(parents=True, exist_ok=True)

    raw_manifest_path = RAW_DIR / "manifest.json"
    raw_manifest = read_json(raw_manifest_path) if use_existing_raw and raw_manifest_path.exists() else {}
    snapshot_utc = raw_manifest.get(
        "snapshot_utc", snapshot_time.isoformat(timespec="seconds")
    )
    manifest = {
        "snapshot_utc": snapshot_utc,
        "snapshot_directory": str(snapshot_dir.relative_to(RAW_DIR.parent.parent)),
        "mode": "existing immutable snapshot" if use_existing_raw else "new API snapshot",
        "sources": SOURCE_URLS,
        "protocols": SELECTED_PROTOCOLS,
    }

    protocol_rows: list[dict] = []
    for protocol in SELECTED_PROTOCOLS:
        slug = protocol["slug"]
        url = SOURCE_URLS["protocol"].format(slug=slug)
        if use_existing_raw:
            payload = read_json(existing_raw_path(snapshot_dir, f"protocol_{slug}"))
        else:
            payload = fetch_json(url)
            write_json(snapshot_dir / f"protocol_{slug}.json.gz", payload)
        protocol_rows.extend(parse_protocol_tvl(slug, protocol["name"], protocol["category"], payload))
        if not use_existing_raw:
            time.sleep(0.15)

    if use_existing_raw:
        stable_payload = read_json(existing_raw_path(snapshot_dir, "stablecoin_supply_all"))
    else:
        stable_payload = fetch_json(SOURCE_URLS["stablecoin_supply"])
        write_json(snapshot_dir / "stablecoin_supply_all.json.gz", stable_payload)
    stable_rows = parse_stablecoin_supply(stable_payload)

    protocol_df = pd.DataFrame(protocol_rows).sort_values(
        ["protocol_slug", "date", "source_timestamp_unix"]
    )
    duplicate_protocol_dates = int(protocol_df.duplicated(["date", "protocol_slug"]).sum())
    protocol_df = protocol_df.drop_duplicates(["date", "protocol_slug"], keep="last")

    stable_df = pd.DataFrame(stable_rows).sort_values(["date", "source_timestamp_unix"])
    duplicate_stablecoin_dates = int(stable_df.duplicated(["date"]).sum())
    stable_df = stable_df.drop_duplicates(["date"], keep="last")

    manifest["rows"] = {
        "protocol_rows_raw": len(protocol_rows),
        "protocol_rows_staged": len(protocol_df),
        "protocol_date_duplicates_removed": duplicate_protocol_dates,
        "stablecoin_rows_raw": len(stable_rows),
        "stablecoin_rows_staged": len(stable_df),
        "stablecoin_date_duplicates_removed": duplicate_stablecoin_dates,
    }

    protocol_df.to_csv(STAGING_DIR / "protocol_tvl.csv", index=False)
    stable_df.to_csv(STAGING_DIR / "stablecoin_supply.csv", index=False)
    write_json(STAGING_DIR / "source_manifest.json", manifest)
    if not use_existing_raw:
        write_json(snapshot_dir / "manifest.json", manifest)
        write_json(RAW_DIR / "manifests" / f"manifest_{snapshot_id}.json", manifest)

    print(
        f"Loaded {len(protocol_rows):,} protocol observations and {len(stable_rows):,} stablecoin observations; "
        f"staged {len(protocol_df):,} unique protocol-days and {len(stable_df):,} unique stablecoin-days."
    )


if __name__ == "__main__":
    main()
