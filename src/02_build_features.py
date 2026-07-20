from __future__ import annotations

import math
import sqlite3

import pandas as pd

from config import PROCESSED_DIR, SQL_DIR, STAGING_DIR, ensure_directories


def load_csv_to_sqlite(conn: sqlite3.Connection, table_name: str, csv_path) -> None:
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)


def main() -> None:
    ensure_directories()

    db_path = PROCESSED_DIR / "defi_contagion.sqlite"
    conn = sqlite3.connect(db_path)
    conn.create_function("LN", 1, lambda value: math.log(value) if value and value > 0 else None)

    load_csv_to_sqlite(conn, "protocol_tvl", STAGING_DIR / "protocol_tvl.csv")
    load_csv_to_sqlite(conn, "stablecoin_supply", STAGING_DIR / "stablecoin_supply.csv")

    sql = (SQL_DIR / "build_features.sql").read_text(encoding="utf-8")
    conn.executescript(sql)

    for table in [
        "clean_protocol_tvl",
        "clean_stablecoin_supply",
        "protocol_returns",
        "system_daily",
        "protocol_metadata",
        "protocol_coverage",
    ]:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        df.to_csv(PROCESSED_DIR / f"{table}.csv", index=False)

    conn.close()
    print(f"Built SQL features in {db_path}.")


if __name__ == "__main__":
    main()
