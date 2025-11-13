#!/usr/bin/env python3
"""
Update Blacklist or Whitelist tables from a CSV file.

- List type chosen with --list-type [blacklist|whitelist]
- CSV must have 'address'; 'reason' is optional and used only for blacklist.
- Default mode: add/update entries from CSV.
- --remove mode: remove addresses from the chosen list.

Example:

  # Replace / upsert blacklist from a prepared CSV
  python update_blacklist.py \
    --db data/blacklist.db \
    --list-type blacklist \
    --csv data/blacklist_full_updated.csv

  # Add to whitelist
  python update_blacklist.py \
    --db data/blacklist.db \
    --list-type whitelist \
    --csv data/some_addresses.csv

  # Remove addresses from blacklist
  python update_blacklist.py \
    --db data/blacklist.db \
    --list-type blacklist \
    --csv data/remove_these.csv \
    --remove
"""

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_address(addr) -> Optional[str]:
    """Simple, forgiving normalizer: strip + lowercase, drop empty/NaN."""
    if pd.isna(addr):
        return None
    s = str(addr).strip()
    if not s:
        return None
    return s.lower()


def load_addresses_from_csv(path: Path, need_reason: bool) -> List[Tuple]:
    """
    Load addresses (and optional reasons) from CSV.

    Returns:
      - if need_reason: list of (address, reason)
      - else: list of (address,)
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.lower().str.strip()

    if "address" not in df.columns:
        raise ValueError(f"CSV {path} must contain an 'address' column")

    if need_reason and "reason" not in df.columns:
        logger.info(f"{path}: no 'reason' column, using 'Manual Import'")
        df["reason"] = "Manual Import"

    records = []
    skipped = 0

    for _, row in df.iterrows():
        addr = normalize_address(row["address"])
        if not addr:
            skipped += 1
            continue

        if need_reason:
            raw_reason = row.get("reason", "Manual Import")
            if pd.isna(raw_reason):
                raw_reason = "Manual Import"
            reason = str(raw_reason).strip() or "Manual Import"
            records.append((addr, reason))
        else:
            records.append((addr,))

    if skipped:
        logger.warning(f"Skipped {skipped} rows with empty/invalid addresses in {path}")

    return records


def add_or_update_blacklist(conn: sqlite3.Connection, entries: List[Tuple[str, str]]) -> int:
    """
    Upsert blacklist entries: address + reason.

    Returns number of rows affected (best-effort estimate).
    """
    if not entries:
        return 0
    cur = conn.cursor()
    # Requires address to be unique / primary key for best effect.
    cur.executemany(
        'INSERT OR REPLACE INTO "Blacklist" (address, reason) VALUES (?, ?)',
        entries,
    )
    conn.commit()
    return cur.rowcount


def remove_from_blacklist(conn: sqlite3.Connection, addresses: List[str]) -> int:
    if not addresses:
        return 0
    cur = conn.cursor()
    cur.executemany(
        'DELETE FROM "Blacklist" WHERE address = ?',
        [(a,) for a in addresses],
    )
    conn.commit()
    return cur.rowcount


def add_or_update_whitelist(conn: sqlite3.Connection, addresses: List[str]) -> int:
    if not addresses:
        return 0
    cur = conn.cursor()
    cur.executemany(
        'INSERT OR IGNORE INTO "Whitelist" (address) VALUES (?)',
        [(a,) for a in addresses],
    )
    conn.commit()
    return cur.rowcount


def remove_from_whitelist(conn: sqlite3.Connection, addresses: List[str]) -> int:
    if not addresses:
        return 0
    cur = conn.cursor()
    cur.executemany(
        'DELETE FROM "Whitelist" WHERE address = ?',
        [(a,) for a in addresses],
    )
    conn.commit()
    return cur.rowcount


def main():
    parser = argparse.ArgumentParser(
        description="Update Blacklist or Whitelist from a CSV file."
    )

    here = Path(__file__).resolve().parent
    default_db = here / "data" / "blacklist.db"

    parser.add_argument(
        "--db",
        type=Path,
        default=default_db,
        help=f"Path to SQLite DB (default: {default_db})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="CSV file with addresses (and optionally reasons)",
    )
    parser.add_argument(
        "--list-type",
        choices=["blacklist", "whitelist"],
        default="blacklist",
        help="Which table to update (default: blacklist)",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="If set, remove the addresses in CSV from the chosen list "
             "instead of adding/updating them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Using DB: {args.db}")
    logger.info(f"CSV file: {args.csv}")
    logger.info(f"List type: {args.list_type}")
    logger.info(f"Mode: {'REMOVE' if args.remove else 'ADD/UPDATE'}")

    # Connect DB
    conn = sqlite3.connect(args.db)

    try:
        if args.list_type == "blacklist":
            need_reason = not args.remove  # reason only needed when inserting
            records = load_addresses_from_csv(args.csv, need_reason=need_reason)

            if args.remove:
                addresses = [addr for (addr,) in records] if records and isinstance(records[0], tuple) and len(records[0]) == 1 else [r[0] for r in records]
                logger.info(f"Removing {len(addresses)} addresses from Blacklist...")
                affected = remove_from_blacklist(conn, addresses)
                logger.info(f"Rows deleted from Blacklist: {affected}")
            else:
                logger.info(f"Upserting {len(records)} entries into Blacklist...")
                affected = add_or_update_blacklist(conn, records)
                logger.info(f"Rows inserted/updated in Blacklist (approx): {affected}")

        else:  # whitelist
            # For whitelist, we only care about addresses.
            raw = load_addresses_from_csv(args.csv, need_reason=False)
            addresses = [addr for (addr,) in raw]

            if args.remove:
                logger.info(f"Removing {len(addresses)} addresses from Whitelist...")
                affected = remove_from_whitelist(conn, addresses)
                logger.info(f"Rows deleted from Whitelist: {affected}")
            else:
                logger.info(f"Adding {len(addresses)} addresses to Whitelist (INSERT OR IGNORE)...")
                affected = add_or_update_whitelist(conn, addresses)
                logger.info(f"Rows inserted into Whitelist (approx): {affected}")

    finally:
        conn.close()

    logger.info("Done.")


if __name__ == "__main__":
    main()
