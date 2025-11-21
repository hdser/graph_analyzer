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
    Ensures uniqueness - if duplicates exist, keeps first occurrence.

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
    seen_addresses = set()
    duplicates = 0

    for _, row in df.iterrows():
        addr = normalize_address(row["address"])
        if not addr:
            skipped += 1
            continue

        # Check for duplicates
        if addr in seen_addresses:
            duplicates += 1
            logger.debug(f"Skipping duplicate address: {addr}")
            continue
        seen_addresses.add(addr)

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
    
    if duplicates:
        logger.warning(f"Skipped {duplicates} duplicate addresses in {path} (kept first occurrence)")

    return records


def add_or_update_blacklist(conn: sqlite3.Connection, entries: List[Tuple[str, str]]) -> Tuple[int, int, int]:
    """
    Upsert blacklist entries: address + reason.
    Skips addresses that are currently in the Whitelist.
    Checks for existing addresses and updates them or inserts new ones.

    Returns tuple: (rows_inserted, rows_updated, skipped_due_to_whitelist)
    """
    if not entries:
        return 0, 0, 0
    
    cur = conn.cursor()
    
    # Get all whitelisted addresses
    cur.execute('SELECT address FROM "Whitelist"')
    whitelist = {row[0] for row in cur.fetchall()}
    
    # Get all existing blacklisted addresses
    cur.execute('SELECT address FROM "Blacklist"')
    existing_blacklist = {row[0] for row in cur.fetchall()}
    
    # Separate into updates and inserts
    to_insert = []
    to_update = []
    skipped_whitelist = 0
    skipped_exists = 0
    
    for addr, reason in entries:
        if addr in whitelist:
            skipped_whitelist += 1
            logger.warning(f"Skipping blacklist entry (already whitelisted): {addr}")
        elif addr in existing_blacklist:
            # Update existing entry
            to_update.append((reason, addr))
        else:
            # Insert new entry
            to_insert.append((addr, reason))
    
    inserted = 0
    updated = 0
    
    # Perform updates
    if to_update:
        cur.executemany(
            'UPDATE "Blacklist" SET reason = ? WHERE address = ?',
            to_update
        )
        updated = cur.rowcount
        logger.info(f"Updated {updated} existing blacklist entries")
    
    # Perform inserts
    if to_insert:
        cur.executemany(
            'INSERT INTO "Blacklist" (address, reason) VALUES (?, ?)',
            to_insert
        )
        inserted = cur.rowcount
        logger.info(f"Inserted {inserted} new blacklist entries")
    
    conn.commit()
    return inserted, updated, skipped_whitelist


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
    default_db = here.parent / "data" / "blacklist.db"

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
                logger.info(f"Processing {len(records)} entries for Blacklist...")
                inserted, updated, skipped = add_or_update_blacklist(conn, records)
                logger.info(f"Inserted {inserted} new addresses into Blacklist")
                logger.info(f"Updated {updated} existing addresses in Blacklist")
                if skipped > 0:
                    logger.warning(f"Skipped {skipped} addresses that are in Whitelist")

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