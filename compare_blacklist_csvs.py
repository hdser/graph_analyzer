#!/usr/bin/env python3
"""
Compare blacklist CSV data with graph_metrics_v1/v2 and then with blacklist.db.

Phase 1: PURE CSV + graph logic (NO DB)
--------------------------------------
- Load graph_metrics_v1.csv and graph_metrics_v2.csv (column 'avatar')
  -> V1 avatars, V2 avatars
- Load multiple blacklist CSV sheets (e.g. blacklist_Sheet1..4.csv)
  -> addresses + reasons
- For each address, collect:
    * canonical_reason: first reason seen (for output)
    * has_circles_v1_reason: True if ANY reason contains 'CirclesV1'/'CiclesV1'
- Filtering rules (only CSV + graphs):
    * addr in V1:
        - keep ONLY if has_circles_v1_reason is True
        - else discard (v1 avatar but not a v1 bot)
    * addr not in V1:
        - keep (v2-only or unknown; v1 logic doesn't restrict them)

Result of Phase 1:
    - filtered CSV blacklist candidate set (addresses + reasons)

Phase 2: DB comparison (blacklist.db)
-------------------------------------
- Load Blacklist and Whitelist from SQLite
- Compare:
    * overlap between existing blacklist and CSV candidates
    * in DB only / in CSV-only
    * whitelist conflicts
    * stats vs graph_metrics_v2 avatars:
        - currently blacklisted
        - CSV-only blacklist
        - merged blacklist
- Extra outputs:
    * list of v2 avatars that would be blacklisted by CSV but are not yet
      in DB (saved as new_v2_blacklist_not_in_db.csv)
    * list of v1 avatars that would be newly blacklisted (not in DB)
      (saved as new_v1_blacklist_not_in_db.csv)
- Build a "full updated blacklist" CSV:
    * Existing DB blacklist entries are canonical (address, reason)
    * CSV-only addresses (not in DB) are appended with their CSV reason
    * EXCLUDES addresses that are in the Whitelist
    * Ensures uniqueness (no duplicate addresses)
    * Written to output CSV (e.g. data/blacklist_full_updated.csv)
"""

import argparse
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Tuple, Set, Dict

import pandas as pd

logger = logging.getLogger(__name__)


# ---------- Helpers (common) ----------

def normalize_address(addr) -> Optional[str]:
    """Simple, forgiving normalizer: strip + lowercase, drop empty/NaN."""
    if pd.isna(addr):
        return None
    s = str(addr).strip()
    if not s:
        return None
    return s.lower()


def load_graph_avatars(path: Path) -> Set[str]:
    """
    Load avatar addresses from a graph_metrics CSV
    (expects a column named 'avatar', case-insensitive).
    """
    logger.info(f"Loading graph metrics from: {path}")
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.lower().str.strip()
    if "avatar" not in df.columns:
        raise ValueError(f"{path} must contain an 'avatar' column")
    avatars = {normalize_address(a) for a in df["avatar"]}
    return {a for a in avatars if a}


def load_csv_blacklists(
    csv_paths: Iterable[Path],
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, int], Dict[str, bool]]:
    """
    Load and merge multiple blacklist CSVs.

    Expected columns:
    - required: 'address' (case insensitive)
    - optional: 'reason' (default 'Manual Import')

    Dedup rule:
    - canonical_reason[address] = first reason encountered
    - address_sheet_counts[address] = number of DISTINCT CSV files where address appears
    - has_circles_v1_reason[address] = True if ANY reason contains 'CirclesV1'/'CiclesV1'

    Returns:
        merged_df: DataFrame with columns [address, reason] (canonical reason)
        address_to_reason: dict[address -> canonical reason]
        address_sheet_counts: dict[address -> number_of_sheets]
        has_circles_v1_reason: dict[address -> bool]
    """
    address_to_reason: Dict[str, str] = {}
    address_sheet_counts: Dict[str, int] = defaultdict(int)
    has_circles_v1_reason: Dict[str, bool] = defaultdict(bool)

    total_rows = 0
    duplicate_addresses = 0
    conflicting_reasons = 0

    for path in csv_paths:
        logger.info(f"Loading CSV: {path}")
        df = pd.read_csv(path, dtype=str)
        df.columns = df.columns.str.lower().str.strip()

        if "address" not in df.columns:
            raise ValueError(f"CSV {path} must contain an 'address' column")

        if "reason" not in df.columns:
            logger.info(f"{path}: no 'reason' column, using 'Manual Import'")
            df["reason"] = "Manual Import"

        total_rows += len(df)

        # Track which addresses appear in this particular sheet (for sheet counts)
        addresses_in_this_sheet: Set[str] = set()

        for _, row in df.iterrows():
            addr = normalize_address(row["address"])
            if not addr:
                continue

            reason_val = row["reason"]
            if pd.isna(reason_val):
                reason_val = "Manual Import"
            reason = str(reason_val).strip() or "Manual Import"

            # For canonical reason: first occurrence wins
            if addr in address_to_reason:
                duplicate_addresses += 1
                if address_to_reason[addr] != reason:
                    conflicting_reasons += 1
            else:
                address_to_reason[addr] = reason

            # Track if ANY reason mentions CirclesV1/CiclesV1 (case-insensitive)
            rl = reason.lower()
            if "circlesv1" in rl or "ciclesv1" in rl:
                has_circles_v1_reason[addr] = True

            # Per-sheet count (count once per sheet)
            if addr not in addresses_in_this_sheet:
                addresses_in_this_sheet.add(addr)
                address_sheet_counts[addr] += 1

    logger.info(f"Total CSV rows read: {total_rows}")
    logger.info(f"Unique addresses from CSVs: {len(address_to_reason)}")
    logger.info(f"Duplicate addresses across CSVs: {duplicate_addresses}")
    logger.info(f"Duplicates with conflicting reasons: {conflicting_reasons}")

    merged_df = pd.DataFrame(
        [{"address": a, "reason": r} for a, r in address_to_reason.items()]
    )
    return merged_df, address_to_reason, address_sheet_counts, has_circles_v1_reason


# ---------- Helpers (DB) ----------

def load_blacklist(conn: sqlite3.Connection) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute('SELECT address, reason FROM "Blacklist"')
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["address", "reason"])
    df["address"] = df["address"].map(normalize_address)
    df = df.dropna(subset=["address"]).drop_duplicates(subset=["address"])
    return df


def load_whitelist(conn: sqlite3.Connection) -> Set[str]:
    cur = conn.cursor()
    cur.execute('SELECT address FROM "Whitelist"')
    rows = cur.fetchall()
    addrs = [normalize_address(r[0]) for r in rows]
    return {a for a in addrs if a}


# ---------- Main logic ----------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare blacklist CSVs using v1/v2 graph logic, then compare with blacklist.db, "
            "and produce a full updated blacklist CSV (excluding whitelisted addresses)."
        )
    )

    here = Path(__file__).resolve().parent
    default_db = here / "data" / "blacklist.db"
    default_graph_v2 = here / "data" / "graph_metrics_v2.csv"
    default_graph_v1 = here / "data" / "graph_metrics_v1.csv"
    default_output = here / "data" / "blacklist_full_updated.csv"

    parser.add_argument(
        "--db",
        type=Path,
        default=default_db,
        help=f"Path to SQLite DB (default: {default_db})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        required=True,
        help="One or more blacklist CSV files to add/compare",
    )
    parser.add_argument(
        "--graph-metrics-v2",
        type=Path,
        default=default_graph_v2,
        help=f"Path to graph_metrics_v2.csv (default: {default_graph_v2})",
    )
    parser.add_argument(
        "--graph-metrics-v1",
        type=Path,
        default=default_graph_v1,
        help=f"Path to graph_metrics_v1.csv (default: {default_graph_v1})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=default_output,
        help=f"Path for full updated blacklist CSV (default: {default_output})",
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

    # ---------------------------
    # PHASE 1: CSV + graph logic
    # ---------------------------

    logger.info("=== PHASE 1: CSV + graph_metrics_v1/v2 logic (NO DB) ===")

    # Load v1 and v2 avatars
    v1_avatars = load_graph_avatars(args.graph_metrics_v1)
    v2_avatars = load_graph_avatars(args.graph_metrics_v2)

    logger.info(f"Total v1 avatars (graph_metrics_v1.csv): {len(v1_avatars)}")
    logger.info(f"Total v2 avatars (graph_metrics_v2.csv): {len(v2_avatars)}")

    # Load CSV blacklist data
    (
        csv_merged_df,
        addr_to_reason,
        addr_sheet_counts,
        has_circles_v1_reason,
    ) = load_csv_blacklists(args.csv)

    all_csv_addrs = set(addr_to_reason.keys())
    logger.info(f"Total CSV addresses (raw): {len(all_csv_addrs)}")

    # Classify addresses by membership
    v1_in_csv = all_csv_addrs & v1_avatars
    v2_in_csv = all_csv_addrs & v2_avatars
    v1_and_v2_in_csv = v1_in_csv & v2_in_csv

    logger.info(f"v1 avatars in CSV (raw): {len(v1_in_csv)}")
    logger.info(f"v2 avatars in CSV (raw): {len(v2_in_csv)}")
    logger.info(f"avatars in BOTH v1 and v2 (in CSV): {len(v1_and_v2_in_csv)}")

    # Apply your rule:
    # - if addr in V1 -> keep only if has_circles_v1_reason[addr] is True
    # - if addr not in V1 -> keep
    kept_entries = []
    discarded_entries = []

    for addr, reason in addr_to_reason.items():
        is_v1 = addr in v1_avatars
        is_v2 = addr in v2_avatars
        has_circles = has_circles_v1_reason.get(addr, False)
        flags = addr_sheet_counts.get(addr, 0)

        if is_v1:
            if has_circles:
                # v1 avatar and explicitly flagged as v1 bot -> keep
                kept_entries.append((addr, reason))
            else:
                # v1 avatar but NOT a v1 bot -> discard
                discarded_entries.append((addr, reason, is_v2, flags))
        else:
            # Not a v1 avatar -> v2-only or unknown -> keep as v2 candidate
            kept_entries.append((addr, reason))

    if kept_entries:
        filtered_csv_df = pd.DataFrame(kept_entries, columns=["address", "reason"])
    else:
        filtered_csv_df = pd.DataFrame(columns=["address", "reason"])

    new_addrs_filtered = set(filtered_csv_df["address"])

    # v1 stats after filtering
    v1_addrs_kept = new_addrs_filtered & v1_avatars
    v1_addrs_discarded = v1_in_csv - v1_addrs_kept

    logger.info("=== CSV v1/v2 FILTERING SUMMARY ===")
    logger.info(f"Total CSV addresses (after v1 filtering): {len(new_addrs_filtered)}")
    logger.info(f"v1 avatars kept after v1 logic: {len(v1_addrs_kept)}")
    logger.info(f"v1 avatars discarded after v1 logic: {len(v1_addrs_discarded)}")

    if discarded_entries:
        logger.info("Sample discarded v1 avatars (addr, reason, is_v2, sheet_flags):")
        shown = 0
        for addr, reason, is_v2, flags in discarded_entries:
            if addr not in v1_avatars:
                # only interested in v1 here for logging
                continue
            logger.info(
                f"  {addr} | is_v2={is_v2} | flags_in_sheets={flags} | reason={reason!r}"
            )
            shown += 1
            if shown >= 10:
                break

    # This set is our final CSV-based candidate blacklist (before DB)
    csv_candidate_addrs = new_addrs_filtered

    # ---------------------------
    # PHASE 2: DB comparison
    # ---------------------------

    logger.info("")
    logger.info("=== PHASE 2: Compare CSV candidates with blacklist.db ===")
    logger.info(f"Using DB: {args.db}")

    conn = sqlite3.connect(args.db)
    try:
        db_blacklist_df = load_blacklist(conn)
        db_whitelist_set = load_whitelist(conn)

        existing_addrs = set(db_blacklist_df["address"])

        logger.info(f"Existing blacklist entries in DB: {len(db_blacklist_df)}")
        logger.info(f"Existing whitelist entries in DB: {len(db_whitelist_set)}")

        # Comparison
        overlap = existing_addrs & csv_candidate_addrs
        db_only = existing_addrs - csv_candidate_addrs
        csv_only = csv_candidate_addrs - existing_addrs

        logger.info("")
        logger.info("=== BLACKLIST COMPARISON (DB vs CSV candidates) ===")
        logger.info(f"Existing blacklist addresses (DB): {len(existing_addrs)}")
        logger.info(f"CSV candidate addresses (after v1 filtering): {len(csv_candidate_addrs)}")
        logger.info(f"Overlap (in both): {len(overlap)}")
        logger.info(f"In DB only (not in CSV candidates): {len(db_only)}")
        logger.info(f"In CSV candidates only (not in DB): {len(csv_only)}")

        # Whitelist conflicts (only at CSV-candidate level)
        new_and_whitelisted = csv_candidate_addrs & db_whitelist_set
        logger.info("")
        logger.info("=== WHITELIST CHECK (ON CSV CANDIDATES) ===")
        logger.info(
            f"CSV candidate blacklist addresses that are currently whitelisted: "
            f"{len(new_and_whitelisted)}"
        )

        if new_and_whitelisted:
            logger.info("Detailed info for whitelist-conflicting addresses:")
            for addr in list(new_and_whitelisted)[:10]:
                flags = addr_sheet_counts.get(addr, 0)
                has_circles = has_circles_v1_reason.get(addr, False)
                is_v1 = addr in v1_avatars
                is_v2 = addr in v2_avatars
                # DB reason if present
                db_reason = None
                if addr in existing_addrs:
                    r = db_blacklist_df.loc[
                        db_blacklist_df["address"] == addr, "reason"
                    ]
                    if not r.empty:
                        db_reason = str(r.iloc[0])
                csv_reason = addr_to_reason.get(addr)
                logger.info(
                    f"  {addr} | v1={is_v1} | v2={is_v2} | has_CirclesV1_reason={has_circles} | "
                    f"flags_in_sheets={flags} | db_reason={db_reason!r} | csv_reason={csv_reason!r}"
                )

        # Stats vs graph_metrics_v2
        logger.info("")
        logger.info("=== STATS vs graph_metrics_v2 (v2 avatars) ===")
        logger.info(f"Total avatars in graph_metrics_v2.csv: {len(v2_avatars)}")

        currently_blacklisted = v2_avatars & existing_addrs
        blacklisted_with_csv_only = v2_avatars & csv_candidate_addrs
        merged_addrs = existing_addrs | csv_candidate_addrs
        blacklisted_with_merged = v2_avatars & merged_addrs

        logger.info(f"Currently blacklisted avatars (DB only): {len(currently_blacklisted)}")
        logger.info(
            f"Would be blacklisted with CSV candidates only: "
            f"{len(blacklisted_with_csv_only)}"
        )
        logger.info(
            f"Would be blacklisted with MERGED blacklist (DB + CSV candidates): "
            f"{len(blacklisted_with_merged)}"
        )

        # --- Extra: new v2 & v1 addresses that would be added by CSV ---

        # v2 addresses that are CSV candidates but NOT in DB yet
        new_v2_not_in_db = (v2_avatars & csv_candidate_addrs) - existing_addrs
        logger.info("")
        logger.info("=== NEW v2 BLACKLIST CANDIDATES (from CSV, not in DB yet) ===")
        logger.info(f"v2 avatars newly blacklisted by CSV (not yet in DB): {len(new_v2_not_in_db)}")

        # v1 addresses that are CSV candidates but NOT in DB yet
        new_v1_not_in_db = (v1_avatars & csv_candidate_addrs) - existing_addrs
        logger.info("=== NEW v1 BLACKLIST CANDIDATES (from CSV, not in DB yet) ===")
        logger.info(f"v1 avatars newly blacklisted by CSV (not yet in DB): {len(new_v1_not_in_db)}")

        # Save lists to CSV for inspection
        out_dir = args.output_csv.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # v2 new blacklist CSV
        if new_v2_not_in_db:
            df_v2_new = pd.DataFrame(
                [
                    {"address": addr, "reason": addr_to_reason.get(addr, "Manual Import")}
                    for addr in sorted(new_v2_not_in_db)
                ]
            )
            v2_out_path = out_dir / "new_v2_blacklist_not_in_db.csv"
            df_v2_new.to_csv(v2_out_path, index=False)
            logger.info(f"Wrote new v2 blacklist candidates (not in DB) to: {v2_out_path}")

        # v1 new blacklist CSV
        if new_v1_not_in_db:
            df_v1_new = pd.DataFrame(
                [
                    {"address": addr, "reason": addr_to_reason.get(addr, "Manual Import")}
                    for addr in sorted(new_v1_not_in_db)
                ]
            )
            v1_out_path = out_dir / "new_v1_blacklist_not_in_db.csv"
            df_v1_new.to_csv(v1_out_path, index=False)
            logger.info(f"Wrote new v1 blacklist candidates (not in DB) to: {v1_out_path}")

        # Build full updated blacklist CSV (existing reason wins, exclude whitelist, ensure uniqueness)
        logger.info("")
        logger.info("=== BUILDING UPDATED BLACKLIST CSV (existing DB reason wins, excluding whitelist) ===")

        # Existing entries as canonical for overlapping addresses
        existing_map = {
            normalize_address(row["address"]): str(row["reason"])
            for _, row in db_blacklist_df.iterrows()
        }

        updated_entries = []
        whitelisted_count = 0

        # 1) All existing DB entries (except those in whitelist)
        for addr, reason in existing_map.items():
            if addr not in db_whitelist_set:
                updated_entries.append((addr, reason))
            else:
                whitelisted_count += 1
                logger.warning(f"Excluding from updated CSV (whitelisted): {addr}")

        # 2) CSV-only candidates (not already in DB and not whitelisted)
        for addr in csv_only:
            if addr not in db_whitelist_set:
                reason = addr_to_reason.get(addr, "Manual Import")
                updated_entries.append((addr, reason))
            else:
                whitelisted_count += 1
                logger.warning(f"Excluding from updated CSV (whitelisted): {addr}")

        # Create dataframe and ensure uniqueness
        updated_df = pd.DataFrame(updated_entries, columns=["address", "reason"])
        
        # Check for duplicates before dropping them
        duplicates_count = updated_df.duplicated(subset=["address"]).sum()
        if duplicates_count > 0:
            logger.warning(f"Found {duplicates_count} duplicate addresses in updated blacklist, keeping first occurrence")
        
        updated_df = updated_df.drop_duplicates(subset=["address"]).sort_values("address")

        logger.info(f"Updated blacklist size (DB + CSV-only candidates, excluding whitelist): {len(updated_df)}")
        logger.info(f"Excluded addresses (whitelisted): {whitelisted_count}")

        # Write to CSV
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        updated_df.to_csv(args.output_csv, index=False)
        logger.info(f"Wrote full updated blacklist to: {args.output_csv}")

    finally:
        conn.close()
        logger.info("Closed DB connection.")

    logger.info("All done.")


if __name__ == "__main__":
    main()