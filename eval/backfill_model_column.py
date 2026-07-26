"""One-shot backfill: label pre-attribution decision rows with the model that
actually produced them.

Deliberately not part of store._migrate. A migration must not guess at data —
it can add a NULL column safely, but deciding that every NULL came from
gemma-3-1b-it is a claim about history that only the operator can make. Running
this is therefore an explicit, auditable choice.

    uv run python -m eval.backfill_model_column --model gemma-3-1b-it
    uv run python -m eval.backfill_model_column --model gemma-3-1b-it --dry-run
"""

from __future__ import annotations

import argparse

from edge_analyst import store

_TABLES = ("decisions", "debate_turns")


def backfill(db_path: str, model: str, dry_run: bool = False) -> dict[str, int]:
    """Sets `model` on rows where it is NULL. Never overwrites an existing
    label, so re-running is a no-op and a mixed-model DB stays intact."""
    conn = store.get_connection(db_path)
    counts = {}
    try:
        for table in _TABLES:
            counts[table] = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE model IS NULL"
            ).fetchone()[0]
            if not dry_run:
                conn.execute(
                    f"UPDATE {table} SET model = ? WHERE model IS NULL", (model,)
                )
        if not dry_run:
            conn.commit()
    finally:
        conn.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/edge_analyst.db")
    parser.add_argument(
        "--model",
        required=True,
        help="label to write, e.g. gemma-3-1b-it — the model that produced "
        "the existing unattributed rows",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report the row counts, change nothing"
    )
    args = parser.parse_args()

    counts = backfill(args.db, args.model, args.dry_run)
    verb = "would set" if args.dry_run else "set"
    for table, count in counts.items():
        print(f"{verb} model={args.model} on {count} {table} row(s)")


if __name__ == "__main__":
    main()
