"""CORTEX Lite CLI."""

import argparse
import json
import sys

from cortex_lite import CortexLite


def main():
    parser = argparse.ArgumentParser(
        prog="cortex-lite",
        description="Lightweight cognitive memory for AI agents.",
    )
    parser.add_argument("--db", default="cortex.db", help="Path to SQLite database")
    parser.add_argument("--agent", default="default", help="Agent namespace")

    sub = parser.add_subparsers(dest="command")

    # store
    p_store = sub.add_parser("store", help="Store text into memory")
    p_store.add_argument("content", help="Text to store")
    p_store.add_argument("--source", default="cli")
    p_store.add_argument("--priority", type=int, default=2)

    # store-file
    p_file = sub.add_parser("store-file", help="Store a file into memory")
    p_file.add_argument("path", help="Path to text/markdown file")
    p_file.add_argument("--priority", type=int, default=2)

    # search
    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--limit", type=int, default=10)

    # recall
    p_recall = sub.add_parser("recall", help="Token-budgeted recall")
    p_recall.add_argument("query", help="Recall query")
    p_recall.add_argument("--budget", type=int, default=4000)

    # status
    sub.add_parser("status", help="Show memory stats")

    # prune
    p_prune = sub.add_parser("prune", help="Remove old low-value memories")
    p_prune.add_argument("--max-age", type=int, default=90, help="Max age in days")
    p_prune.add_argument("--min-resonance", type=float, default=2.0)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    cx = CortexLite(db_path=args.db, agent=args.agent)

    if args.command == "store":
        result = cx.store(args.content, source=args.source, priority=args.priority)
        print(f"Stored {result.chunks} chunk(s): ids={result.memory_ids}")

    elif args.command == "store-file":
        result = cx.store_file(args.path, priority=args.priority)
        print(f"Stored {result.chunks} chunk(s) from {args.path}: ids={result.memory_ids}")

    elif args.command == "search":
        results = cx.search(args.query, limit=args.limit)
        if not results:
            print("No results.")
        for r in results:
            print(f"[{r.id}] score={r.score:.3f} | {r.content[:120]}")

    elif args.command == "recall":
        context = cx.recall(args.query, token_budget=args.budget)
        if context:
            print(context)
        else:
            print("No memories found.")

    elif args.command == "status":
        stats = cx.status()
        print(json.dumps(stats, indent=2, default=str))

    elif args.command == "prune":
        count = cx.prune(max_age_days=args.max_age, min_resonance=args.min_resonance)
        print(f"Pruned {count} memories.")

    cx.close()


if __name__ == "__main__":
    main()
