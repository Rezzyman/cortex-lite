"""
CORTEX Lite — Quickstart Example

Run this script to see CORTEX Lite in action:

    python examples/quickstart.py

Prerequisites:
    pip install cortex-lite
    ollama pull mxbai-embed-large
"""

from cortex_lite import CortexLite


def main():
    # Create a memory store (one SQLite file, that's it)
    cx = CortexLite("quickstart.db")

    print("Storing memories...")
    cx.store("The quarterly board meeting is scheduled for March 15th at 3pm in Denver.")
    cx.store("Jeff Zorbo is the equity partner. He handles roofing leads through GHL.")
    cx.store("CORTEX V2.4 scored 500 out of 500 on LongMemEval benchmarks.")
    cx.store("Aria is the content strategist. She drafts daily social media batches.")
    cx.store("The API rate limit for the production endpoint is 100 requests per minute.")
    cx.store("Always use the affiliate link for Wispr Flow, never the generic URL.")

    print(f"Stored 6 memories.\n")

    # Search for something specific
    print("--- Search: 'when is the board meeting' ---")
    for r in cx.search("when is the board meeting", limit=3):
        print(f"  [{r.score:.2f}] {r.content}")

    print()

    # Search for something semantic (no keyword overlap)
    print("--- Search: 'benchmark results' ---")
    for r in cx.search("benchmark results", limit=3):
        print(f"  [{r.score:.2f}] {r.content}")

    print()

    # Recall with a token budget (for injecting into LLM prompts)
    print("--- Recall: 'Jeff' (budget=500 tokens) ---")
    context = cx.recall("Jeff", token_budget=500)
    print(f"  {context}")

    print()

    # Stats
    print("--- Status ---")
    stats = cx.status()
    print(f"  {stats['active']} active memories in quickstart.db")

    cx.close()

    # Cleanup
    import os
    os.remove("quickstart.db")
    print("\nCleaned up quickstart.db. Done!")


if __name__ == "__main__":
    main()
