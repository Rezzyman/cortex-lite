"""
CORTEX Lite — Agent Memory Example

Shows how to use CORTEX Lite as the memory layer for an AI agent.
The agent stores what it learns and recalls context before responding.

    python examples/agent_memory.py
"""

from cortex_lite import CortexLite


def agent_respond(cx: CortexLite, user_message: str) -> str:
    """
    Simulate an agent that recalls context before responding.

    In production, you'd pass the recalled context to your LLM
    as part of the system prompt or message history.
    """
    # Step 1: Recall relevant memories (fit in 2000 tokens)
    context = cx.recall(user_message, token_budget=2000)

    # Step 2: Store this interaction as a new memory
    cx.store(
        f"User said: {user_message}",
        source="conversation",
        priority=2,
    )

    # Step 3: In production, you'd send this to your LLM:
    #
    #   prompt = f"""You have the following context from memory:
    #   {context}
    #
    #   User: {user_message}
    #   """
    #   response = llm.generate(prompt)

    return f"[Recalled {len(context)} chars of context]\n{context if context else '(no prior context)'}"


def main():
    cx = CortexLite("agent-demo.db", agent="demo-agent")

    # Seed the agent with some knowledge
    cx.store("The client's name is Sarah Chen. She runs a roofing company in Denver.")
    cx.store("Sarah's company does 200 jobs per month. Average ticket is $8,500.")
    cx.store("Sarah prefers text messages over email for urgent updates.")
    cx.store("The CRM integration uses GHL (GoHighLevel) webhooks.")
    cx.store("Sarah's contract renews on June 1st. She's been a client since January.")

    print("Agent seeded with 5 memories.\n")

    # Simulate a conversation
    queries = [
        "What do we know about Sarah?",
        "How does she prefer to be contacted?",
        "When does her contract come up?",
    ]

    for q in queries:
        print(f"User: {q}")
        response = agent_respond(cx, q)
        print(f"Agent: {response}\n")

    cx.close()

    import os
    os.remove("agent-demo.db")
    print("Cleaned up. Done!")


if __name__ == "__main__":
    main()
