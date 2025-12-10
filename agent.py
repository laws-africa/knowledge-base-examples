import asyncio
import dotenv

from langchain_core.messages import HumanMessage

from kb_agent.graph import graph


async def main():
    try:
        question = input("Enter a question (or 'quit' to exit): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")

    if not question:
        return

    if question.lower() in {"quit", "exit", "q"}:
        print("Goodbye!")
        return

    state = {"messages": [HumanMessage(content=question)]}
    async for update in graph.astream(
        state,
        stream_mode="updates",
    ):
        print(update)

    print(state['messages'][-1].pretty_print(), end="", flush=True)


if __name__ == "__main__":
    dotenv.load_dotenv("env")
    asyncio.run(main())
