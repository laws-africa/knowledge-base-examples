import argparse
import asyncio
import dotenv

from langchain_core.messages import HumanMessage

from forecast_agent.graph import forecast_graph
from kb_agent.graph import judgment_graph, legislation_graph

GRAPHS = {
    "judgment": judgment_graph,
    "legislation": legislation_graph,
    "forecast": forecast_graph,
}


async def main(graph):
    try:
        question = input("Enter a question (or 'quit' to exit): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        return

    if not question:
        return

    if question.lower() in {"quit", "exit", "q"}:
        print("Goodbye!")
        return

    state = {"messages": [HumanMessage(content=question)]}
    state = await graph.ainvoke(state)
    print(state['messages'][-1].content, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a knowledge base agent.")
    parser.add_argument(
        "agent",
        choices=sorted(GRAPHS),
        help="Which agent to run",
    )
    args = parser.parse_args()

    dotenv.load_dotenv(".env")
    asyncio.run(main(GRAPHS[args.agent]))
