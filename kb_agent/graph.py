import os

import httpx
from langgraph.graph import StateGraph
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

# Change this to use your desired model
MODEL = "openai/gpt-5-mini"
KB_NAME = "legislation-za-municipal"
LAWSAFRICA_KB_URL = f"https://api.laws.africa/ai/v1/knowledge-bases/{KB_NAME}/retrieve"


class State(MessagesState):
    user_question: str
    search_query: str
    document_portions: list[str]


class SearchQuery(BaseModel):
    search_query: str


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a name in the format 'provider/model'."""
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider)


async def search_query(state: State):
    """Determine the search query to use based on the user's question."""
    llm = load_chat_model(MODEL).with_structured_output(SearchQuery)

    # get the user's query
    if not state.get('user_question'):
        state['user_question'] = state['messages'][-1].content

    # come up with search queries if we don't have them yet
    if not state.get('search_query'):
        print("Generating search query...")
        output = await llm.ainvoke([
            {"role": "system", "content": """
Generate a concise search query for a symantic search on a document corpus to find legislation to help answer the
following legal research question. Your search query should focus on terms and phrases that are likely
to appear in relevant legislation. Avoid including unnecessary words or phrases, and don't use any boolean operators
or special search syntax.
"""},
            {"role": "user", "content": f"Legal research question: {state['user_question']}\nSearch query:"}
        ])
        state["search_query"] = output.search_query
        print(f"Generated search query: {state['search_query']}")

    return {
        "user_question": state['user_question'],
        "search_query": state['search_query'],
    }


async def rag(state: State):
    """Load relevant document portions based on the search query."""
    if not state.get('document_portions'):
        documents = await get_legislation_portions(state['search_query'])
        return {"document_portions": documents}

    return {}


async def answer(state: State):
    """Answer the user's legal research question based on the retrieved document portions."""
    prompt = """You are a legal research assistant who helps users find relevant legal information based on their
queries. Only reply with information provided here, not from your background knowledge.

You are answering questions specifically about Cape Town, South Africa.
"""

    document_context = """Use the following legal document portions as context to answer the user's question.
When answering, refer to the document's title and section numbers where relevant.""" + "\n\n" + "\n\n".join(state['document_portions'])

    llm = load_chat_model(MODEL)
    response = await llm.ainvoke([
        {"role": "system", "content": prompt},
        {"role": "user", "content": document_context},
        {"role": "user", "content": f"Answer the user's legal research question: {state['user_question']}"},
    ])

    return {"messages": [response]}


async def get_legislation_portions(query: str) -> list[str]:
    """Helper function to get legislation portions from Laws.Africa Knowledge Base API, format them into per-document
    chunks, and return a list of document strings."""
    la_api_token = os.environ.get("LAWSAFRICA_API_TOKEN")
    headers = {
        "Authorization": f"Token {la_api_token}"
    }

    async with httpx.AsyncClient() as client:
        print("Querying Laws.Africa Knowledge Base...")
        resp = await client.post(
            LAWSAFRICA_KB_URL,
            headers=headers,
            json={
                "text": query,
                "top_k": "5",
                "filters": {
                    "principal": True,
                    "repealed": False,
                    "frbr_place": "za-cpt",
                }
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        print(f"Received {len(data['results'])} results from Laws.Africa Knowledge Base.")

    # gather document chunks together for each document, grouping on work_frbr_uri, but preserving their ordering
    results = {}
    for result in data["results"]:
        work_frbr_uri = result["metadata"]["work_frbr_uri"]
        if work_frbr_uri not in results:
            results[work_frbr_uri] = []
        results[work_frbr_uri].append(result)

    documents = []
    for work_frbr_uri, items in results.items():
        # only keep provision portions
        items = [x for x in items if x["metadata"]["portion_type"] == "provision"]
        if not items:
            continue

        first_item = items[0]
        lines = [
            f"# Document {len(documents) + 1}"
            "",
            "title: " + first_item["metadata"]["title"],
            "work_frbr_uri: " + first_item["metadata"]["work_frbr_uri"],
            "date: " + first_item["metadata"]["expression_date"],
            "public_url: " + first_item["metadata"]["public_url"],
            "",
            ]

        for chunk in items:
            lines.append("")
            lines.append(f'<portion id="{chunk["metadata"]["portion_id"]}" title="{chunk["metadata"].get("portion_title", "")}">')
            lines.append(chunk["content"]["text"])
            lines.append("</portion>")

        documents.append("\n".join(lines))

    return documents


# Define a new graph
builder = StateGraph(State)

builder.add_node("search_query", search_query)
builder.add_node("rag", rag)
builder.add_node("answer", answer)
builder.add_edge("__start__", "search_query")
builder.add_edge("search_query", "rag")
builder.add_edge("rag", "answer")

# Compile the builder into an executable graph
graph = builder.compile(name="KB Agent")
