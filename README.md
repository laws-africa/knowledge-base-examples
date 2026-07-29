# Laws.Africa Knowledge Base Examples

This repo contains examples of using the Laws.Africa [Knowledge Base API](https://developers.laws.africa/ai-api/knowledge-bases) with LangGraph.

| Agent | What it does |
|---|---|
| `legislation_agent` | RAG over Cape Town by-laws (place code `za-cpt`) |
| `judgment_agent` | RAG over the South African judgments knowledge base |
| `forecast_agent` | Answers "what is the law here, and what is about to change?" |

## What the RAG examples do

1. Take a user **query** as input
2. Use an LLM to come up with a search query for the Knowledge Base, based on the user's query.
3. Search the Knowledge Base using the generated search query to retrieve relevant documents.
4. Answer the user's query using the retrieved documents as context.

## What the forecast example does

Rather than searching once, it runs three filtered retrievals over the same
knowledge base — legislation in force, legislation passed but not yet
commenced, and legislation recently repealed — and synthesizes them into a
forecast for a legal area, with every source the model cites verified against
what was retrieved. It also ships a `monitor` that re-runs the retrieval on a
schedule and reports only what's new since the last run.

**[TUTORIAL.md](TUTORIAL.md) walks through building it from scratch.**

## Requirements

1. Python 3.11 or later
2. An OpenAI API key.
3. A Laws.Africa API key. You can get one by [following these instructions](https://developers.laws.africa/ai-api/authentication).

## Setup

1. Clone this repository and navigate to the project directory:

   ```bash
   git clone
   cd laws-africa-knowledge-base-examples
    ```

2. Setup a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Set your OpenAI and Laws.Africa API keys as environment variables, or add them to a new `.env` file:

   ```bash
   export OPENAI_API_KEY='your-openai-api-key'
   export LAWSAFRICA_API_TOKEN='your-laws-africa-api-key'
   ```

## Running the agent

### Running the agent with LangGraph CLI

This is a simple visual way to run the agent, using the LangGraph CLI and LangGraph's [Agent Chat UI](https://docs.langchain.com/oss/python/langgraph/ui).

```bash
langgraph dev --no-browser
```

Open your browser and go to https://agentchat.vercel.app/?apiUrl=http://localhost:2024

Type in `legislation_agent`, `judgment_agent` or `forecast_agent` to choose the respective agent and click Continue.

That will present you with a chat interface where you can interact with the agent.

Ask: `How many dogs can I own?`, `Cases for delict in a slip and trip scenario`, or — for the forecast
agent — `electricity generation licensing`

### Running the agent with a Python script

You can also run the agent directly using the provided `agent.py` script. It
requires a single argument to choose which knowledge base agent to run:

```bash
python agent.py legislation
python agent.py judgment
python agent.py forecast
```

Choose `legislation` for the Cape Town legislation RAG flow, `judgment` to query the judgments
knowledge base, or `forecast` for the legislation forecast.

Ask: `How many dogs can I own?`, `Cases for delict in a slip and trip scenario`, or — for the forecast
agent — `electricity generation licensing`

## Watching a legal area for changes

The forecast agent's retrieval can be run on a schedule, reporting only works
that are new since the last run:

```bash
python -m forecast_agent.monitor --area "electricity generation licensing"
```

State is kept in `monitor_state.json`. Pass `--notify you@example.com` (with
`SMTP_HOST` set) to email the digest instead of just printing it.

## Tests

The forecast agent's tests fake both the Knowledge Base and the model, so they
need no API keys and make no network calls:

```bash
pip install -r requirements-dev.txt
pytest
```
