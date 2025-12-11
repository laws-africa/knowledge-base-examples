# Laws.Africa Knowledge Base Examples

This repo contains an example of using the Laws.Africa [Knowledge Base API](https://developers.laws.africa/ai-api/knowledge-bases) in a simple RAG setup with LangGraph.

The example restricts its queries to Cape Town using the place code `za-cpt`.

## What it does

1. Takes a user **query** as input
2. Uses an LLM to come up with a search query for the Knowledge Base, based on the user's query.
3. Searches the Knowledge Base using the generated search query to retrieve relevant documents.
4. Answers the user's query using the retrieved documents as context.

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
   export LAWS_AFRICA_API_KEY='your-laws-africa-api-key'
   ```

## Running the agent

### Running the agent with LangGraph CLI

This is a simple visual way to run the agent, using the LangGraph CLI and LangGraph's [Agent Chat UI](https://docs.langchain.com/oss/python/langgraph/ui).

```bash
langgraph dev --no-browser
```

Open your browser and go to https://agentchat.vercel.app/?apiUrl=http://localhost:2024

Type in `legislation_agent` or `judgment_agent` to choose the respective agent and click Continue.

That will present you with a chat interface where you can interact with the agent.

Ask: `How many dogs can I own?` or `Cases for delict in a slip and trip scenario`

### Running the agent with a Python script

This is a more programmatic way to run the agent, using a simple Python script. The script now
requires a single argument to choose which knowledge base agent to run:

```bash
python agent.py legislation
```

Choose `legislation` for the Cape Town legislation RAG flow, or `judgment` to query the judgments
knowledge base instead:

```bash
python agent.py judgment
```

Ask: `How many dogs can I own?` or `Cases for delict in a slip and trip scenario`
