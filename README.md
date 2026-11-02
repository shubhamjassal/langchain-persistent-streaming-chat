# LangChain Persistent Streaming AI Chat

A learning project that combines LangChain, Google Gemini, ChromaDB, FastAPI, SQLite, and a browser-based chat interface.

The application supports general questions through Gemini and document-related questions through a persistent ChromaDB vector store. It also includes streaming chat responses and persistent conversation history.

## Features

- **LangChain + Gemini:** Uses LangChain integrations to call Google Gemini.
- **Question routing:** Routes general questions to the Gemini chain and document questions to the ChromaDB retrieval chain.
- **Persistent vector store:** Uses ChromaDB with a local persistence directory (`chroma_routing_db/`).
- **Streaming responses:** Exposes a streaming chat endpoint using Server-Sent Events (SSE).
- **Conversation persistence:** Stores conversations and messages in SQLite.
- **Conversation management:** Supports creating, listing, searching, renaming, deleting, and loading conversations.
- **Conversation history control:** Trims older history using a configurable approximate token budget.
- **Tests:** Includes pytest tests for API, database, memory, LCEL, and routing behavior.

## Technology stack

- Python
- LangChain Core
- LangChain Google GenAI
- LangChain Chroma
- Google Gemini
- ChromaDB
- FastAPI
- SQLite
- HTML, CSS, and JavaScript
- pytest

## Project structure

```text
.
├── api.py
├── main.py
├── chat.html
├── database.py
├── memory.py
├── routing_logic.py
├── routing_models.py
├── routing_with_chroma.py
├── routing_with_gemini.py
├── routing_with_retriever.py
├── rag_chain.py
├── semantic_retriever.py
├── test_api.py
├── test_database.py
├── test_lcel.py
├── test_memory.py
├── test_routing_with_chroma.py
├── requirements.txt
├── .env.example
└── .gitignore
```

The repository also contains smaller standalone examples covering prompts, LCEL, runnables, streaming, structured output, retrieval, routing, and retry behavior.

## Prerequisites

- Python 3.14 (the uploaded project environment was using Python 3.14.6).
- A Google Gemini API key.
- Internet access when calling Gemini or installing dependencies.

## Setup on Windows (PowerShell)

### 1. Clone the repository

```powershell
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

Replace the repository URL and folder name with the actual GitHub repository details.

### 2. Create and activate a virtual environment

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder with your own Google Gemini API key:

```dotenv
GOOGLE_API_KEY=your_actual_key_here
```

Keep `.env` private. It is excluded by `.gitignore`. Do not paste API keys into source files, README files, screenshots, or GitHub commits.

### 5. Start the FastAPI backend

From the project root:

```powershell
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at:

- API base: `http://127.0.0.1:8000`
- Interactive API documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### 6. Open the chat interface

With the backend running, open `chat.html` in a browser.

The current frontend is configured to call `http://127.0.0.1:8000`. Keep the backend running while using the chat interface.

## Run tests

From the project root, with the virtual environment activated:

```powershell
pytest -q
```

Some tests mock external model calls; a passing test suite does not by itself verify that a real Gemini API key, model access, or external network connection is working.

## Configuration

| Variable | Required | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | Yes for Gemini calls | Authenticates requests to Google Gemini |
| `GEMINI_MODEL` | No | Overrides the default model used by the main routing application |
| `HISTORY_TOKEN_BUDGET` | No | Sets the approximate conversation-history token budget (default: `2000`) |
| `CHAT_HISTORY_DB` | No | Overrides the SQLite database path |

## Local data and generated files

The following are local runtime data and should generally not be committed:

- `.venv/`
- `.env`
- `chat_history.db` and other SQLite database files
- `chroma_routing_db/`
- `__pycache__/`
- `.pytest_cache/`

The ChromaDB directory stores the local vector collection. If you exclude it from GitHub, users need to recreate/populate the collection using the project's document-ingestion process before document-question retrieval can work. Do not assume a fresh clone contains your local indexed documents.

## Notes

- This is a learning and portfolio project, not a hosted production service.
- The frontend currently uses a local API address. A deployed frontend would need its API base URL configured for the deployed backend.
- Model availability and Google API quotas can change. If the configured model is unavailable to your account, set `GEMINI_MODEL` to a model currently supported by your Google AI project.
- Do not commit API keys, personal conversation databases, private documents, or other sensitive data.

## License

No license has been specified yet. Add a license only after choosing the terms under which you want others to use the project.
