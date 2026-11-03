# Claude RAG API

A FastAPI customer-support service that answers questions from client-specific knowledge bases. It combines semantic and Arabic-aware keyword retrieval with Claude, supports short conversation histories, and can return a human handoff action.

## What it does

- Authenticates chat and knowledge-upload requests with a client Bearer token stored in the database.
- Accepts knowledge as JSON, converts entries to text, creates multilingual E5 embeddings locally, and stores the content and vectors in PostgreSQL.
- Retrieves knowledge using both vector similarity and PostgreSQL full-text search, scoped to the authenticated client.
- Uses Claude to interpret follow-up questions, classify intent, and generate an answer grounded in retrieved content and the client's configured business rules.
- Keeps recent conversation turns in process memory when a `conversation_id` is provided. Sessions expire after 30 minutes.
- Records questions that could not be answered in the `ai_gaps` table when the response status is `RELATED_BUT_UNKNOWN`.

## Technology

- Python and FastAPI
- PostgreSQL with the `pgvector` extension and PostgreSQL full-text search
- `intfloat/multilingual-e5-large` via Sentence Transformers for embeddings (runs on CPU)
- Anthropic Claude for intent classification, query rewriting, and response generation

The embedding module is named `app/embeddings/voyage.py`, but its current implementation uses the local Sentence Transformers model. Voyage reranking is optional and disabled by default.

## Requirements and configuration

Use Python 3.10 or newer, a PostgreSQL database with `pgvector`, and an Anthropic API key. The database must already contain the tables and columns used by the application: `clients` (`id`, `name`, `token`, `ai_config`), `knowledge` (`id`, `client_id`, `content`, `embedding`), and `ai_gaps` (`client_id`, `message`, `type`). This repository does not include database migrations or schema creation scripts.

Create a `.env` file in the project root:

```dotenv
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE
ANTHROPIC_API_KEY=your-anthropic-api-key
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_INTENT_MODEL=claude-haiku-4-5-20251001
JWT_SECRET=replace-with-a-long-random-secret
```

`ANTHROPIC_MODEL` and `ANTHROPIC_INTENT_MODEL` are optional; the shown model is the default for both. `JWT_SECRET` is required if importing `app.auth.security`; the current chat and upload routes authenticate with client tokens directly. Keep `.env` out of version control.

Install the dependencies and run the API:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Dependency note:** the current `requirements.txt` does not list all direct imports used by the application. At minimum, a clean install also needs SQLAlchemy, a PostgreSQL driver, Sentence Transformers, PyJWT, and `pwdlib`. Add and pin the dependencies required for your environment before relying on the install command above. The embedding model is downloaded by Sentence Transformers on first use, so the host needs access to the model files or a pre-cached copy.

Once running, FastAPI's interactive API documentation is available at `/docs`.

## HTTP API

### `POST /chat`

Requires `Authorization: Bearer <client-token>`.

Example request:

```json
{
  "message": "كم سعر الخدمة؟",
  "conversation_id": "session-123"
}
```

The request also accepts optional `conversation_context` and `reset_conversation` fields. Additional fields are accepted to accommodate varied chat integrations; several common history and session field names are inspected automatically.

Example response:

```json
{
  "action": "reply",
  "answer": "...",
  "best_distance": 0.31,
  "status": "KNOWN"
}
```

`action` is currently the classifier's `reply`, `api`, or `handoff` value. `status` describes the answer path, such as `KNOWN`, `CASUAL`, `CONFIG_ONLY`, `RELATED_BUT_UNKNOWN`, or `SEARCH_ERROR`. `best_distance` is the closest vector distance when search returned a result; it is `null` when no distance is available.

### `POST /knowledge/upload`

Requires `Authorization: Bearer <client-token>`. The JSON body can be a list of entries, or an object containing a list under `entries`, `items`, `data`, `documents`, or `knowledge`. An entry can be plain text, contain a `content`, `text`, `description`, or `information` string, or contain both `question` and `answer` strings.

Example:

```json
{
  "entries": [
    {
      "question": "What are your opening hours?",
      "answer": "We are open Sunday through Thursday, 9 AM to 5 PM."
    }
  ]
}
```

The response includes the number of text entries ingested. Uploading adds rows; it does not replace or deduplicate existing knowledge.

## Project layout

```text
app/
  auth/         Password and JWT helpers
  clients/      Client token lookup
  database/     SQLAlchemy database connection
  documents/    JSON knowledge parsing
  embeddings/   Local multilingual embedding model
  knowledge/    Ingestion, retrieval, and unanswered-question storage
  llm/          Claude integration
  routes/       Chat and knowledge-upload endpoints
  services/     RAG pipeline, intent, and conversation memory
data/           Example knowledge data
reembed_knowledge.py  Local-database embedding migration utility
```

## Known limitations and weaknesses

1. **The dependency list is incomplete.** A fresh environment can fail to import modules such as SQLAlchemy or Sentence Transformers because their packages are absent from `requirements.txt`. This makes setup unreliable and prevents reproducible deployments. Pin all direct dependencies and validate installation in a clean environment.
2. **Conversation memory is per process.** It disappears on restart and is not shared between multiple Uvicorn workers or server instances. A follow-up can therefore lose its history depending on which process handles it. Use a shared store such as Redis or PostgreSQL if persistent or multi-instance conversations are needed.
3. **The `api` action does not execute an operation.** The intent prompt describes live operations, but `/chat` sends every non-handoff request to the RAG answer path. The response may say `"action": "api"`, but no booking, order lookup, or other API tool is called. Until such handlers exist, remove or constrain this intent, or add explicit operation dispatch and error handling.
4. **No schema migrations or bootstrap instructions are included.** The service assumes particular database tables and a compatible vector column already exist. A new developer or deployment must reconstruct that schema before the service can work. Add versioned migrations and a documented database bootstrap process.
5. **Operational errors and content can end up in logs.** The RAG pipeline prints retrieved knowledge text and exception messages. If that content contains customer or business-sensitive information, logs can expose it and become noisy. Use structured logging, avoid printing retrieved content, and define appropriate log retention and access controls.

## Maintenance utility

`reembed_knowledge.py` previews the existing knowledge records and then, when invoked with `--apply`, updates their vectors in batches. It deliberately refuses databases whose host is not localhost or whose database name is not `ragdb`. Back up the database before applying a migration, and confirm those safeguards match your local setup.

## Tests and scripts

The repository contains standalone Python scripts named `test_*.py`, but they are not organized as a standard automated test suite and some are manual API/model checks. Review their inputs and environment requirements before running them; they may call external services or require a configured database.
