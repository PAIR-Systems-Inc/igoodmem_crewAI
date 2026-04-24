# GoodMem Tools for CrewAI

GoodMem is a memory layer for AI agents with support for semantic storage,
retrieval, and summarization. This package exposes GoodMem operations as
CrewAI tools that can be used with any CrewAI agent.

## Prerequisites

### 1. Install GoodMem

You need a running GoodMem instance. Install it on your VM or local machine:

**Visit:** [https://goodmem.ai/](https://goodmem.ai/)

Follow the installation instructions for your platform (Docker, local installation,
or cloud deployment).

### 2. Create an Embedder

Before you can create spaces and memories, register an embedder model in your
GoodMem instance. `GoodMemListEmbeddersTool` lets you discover the IDs of
embedders that are already registered on the server.

### 3. Get Your API Key

Obtain an API key from your GoodMem instance (keys start with `gm_`).

## Configuration

Each tool accepts `base_url`, `api_key`, and `verify_ssl` as constructor arguments,
or reads from environment variables:

- `GOODMEM_BASE_URL` — The base URL of your GoodMem instance (e.g.
  `http://localhost:8080`, `https://api.goodmem.ai`)
- `GOODMEM_API_KEY` — Your GoodMem API key (starts with `gm_`)

```python
from crewai_tools import GoodMemCreateSpaceTool

tool = GoodMemCreateSpaceTool(
    base_url="http://localhost:8080",
    api_key="gm_your_key_here",
)
```

Or set environment variables and instantiate without arguments:

```bash
export GOODMEM_BASE_URL="http://localhost:8080"
export GOODMEM_API_KEY="gm_your_key_here"
```

```python
tool = GoodMemCreateSpaceTool()
```

## Available Tools

There are 11 tools, covering the complete GoodMem v1 REST surface: embedders,
spaces (CRUD), memories (CRUD), and semantic retrieval.

| Tool | Operation |
|---|---|
| `GoodMemListEmbeddersTool` | `GET /v1/embedders` |
| `GoodMemListSpacesTool` | `GET /v1/spaces` |
| `GoodMemGetSpaceTool` | `GET /v1/spaces/{id}` |
| `GoodMemCreateSpaceTool` | `POST /v1/spaces` (reuses if name exists) |
| `GoodMemUpdateSpaceTool` | `PUT /v1/spaces/{id}` |
| `GoodMemDeleteSpaceTool` | `DELETE /v1/spaces/{id}` |
| `GoodMemCreateMemoryTool` | `POST /v1/memories` |
| `GoodMemListMemoriesTool` | `GET /v1/spaces/{space_id}/memories` |
| `GoodMemRetrieveMemoriesTool` | `POST /v1/memories:retrieve` (NDJSON stream) |
| `GoodMemGetMemoryTool` | `GET /v1/memories/{id}` + `/content` |
| `GoodMemDeleteMemoryTool` | `DELETE /v1/memories/{id}` |

### GoodMemCreateSpaceTool

Create a new space (container for memories) with configurable settings. If a
space with the same name already exists it is reused and the caller receives
the embedder actually attached to that space.

| Argument | Required | Default | Description |
|---|---|---|---|
| `name` | Yes | — | Unique name for the space |
| `embedder_id` | Yes | — | ID of the embedder model for vector embeddings |
| `chunk_size` | No | 256 | Characters per chunk when splitting documents |
| `chunk_overlap` | No | 25 | Overlapping characters between consecutive chunks |
| `keep_strategy` | No | `KEEP_END` | Separator placement: `KEEP_END`, `KEEP_START`, `DISCARD` |
| `length_measurement` | No | `CHARACTER_COUNT` | `CHARACTER_COUNT` or `TOKEN_COUNT` |

### GoodMemUpdateSpaceTool

Update a space's name, labels, or `publicRead` flag. Only fields you pass are
changed; omitted fields keep their current values.

| Argument | Required | Default | Description |
|---|---|---|---|
| `space_id` | Yes | — | UUID of the space to update |
| `name` | No | — | New name for the space |
| `public_read` | No | — | Toggle unauthenticated read access |
| `replace_labels_json` | No | — | JSON string that replaces all existing labels |
| `merge_labels_json` | No | — | JSON string that merges into existing labels |

`replace_labels_json` and `merge_labels_json` are mutually exclusive.

### GoodMemDeleteSpaceTool

Permanently delete a space and all its memories, chunks, and embeddings.

| Argument | Required | Description |
|---|---|---|
| `space_id` | Yes | UUID of the space to delete |

### GoodMemGetSpaceTool

Fetch a single space by ID. Returns the full space configuration (embedders,
chunking settings, labels).

| Argument | Required | Description |
|---|---|---|
| `space_id` | Yes | UUID of the space to fetch |

### GoodMemCreateMemoryTool

Store a document or plain text as a memory in a space. Content is automatically
chunked and embedded for semantic search. Binary content is base64-encoded on
upload.

| Argument | Required | Default | Description |
|---|---|---|---|
| `space_id` | Yes | — | Space to store the memory in |
| `text_content` | No | — | Plain text content (used when no file is provided) |
| `file_path` | No | — | Local file path (PDF, DOCX, image, etc.). Takes priority over `text_content` |
| `metadata` | No | — | Key-value metadata as a JSON object |

### GoodMemListMemoriesTool

List memories in a space with optional filtering and sorting.

| Argument | Required | Default | Description |
|---|---|---|---|
| `space_id` | Yes | — | UUID of the space to list memories from |
| `status_filter` | No | — | One of `PENDING`, `PROCESSING`, `COMPLETED`, `FAILED` |
| `include_content` | No | `false` | Include each memory's original content alongside metadata |
| `sort_by` | No | — | `created_at` or `updated_at` |
| `sort_order` | No | — | `ASCENDING` or `DESCENDING` |

### GoodMemRetrieveMemoriesTool

Perform semantic search across one or more spaces. Supports reranking, LLM
summarization, chronological resort, and server-side metadata filtering via
SQL-style JSONPath expressions.

| Argument | Required | Default | Description |
|---|---|---|---|
| `query` | Yes | — | Natural language search query |
| `space_ids` | Yes | — | List of space IDs to search across |
| `max_results` | No | 5 | Maximum number of results |
| `include_memory_definition` | No | `true` | Fetch full memory metadata alongside chunks |
| `wait_for_indexing` | No | `true` | Retry polling when no results found |
| `max_wait_seconds` | No | 10 | Maximum seconds to poll when `wait_for_indexing` is `true` |
| `poll_interval` | No | 2 | Seconds between polling attempts |
| `reranker_id` | No | — | Reranker model ID for improved ordering |
| `llm_id` | No | — | LLM ID for contextual response generation |
| `relevance_threshold` | No | — | Minimum score (0–1). Used with reranker/LLM |
| `llm_temperature` | No | — | LLM creativity (0–2). Used when `llm_id` is set |
| `chronological_resort` | No | `false` | Reorder by creation time instead of relevance |
| `metadata_filter` | No | — | SQL-style JSONPath expression applied server-side (e.g. `CAST(val('$.category') AS TEXT) = 'feat'`). Applied to every space in `space_ids` |

### GoodMemGetMemoryTool

Retrieve a specific memory by ID, with metadata and optionally the original
content. Text content is returned verbatim (`contentEncoding: "text"`); binary
content is base64-encoded (`contentEncoding: "base64"`).

| Argument | Required | Default | Description |
|---|---|---|---|
| `memory_id` | Yes | — | UUID of the memory to fetch |
| `include_content` | No | `true` | Also fetch the original document content |

### GoodMemDeleteMemoryTool

Permanently delete a memory and its chunks and embeddings.

| Argument | Required | Description |
|---|---|---|
| `memory_id` | Yes | UUID of the memory to delete |

### GoodMemListSpacesTool / GoodMemListEmbeddersTool

List all spaces or embedders. Take no arguments.

## Usage Example

```python
from crewai import Agent, Task, Crew
from crewai_tools import (
    GoodMemCreateSpaceTool,
    GoodMemCreateMemoryTool,
    GoodMemRetrieveMemoriesTool,
    GoodMemListEmbeddersTool,
)

embedders = GoodMemListEmbeddersTool()
create_space = GoodMemCreateSpaceTool()
create_memory = GoodMemCreateMemoryTool()
retrieve = GoodMemRetrieveMemoriesTool()

researcher = Agent(
    role="Knowledge Manager",
    goal="Store and retrieve information using GoodMem",
    tools=[embedders, create_space, create_memory, retrieve],
    verbose=True,
)

task = Task(
    description=(
        "List embedders, create a space called 'research-notes', store the "
        "note 'CrewAI v1.14 ships agent skills.', then search for "
        "'agent skills' in that space."
    ),
    agent=researcher,
    expected_output="Search results from the stored document.",
)

crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
```

## Metadata Filtering

Tag memories at creation time and filter server-side at retrieval time:

```python
create_memory._run(
    space_id=space_id,
    text_content="Shipped the CSV export feature.",
    metadata={"category": "feat"},
)

retrieve._run(
    query="new features",
    space_ids=[space_id],
    metadata_filter="CAST(val('$.category') AS TEXT) = 'feat'",
)
```

The filter is a SQL-style JSONPath expression evaluated by GoodMem. When set,
the same filter is applied to every space in `space_ids`.

## SSL Verification

All tools accept a `verify_ssl` parameter (default `True`). Set to `False` when
connecting to a local development server with self-signed certificates:

```python
tool = GoodMemCreateSpaceTool(
    base_url="https://localhost:8080",
    api_key="gm_your_key_here",
    verify_ssl=False,
)
```

## Runnable Example

[`example.py`](example.py) exercises every tool in this package against a live
GoodMem server across three scenarios: persistent context with operator-level
inspection, a Scribe/Analyst pipeline, and metadata-filtered retrieval.

Run from the repo root (the script auto-loads `.env` via `python-dotenv`):

```bash
uv run python lib/crewai-tools/src/crewai_tools/tools/goodmem_tool/example.py
```
