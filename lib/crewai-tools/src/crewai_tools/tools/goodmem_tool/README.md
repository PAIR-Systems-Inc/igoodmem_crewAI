# GoodMem Tools for CrewAI

GoodMem is memory layer for AI agents with support for semantic storage, retrieval, and summarization. This package exposes GoodMem operations as CrewAI tools that can be used with any CrewAI agent.

## Prerequisites

### 1. Install GoodMem

You need a running GoodMem instance. Install it on your VM or local machine:

**Visit:** [https://goodmem.ai/](https://goodmem.ai/)

Follow the installation instructions for your platform (Docker, local installation, or cloud deployment).

### 2. Create an Embedder

Before you can create spaces and memories, you need to set up an embedder model in your GoodMem instance.

### 3. Get Your API Key

Obtain an API key from your GoodMem instance (keys start with `gm_`).

## Configuration

Each tool accepts `base_url` and `api_key` as constructor arguments, or reads from
environment variables:

- **`GOODMEM_BASE_URL`** -- The base URL of your GoodMem instance (e.g., `http://localhost:8080`, `https://api.goodmem.ai`)
- **`GOODMEM_API_KEY`** -- Your GoodMem API key (starts with `gm_`)

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

### GoodMemCreateSpaceTool

Create a new space (container for memories) with configurable settings. If a
space with the same name already exists, it will be reused instead of creating
a duplicate.

**Arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `name` | Yes | -- | Unique name for the space |
| `embedder_id` | Yes | -- | ID of the embedder model for vector embeddings |
| `chunk_size` | No | 256 | Characters per chunk when splitting documents |
| `chunk_overlap` | No | 25 | Overlapping characters between consecutive chunks |
| `keep_strategy` | No | `KEEP_END` | Separator placement: `KEEP_END`, `KEEP_START`, or `DISCARD` |
| `length_measurement` | No | `CHARACTER_COUNT` | How chunk size is measured: `CHARACTER_COUNT` or `TOKEN_COUNT` |

### GoodMemCreateMemoryTool

Store a document or plain text as a memory in a space. Content is automatically
chunked and embedded for semantic search.

**Arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `space_id` | Yes | -- | Space ID to store the memory in |
| `text_content` | No | -- | Plain text content (used when no file is provided) |
| `file_path` | No | -- | Local file path (PDF, DOCX, images, etc.). Takes priority over `text_content` |
| `metadata` | No | -- | Key-value metadata as a JSON object |

### GoodMemRetrieveMemoriesTool

Perform semantic search across one or more spaces to find relevant memory
chunks. Supports advanced post-processing with reranking and LLM-generated
contextual responses.

**Arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `query` | Yes | -- | Natural language search query |
| `space_ids` | Yes | -- | List of space IDs to search across |
| `max_results` | No | 5 | Maximum number of results |
| `include_memory_definition` | No | `true` | Fetch full memory metadata alongside chunks |
| `wait_for_indexing` | No | `true` | Retry up to 60s when no results found |
| `reranker_id` | No | -- | Reranker model ID for improved ordering |
| `llm_id` | No | -- | LLM ID for contextual response generation |
| `relevance_threshold` | No | -- | Minimum score (0-1). Used with reranker/LLM |
| `llm_temperature` | No | -- | LLM creativity (0-2). Used when `llm_id` is set |
| `chronological_resort` | No | `false` | Reorder by creation time instead of relevance |

### GoodMemGetMemoryTool

Retrieve a specific memory by its ID, including metadata, processing status,
and optionally the original content.

**Arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `memory_id` | Yes | -- | UUID of the memory to fetch |
| `include_content` | No | `true` | Also fetch the original document content |

### GoodMemDeleteMemoryTool

Permanently delete a memory and all its associated chunks and vector
embeddings.

**Arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `memory_id` | Yes | -- | UUID of the memory to delete |

### GoodMemListSpacesTool

List all available spaces. Useful for discovering space IDs before performing
other operations.

**Arguments:** None.

## Usage Example

```python
from crewai import Agent, Task, Crew
from crewai_tools import (
    GoodMemCreateSpaceTool,
    GoodMemCreateMemoryTool,
    GoodMemRetrieveMemoriesTool,
    GoodMemGetMemoryTool,
    GoodMemDeleteMemoryTool,
    GoodMemListSpacesTool,
)

# Instantiate tools
create_space = GoodMemCreateSpaceTool(
    base_url="http://localhost:8080",
    api_key="gm_your_key_here",
)
create_memory = GoodMemCreateMemoryTool(
    base_url="http://localhost:8080",
    api_key="gm_your_key_here",
)
retrieve = GoodMemRetrieveMemoriesTool(
    base_url="http://localhost:8080",
    api_key="gm_your_key_here",
)

# Give tools to an agent
researcher = Agent(
    role="Knowledge Manager",
    goal="Store and retrieve information using GoodMem",
    tools=[create_space, create_memory, retrieve],
    verbose=True,
)

task = Task(
    description="Create a space, store a document, then search for relevant info.",
    agent=researcher,
    expected_output="Search results from the stored document.",
)

crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
```

## SSL Verification

All tools accept a `verify_ssl` parameter (default `True`). Set to `False`
when connecting to a local development server with self-signed certificates:

```python
tool = GoodMemCreateSpaceTool(
    base_url="https://localhost:8080",
    api_key="gm_your_key_here",
    verify_ssl=False,
)
```
