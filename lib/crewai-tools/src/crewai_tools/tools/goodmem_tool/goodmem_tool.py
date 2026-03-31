"""GoodMem tools for CrewAI.

Provides CrewAI tools for interacting with a GoodMem API server to create
spaces, store memories (text and files), perform semantic retrieval, inspect
individual memories, and delete memories.
"""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Literal

import requests
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MIME_TYPES: dict[str, str] = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "txt": "text/plain",
    "html": "text/html",
    "md": "text/markdown",
    "csv": "text/csv",
    "json": "application/json",
    "xml": "application/xml",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "ppt": "application/vnd.ms-powerpoint",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def _mime_from_extension(ext: str) -> str | None:
    return _MIME_TYPES.get(ext.lower().lstrip("."))


def _base_url(raw: str) -> str:
    return raw.rstrip("/")


def _headers(api_key: str) -> dict[str, str]:
    return {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _resolve_config(
    *,
    base_url: str | None,
    api_key: str | None,
) -> tuple[str, str]:
    """Return (base_url, api_key) from explicit args or env vars."""
    url = base_url or os.getenv("GOODMEM_BASE_URL", "")
    key = api_key or os.getenv("GOODMEM_API_KEY", "")
    if not url:
        raise ValueError(
            "GoodMem base URL is required. Pass base_url to the tool "
            "constructor or set the GOODMEM_BASE_URL environment variable."
        )
    if not key:
        raise ValueError(
            "GoodMem API key is required. Pass api_key to the tool "
            "constructor or set the GOODMEM_API_KEY environment variable."
        )
    return _base_url(url), key


# ---------------------------------------------------------------------------
# Internal helpers -- List Embedders / List Spaces
# ---------------------------------------------------------------------------

def _list_embedders(base_url: str, api_key: str, *, verify_ssl: bool = True) -> list[dict[str, Any]]:
    """Fetch available embedders from the GoodMem API (internal helper)."""
    resp = requests.get(
        f"{base_url}/v1/embedders",
        headers=_headers(api_key),
        verify=verify_ssl,
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    return body if isinstance(body, list) else body.get("embedders", [])


def _list_spaces(base_url: str, api_key: str, *, verify_ssl: bool = True) -> list[dict[str, Any]]:
    """Fetch available spaces from the GoodMem API (internal helper)."""
    resp = requests.get(
        f"{base_url}/v1/spaces",
        headers=_headers(api_key),
        verify=verify_ssl,
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    return body if isinstance(body, list) else body.get("spaces", [])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CreateSpaceSchema(BaseModel):
    name: str = Field(
        ...,
        description=(
            "A unique name for the space. If a space with this name already "
            "exists, its ID is returned instead of creating a duplicate."
        ),
    )
    embedder_id: str = Field(
        ...,
        description=(
            "The ID of the embedder model that converts text into vector "
            "representations for similarity search."
        ),
    )
    chunk_size: int = Field(
        default=256,
        description="Number of characters per chunk when splitting documents.",
    )
    chunk_overlap: int = Field(
        default=25,
        description="Number of overlapping characters between consecutive chunks.",
    )
    keep_strategy: Literal["KEEP_END", "KEEP_START", "DISCARD"] = Field(
        default="KEEP_END",
        description=(
            "Where to attach the separator when splitting. "
            "One of: KEEP_END, KEEP_START, DISCARD."
        ),
    )
    length_measurement: Literal["CHARACTER_COUNT", "TOKEN_COUNT"] = Field(
        default="CHARACTER_COUNT",
        description=(
            "How chunk size is measured. "
            "One of: CHARACTER_COUNT, TOKEN_COUNT."
        ),
    )


class CreateMemorySchema(BaseModel):
    space_id: str = Field(
        ...,
        description="The ID of the space to store the memory in.",
    )
    text_content: str | None = Field(
        default=None,
        description=(
            "Plain text content to store as memory. "
            "If both file_path and text_content are provided, file takes priority."
        ),
    )
    file_path: str | None = Field(
        default=None,
        description=(
            "Local file path to store as memory (PDF, DOCX, image, etc.). "
            "Content type is auto-detected from the file extension."
        ),
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional key-value metadata as a JSON object.",
    )


class RetrieveMemoriesSchema(BaseModel):
    query: str = Field(
        ...,
        description="A natural language query used to find semantically similar memory chunks.",
    )
    space_ids: list[str] = Field(
        ...,
        description="List of space IDs to search across.",
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return.",
    )
    include_memory_definition: bool = Field(
        default=True,
        description="Fetch full memory metadata alongside matched chunks.",
    )
    wait_for_indexing: bool = Field(
        default=True,
        description=(
            "Retry for up to 60 seconds when no results are found. "
            "Useful when memories were just added and may still be processing."
        ),
    )
    reranker_id: str | None = Field(
        default=None,
        description="Optional reranker model ID to improve result ordering.",
    )
    llm_id: str | None = Field(
        default=None,
        description="Optional LLM ID to generate contextual responses alongside retrieved chunks.",
    )
    relevance_threshold: float | None = Field(
        default=None,
        description="Minimum score (0-1) for including results. Only used with reranker or LLM.",
    )
    llm_temperature: float | None = Field(
        default=None,
        description="Creativity setting for LLM generation (0-2). Only used when llm_id is set.",
    )
    chronological_resort: bool = Field(
        default=False,
        description="Reorder results by creation time instead of relevance score.",
    )


class GetMemorySchema(BaseModel):
    memory_id: str = Field(
        ...,
        description="The UUID of the memory to fetch.",
    )
    include_content: bool = Field(
        default=True,
        description="Fetch the original document content in addition to metadata.",
    )


class DeleteMemorySchema(BaseModel):
    memory_id: str = Field(
        ...,
        description="The UUID of the memory to delete.",
    )


class ListSpacesSchema(BaseModel):
    pass


class ListEmbeddersSchema(BaseModel):
    pass


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class GoodMemCreateSpaceTool(BaseTool):
    """Create a new GoodMem space or reuse an existing one.

    A space is a logical container for organising related memories, configured
    with an embedder that converts text to vector embeddings.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "GoodMemCreateSpace"
    description: str = (
        "Create a new GoodMem space (container for memories) with an embedder "
        "and chunking configuration. If a space with the same name already "
        "exists it is reused."
    )
    args_schema: type[BaseModel] = CreateSpaceSchema
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(name="GOODMEM_BASE_URL", description="GoodMem API base URL", required=True),
            EnvVar(name="GOODMEM_API_KEY", description="GoodMem API key", required=True),
        ]
    )
    base_url: str | None = None
    api_key: str | None = None
    verify_ssl: bool = True

    def _run(
        self,
        name: str,
        embedder_id: str,
        chunk_size: int = 256,
        chunk_overlap: int = 25,
        keep_strategy: Literal["KEEP_END", "KEEP_START", "DISCARD"] = "KEEP_END",
        length_measurement: Literal["CHARACTER_COUNT", "TOKEN_COUNT"] = "CHARACTER_COUNT",
    ) -> str:
        try:
            url, key = _resolve_config(base_url=self.base_url, api_key=self.api_key)
        except ValueError as exc:
            return json.dumps({"success": False, "error": str(exc)})

        # Check for existing space with same name
        try:
            spaces = _list_spaces(url, key, verify_ssl=self.verify_ssl)
            for s in spaces:
                if s.get("name") == name:
                    return json.dumps({
                        "success": True,
                        "spaceId": s.get("spaceId") or s.get("id"),
                        "name": s["name"],
                        "embedderId": embedder_id,
                        "message": "Space already exists, reusing existing space",
                        "reused": True,
                    })
        except requests.RequestException as exc:
            error_detail = ""
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    error_detail = exc.response.text
                except Exception:
                    pass
            if exc.response is not None and exc.response.status_code in (401, 403):
                return json.dumps({
                    "success": False,
                    "error": f"Authentication failed while checking existing spaces: {exc}",
                    "details": error_detail,
                })

        body: dict[str, Any] = {
            "name": name,
            "spaceEmbedders": [{"embedderId": embedder_id, "defaultRetrievalWeight": 1.0}],
            "defaultChunkingConfig": {
                "recursive": {
                    "chunkSize": chunk_size,
                    "chunkOverlap": chunk_overlap,
                    "separators": ["\n\n", "\n", ". ", " ", ""],
                    "keepStrategy": keep_strategy,
                    "separatorIsRegex": False,
                    "lengthMeasurement": length_measurement,
                },
            },
        }

        try:
            resp = requests.post(
                f"{url}/v1/spaces",
                headers=_headers(key),
                json=body,
                verify=self.verify_ssl,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return json.dumps({
                "success": True,
                "spaceId": data.get("spaceId"),
                "name": data.get("name"),
                "embedderId": embedder_id,
                "chunkingConfig": body["defaultChunkingConfig"],
                "message": "Space created successfully",
                "reused": False,
            })
        except requests.RequestException as exc:
            error_detail = ""
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    error_detail = exc.response.text
                except Exception:
                    pass
            return json.dumps({
                "success": False,
                "error": f"Failed to create space: {exc}",
                "details": error_detail,
            })


class GoodMemCreateMemoryTool(BaseTool):
    """Store a document or plain text as a new memory in a GoodMem space.

    The memory is processed asynchronously -- chunked into searchable pieces
    and embedded into vectors.  Accepts a local file path or plain text.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "GoodMemCreateMemory"
    description: str = (
        "Store a document or plain text as a memory in a GoodMem space. "
        "Provide either a local file path (PDF, DOCX, images, etc.) or "
        "plain text content. The memory is chunked and embedded for "
        "semantic search."
    )
    args_schema: type[BaseModel] = CreateMemorySchema
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(name="GOODMEM_BASE_URL", description="GoodMem API base URL", required=True),
            EnvVar(name="GOODMEM_API_KEY", description="GoodMem API key", required=True),
        ]
    )
    base_url: str | None = None
    api_key: str | None = None
    verify_ssl: bool = True

    def _run(
        self,
        space_id: str,
        text_content: str | None = None,
        file_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        try:
            url, key = _resolve_config(base_url=self.base_url, api_key=self.api_key)
        except ValueError as exc:
            return json.dumps({"success": False, "error": str(exc)})

        request_body: dict[str, Any] = {"spaceId": space_id}
        file_name: str | None = None

        if file_path:
            # Read file and encode
            ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
            mime = _mime_from_extension(ext) or "application/octet-stream"
            try:
                with open(file_path, "rb") as fh:
                    raw = fh.read()
            except OSError as exc:
                return json.dumps({
                    "success": False,
                    "error": f"Cannot read file '{file_path}': {exc}",
                })

            file_name = os.path.basename(file_path)

            if mime.startswith("text/"):
                request_body["contentType"] = mime
                request_body["originalContent"] = raw.decode("utf-8", errors="replace")
            else:
                request_body["contentType"] = mime
                request_body["originalContentB64"] = base64.b64encode(raw).decode("ascii")
        elif text_content:
            request_body["contentType"] = "text/plain"
            request_body["originalContent"] = text_content
        else:
            return json.dumps({
                "success": False,
                "error": "No content provided. Supply either file_path or text_content.",
            })

        if metadata and isinstance(metadata, dict) and len(metadata) > 0:
            request_body["metadata"] = metadata

        try:
            resp = requests.post(
                f"{url}/v1/memories",
                headers=_headers(key),
                json=request_body,
                verify=self.verify_ssl,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return json.dumps({
                "success": True,
                "memoryId": data.get("memoryId"),
                "spaceId": data.get("spaceId"),
                "status": data.get("processingStatus", "PENDING"),
                "contentType": request_body.get("contentType"),
                "fileName": file_name,
                "message": "Memory created successfully",
            })
        except requests.RequestException as exc:
            error_detail = ""
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    error_detail = exc.response.text
                except Exception:
                    pass
            return json.dumps({
                "success": False,
                "error": f"Failed to create memory: {exc}",
                "details": error_detail,
            })


class GoodMemRetrieveMemoriesTool(BaseTool):
    """Perform similarity-based semantic retrieval across GoodMem spaces.

    Returns matching chunks ranked by relevance, with optional full memory
    definitions, reranking, and LLM-generated contextual responses.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "GoodMemRetrieveMemories"
    description: str = (
        "Perform semantic search across one or more GoodMem spaces to find "
        "relevant memory chunks. Returns chunks ranked by relevance with "
        "optional reranking and LLM-generated responses."
    )
    args_schema: type[BaseModel] = RetrieveMemoriesSchema
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(name="GOODMEM_BASE_URL", description="GoodMem API base URL", required=True),
            EnvVar(name="GOODMEM_API_KEY", description="GoodMem API key", required=True),
        ]
    )
    base_url: str | None = None
    api_key: str | None = None
    verify_ssl: bool = True

    def _run(
        self,
        query: str,
        space_ids: list[str],
        max_results: int = 5,
        include_memory_definition: bool = True,
        wait_for_indexing: bool = True,
        reranker_id: str | None = None,
        llm_id: str | None = None,
        relevance_threshold: float | None = None,
        llm_temperature: float | None = None,
        chronological_resort: bool = False,
    ) -> str:
        try:
            url, key = _resolve_config(base_url=self.base_url, api_key=self.api_key)
        except ValueError as exc:
            return json.dumps({"success": False, "error": str(exc)})

        space_keys = [{"spaceId": sid} for sid in space_ids if sid]
        if not space_keys:
            return json.dumps({
                "success": False,
                "error": "At least one space ID must be provided.",
            })

        request_body: dict[str, Any] = {
            "message": query,
            "spaceKeys": space_keys,
            "requestedSize": max_results,
            "fetchMemory": include_memory_definition,
        }

        # Post-processor config
        if reranker_id or llm_id or chronological_resort:
            config: dict[str, Any] = {}
            if reranker_id:
                config["reranker_id"] = reranker_id
            if llm_id:
                config["llm_id"] = llm_id
            if relevance_threshold is not None:
                config["relevance_threshold"] = relevance_threshold
            if llm_temperature is not None:
                config["llm_temp"] = llm_temperature
            if max_results:
                config["max_results"] = max_results
            if chronological_resort:
                config["chronological_resort"] = True
            request_body["postProcessor"] = {
                "name": "com.goodmem.retrieval.postprocess.ChatPostProcessorFactory",
                "config": config,
            }

        max_wait = 60
        poll_interval = 5
        start = time.monotonic()
        last_result: dict[str, Any] | None = None

        try:
            while True:
                retrieve_headers = {**_headers(key), "Accept": "application/x-ndjson"}
                resp = requests.post(
                    f"{url}/v1/memories:retrieve",
                    headers=retrieve_headers,
                    json=request_body,
                    verify=self.verify_ssl,
                    timeout=90,
                )
                resp.raise_for_status()

                results: list[dict[str, Any]] = []
                memories: list[dict[str, Any]] = []
                result_set_id = ""
                abstract_reply: dict[str, Any] | None = None

                response_text = resp.text
                for line in response_text.strip().split("\n"):
                    json_str = line.strip()
                    if not json_str:
                        continue
                    if json_str.startswith("data:"):
                        json_str = json_str[5:].strip()
                    if json_str.startswith("event:") or json_str == "":
                        continue
                    try:
                        item = json.loads(json_str)
                        if item.get("resultSetBoundary"):
                            result_set_id = item["resultSetBoundary"].get("resultSetId", "")
                        elif item.get("memoryDefinition"):
                            memories.append(item["memoryDefinition"])
                        elif item.get("abstractReply"):
                            abstract_reply = item["abstractReply"]
                        elif item.get("retrievedItem"):
                            chunk_data = item["retrievedItem"].get("chunk", {})
                            chunk = chunk_data.get("chunk", {})
                            results.append({
                                "chunkId": chunk.get("chunkId"),
                                "chunkText": chunk.get("chunkText"),
                                "memoryId": chunk.get("memoryId"),
                                "relevanceScore": chunk_data.get("relevanceScore"),
                                "memoryIndex": chunk_data.get("memoryIndex"),
                            })
                    except (json.JSONDecodeError, KeyError):
                        continue

                last_result = {
                    "success": True,
                    "resultSetId": result_set_id,
                    "results": results,
                    "memories": memories,
                    "totalResults": len(results),
                    "query": query,
                }
                if abstract_reply:
                    last_result["abstractReply"] = abstract_reply

                if results or not wait_for_indexing:
                    return json.dumps(last_result)

                elapsed = time.monotonic() - start
                if elapsed >= max_wait:
                    last_result["message"] = (
                        "No results found after waiting 60 seconds for indexing. "
                        "Memories may still be processing."
                    )
                    return json.dumps(last_result)

                time.sleep(poll_interval)

        except requests.RequestException as exc:
            error_detail = ""
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    error_detail = exc.response.text
                except Exception:
                    pass
            return json.dumps({
                "success": False,
                "error": f"Failed to retrieve memories: {exc}",
                "details": error_detail,
            })


class GoodMemGetMemoryTool(BaseTool):
    """Fetch a specific memory record by its ID from GoodMem.

    Includes metadata, processing status, and optionally the original content.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "GoodMemGetMemory"
    description: str = (
        "Fetch a specific memory by its ID from GoodMem, including metadata, "
        "processing status, and optionally the original content."
    )
    args_schema: type[BaseModel] = GetMemorySchema
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(name="GOODMEM_BASE_URL", description="GoodMem API base URL", required=True),
            EnvVar(name="GOODMEM_API_KEY", description="GoodMem API key", required=True),
        ]
    )
    base_url: str | None = None
    api_key: str | None = None
    verify_ssl: bool = True

    def _run(
        self,
        memory_id: str,
        include_content: bool = True,
    ) -> str:
        try:
            url, key = _resolve_config(base_url=self.base_url, api_key=self.api_key)
        except ValueError as exc:
            return json.dumps({"success": False, "error": str(exc)})

        try:
            resp = requests.get(
                f"{url}/v1/memories/{memory_id}",
                headers=_headers(key),
                verify=self.verify_ssl,
                timeout=30,
            )
            resp.raise_for_status()
            result: dict[str, Any] = {
                "success": True,
                "memory": resp.json(),
            }

            if include_content:
                try:
                    content_resp = requests.get(
                        f"{url}/v1/memories/{memory_id}/content",
                        headers=_headers(key),
                        verify=self.verify_ssl,
                        timeout=30,
                    )
                    content_resp.raise_for_status()
                    content_type = content_resp.headers.get("Content-Type", "")
                    if "json" in content_type:
                        result["content"] = content_resp.json()
                    else:
                        result["content"] = content_resp.text
                except requests.RequestException as content_exc:
                    result["contentError"] = f"Failed to fetch content: {content_exc}"

            return json.dumps(result)
        except requests.RequestException as exc:
            error_detail = ""
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    error_detail = exc.response.text
                except Exception:
                    pass
            return json.dumps({
                "success": False,
                "error": f"Failed to get memory: {exc}",
                "details": error_detail,
            })


class GoodMemDeleteMemoryTool(BaseTool):
    """Permanently delete a memory and its associated chunks and embeddings."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "GoodMemDeleteMemory"
    description: str = (
        "Permanently delete a memory and all its associated chunks and "
        "vector embeddings from GoodMem."
    )
    args_schema: type[BaseModel] = DeleteMemorySchema
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(name="GOODMEM_BASE_URL", description="GoodMem API base URL", required=True),
            EnvVar(name="GOODMEM_API_KEY", description="GoodMem API key", required=True),
        ]
    )
    base_url: str | None = None
    api_key: str | None = None
    verify_ssl: bool = True

    def _run(self, memory_id: str) -> str:
        try:
            url, key = _resolve_config(base_url=self.base_url, api_key=self.api_key)
        except ValueError as exc:
            return json.dumps({"success": False, "error": str(exc)})

        try:
            resp = requests.delete(
                f"{url}/v1/memories/{memory_id}",
                headers=_headers(key),
                verify=self.verify_ssl,
                timeout=30,
            )
            resp.raise_for_status()
            return json.dumps({
                "success": True,
                "memoryId": memory_id,
                "message": "Memory deleted successfully",
            })
        except requests.RequestException as exc:
            error_detail = ""
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    error_detail = exc.response.text
                except Exception:
                    pass
            return json.dumps({
                "success": False,
                "error": f"Failed to delete memory: {exc}",
                "details": error_detail,
            })


class GoodMemListSpacesTool(BaseTool):
    """List all available spaces in GoodMem."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "GoodMemListSpaces"
    description: str = (
        "List all available spaces in GoodMem. Returns space IDs and names "
        "that can be used with other GoodMem tools."
    )
    args_schema: type[BaseModel] = ListSpacesSchema
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(name="GOODMEM_BASE_URL", description="GoodMem API base URL", required=True),
            EnvVar(name="GOODMEM_API_KEY", description="GoodMem API key", required=True),
        ]
    )
    base_url: str | None = None
    api_key: str | None = None
    verify_ssl: bool = True

    def _run(self) -> str:
        try:
            url, key = _resolve_config(base_url=self.base_url, api_key=self.api_key)
        except ValueError as exc:
            return json.dumps({"success": False, "error": str(exc)})

        try:
            spaces = _list_spaces(url, key, verify_ssl=self.verify_ssl)
            return json.dumps({
                "success": True,
                "spaces": [
                    {
                        "spaceId": s.get("spaceId") or s.get("id"),
                        "name": s.get("name", "Unnamed"),
                    }
                    for s in spaces
                ],
                "totalSpaces": len(spaces),
            })
        except requests.RequestException as exc:
            error_detail = ""
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    error_detail = exc.response.text
                except Exception:
                    pass
            return json.dumps({
                "success": False,
                "error": f"Failed to list spaces: {exc}",
                "details": error_detail,
            })


class GoodMemListEmbeddersTool(BaseTool):
    """List all available embedders in GoodMem.

    Embedder IDs are required when creating a new space. Use this tool
    to discover which embedder models are available on the server.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "GoodMemListEmbedders"
    description: str = (
        "List all available embedders in GoodMem. Returns embedder IDs "
        "and names that are required when creating a new space."
    )
    args_schema: type[BaseModel] = ListEmbeddersSchema
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(name="GOODMEM_BASE_URL", description="GoodMem API base URL", required=True),
            EnvVar(name="GOODMEM_API_KEY", description="GoodMem API key", required=True),
        ]
    )
    base_url: str | None = None
    api_key: str | None = None
    verify_ssl: bool = True

    def _run(self) -> str:
        try:
            url, key = _resolve_config(base_url=self.base_url, api_key=self.api_key)
        except ValueError as exc:
            return json.dumps({"success": False, "error": str(exc)})

        try:
            embedders = _list_embedders(url, key, verify_ssl=self.verify_ssl)
            return json.dumps({
                "success": True,
                "embedders": [
                    {
                        "embedderId": e.get("embedderId") or e.get("id"),
                        "name": e.get("name", "Unnamed"),
                        "provider": e.get("provider", ""),
                    }
                    for e in embedders
                ],
                "totalEmbedders": len(embedders),
            })
        except requests.RequestException as exc:
            error_detail = ""
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    error_detail = exc.response.text
                except Exception:
                    pass
            return json.dumps({
                "success": False,
                "error": f"Failed to list embedders: {exc}",
                "details": error_detail,
            })
