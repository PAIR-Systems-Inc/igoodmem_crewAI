"""Unit tests for GoodMem CrewAI tools.

Uses mocks to test tool logic without requiring a live GoodMem server.
Follows the same pattern as other CrewAI tool tests (e.g. test_oxylabs_tools,
test_mongodb_vector_search_tool).
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import requests

from crewai_tools import (
    GoodMemCreateMemoryTool,
    GoodMemCreateSpaceTool,
    GoodMemDeleteMemoryTool,
    GoodMemGetMemoryTool,
    GoodMemListEmbeddersTool,
    GoodMemListSpacesTool,
    GoodMemRetrieveMemoriesTool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_URL = "https://test.goodmem.ai"
API_KEY = "gm_test_key_123"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Ensure env vars don't leak between tests."""
    monkeypatch.delenv("GOODMEM_BASE_URL", raising=False)
    monkeypatch.delenv("GOODMEM_API_KEY", raising=False)


def _tool_kwargs():
    return dict(base_url=BASE_URL, api_key=API_KEY, verify_ssl=False)


# ---------------------------------------------------------------------------
# Initialization / Config resolution
# ---------------------------------------------------------------------------

class TestToolInitialization:
    """All 7 tools should instantiate with explicit args or env vars."""

    TOOL_CLASSES = [
        GoodMemCreateSpaceTool,
        GoodMemCreateMemoryTool,
        GoodMemRetrieveMemoriesTool,
        GoodMemGetMemoryTool,
        GoodMemDeleteMemoryTool,
        GoodMemListSpacesTool,
        GoodMemListEmbeddersTool,
    ]

    @pytest.mark.parametrize("tool_class", TOOL_CLASSES)
    def test_init_with_explicit_args(self, tool_class):
        tool = tool_class(**_tool_kwargs())
        assert isinstance(tool, tool_class)
        assert tool.base_url == BASE_URL
        assert tool.api_key == API_KEY

    @pytest.mark.parametrize("tool_class", TOOL_CLASSES)
    def test_init_with_env_vars(self, tool_class, monkeypatch):
        monkeypatch.setenv("GOODMEM_BASE_URL", BASE_URL)
        monkeypatch.setenv("GOODMEM_API_KEY", API_KEY)
        tool = tool_class()
        assert isinstance(tool, tool_class)

    @pytest.mark.parametrize("tool_class", TOOL_CLASSES)
    def test_missing_config_returns_error_json(self, tool_class):
        """Tools should return a JSON error (not raise) when config is missing."""
        tool = tool_class()
        # Call the simplest _run signature per tool
        if tool_class == GoodMemCreateSpaceTool:
            result = tool._run(name="test", embedder_id="emb-1")
        elif tool_class == GoodMemCreateMemoryTool:
            result = tool._run(space_id="sp-1", text_content="hello")
        elif tool_class == GoodMemRetrieveMemoriesTool:
            result = tool._run(query="test", space_ids=["sp-1"])
        elif tool_class == GoodMemGetMemoryTool:
            result = tool._run(memory_id="mem-1")
        elif tool_class == GoodMemDeleteMemoryTool:
            result = tool._run(memory_id="mem-1")
        elif tool_class == GoodMemListSpacesTool:
            result = tool._run()
        elif tool_class == GoodMemListEmbeddersTool:
            result = tool._run()
        else:
            pytest.fail(f"Unknown tool class: {tool_class}")

        data = json.loads(result)
        assert data["success"] is False
        assert "error" in data
        assert "GOODMEM" in data["error"]


# ---------------------------------------------------------------------------
# GoodMemListEmbeddersTool
# ---------------------------------------------------------------------------

class TestListEmbeddersTool:
    @patch("crewai_tools.tools.goodmem_tool.goodmem_tool.requests.get")
    def test_list_embedders_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "embedders": [
                {"embedderId": "emb-1", "name": "text-embedding-3-small", "provider": "openai"},
                {"embedderId": "emb-2", "name": "bge-m3", "provider": "huggingface"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        tool = GoodMemListEmbeddersTool(**_tool_kwargs())
        result = json.loads(tool._run())

        assert result["success"] is True
        assert result["totalEmbedders"] == 2
        assert result["embedders"][0]["embedderId"] == "emb-1"

    @patch("crewai_tools.tools.goodmem_tool.goodmem_tool.requests.get")
    def test_list_embedders_api_error(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("Connection refused")

        tool = GoodMemListEmbeddersTool(**_tool_kwargs())
        result = json.loads(tool._run())

        assert result["success"] is False
        assert "Failed to list embedders" in result["error"]


# ---------------------------------------------------------------------------
# GoodMemListSpacesTool
# ---------------------------------------------------------------------------

class TestListSpacesTool:
    @patch("crewai_tools.tools.goodmem_tool.goodmem_tool.requests.get")
    def test_list_spaces_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "spaces": [
                {"spaceId": "sp-1", "name": "My Space"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        tool = GoodMemListSpacesTool(**_tool_kwargs())
        result = json.loads(tool._run())

        assert result["success"] is True
        assert result["totalSpaces"] == 1
        assert result["spaces"][0]["spaceId"] == "sp-1"

    @patch("crewai_tools.tools.goodmem_tool.goodmem_tool.requests.get")
    def test_list_spaces_api_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        http_err = requests.HTTPError(response=mock_resp)
        mock_get.side_effect = http_err

        tool = GoodMemListSpacesTool(**_tool_kwargs())
        result = json.loads(tool._run())

        assert result["success"] is False
        assert "Failed to list spaces" in result["error"]


# ---------------------------------------------------------------------------
# GoodMemCreateSpaceTool
# ---------------------------------------------------------------------------

class TestCreateSpaceTool:
    @patch("crewai_tools.tools.goodmem_tool.goodmem_tool.requests.post")
    @patch("crewai_tools.tools.goodmem_tool.goodmem_tool.requests.get")
    def test_create_space_new(self, mock_get, mock_post):
        # _list_spaces returns empty
        list_resp = MagicMock()
        list_resp.json.return_value = {"spaces": []}
        list_resp.raise_for_status = MagicMock()
        mock_get.return_value = list_resp

        # create returns new space
        create_resp = MagicMock()
        create_resp.json.return_value = {"spaceId": "sp-new", "name": "test-space"}
        create_resp.raise_for_status = MagicMock()
        mock_post.return_value = create_resp

        tool = GoodMemCreateSpaceTool(**_tool_kwargs())
        result = json.loads(tool._run(name="test-space", embedder_id="emb-1"))

        assert result["success"] is True
        assert result["spaceId"] == "sp-new"
        assert result["reused"] is False

    @patch("crewai_tools.tools.goodmem_tool.goodmem_tool.requests.get")
    def test_create_space_reuses_existing(self, mock_get):
        list_resp = MagicMock()
        list_resp.json.return_value = {
            "spaces": [{"spaceId": "sp-existing", "name": "my-space"}]
        }
        list_resp.raise_for_status = MagicMock()
        mock_get.return_value = list_resp

        tool = GoodMemCreateSpaceTool(**_tool_kwargs())
        result = json.loads(tool._run(name="my-space", embedder_id="emb-1"))

        assert result["success"] is True
        assert result["spaceId"] == "sp-existing"
        assert result["reused"] is True

    @patch("crewai_tools.tools.goodmem_tool.goodmem_tool.requests.post")
    @patch("crewai_tools.tools.goodmem_tool.goodmem_tool.requests.get")
    def test_create_space_auth_failure_on_list(self, mock_get, mock_post):
        """Auth errors listing spaces should be returned, not silently swallowed."""
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        http_err = requests.HTTPError(response=mock_resp)
        mock_resp.raise_for_status.side_effect = http_err
        mock_get.return_value = mock_resp

        # The _list_spaces helper calls raise_for_status, which raises
        mock_get.side_effect = http_err

        tool = GoodMemCreateSpaceTool(**_tool_kwargs())
        result = json.loads(tool._run(name="test", embedder_id="emb-1"))

        assert result["success"] is False
        assert "Authentication failed" in result["error"]


# ---------------------------------------------------------------------------
# GoodMemCreateMemoryTool
# ---------------------------------------------------------------------------

class TestCreateMemoryTool:
    @patch("crewai_tools.tools.goodmem_tool.goodmem_tool.requests.post")
    def test_create_memory_text(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "memoryId": "mem-1",
            "processingStatus": "PROCESSING",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        tool = GoodMemCreateMemoryTool(**_tool_kwargs())
        result = json.loads(tool._run(space_id="sp-1", text_content="Hello world"))

        assert result["success"] is True
        assert result["memoryId"] == "mem-1"

        # Verify the request body
        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert body["spaceId"] == "sp-1"
        assert body["contentType"] == "text/plain"
        assert body["originalContent"] == "Hello world"

    @patch("crewai_tools.tools.goodmem_tool.goodmem_tool.requests.post")
    def test_create_memory_file(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "memoryId": "mem-2",
            "processingStatus": "PROCESSING",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("File content here")
            tmp_path = f.name

        try:
            tool = GoodMemCreateMemoryTool(**_tool_kwargs())
            result = json.loads(tool._run(space_id="sp-1", file_path=tmp_path))

            assert result["success"] is True
            assert result["memoryId"] == "mem-2"
        finally:
            os.unlink(tmp_path)

    def test_create_memory_no_content(self):
        tool = GoodMemCreateMemoryTool(**_tool_kwargs())
        result = json.loads(tool._run(space_id="sp-1"))

        assert result["success"] is False
        assert "No content provided" in result["error"]

    def test_create_memory_bad_file_path(self):
        tool = GoodMemCreateMemoryTool(**_tool_kwargs())
        result = json.loads(tool._run(space_id="sp-1", file_path="/nonexistent/file.pdf"))

        assert result["success"] is False
        assert "Cannot read file" in result["error"]


# ---------------------------------------------------------------------------
# GoodMemRetrieveMemoriesTool
# ---------------------------------------------------------------------------

class TestRetrieveMemoriesTool:
    @patch("crewai_tools.tools.goodmem_tool.goodmem_tool.requests.post")
    def test_retrieve_success(self, mock_post):
        ndjson_body = (
            '{"resultSetBoundary": {"resultSetId": "rs-1"}}\n'
            '{"retrievedItem": {"chunk": {"chunk": {"chunkId": "c-1", "chunkText": "hello world", "memoryId": "mem-1"}, "relevanceScore": 0.95}}}\n'
            '{"memoryDefinition": {"memoryId": "mem-1", "contentType": "text/plain"}}\n'
        )
        mock_resp = MagicMock()
        mock_resp.text = ndjson_body
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        tool = GoodMemRetrieveMemoriesTool(**_tool_kwargs())
        result = json.loads(tool._run(
            query="hello",
            space_ids=["sp-1"],
            wait_for_indexing=False,
        ))

        assert result["success"] is True
        assert result["totalResults"] == 1
        assert result["results"][0]["chunkText"] == "hello world"
        assert result["results"][0]["relevanceScore"] == 0.95
        assert len(result["memories"]) == 1

    def test_retrieve_empty_space_ids(self):
        tool = GoodMemRetrieveMemoriesTool(**_tool_kwargs())
        result = json.loads(tool._run(query="test", space_ids=[]))

        assert result["success"] is False
        assert "At least one space ID" in result["error"]

    @patch("crewai_tools.tools.goodmem_tool.goodmem_tool.requests.post")
    def test_retrieve_no_results_no_wait(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.text = '{"resultSetBoundary": {"resultSetId": "rs-1"}}\n'
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        tool = GoodMemRetrieveMemoriesTool(**_tool_kwargs())
        result = json.loads(tool._run(
            query="nothing",
            space_ids=["sp-1"],
            wait_for_indexing=False,
        ))

        assert result["success"] is True
        assert result["totalResults"] == 0

    @patch("crewai_tools.tools.goodmem_tool.goodmem_tool.requests.post")
    def test_retrieve_api_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Server Error"
        http_err = requests.HTTPError(response=mock_resp)
        mock_post.side_effect = http_err

        tool = GoodMemRetrieveMemoriesTool(**_tool_kwargs())
        result = json.loads(tool._run(
            query="test",
            space_ids=["sp-1"],
            wait_for_indexing=False,
        ))

        assert result["success"] is False
        assert "Failed to retrieve memories" in result["error"]


# ---------------------------------------------------------------------------
# GoodMemGetMemoryTool
# ---------------------------------------------------------------------------

class TestGetMemoryTool:
    @patch("crewai_tools.tools.goodmem_tool.goodmem_tool.requests.get")
    def test_get_memory_success(self, mock_get):
        memory_resp = MagicMock()
        memory_resp.json.return_value = {
            "memoryId": "mem-1",
            "processingStatus": "COMPLETED",
            "contentType": "text/plain",
        }
        memory_resp.raise_for_status = MagicMock()

        content_resp = MagicMock()
        content_resp.json.return_value = {"originalContent": "Hello"}
        content_resp.headers = {"Content-Type": "application/json"}
        content_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [memory_resp, content_resp]

        tool = GoodMemGetMemoryTool(**_tool_kwargs())
        result = json.loads(tool._run(memory_id="mem-1"))

        assert result["success"] is True
        assert result["memory"]["memoryId"] == "mem-1"
        assert result["content"]["originalContent"] == "Hello"

    @patch("crewai_tools.tools.goodmem_tool.goodmem_tool.requests.get")
    def test_get_memory_without_content(self, mock_get):
        memory_resp = MagicMock()
        memory_resp.json.return_value = {"memoryId": "mem-1"}
        memory_resp.raise_for_status = MagicMock()
        mock_get.return_value = memory_resp

        tool = GoodMemGetMemoryTool(**_tool_kwargs())
        result = json.loads(tool._run(memory_id="mem-1", include_content=False))

        assert result["success"] is True
        assert "content" not in result

    @patch("crewai_tools.tools.goodmem_tool.goodmem_tool.requests.get")
    def test_get_memory_not_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not Found"
        http_err = requests.HTTPError(response=mock_resp)
        mock_get.side_effect = http_err

        tool = GoodMemGetMemoryTool(**_tool_kwargs())
        result = json.loads(tool._run(memory_id="nonexistent"))

        assert result["success"] is False
        assert "Failed to get memory" in result["error"]

    @patch("crewai_tools.tools.goodmem_tool.goodmem_tool.requests.get")
    def test_get_memory_content_non_json(self, mock_get):
        """Content endpoint returning non-JSON (e.g. plain text) should not crash."""
        memory_resp = MagicMock()
        memory_resp.json.return_value = {"memoryId": "mem-1"}
        memory_resp.raise_for_status = MagicMock()

        content_resp = MagicMock()
        content_resp.text = "Plain text content here"
        content_resp.headers = {"Content-Type": "text/plain"}
        content_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [memory_resp, content_resp]

        tool = GoodMemGetMemoryTool(**_tool_kwargs())
        result = json.loads(tool._run(memory_id="mem-1"))

        assert result["success"] is True
        assert result["content"] == "Plain text content here"


# ---------------------------------------------------------------------------
# GoodMemDeleteMemoryTool
# ---------------------------------------------------------------------------

class TestDeleteMemoryTool:
    @patch("crewai_tools.tools.goodmem_tool.goodmem_tool.requests.delete")
    def test_delete_success(self, mock_delete):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_delete.return_value = mock_resp

        tool = GoodMemDeleteMemoryTool(**_tool_kwargs())
        result = json.loads(tool._run(memory_id="mem-1"))

        assert result["success"] is True
        assert result["memoryId"] == "mem-1"
        assert "deleted" in result["message"].lower()

    @patch("crewai_tools.tools.goodmem_tool.goodmem_tool.requests.delete")
    def test_delete_not_found(self, mock_delete):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not Found"
        http_err = requests.HTTPError(response=mock_resp)
        mock_delete.side_effect = http_err

        tool = GoodMemDeleteMemoryTool(**_tool_kwargs())
        result = json.loads(tool._run(memory_id="nonexistent"))

        assert result["success"] is False
        assert "Failed to delete memory" in result["error"]
