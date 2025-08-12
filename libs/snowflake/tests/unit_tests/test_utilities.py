"""Unit tests for utility components."""

import json

from langchain_core.documents import Document

from langchain_snowflake._error_handling import SnowflakeErrorHandler
from langchain_snowflake.formatters import format_cortex_search_documents
from langchain_snowflake.tools._base import SnowflakeToolResponse


class TestFormatters:
    """Test document formatting utilities."""

    def test_format_cortex_search_documents_basic(self):
        """Test basic document formatting functionality."""
        docs = [
            Document(
                page_content="Test content 1",
                metadata={"TRANSCRIPT_TEXT": "Formatted content 1", "score": 0.95},
            ),
            Document(
                page_content="Test content 2",
                metadata={"TRANSCRIPT_TEXT": "Formatted content 2", "score": 0.87},
            ),
        ]

        result = format_cortex_search_documents(docs)

        assert isinstance(result, str)
        assert "Formatted content 1" in result
        assert "Formatted content 2" in result
        assert "\n\n" in result  # Default separator

    def test_format_cortex_search_documents_empty(self):
        """Test formatting with empty document list."""
        result = format_cortex_search_documents([])
        assert result == ""

    def test_format_cortex_search_documents_fallback(self):
        """Test fallback to page_content when metadata field missing."""
        docs = [
            Document(page_content="Fallback content", metadata={}),
        ]

        result = format_cortex_search_documents(docs, fallback_to_page_content=True)
        assert "Fallback content" in result


class TestErrorHandling:
    """Test error handling utilities."""

    def test_log_and_return_json_error(self):
        """Test JSON error response generation."""
        error = ValueError("Test error message")

        result = SnowflakeErrorHandler.log_and_return_json_error(
            error=error,
            operation="test operation",
            additional_context={"context_key": "context_value"},
        )

        # Should return valid JSON
        parsed = json.loads(result)
        assert "error" in parsed
        assert "test operation" in parsed["error"]
        assert "Test error message" in parsed["error"]
        assert parsed["operation"] == "test operation"
        assert parsed["context_key"] == "context_value"

    def test_error_handler_basic_functionality(self):
        """Test basic error handler functionality."""
        # Test that the error handler class can be imported and used
        handler = SnowflakeErrorHandler()
        assert hasattr(handler, "log_and_return_json_error")

        # Test static method access
        assert callable(SnowflakeErrorHandler.log_and_return_json_error)


class TestSchemas:
    """Test input/output schema definitions."""

    def test_snowflake_tool_response_schema(self):
        """Test SnowflakeToolResponse schema validation."""
        # Test successful response
        response = SnowflakeToolResponse(success=True, error=None)

        assert response.success is True
        assert response.error is None

    def test_snowflake_tool_response_error(self):
        """Test SnowflakeToolResponse with error."""
        response = SnowflakeToolResponse(success=False, error="Test error message")

        assert response.success is False
        assert response.error == "Test error message"

    def test_schema_serialization(self):
        """Test schema serialization/deserialization."""
        response = SnowflakeToolResponse(success=True, error=None)

        # Test that it can be serialized to dict
        as_dict = response.model_dump()
        assert isinstance(as_dict, dict)
        assert as_dict["success"] is True
        assert as_dict["error"] is None
