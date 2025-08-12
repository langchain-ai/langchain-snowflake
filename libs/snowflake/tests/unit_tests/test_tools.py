"""Unit tests for Snowflake tools."""

from unittest.mock import Mock, patch

import pytest

from langchain_snowflake.tools import CortexCompleteTool, SnowflakeQueryTool


class MockSession:
    """Mock Snowflake session for testing."""

    def sql(self, query):
        mock_result = Mock()
        if "CORTEX.COMPLETE" in query:
            mock_result.collect.return_value = [["Generated text response"]]
        else:
            # Create a simple mock that returns empty results to avoid complex row mocking
            mock_result.collect.return_value = []

        # Add limit method
        mock_result.limit.return_value = mock_result
        return mock_result


@pytest.fixture
def mock_session():
    """Fixture for mock Snowflake session."""
    return MockSession()


class TestCortexCompleteTool:
    """Test CortexCompleteTool functionality."""

    def test_cortex_complete_tool_initialization(self, mock_session):
        """Test CortexCompleteTool initializes correctly."""
        tool = CortexCompleteTool(session=mock_session)
        assert tool.name == "cortex_complete"
        assert "complete" in tool.description.lower()
        assert tool._session is None  # Initially None, set on first use

    @patch(
        "langchain_snowflake.tools.cortex_functions.SnowflakeConnectionMixin._get_session"
    )
    def test_cortex_complete_tool_run(self, mock_get_session, mock_session):
        """Test CortexCompleteTool._run method."""
        mock_get_session.return_value = mock_session

        tool = CortexCompleteTool()
        result = tool._run(prompt="What is machine learning?", model="llama3.1-70b")

        # Should return some result (mocked as JSON string)
        assert isinstance(result, str)
        assert len(result) > 0


class TestSnowflakeQueryTool:
    """Test SnowflakeQueryTool functionality."""

    def test_query_tool_initialization(self, mock_session):
        """Test SnowflakeQueryTool initializes correctly."""
        tool = SnowflakeQueryTool(session=mock_session)
        assert tool.name == "snowflake_query"
        assert "sql" in tool.description.lower()
        assert tool.max_rows == 100  # Default value

    @patch("langchain_snowflake.tools.query.SnowflakeConnectionMixin._get_session")
    def test_query_tool_run(self, mock_get_session, mock_session):
        """Test SnowflakeQueryTool._run method."""
        mock_get_session.return_value = mock_session

        tool = SnowflakeQueryTool(max_rows=50)
        result = tool._run(query="SELECT COUNT(*) FROM test_table")

        # Should return some result (mocked as "no results" message)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "no results" in result.lower()
