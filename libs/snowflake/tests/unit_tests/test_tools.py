"""Unit tests for Snowflake tools."""

from unittest.mock import Mock, patch

import pytest

from langchain_snowflake._connection import SqlExecutionClient
from langchain_snowflake.tools import CortexCompleteTool, SnowflakeQueryTool


class MockSession:
    """Mock Snowflake session for testing."""

    def __init__(self):
        """Initialize with mock sql method."""
        self.sql = Mock()
        # Set default behavior
        self._setup_default_sql_behavior()

    def _setup_default_sql_behavior(self):
        """Set up default SQL mock behavior."""

        def sql_side_effect(query, params=None):
            mock_result = Mock()
            if "CORTEX.COMPLETE" in query:
                mock_result.collect.return_value = [{"RESULT": "Generated text response"}]
            elif "CORTEX.SENTIMENT" in query:
                mock_result.collect.return_value = [{"RESULT": 0.8}]
            elif "CORTEX.SUMMARIZE" in query:
                mock_result.collect.return_value = [{"RESULT": "This is a summary"}]
            elif "CORTEX.TRANSLATE" in query:
                mock_result.collect.return_value = [{"RESULT": "Translated text"}]
            else:
                # Create a simple mock that returns empty results to avoid complex row mocking
                mock_result.collect.return_value = []

            # Add limit method
            mock_result.limit.return_value = mock_result
            return mock_result

        self.sql.side_effect = sql_side_effect


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
        assert tool._session == mock_session  # Session should be set to provided session

    @patch("langchain_snowflake.tools.cortex_functions.SnowflakeConnectionMixin._get_session")
    def test_cortex_complete_tool_run(self, mock_get_session, mock_session):
        """Test CortexCompleteTool._run method."""
        mock_get_session.return_value = mock_session

        tool = CortexCompleteTool()
        result = tool._run(prompt="What is machine learning?", model="llama3.1-70b")

        # Should return some result (mocked as JSON string)
        assert isinstance(result, str)
        assert len(result) > 0


class TestSqlExecutionClient:
    """Test SqlExecutionClient functionality."""

    def test_execute_sync_success(self, mock_session):
        """Test successful SQL execution."""
        result = SqlExecutionClient.execute_sync(
            session=mock_session, sql="SELECT CURRENT_TIMESTAMP()", operation_name="test operation"
        )

        assert result["success"] is True
        assert "result" in result

    def test_execute_cortex_function_success(self, mock_session):
        """Test successful Cortex function execution."""
        result = SqlExecutionClient.execute_cortex_function(
            session=mock_session, function_name="SENTIMENT", args=["test text"], operation_name="test sentiment"
        )

        assert result["success"] is True
        assert "result" in result

    @patch("langchain_snowflake._connection.sql_client.SnowflakeToolErrorHandler.handle_tool_error")
    def test_execute_sync_error_handling(self, mock_error_handler, mock_session):
        """Test error handling in SQL execution."""
        # Mock an exception
        mock_session.sql.side_effect = Exception("SQL execution failed")
        mock_error_handler.return_value = "Handled error"

        result = SqlExecutionClient.execute_sync(
            session=mock_session, sql="INVALID SQL", operation_name="test operation"
        )

        assert result["success"] is False
        assert result["error"] == "Handled error"
        mock_error_handler.assert_called_once()


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
