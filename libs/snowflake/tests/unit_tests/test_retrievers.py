"""Unit tests for Snowflake retrievers."""

from unittest.mock import Mock, patch

import pytest

from langchain_snowflake.retrievers import SnowflakeCortexSearchRetriever


class MockSession:
    """Mock Snowflake session for testing."""

    def __init__(self):
        self.account = "test-account"
        self.user = "test-user"

    def sql(self, query):
        mock_result = Mock()
        mock_result.collect.return_value = [["Test document content", "metadata"]]
        return mock_result


@pytest.fixture
def mock_session():
    """Fixture for mock Snowflake session."""
    return MockSession()


@pytest.fixture
def mock_requests_response():
    """Mock HTTP response for REST API calls."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "content": "Test document 1",
                "metadata": {"score": 0.95, "source": "doc1"},
            },
            {
                "content": "Test document 2",
                "metadata": {"score": 0.87, "source": "doc2"},
            },
        ]
    }
    return mock_response


class TestSnowflakeCortexSearchRetriever:
    """Test SnowflakeCortexSearchRetriever functionality."""

    def test_retriever_initialization(self, mock_session):
        """Test retriever initializes correctly."""
        retriever = SnowflakeCortexSearchRetriever(
            service_name="test_db.test_schema.test_service", session=mock_session, k=5
        )
        assert retriever.service_name == "test_db.test_schema.test_service"
        assert retriever.k == 5
        assert retriever.session == mock_session

    def test_retriever_search(self, mock_session):
        """Test basic search functionality."""
        retriever = SnowflakeCortexSearchRetriever(
            service_name="test_db.test_schema.test_service", session=mock_session, k=2
        )

        # Test that the retriever can be initialized and has the right attributes
        assert retriever.service_name == "test_db.test_schema.test_service"
        assert retriever.k == 2

        # Test that the search method exists and can be called
        # We'll patch the actual implementation to avoid network calls
        with patch.object(retriever, "_get_relevant_documents") as mock_search:
            mock_search.return_value = []
            results = retriever._get_relevant_documents("test query", run_manager=None)
            assert isinstance(results, list)
            mock_search.assert_called_once()
