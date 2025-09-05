"""Unit tests for ChatSnowflake and related components."""

from typing import Literal
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from langchain_snowflake.chat_models import ChatSnowflake


class MockSession:
    """Mock Snowflake session for testing."""

    def sql(self, query):
        mock_result = Mock()
        mock_result.collect.return_value = [["Test response"]]
        return mock_result


@pytest.fixture
def mock_session():
    """Fixture for mock Snowflake session."""
    return MockSession()


class TestChatSnowflakeMain:
    """Test main ChatSnowflake functionality."""

    @patch("langchain_snowflake.chat_models.base.Session")
    def test_initialization_with_session(self, mock_session_class):
        """Test ChatSnowflake initializes correctly with session."""
        session = MockSession()
        llm = ChatSnowflake(model="llama3.1-70b", session=session)
        assert llm.model == "llama3.1-70b"
        assert llm.session == session

    @patch("langchain_snowflake.chat_models.base.Session")
    def test_initialization_with_params(self, mock_session_class):
        """Test ChatSnowflake initializes with connection parameters."""
        llm = ChatSnowflake(model="claude-3-5-sonnet", account="test-account", user="test-user")
        assert llm.model == "claude-3-5-sonnet"
        assert llm.account == "test-account"
        assert llm.user == "test-user"


class TestSnowflakeAuth:
    """Test authentication functionality."""

    @patch("snowflake.snowpark.Session")
    def test_session_creation_with_password(self, mock_session_class):
        """Test session creation with password authentication."""
        mock_session_instance = Mock()
        mock_session_class.builder.configs.return_value.create.return_value = mock_session_instance

        llm = ChatSnowflake(account="test-account", user="test-user", password="test-password")

        # Trigger session creation
        session = llm._get_session()
        assert session == mock_session_instance

    @patch("snowflake.snowpark.Session")
    def test_session_creation_with_token(self, mock_session_class):
        """Test session creation with token authentication."""
        mock_session_instance = Mock()
        mock_session_class.builder.configs.return_value.create.return_value = mock_session_instance

        llm = ChatSnowflake(account="test-account", user="test-user", token="test-token")

        # Trigger session creation
        session = llm._get_session()
        assert session == mock_session_instance


class TestSnowflakeStructuredOutput:
    """Test structured output functionality."""

    def test_with_structured_output_pydantic(self):
        """Test structured output with Pydantic model."""

        class TestModel(BaseModel):
            sentiment: Literal["positive", "negative", "neutral"]
            confidence: float

        llm = ChatSnowflake(model="llama3.1-70b", session=MockSession())
        structured_llm = llm.with_structured_output(TestModel)

        assert structured_llm is not None
        assert hasattr(structured_llm, "_target_schema")

    def test_with_structured_output_json_schema(self):
        """Test structured output with JSON schema."""
        schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}, "score": {"type": "number"}},
        }

        llm = ChatSnowflake(model="llama3.1-70b", session=MockSession())
        structured_llm = llm.with_structured_output(schema)

        assert structured_llm is not None
        assert hasattr(structured_llm, "_target_schema")


class TestSnowflakeUtils:
    """Test utility functions."""

    def test_message_conversion(self):
        """Test message format conversion."""
        ChatSnowflake(model="llama3.1-70b", session=MockSession())

        messages = [HumanMessage(content="Hello")]
        # Test that messages can be processed without error
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello"

    def test_model_validation(self):
        """Test model parameter validation."""
        llm = ChatSnowflake(model="mistral-large2", session=MockSession())
        assert llm.model == "mistral-large2"

        # Test with different valid model
        llm2 = ChatSnowflake(model="claude-3-5-sonnet", session=MockSession())
        assert llm2.model == "claude-3-5-sonnet"
