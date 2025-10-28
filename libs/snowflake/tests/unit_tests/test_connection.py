"""Unit tests for connection management components."""

from unittest.mock import patch

from pydantic import BaseModel, SecretStr

from langchain_snowflake._connection import (
    SnowflakeConnectionMixin,
    SnowflakeSessionManager,
)


class MockSession:
    """Mock Snowflake session for testing."""

    def __init__(self):
        self.account = "test-account"
        self.user = "test-user"


class ConnectionMixinTestImpl(SnowflakeConnectionMixin, BaseModel):
    """Test class that implements SnowflakeConnectionMixin."""

    pass


class TestSnowflakeConnectionMixin:
    """Test SnowflakeConnectionMixin functionality."""

    def test_connection_mixin_initialization(self):
        """Test mixin initializes with connection parameters."""
        mixin = ConnectionMixinTestImpl(
            account="test-account",
            user="test-user",
            password=SecretStr("test-password"),
            warehouse="test-warehouse",
        )

        assert mixin.account == "test-account"
        assert mixin.user == "test-user"
        assert mixin.password.get_secret_value() == "test-password"
        assert mixin.warehouse == "test-warehouse"
        assert mixin._session is None

    @patch("langchain_snowflake._connection.session_manager.SnowflakeSessionManager.get_or_create_session")
    def test_get_session(self, mock_get_or_create):
        """Test _get_session method."""
        mock_session = MockSession()
        mock_get_or_create.return_value = mock_session

        mixin = ConnectionMixinTestImpl(account="test-account")
        session = mixin._get_session()

        assert session == mock_session
        mock_get_or_create.assert_called_once()


class TestSnowflakeSessionManager:
    """Test SnowflakeSessionManager functionality."""

    def test_build_connection_config(self):
        """Test connection configuration building."""
        config = SnowflakeSessionManager.build_connection_config(
            account="test-account",
            user="test-user",
            password="test-password",
            warehouse="test-warehouse",
        )

        assert config["account"] == "test-account"
        assert config["user"] == "test-user"
        assert config["password"] == "test-password"
        assert config["warehouse"] == "test-warehouse"

    @patch("langchain_snowflake._connection.session_manager.SnowflakeSessionManager.test_session_connection")
    @patch("langchain_snowflake._connection.session_manager.Session")
    def test_get_or_create_session_with_existing(self, mock_session_class, mock_test_connection):
        """Test session creation with existing session."""
        existing_session = MockSession()
        mock_test_connection.return_value = True  # Mock successful connection test

        result = SnowflakeSessionManager.get_or_create_session(existing_session=existing_session)

        assert result == existing_session
        # Should not create new session
        mock_session_class.builder.configs.assert_not_called()
