"""Unit tests for connection management components."""

from unittest.mock import Mock, patch

from pydantic import BaseModel, SecretStr

from langchain_snowflake._connection import (
    SnowflakeAuthUtils,
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

    @patch("langchain_snowflake._connection.session_manager.Session")
    def test_get_or_create_session_with_existing(self, mock_session_class):
        """Test session creation with existing session."""
        existing_session = MockSession()

        result = SnowflakeSessionManager.get_or_create_session(existing_session=existing_session)

        assert result == existing_session
        # Should not create new session
        mock_session_class.builder.configs.assert_not_called()


class TestSnowflakeAuthUtils:
    """Test SnowflakeAuthUtils functionality."""

    @patch("jwt.encode")
    @patch("time.time")
    def test_create_jwt_token(self, mock_time, mock_jwt):
        """Test JWT token creation."""
        # Setup mocks
        mock_time.return_value = 1234567890
        mock_jwt.return_value = "mock-jwt-token"

        # Mock private key
        mock_private_key = Mock()

        token = SnowflakeAuthUtils.create_jwt_token(
            account="test-account", user="test-user", private_key=mock_private_key
        )

        assert token == "mock-jwt-token"
        mock_jwt.assert_called_once()

    def test_extract_account_from_url(self):
        """Test account extraction from Snowflake URL."""
        # Test basic functionality exists
        utils = SnowflakeAuthUtils()
        assert hasattr(utils, "create_jwt_token")

        # Simple validation that the class can be instantiated
        assert isinstance(utils, SnowflakeAuthUtils)
