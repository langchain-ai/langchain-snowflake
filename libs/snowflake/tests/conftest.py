"""Shared test fixtures and configuration."""

from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_snowflake_session():
    """Mock Snowflake session for all tests."""
    session = Mock()
    session.sql.return_value.collect.return_value = [["test_result"]]
    session.account = "test-account"
    session.user = "test-user"
    return session


@pytest.fixture(autouse=True)
def disable_network_calls():
    """Automatically disable network calls for unit tests."""
    # This fixture runs automatically for all tests
    # pytest-socket is configured in pyproject.toml to disable network calls
    pass
