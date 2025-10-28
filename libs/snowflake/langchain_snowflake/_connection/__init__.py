"""Shared connection and authentication utilities for langchain-snowflake.

This internal module provides centralized connection management, session creation,
and authentication utilities that are used across the langchain-snowflake package.

Key Components:
- SnowflakeConnectionMixin: Base mixin for consistent connection handling
- SnowflakeSessionManager: Centralized session creation and management
- SnowflakeAuthUtils: Authentication utilities for REST API and SQL connections
"""

from .auth_utils import SnowflakeAuthUtils
from .base import SnowflakeConnectionMixin
from .rest_client import RestApiClient, RestApiRequestBuilder
from .session_manager import SnowflakeSessionManager
from .sql_client import SqlExecutionClient

__all__ = [
    "SnowflakeConnectionMixin",
    "SnowflakeSessionManager",
    "SnowflakeAuthUtils",
    "RestApiClient",
    "RestApiRequestBuilder",
    "SqlExecutionClient",
]
