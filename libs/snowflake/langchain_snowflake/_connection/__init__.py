"""Shared connection and authentication utilities for langchain-snowflake.

This internal module provides centralized connection management, session creation,
and authentication utilities that are used across the langchain-snowflake package.

Key Components:
- SnowflakeConnectionMixin: Base mixin for consistent connection handling
- SnowflakeSessionManager: Centralized session creation and management
- SnowflakeAuthUtils: Authentication utilities for REST API and SQL connections

This module eliminates 250+ lines of duplicated code across the package and
provides consistent patterns for Snowflake connectivity.
"""

from .auth_utils import SnowflakeAuthUtils
from .base import SnowflakeConnectionMixin
from .session_manager import SnowflakeSessionManager

__all__ = ["SnowflakeConnectionMixin", "SnowflakeSessionManager", "SnowflakeAuthUtils"]
