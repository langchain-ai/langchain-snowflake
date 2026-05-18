"""Base classes and mixins for Snowflake connection management."""

import logging
from typing import Any, Optional

from pydantic import ConfigDict, Field, SecretStr
from snowflake.snowpark import Session

from .._error_handling import SnowflakeErrorHandler
from .session_manager import SnowflakeSessionManager

logger = logging.getLogger(__name__)


class SnowflakeConnectionMixin:
    """Mixin providing common Snowflake connection functionality.

    This mixin provides:
    - Shared Pydantic field definitions for connection parameters
    - Common session management logic
    - Standardized connection configuration building

    Classes inheriting from this mixin get consistent connection handling
    across the langchain-snowflake package.
    """

    model_config = ConfigDict(populate_by_name=True)
    # Shared Pydantic field definitions
    session: Any = Field(default=None, exclude=True, description="Active Snowflake session")
    account: Optional[str] = Field(default=None, description="Snowflake account identifier")
    user: Optional[str] = Field(default=None, description="Snowflake username")
    password: Optional[SecretStr] = Field(default=None, description="Snowflake password")
    token: Optional[str] = Field(default=None, description="OAuth token for authentication")
    private_key_path: Optional[str] = Field(default=None, description="Path to private key file")
    private_key_passphrase: Optional[str] = Field(default=None, description="Private key passphrase")
    warehouse: Optional[str] = Field(default=None, description="Snowflake warehouse")
    database: Optional[str] = Field(default=None, alias="database", description="Snowflake database")
    snowflake_schema: Optional[str] = Field(default=None, alias="schema", description="Snowflake schema")

    # Timeout configuration - follows Snowflake parameter hierarchy
    request_timeout: int = Field(default=30, description="HTTP request timeout in seconds (default: 30)")
    respect_session_timeout: bool = Field(
        default=True,
        description="Whether to respect session STATEMENT_TIMEOUT_IN_SECONDS parameter (default: True)",
    )

    # SSL configuration
    verify_ssl: bool = Field(default=True, description="Whether to verify SSL certificates for HTTPS requests")

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the mixin with session caching."""
        super().__init__(**kwargs)
        self._session = None

    def _get_session(self) -> Session:
        """Get or create Snowflake session using centralized session manager.

        This method eliminates 150+ lines of duplicated session creation logic
        across the tools.py file by using the shared SnowflakeSessionManager.

        Returns:
            Active Snowflake session

        Raises:
            ValueError: If session creation fails
        """
        session = SnowflakeSessionManager.get_or_create_session(
            existing_session=self.session,
            cached_session=self._session,
            account=self.account,
            user=self.user,
            password=self.password,
            token=getattr(self, "token", None),
            private_key_path=getattr(self, "private_key_path", None),
            private_key_passphrase=getattr(self, "private_key_passphrase", None),
            warehouse=self.warehouse,
            database=self.database,
            schema=self.snowflake_schema,
        )

        # Cache the session for reuse
        self._session = session
        return session

    def _build_connection_config(self) -> dict:
        """Build connection configuration dictionary from instance attributes.

        Returns:
            Connection configuration dictionary
        """
        return SnowflakeSessionManager.build_connection_config(
            account=self.account,
            user=self.user,
            password=self.password,
            token=getattr(self, "token", None),
            private_key_path=getattr(self, "private_key_path", None),
            private_key_passphrase=getattr(self, "private_key_passphrase", None),
            warehouse=self.warehouse,
            database=self.database,
            schema=self.snowflake_schema,
        )

    def _get_effective_timeout(self) -> int:
        """Get effective timeout respecting Snowflake parameter hierarchy.

        Returns:
            Effective timeout in seconds following Snowflake best practices:
            1. Session STATEMENT_TIMEOUT_IN_SECONDS (if enabled and > 0)
            2. Tool-specific request_timeout (fallback)
        """
        from .auth_utils import SnowflakeAuthUtils

        session = self._get_session()
        return SnowflakeAuthUtils.get_effective_timeout(
            session=session,
            request_timeout=self.request_timeout,
            respect_session_timeout=self.respect_session_timeout,
        )

    def _count_tokens(self, text: str, model: str = "llama3.1-70b") -> int:
        """Count tokens using official Snowflake CORTEX.COUNT_TOKENS function.

        Args:
            text: Text to count tokens for
            model: Model name to use for tokenization (affects token count)

        Returns:
            Number of tokens according to the specified model's tokenizer

        Note:
            Uses SNOWFLAKE.CORTEX.COUNT_TOKENS for accurate counting.
            Falls back to word-count estimation if Cortex function fails.
        """
        session = self._get_session()

        try:
            # Use official Snowflake tokenizer
            escaped_text = text.replace("'", "''")
            sql = f"""
            SELECT SNOWFLAKE.CORTEX.COUNT_TOKENS(
                '{model}',
                '{escaped_text}'
            ) as token_count
            """

            result = session.sql(sql).collect()

            if result and len(result) > 0:
                token_count = result[0]["TOKEN_COUNT"]
                return int(token_count) if token_count is not None else len(text.split())
            else:
                # Fallback to word count estimation
                return len(text.split())

        except Exception as e:
            # Fallback to word count estimation if Cortex function unavailable
            SnowflakeErrorHandler.log_debug(
                "token counting fallback", f"Could not use CORTEX.COUNT_TOKENS, falling back to word count: {e}", logger
            )
            return len(text.split())
