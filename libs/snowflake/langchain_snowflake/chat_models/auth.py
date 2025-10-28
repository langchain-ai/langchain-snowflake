"""Authentication utilities for Snowflake chat models."""

import logging
from typing import Any, Dict, Optional

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


class SnowflakeAuth:
    """Mixin class for Snowflake authentication functionality."""

    session: Optional[Session]

    def _get_session(self) -> Session:
        """Get or create Snowflake session using LLM intelligence for credential prioritization."""
        # Use existing session if available
        if self.session is not None:
            return self.session

        # Build connection parameters from provided credentials
        connection_params = {}

        if self.account:
            connection_params["account"] = self.account
        if self.user:
            connection_params["user"] = self.user
        if self.password:
            if hasattr(self.password, "get_secret_value"):
                # Pydantic SecretStr
                connection_params["password"] = self.password.get_secret_value()
            else:
                # Regular string (for tests and direct usage)
                connection_params["password"] = str(self.password)
        if self.token:
            connection_params["token"] = self.token
        if self.private_key_path:
            connection_params["private_key_path"] = self.private_key_path
        if self.private_key_passphrase:
            connection_params["private_key_passphrase"] = self.private_key_passphrase
        if self.warehouse:
            connection_params["warehouse"] = self.warehouse
        if self.database:
            connection_params["database"] = self.database
        if self.schema:
            connection_params["schema"] = self.schema

        try:
            from snowflake.snowpark import Session

            # If we have specific connection parameters, use them
            if connection_params:
                self.session = Session.builder.configs(connection_params).create()
                return self.session

            # Fallback: Use production session management
            try:
                from .. import get_default_session

                logger.info("Using default session creation from langchain_snowflake")
                self.session = get_default_session()
                return self.session
            except Exception as e:
                logger.warning(f"Failed to create default session: {e}")
                # Final fallback to basic session
                self.session = Session.builder.configs({}).create()
                return self.session

        except Exception as e:
            raise ValueError(f"Failed to create Snowflake session: {e}")
