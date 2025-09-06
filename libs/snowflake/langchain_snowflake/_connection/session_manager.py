"""Centralized Snowflake session creation and management utilities."""

import json
import logging
from typing import Any, Dict, Optional

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


class SnowflakeSessionManager:
    """Centralized Snowflake session creation and management."""

    @staticmethod
    def _create_query_tag() -> str:
        """Create standardized query tag for tracking.
        Returns:
            JSON string with integration metadata
        """
        try:
            # Get package version
            from importlib import metadata

            version = metadata.version("langchain-snowflake")
        except Exception:
            version = "0.2.1"  # Fallback version

        # Create simple tracking tag
        tag_data = {"integration": "langchain-snowflake", "version": version}
        return json.dumps(tag_data, separators=(",", ":"))

    @staticmethod
    def build_connection_config(
        account: Optional[str] = None,
        user: Optional[str] = None,
        password: Any = None,
        token: Optional[str] = None,
        private_key_path: Optional[str] = None,
        private_key_passphrase: Optional[str] = None,
        warehouse: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        **extra_params: Any,
    ) -> Dict[str, Any]:
        """Build standardized connection configuration dictionary.

        Args:
            account: Snowflake account identifier
            user: Snowflake username
            password: Password (can be SecretStr or string)
            token: OAuth token for authentication
            private_key_path: Path to private key file for key-pair auth
            private_key_passphrase: Passphrase for private key
            warehouse: Warehouse to use
            database: Database to connect to
            schema: Schema to use
            **extra_params: Additional connection parameters

        Returns:
            Dictionary of connection parameters for Snowflake session
        """
        config = {}

        # Core connection parameters
        if account:
            config["account"] = account
        if user:
            config["user"] = user
        if password:
            if hasattr(password, "get_secret_value"):
                # Pydantic SecretStr
                config["password"] = password.get_secret_value()
            else:
                # Regular string (for tests and direct usage)
                config["password"] = str(password)
        if token:
            config["token"] = token
        if private_key_path:
            config["private_key_path"] = private_key_path
        if private_key_passphrase:
            config["private_key_passphrase"] = private_key_passphrase

        # Context parameters
        if warehouse:
            config["warehouse"] = warehouse
        if database:
            config["database"] = database
        if schema:
            config["schema"] = schema

        # Additional parameters
        config.update(extra_params)

        return config

    @staticmethod
    def create_session(config: Dict[str, Any]) -> Session:
        """Create a Snowflake session from configuration dictionary.

        Args:
            config: Connection configuration dictionary

        Returns:
            Active Snowflake session

        Raises:
            ValueError: If session creation fails
        """
        try:
            session = Session.builder.configs(config).create()
            # Apply query tag
            query_tag = SnowflakeSessionManager._create_query_tag()
            session.query_tag = query_tag

            logger.debug("Created Snowflake session")
            return session

        except Exception as e:
            raise ValueError(f"Failed to create Snowflake session: {e}")

    @staticmethod
    def get_or_create_session(
        existing_session: Optional[Session] = None,
        cached_session: Optional[Session] = None,
        config: Optional[Dict[str, Any]] = None,
        **connection_params: Any,
    ) -> Session:
        """Get existing session or create new one with fallback logic.

        Priority order:
        1. Use existing_session if provided
        2. Use cached_session if available
        3. Create new session from config
        4. Create new session from connection_params

        Args:
            existing_session: Pre-existing session to use
            cached_session: Cached session to reuse
            config: Pre-built connection configuration
            **connection_params: Individual connection parameters

        Returns:
            Active Snowflake session

        Raises:
            ValueError: If no session can be created
        """
        # Use existing session if available
        if existing_session is not None:
            # Apply query tag if not already set
            if not hasattr(existing_session, "query_tag") or not existing_session.query_tag:
                query_tag = SnowflakeSessionManager._create_query_tag()
                existing_session.query_tag = query_tag
            return existing_session

        # Use cached session if available
        if cached_session is not None:
            # Apply query tag if not already set
            if not hasattr(cached_session, "query_tag") or not cached_session.query_tag:
                query_tag = SnowflakeSessionManager._create_query_tag()
                cached_session.query_tag = query_tag
            return cached_session

        # Build config if not provided
        if config is None:
            config = SnowflakeSessionManager.build_connection_config(**connection_params)

        # Create new session
        return SnowflakeSessionManager.create_session(config)
