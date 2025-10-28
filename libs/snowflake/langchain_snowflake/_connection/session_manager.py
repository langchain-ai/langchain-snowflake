"""Centralized Snowflake session creation and management utilities."""

import json
import logging
from typing import Any, Dict, Optional

from snowflake.snowpark import Session

from .._error_handling import SnowflakeErrorHandler

logger = logging.getLogger(__name__)


class SnowflakeSessionManager:
    """Centralized Snowflake session creation and management."""

    @staticmethod
    def _get_package_version() -> str:
        """Get the current package version for query tagging."""
        try:
            import importlib.metadata

            return importlib.metadata.version("langchain-snowflake")
        except Exception as e:
            # Log version detection failure but continue with fallback
            SnowflakeErrorHandler.log_error("detect package version", e)
            version = "0.2.1"  # Fallback version
            return version

    @staticmethod
    def _create_query_tag() -> str:
        """Create a standardized query tag for Snowflake sessions.

        Returns:
            Query tag string (JSON format for tracking)
        """
        # Get actual package version
        version_str = SnowflakeSessionManager._get_package_version()

        # Parse version string (e.g., "0.2.1" -> {"major": 0, "minor": 2})
        try:
            version_parts = version_str.split(".")
            major = int(version_parts[0]) if len(version_parts) > 0 else 0
            minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        except (ValueError, IndexError):
            # Fallback if version parsing fails
            major, minor = 0, 2

        # Create custom tracking tag
        name = "langchain_snowflake_usage"
        tag_dict = {"origin": "sf_pse", "name": name, "version": {"major": major, "minor": minor}}
        return json.dumps(tag_dict, separators=(",", ":"))

    @staticmethod
    def _validate_connection_params(connection_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize connection parameters.

        Args:
            connection_params: Raw connection parameters

        Returns:
            Validated connection parameters

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Use centralized validation utilities
        from .._validation_utils import SnowflakeValidationUtils

        return SnowflakeValidationUtils.validate_connection_params(connection_params)

    @staticmethod
    def _normalize_connection_params(connection_params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize connection parameters for Snowpark Session.

        Args:
            connection_params: Connection parameters

        Returns:
            Normalized parameters suitable for Session.builder.configs()
        """
        normalized = connection_params.copy()

        # Ensure required parameters are strings
        if "account" in normalized:
            normalized["account"] = str(normalized["account"]).strip()
        if "user" in normalized:
            normalized["user"] = str(normalized["user"]).strip()

        # Handle optional parameters
        if "database" in normalized and normalized["database"]:
            normalized["database"] = str(normalized["database"]).strip()
        if "schema" in normalized and normalized["schema"]:
            normalized["schema"] = str(normalized["schema"]).strip()
        if "warehouse" in normalized and normalized["warehouse"]:
            normalized["warehouse"] = str(normalized["warehouse"]).strip()
        if "role" in normalized and normalized["role"]:
            normalized["role"] = str(normalized["role"]).strip()

        return normalized

    @staticmethod
    def create_session(connection_params: Dict[str, Any]) -> Session:
        """Create a Snowflake session with standardized configuration.

        Args:
            connection_params: Dictionary containing connection parameters

        Returns:
            Configured Snowflake session

        Raises:
            ValueError: If connection parameters are invalid
            Exception: If session creation fails

        Example:
            >>> params = {
            ...     "account": "myaccount",
            ...     "user": "myuser",
            ...     "password": "mypassword",
            ...     "database": "mydatabase",
            ...     "schema": "myschema"
            ... }
            >>> session = SnowflakeSessionManager.create_session(params)
        """
        try:
            # Validate connection parameters
            validated_params = SnowflakeSessionManager._validate_connection_params(connection_params)

            # Normalize parameters
            config = SnowflakeSessionManager._normalize_connection_params(validated_params)

            # Create session
            session = Session.builder.configs(config).create()
            # Apply query tag
            query_tag = SnowflakeSessionManager._create_query_tag()
            session.query_tag = query_tag

            SnowflakeErrorHandler.log_debug("session creation", f"created Snowflake session: {query_tag}")
            return session

        except Exception as e:
            # Use centralized error handling for session creation failures
            SnowflakeErrorHandler.log_and_raise(e, "create Snowflake session")

    @staticmethod
    def create_session_from_config_file(config_file_path: str) -> Session:
        """Create a session from a JSON configuration file.

        Args:
            config_file_path: Path to JSON configuration file

        Returns:
            Configured Snowflake session

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
            Exception: If session creation fails
        """
        try:
            with open(config_file_path, "r") as f:
                config = json.load(f)

            return SnowflakeSessionManager.create_session(config)

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            SnowflakeErrorHandler.log_and_raise(e, f"create session from config file {config_file_path}")

    @staticmethod
    def test_session_connection(session: Session) -> bool:
        """Test if a session connection is working.

        Args:
            session: Snowflake session to test

        Returns:
            True if connection is working, False otherwise
        """
        try:
            # Simple query to test connection
            result = session.sql("SELECT 1 as test").collect()
            return len(result) > 0 and result[0]["TEST"] == 1

        except Exception:
            return False

    @staticmethod
    def get_session_info(session: Session) -> Dict[str, Any]:
        """Get information about the current session.

        Args:
            session: Snowflake session

        Returns:
            Dictionary containing session information
        """
        try:
            info = {}

            # Get current context
            context_result = session.sql(
                "SELECT CURRENT_ACCOUNT(), CURRENT_USER(), CURRENT_DATABASE(), "
                "CURRENT_SCHEMA(), CURRENT_WAREHOUSE(), CURRENT_ROLE()"
            ).collect()

            if context_result:
                row = context_result[0]
                info.update(
                    {
                        "account": row[0],
                        "user": row[1],
                        "database": row[2],
                        "schema": row[3],
                        "warehouse": row[4],
                        "role": row[5],
                    }
                )

            # Get session ID if available
            try:
                session_result = session.sql("SELECT CURRENT_SESSION()").collect()
                if session_result:
                    info["session_id"] = session_result[0][0]
            except Exception:
                pass  # Session ID not critical

            return info

        except Exception as e:
            SnowflakeErrorHandler.log_error("get session info", e)
            return {"error": str(e)}

    @staticmethod
    def build_connection_config(
        account: str,
        user: str,
        password: Optional[str] = None,
        token: Optional[str] = None,
        private_key_path: Optional[str] = None,
        private_key_passphrase: Optional[str] = None,
        warehouse: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        role: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build connection configuration dictionary from parameters.

        Args:
            account: Snowflake account identifier
            user: Username for authentication
            password: Password for authentication (optional)
            token: OAuth token for authentication (optional)
            private_key_path: Path to private key file (optional)
            private_key_passphrase: Private key passphrase (optional)
            warehouse: Default warehouse (optional)
            database: Default database (optional)
            schema: Default schema (optional)
            role: Default role (optional)
            **kwargs: Additional connection parameters

        Returns:
            Dictionary containing connection configuration

        Raises:
            ValueError: If required parameters are missing
        """
        # Build base configuration
        config = {
            "account": account,
            "user": user,
        }

        # Add authentication method
        if password:
            config["password"] = password
        elif token:
            config["token"] = token
        elif private_key_path:
            config["private_key_path"] = private_key_path
            if private_key_passphrase:
                config["private_key_passphrase"] = private_key_passphrase

        # Add optional parameters
        if warehouse:
            config["warehouse"] = warehouse
        if database:
            config["database"] = database
        if schema:
            config["schema"] = schema
        if role:
            config["role"] = role

        # Add any additional parameters
        config.update(kwargs)

        return config

    @staticmethod
    def get_or_create_session(
        existing_session: Optional[Session] = None,
        cached_session: Optional[Session] = None,
        **connection_params: Any,
    ) -> Session:
        """Get existing session or create a new one.

        Args:
            existing_session: Existing session to use if available
            cached_session: Cached session to reuse if valid
            **connection_params: Parameters for creating new session

        Returns:
            Snowflake session (existing, cached, or newly created)

        Raises:
            ValueError: If session creation fails
        """
        # Use existing session if provided and valid
        if existing_session:
            try:
                if SnowflakeSessionManager.test_session_connection(existing_session):
                    SnowflakeErrorHandler.log_debug("session management", "using provided existing session")
                    return existing_session
            except Exception:
                SnowflakeErrorHandler.log_debug("session management", "existing session is invalid, creating new one")

        # Use cached session if available and valid
        if cached_session:
            try:
                if SnowflakeSessionManager.test_session_connection(cached_session):
                    SnowflakeErrorHandler.log_debug("session management", "using cached session")
                    return cached_session
            except Exception:
                SnowflakeErrorHandler.log_debug("session management", "cached session is invalid, creating new one")

        # Create new session from connection parameters
        if connection_params:
            SnowflakeErrorHandler.log_debug("session management", "creating new session from parameters")
            return SnowflakeSessionManager.create_session(connection_params)

        # If no parameters provided, use centralized error handling
        error = ValueError("No valid session available and no connection parameters provided")
        SnowflakeErrorHandler.log_and_raise(error, "get or create session")
