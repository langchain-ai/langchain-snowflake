"""Authentication utilities for Snowflake REST API and connections."""

import logging
from typing import Any, Dict, Optional

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


class SnowflakeAuthUtils:
    """Shared authentication utilities for Snowflake REST API and SQL connections."""

    @staticmethod
    def create_jwt_token(
        account: str,
        user: str,
        private_key: Any,
        passphrase: Optional[str] = None,
        expiry_hours: int = 1,
    ) -> str:
        """Create JWT token for REST API authentication.

        Args:
            account: Snowflake account identifier
            user: Snowflake username
            private_key: Private key data (string, bytes, or key object)
            passphrase: Optional passphrase for private key
            expiry_hours: Token expiry time in hours (default: 1)

        Returns:
            JWT token string

        Raises:
            ValueError: If JWT token creation fails
        """
        try:
            import time

            import jwt
            from cryptography.hazmat.primitives import serialization

            # Prepare claims
            now = int(time.time())
            payload = {
                "iss": f"{account}.{user}".upper(),
                "sub": user.upper(),
                "iat": now,
                "exp": now + (expiry_hours * 3600),  # Configurable expiry
            }

            # Load private key with optional passphrase
            if isinstance(private_key, (str, bytes)):
                if isinstance(private_key, str):
                    private_key = private_key.encode()

                password = passphrase.encode() if passphrase else None
                private_key_obj = serialization.load_pem_private_key(private_key, password=password)
            else:
                private_key_obj = private_key

            # Create JWT
            token = jwt.encode(payload, private_key_obj, algorithm="RS256")
            return token

        except Exception as e:
            logger.error(f"Error creating JWT token: {e}")
            raise ValueError(f"Failed to create JWT token: {e}")

    @staticmethod
    def get_rest_api_headers(
        session: Session,
        account: Optional[str] = None,
        user: Optional[str] = None,
        token: Optional[str] = None,
        private_key_path: Optional[str] = None,
        private_key_passphrase: Optional[str] = None,
    ) -> Dict[str, str]:
        """Get REST API headers with authentication.

        Supports multiple authentication methods in priority order:
        1. Personal Access Token (PAT)
        2. Key-pair authentication (JWT)
        3. Session-based token extraction (fallback)

        Args:
            session: Active Snowflake session
            account: Snowflake account identifier (for JWT)
            user: Snowflake username (for JWT)
            token: Personal Access Token (highest priority)
            private_key_path: Path to private key file (for JWT)
            private_key_passphrase: Private key passphrase (for JWT)

        Returns:
            Dictionary of HTTP headers for REST API requests

        Raises:
            ValueError: If no valid authentication method is available
        """
        # Base headers
        base_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "langchain-snowflake-chat",
        }

        # Method 1: Personal Access Token (PAT) - highest priority
        if token:
            return {**base_headers, "Authorization": f"Bearer {token}"}

        # Method 2: Key-pair authentication (JWT)
        if private_key_path and account and user:
            try:
                # Read private key from file
                with open(private_key_path, "rb") as key_file:
                    private_key_data = key_file.read()

                # Create JWT token
                jwt_token = SnowflakeAuthUtils.create_jwt_token(
                    account=account,
                    user=user,
                    private_key=private_key_data,
                    passphrase=private_key_passphrase,
                )

                return {**base_headers, "Authorization": f"Bearer {jwt_token}"}

            except Exception as e:
                logger.error(f"Error creating JWT from key-pair: {e}")
                raise ValueError(f"Failed to create JWT token from private key: {e}")

        # Method 3: Fallback to session-based token extraction (for compatibility)
        try:
            conn = session._conn._conn
            rest = conn.rest

            # Extract token from session if available
            if rest and hasattr(rest, "token") and rest.token:
                return {
                    **base_headers,
                    "Authorization": f'Snowflake Token="{rest.token}"',
                }

        except Exception as e:
            logger.debug(f"Error extracting token from session: {e}")

        # No valid authentication found
        raise ValueError(
            "Unable to authenticate REST API request. Please provide either:\n"
            "1. Personal Access Token (PAT) via 'token' parameter\n"
            "2. Key-pair authentication via 'private_key_path', 'account', and 'user' parameters\n"
            "3. A properly authenticated Snowpark session"
        )

    @staticmethod
    def build_rest_api_url(session: Session) -> str:
        """Build the REST API URL for Snowflake Cortex Complete.

        Args:
            session: Active Snowflake session

        Returns:
            Complete REST API URL

        Raises:
            ValueError: If URL cannot be constructed
        """
        try:
            # Use session method to get account (more reliable)
            account = session.get_current_account()
            
            # Handle None case
            if not account:
                raise ValueError("Unable to retrieve account information from session")
            
            # Remove quotes if present (Snowflake returns quoted account names)
            if account.startswith('"') and account.endswith('"'):
                account = account[1:-1]
            
            # Extract account name (remove region/cloud info if present)
            account_name = account.split(".")[0] if "." in account else account

            # Build URL with correct hostname format
            base_url = f"https://{account_name}.snowflakecomputing.com"
            return f"{base_url}/api/v2/cortex/inference:complete"

        except Exception as e:
            # Use centralized error handling
            from .._error_handling import SnowflakeErrorHandler
            SnowflakeErrorHandler.log_and_raise(e, "build REST API URL")

    @staticmethod
    def get_effective_timeout(
        session: Session,
        request_timeout: int = 30,
        respect_session_timeout: bool = True,
    ) -> int:
        """Get effective timeout respecting Snowflake parameter hierarchy.

        This follows Snowflake's official parameter management best practices:
        1. Check session STATEMENT_TIMEOUT_IN_SECONDS parameter first
        2. Fall back to tool-specific request_timeout

        Args:
            session: Active Snowflake session
            request_timeout: Default HTTP request timeout in seconds
            respect_session_timeout: Whether to check session parameters

        Returns:
            Effective timeout in seconds
        """
        if not respect_session_timeout:
            return request_timeout

        try:
            # Query session parameter for statement timeout
            result = session.sql("SHOW PARAMETERS LIKE 'STATEMENT_TIMEOUT_IN_SECONDS'").collect()

            if result and len(result) > 0:
                timeout_value = result[0]["value"]
                # Convert to int, 0 means disabled (unlimited)
                session_timeout = int(timeout_value)
                if session_timeout > 0:
                    logger.debug(f"Using session STATEMENT_TIMEOUT_IN_SECONDS: {session_timeout}")
                    return session_timeout

        except Exception as e:
            logger.debug(f"Could not retrieve session timeout parameter: {e}")

        logger.debug(f"Using default request timeout: {request_timeout}")
        return request_timeout
