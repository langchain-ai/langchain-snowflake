"""Authentication utilities for Snowflake chat models."""

import logging
from typing import Any, Dict

import aiohttp
import requests
from snowflake.snowpark import Session

from .._connection import SnowflakeAuthUtils

logger = logging.getLogger(__name__)


class SnowflakeAuth:
    """Mixin class for Snowflake authentication functionality."""

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

    def _make_rest_api_request(self, payload: Dict[str, Any]) -> requests.Response:
        """Make the actual REST API request using shared utilities.

        This method now uses the centralized SnowflakeAuthUtils.make_rest_api_request
        which handles authentication, URL building, and request execution internally.
        """
        session = self._get_session()

        # Use shared utility for making REST API requests (handles auth internally)
        return SnowflakeAuthUtils.make_rest_api_request(
            session=session,
            payload=payload,
            account=getattr(self, "account", None),
            user=getattr(self, "user", None),
            token=getattr(self, "token", None),
            private_key_path=getattr(self, "private_key_path", None),
            private_key_passphrase=getattr(self, "private_key_passphrase", None),
            request_timeout=getattr(self, "request_timeout", 30),
            verify_ssl=getattr(self, "verify_ssl", True),
            stream=True,  # Enable streaming for tool calling
        )

    async def _make_rest_api_request_async(
        self, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make async REST API request using aiohttp for true async patterns.

        This method provides native async HTTP for chat model REST API calls,
        avoiding the blocking behavior of synchronous requests.

        Returns:
            The JSON response data as a dictionary
        """
        session = self._get_session()

        try:
            # Build URL using shared utilities
            url = (
                SnowflakeAuthUtils.build_rest_api_url(session)
                + "/api/v2/cortex/inference:complete"
            )

            # Get authentication headers using shared utilities
            headers = SnowflakeAuthUtils.get_rest_api_headers(
                session=session,
                account=getattr(self, "account", None),
                user=getattr(self, "user", None),
            )

            # Get timeout configuration
            request_timeout = getattr(self, "request_timeout", 30)
            verify_ssl = getattr(self, "verify_ssl", True)

            # Use aiohttp for true async HTTP
            async with aiohttp.ClientSession() as client:
                async with client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=request_timeout),
                    ssl=verify_ssl,
                ) as response:
                    response.raise_for_status()
                    # Return the JSON data directly since we can't return the response object
                    return await response.json()

        except Exception as e:
            logger.error(f"Async REST API request failed: {e}")
            raise
