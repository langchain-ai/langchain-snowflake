"""Unified REST API client for Snowflake Cortex operations.

This module provides a centralized client for all REST API operations.
"""

import json
import logging
from typing import Any, Dict, Optional
from urllib.parse import quote

import aiohttp
import requests

from .._error_handling import SnowflakeErrorHandler, SnowflakeRestApiErrorHandler
from .auth_utils import SnowflakeAuthUtils

logger = logging.getLogger(__name__)


class RestApiClient:
    """Unified REST API client for all Snowflake Cortex operations.

    This client provides a consistent interface for both sync and async
    REST API operations while preserving existing fallback mechanisms.
    """

    @staticmethod
    def prepare_request(
        session,
        endpoint: str,
        method: str = "POST",
        payload: Optional[Dict] = None,
        url_params: Optional[Dict] = None,
        query_params: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare unified request parameters for any Snowflake REST API endpoint.

        Args:
            session: Active Snowflake session
            endpoint: API endpoint (e.g., "/cortex/threads", "/databases/{db}/schemas/{schema}/agents")
            method: HTTP method (GET, POST, PUT, DELETE)
            payload: Request body payload
            url_params: Parameters for URL templating (e.g., {"db": "mydb", "schema": "myschema"})
            query_params: URL query parameters
            **kwargs: Additional config (request_timeout, verify_ssl, etc.)

        Returns:
            Dict containing all request configuration
        """
        # Build URL based on endpoint type
        if url_params:
            # Handle parameterized URLs like /databases/{db}/schemas/{schema}/agents
            url = RestApiClient._build_parameterized_url(session, endpoint, url_params)
        else:
            # Handle simple endpoints like /cortex/threads
            url = RestApiClient._build_simple_url(session, endpoint)

        # Get authentication headers
        headers = SnowflakeAuthUtils.get_rest_api_headers(
            session=session,
            account=kwargs.get("account"),
            user=kwargs.get("user"),
            token=kwargs.get("token"),
            private_key_path=kwargs.get("private_key_path"),
            private_key_passphrase=kwargs.get("private_key_passphrase"),
        )

        # Prepare unified request configuration
        request_config = {
            "url": url,
            "method": method.upper(),
            "headers": headers,
            "timeout": kwargs.get("request_timeout", 30),
            "verify": kwargs.get("verify_ssl", True),
        }

        # Add payload if provided
        if payload:
            request_config["json"] = payload

        # Add query parameters if provided
        if query_params:
            request_config["params"] = query_params

        return request_config

    @staticmethod
    def make_sync_request(request_config: Dict[str, Any], operation_name: str = "REST API request") -> Dict[str, Any]:
        """Make synchronous REST API request using requests library.

        Args:
            request_config: Request configuration from prepare_request()
            operation_name: Description for error logging

        Returns:
            Parsed JSON response data

        Raises:
            requests.RequestException: If request fails
        """
        method = request_config.pop("method")
        timeout = request_config.pop("timeout")
        verify = request_config.pop("verify")

        try:
            # Log request details for debugging
            logger.debug(f"Making {method} request to: {request_config.get('url', 'Unknown URL')}")
            if "json" in request_config:
                logger.debug(f"Request payload keys: {list(request_config['json'].keys())}")
                if "stream" in request_config["json"]:
                    logger.debug(f"Stream parameter: {request_config['json']['stream']}")

            # Use requests for sync HTTP
            response = getattr(requests, method.lower())(**request_config, timeout=timeout, verify=verify)
            response.raise_for_status()

            # Log response details for debugging
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response content-type: {response.headers.get('content-type', 'Unknown')}")

            # Check for Snowflake request ID in headers (used as run_id for feedback)
            snowflake_request_id = response.headers.get("X-Snowflake-Request-Id")
            if snowflake_request_id:
                logger.debug(f"Snowflake Request ID: {snowflake_request_id}")

            # Use centralized response parsing
            response_data = SnowflakeRestApiErrorHandler.safe_parse_json_response(response, operation_name, logger)

            # Add Snowflake request ID to response data if available (for feedback tracking)
            if snowflake_request_id and isinstance(response_data, dict):
                response_data["_snowflake_request_id"] = snowflake_request_id

            return response_data

        except Exception as e:
            SnowflakeErrorHandler.log_and_raise(e, f"sync {operation_name}")

    @staticmethod
    async def make_async_request(
        request_config: Dict[str, Any], operation_name: str = "REST API request"
    ) -> Dict[str, Any]:
        """Make asynchronous REST API request using aiohttp library.

        Args:
            request_config: Request configuration from prepare_request()
            operation_name: Description for error logging

        Returns:
            Parsed JSON response data

        Raises:
            aiohttp.ClientError: If request fails
        """
        method = request_config.pop("method")
        timeout = aiohttp.ClientTimeout(total=request_config.pop("timeout"))
        ssl = request_config.pop("verify")

        try:
            # Use aiohttp for async HTTP
            async with aiohttp.ClientSession() as client:
                async with getattr(client, method.lower())(**request_config, timeout=timeout, ssl=ssl) as response:
                    response.raise_for_status()

                    # Check for Snowflake request ID in headers (used as run_id for feedback)
                    snowflake_request_id = response.headers.get("X-Snowflake-Request-Id")

                    response_data = await response.json()

                    # Add Snowflake request ID to response data if available (for feedback tracking)
                    if snowflake_request_id and isinstance(response_data, dict):
                        response_data["_snowflake_request_id"] = snowflake_request_id

                    SnowflakeErrorHandler.log_debug(f"async {operation_name}", "completed successfully")
                    return response_data

        except Exception as e:
            SnowflakeErrorHandler.log_and_raise(e, f"async {operation_name}")

    @staticmethod
    def make_sync_streaming_request(request_config: Dict[str, Any], operation_name: str = "streaming REST API request"):
        """Make synchronous streaming REST API request using requests library.

        Args:
            request_config: Request configuration from prepare_request()
            operation_name: Description for error logging

        Yields:
            Parsed streaming chunks

        Raises:
            requests.RequestException: If request fails
        """
        method = request_config.pop("method")
        timeout = request_config.pop("timeout")
        verify = request_config.pop("verify")

        try:
            # Use requests for sync streaming HTTP
            with getattr(requests, method.lower())(
                **request_config, timeout=timeout, verify=verify, stream=True
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]  # Remove 'data: ' prefix
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                chunk_data = json.loads(data_str)
                                # Yield complete chunk data as JSON string for consumer to parse
                                # This preserves all information (thinking, tool_use, etc.)
                                yield json.dumps(chunk_data)
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            SnowflakeErrorHandler.log_and_raise(e, f"sync {operation_name}")

    @staticmethod
    async def make_async_streaming_request(
        request_config: Dict[str, Any], operation_name: str = "streaming REST API request"
    ):
        """Make asynchronous streaming REST API request using aiohttp library.

        Args:
            request_config: Request configuration from prepare_request()
            operation_name: Description for error logging

        Yields:
            Parsed streaming chunks

        Raises:
            aiohttp.ClientError: If request fails
        """
        method = request_config.pop("method")
        timeout = aiohttp.ClientTimeout(total=request_config.pop("timeout"))
        ssl = request_config.pop("verify")

        try:
            # Use aiohttp for async streaming HTTP
            async with aiohttp.ClientSession() as client:
                async with getattr(client, method.lower())(**request_config, timeout=timeout, ssl=ssl) as response:
                    response.raise_for_status()

                    # Process streaming response
                    async for line in response.content:
                        if line:
                            line_str = line.decode("utf-8").strip()
                            if line_str.startswith("data: "):
                                data_str = line_str[6:]  # Remove 'data: ' prefix
                                if data_str.strip() == "[DONE]":
                                    break
                                try:
                                    chunk_data = json.loads(data_str)
                                    # Yield complete chunk data as JSON string for consumer to parse
                                    # This preserves all information (thinking, tool_use, etc.)
                                    yield json.dumps(chunk_data)
                                except json.JSONDecodeError:
                                    continue

        except Exception as e:
            SnowflakeErrorHandler.log_and_raise(e, f"async {operation_name}")

    @staticmethod
    def _build_simple_url(session, endpoint: str) -> str:
        """Build URL for simple endpoints like /cortex/threads."""
        base_url = RestApiClient._get_base_url(session)

        # Ensure endpoint starts with /
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint

        # Ensure endpoint starts with /api/v2
        if not endpoint.startswith("/api/v2"):
            endpoint = "/api/v2" + endpoint

        return base_url + endpoint

    @staticmethod
    def _build_parameterized_url(session, endpoint: str, url_params: Dict[str, str]) -> str:
        """Build URL for parameterized endpoints like /databases/{db}/schemas/{schema}/agents."""
        base_url = RestApiClient._get_base_url(session)

        # Ensure endpoint starts with /
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint

        # Ensure endpoint starts with /api/v2
        if not endpoint.startswith("/api/v2"):
            endpoint = "/api/v2" + endpoint

        # Replace URL parameters with quoted values
        for param, value in url_params.items():
            placeholder = "{" + param + "}"
            if placeholder in endpoint:
                endpoint = endpoint.replace(placeholder, quote(str(value)))

        return base_url + endpoint

    @staticmethod
    def _get_base_url(session) -> str:
        """Get base URL for Snowflake REST API."""
        try:
            # Extract account info from session
            account_info = session.get_current_account()

            # Handle None case
            if not account_info:
                SnowflakeErrorHandler.log_and_raise(
                    ValueError("Unable to retrieve account information from session"), "get base URL"
                )

            # Remove quotes if present (Snowflake returns quoted account names)
            if account_info.startswith('"') and account_info.endswith('"'):
                account_info = account_info[1:-1]

            # Extract account name (remove region/cloud info if present)
            account_name = account_info.split(".")[0] if "." in account_info else account_info

            # Snowflake TLS certificates use hyphens, not underscores, in hostnames.
            account_name = account_name.replace("_", "-")

            # Build base URL following Snowflake REST API patterns
            return f"https://{account_name}.snowflakecomputing.com"

        except Exception as e:
            SnowflakeErrorHandler.log_and_raise(e, "get base URL")


class RestApiRequestBuilder:
    """Helper class for building common REST API request configurations."""

    @staticmethod
    def agent_request(session, database: str, schema: str, name: str, action: str = "", **kwargs) -> Dict[str, Any]:
        """Build request config for agent operations.

        This method handles both management operations and execution.
        - Management operations (create, describe, update, delete, list) use slash prefix
        - Execution operations (run) use colon prefix as per Snowflake API specification
        """
        endpoint = "/databases/{database}/schemas/{schema}/agents"
        if name:
            endpoint += "/{name}"
        if action:
            # Use colon prefix for execution actions (e.g., :run)
            if action in ["run"]:
                endpoint += ":" + action
            else:
                # For management operations, use slash prefix
                endpoint += "/" + action

        url_params = {"database": database, "schema": schema, "name": name}

        return RestApiClient.prepare_request(session=session, endpoint=endpoint, url_params=url_params, **kwargs)

    @staticmethod
    def thread_request(session, thread_id: str = "", **kwargs) -> Dict[str, Any]:
        """Build request config for thread operations."""
        endpoint = "/cortex/threads"
        if thread_id:
            endpoint += "/" + thread_id

        return RestApiClient.prepare_request(session=session, endpoint=endpoint, **kwargs)

    @staticmethod
    def thread_run_request(session, thread_id: str, **kwargs) -> Dict[str, Any]:
        """Build request config for agent execution within threads."""
        endpoint = f"/cortex/threads/{thread_id}/runs"
        return RestApiClient.prepare_request(session=session, endpoint=endpoint, **kwargs)

    @staticmethod
    def feedback_request(session, database: str, schema: str, name: str, **kwargs) -> Dict[str, Any]:
        """Build request config for feedback operations."""
        method = kwargs.get("method", "POST")

        if method != "POST":
            raise ValueError("Only POST method is supported for feedback operations")

        endpoint = "/databases/{database}/schemas/{schema}/agents/{name}:feedback"
        url_params = {"database": database, "schema": schema, "name": name}
        return RestApiClient.prepare_request(session=session, endpoint=endpoint, url_params=url_params, **kwargs)

    @staticmethod
    def cortex_complete_request(session, **kwargs) -> Dict[str, Any]:
        """Build request config for Cortex Complete operations."""
        return RestApiClient.prepare_request(session=session, endpoint="/cortex/inference:complete", **kwargs)

    @staticmethod
    def cortex_analyst_request(session, **kwargs) -> Dict[str, Any]:
        """Build request config for Cortex Analyst operations."""
        return RestApiClient.prepare_request(session=session, endpoint="/cortex/analyst/message", **kwargs)

    @staticmethod
    def cortex_search_request(session, database: str, schema: str, service: str, **kwargs) -> Dict[str, Any]:
        """Build request config for Cortex Search operations."""
        url_params = {"database": database, "schema": schema, "service": service}
        endpoint = "/databases/{database}/schemas/{schema}/cortex-search-services/{service}:query"

        return RestApiClient.prepare_request(session=session, endpoint=endpoint, url_params=url_params, **kwargs)
