"""Thread management functionality for Snowflake Cortex Agents.

This module provides thread CRUD operations for conversation management.
"""

from typing import Any, Dict, List, Optional

from .._connection import RestApiClient, RestApiRequestBuilder
from .schemas import ThreadUpdateInput


class ThreadManagement:
    """Mixin class providing thread management functionality."""

    # ============================================================================
    # Request Configuration Methods
    # ============================================================================

    def _build_thread_request_config(
        self, method: str, thread_id: Optional[str] = None, payload: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Build request configuration for thread operations.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            thread_id: Thread ID (for specific thread operations)
            payload: Request payload (None for GET/DELETE)

        Returns:
            Request configuration dictionary
        """
        session = self._get_session()

        return RestApiRequestBuilder.thread_request(
            session=session,
            thread_id=thread_id,
            method=method,
            payload=payload,
            request_timeout=self.request_timeout,
            verify_ssl=self.verify_ssl,
        )

    def _process_thread_list_response(self, response_data: Any) -> List[Dict[str, Any]]:
        """Process thread list response.

        Args:
            response_data: Response data from API (can be list or dict)

        Returns:
            List of thread information dictionaries
        """
        # Handle both list and dictionary formats
        if isinstance(response_data, list):
            return response_data
        elif isinstance(response_data, dict):
            return response_data.get("threads", [])
        else:
            return []

    # ============================================================================
    # CRUD Operations (Sync)
    # ============================================================================

    def create_thread(self, metadata: Optional[Dict] = None) -> str:
        """Create a new conversation thread.

        Args:
            metadata: Optional metadata to associate with the thread

        Returns:
            Thread ID for the created thread

        Raises:
            SnowflakeRestApiError: If thread creation fails
        """
        payload = {}
        if metadata:
            payload["metadata"] = metadata

        request_config = self._build_thread_request_config("POST", payload=payload)
        operation_name = "create thread"

        response_data = RestApiClient.make_sync_request(request_config, operation_name)
        return response_data.get("thread_id")

    def update_thread(self, thread_id: str, updates: ThreadUpdateInput) -> Dict[str, Any]:
        """Update thread configuration.

        Args:
            thread_id: Thread ID to update
            updates: Thread updates following Snowflake schema

        Returns:
            Dict containing updated thread details

        Raises:
            SnowflakeRestApiError: If thread update fails
        """
        payload = updates.model_dump(exclude_none=True)
        request_config = self._build_thread_request_config("POST", thread_id=thread_id, payload=payload)
        operation_name = f"update thread '{thread_id}'"

        return RestApiClient.make_sync_request(request_config, operation_name)

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a conversation thread.

        Args:
            thread_id: Thread ID to delete

        Returns:
            True if deletion was successful

        Raises:
            SnowflakeRestApiError: If thread deletion fails
        """
        request_config = self._build_thread_request_config("DELETE", thread_id=thread_id)
        operation_name = f"delete thread '{thread_id}'"

        RestApiClient.make_sync_request(request_config, operation_name)
        return True

    def list_threads(self) -> List[Dict[str, Any]]:
        """List all conversation threads (global, not agent-specific).

        Returns:
            List of thread information dictionaries

        Raises:
            SnowflakeRestApiError: If thread listing fails
        """
        request_config = self._build_thread_request_config("GET")
        operation_name = "list threads"

        response_data = RestApiClient.make_sync_request(request_config, operation_name)
        return self._process_thread_list_response(response_data)

    def describe_thread(self, thread_id: str) -> Dict[str, Any]:
        """Get thread details and configuration.

        Args:
            thread_id: Thread ID to describe

        Returns:
            Dict containing thread details

        Raises:
            SnowflakeRestApiError: If thread retrieval fails
        """
        request_config = self._build_thread_request_config("GET", thread_id=thread_id)
        operation_name = f"describe thread '{thread_id}'"

        return RestApiClient.make_sync_request(request_config, operation_name)

    # ============================================================================
    # CRUD Operations (Async)
    # ============================================================================

    async def create_thread_async(self, metadata: Optional[Dict] = None) -> str:
        """Create a new conversation thread (async).

        Args:
            metadata: Optional metadata to associate with the thread

        Returns:
            Thread ID for the created thread

        Raises:
            SnowflakeRestApiError: If thread creation fails
        """
        payload = {}
        if metadata:
            payload["metadata"] = metadata

        request_config = self._build_thread_request_config("POST", payload=payload)
        operation_name = "create thread (async)"

        response_data = await RestApiClient.make_async_request(request_config, operation_name)
        return response_data.get("thread_id")

    async def update_thread_async(self, thread_id: str, updates: ThreadUpdateInput) -> Dict[str, Any]:
        """Update thread configuration (async).

        Args:
            thread_id: Thread ID to update
            updates: Thread updates following Snowflake schema

        Returns:
            Dict containing updated thread details

        Raises:
            SnowflakeRestApiError: If thread update fails
        """
        payload = updates.model_dump(exclude_none=True)
        request_config = self._build_thread_request_config("POST", thread_id=thread_id, payload=payload)
        operation_name = f"update thread '{thread_id}' (async)"

        return await RestApiClient.make_async_request(request_config, operation_name)

    async def delete_thread_async(self, thread_id: str) -> bool:
        """Delete a conversation thread (async).

        Args:
            thread_id: Thread ID to delete

        Returns:
            True if deletion was successful

        Raises:
            SnowflakeRestApiError: If thread deletion fails
        """
        request_config = self._build_thread_request_config("DELETE", thread_id=thread_id)
        operation_name = f"delete thread '{thread_id}' (async)"

        await RestApiClient.make_async_request(request_config, operation_name)
        return True

    async def list_threads_async(self) -> List[Dict[str, Any]]:
        """List all conversation threads (global, not agent-specific) (async).

        Returns:
            List of thread information dictionaries

        Raises:
            SnowflakeRestApiError: If thread listing fails
        """
        request_config = self._build_thread_request_config("GET")
        operation_name = "list threads (async)"

        response_data = await RestApiClient.make_async_request(request_config, operation_name)
        return self._process_thread_list_response(response_data)

    async def describe_thread_async(self, thread_id: str) -> Dict[str, Any]:
        """Get thread details and configuration (async).

        Args:
            thread_id: Thread ID to describe

        Returns:
            Dict containing thread details

        Raises:
            SnowflakeRestApiError: If thread retrieval fails
        """
        request_config = self._build_thread_request_config("GET", thread_id=thread_id)
        operation_name = f"describe thread '{thread_id}' (async)"

        return await RestApiClient.make_async_request(request_config, operation_name)
