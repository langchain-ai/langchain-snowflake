"""Agent execution functionality for Snowflake Cortex Agents.

This module provides agent execution within threads using the REST API.
"""

import logging
from typing import Any, Dict

from .._connection import RestApiClient, RestApiRequestBuilder

logger = logging.getLogger(__name__)


class RunManagement:
    """Mixin class providing agent execution functionality within threads."""

    # ============================================================================
    # Request Configuration Methods
    # ============================================================================

    def _build_thread_run_request_config(
        self,
        thread_id: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build request configuration for agent execution within a thread.

        Args:
            thread_id: Thread ID where the agent should run
            payload: Request payload containing the query and parameters

        Returns:
            Request configuration dictionary
        """
        session = self._get_session()

        # Use the thread-based run endpoint
        return RestApiRequestBuilder.thread_run_request(
            session=session,
            thread_id=thread_id,
            method="POST",
            payload=payload,
            request_timeout=self.request_timeout,
            verify_ssl=self.verify_ssl,
        )

    # ============================================================================
    # Agent Execution Operations (Sync)
    # ============================================================================

    def run_in_thread(self, thread_id: str, query: str, **kwargs) -> Dict[str, Any]:
        """Execute agent within a specific thread.

        Args:
            thread_id: Thread ID where the agent should run
            query: Query/message to send to the agent
            **kwargs: Additional parameters for the run

        Returns:
            Agent response data

        Raises:
            SnowflakeRestApiError: If agent execution fails
        """
        payload = {"query": query, **kwargs}

        request_config = self._build_thread_run_request_config(thread_id, payload)
        operation_name = f"run agent '{self.name}' in thread '{thread_id}'"

        # RestApiClient handles all error cases internally
        return RestApiClient.make_sync_request(request_config, operation_name)

    # ============================================================================
    # Agent Execution Operations (Async)
    # ============================================================================

    async def run_in_thread_async(self, thread_id: str, query: str, **kwargs) -> Dict[str, Any]:
        """Execute agent within a specific thread (async).

        Args:
            thread_id: Thread ID where the agent should run
            query: Query/message to send to the agent
            **kwargs: Additional parameters for the run

        Returns:
            Agent response data

        Raises:
            SnowflakeRestApiError: If agent execution fails
        """
        payload = {"query": query, **kwargs}

        request_config = self._build_thread_run_request_config(thread_id, payload)
        operation_name = f"run agent '{self.name}' in thread '{thread_id}'"

        # RestApiClient handles all error cases internally
        return await RestApiClient.make_async_request(request_config, operation_name)
