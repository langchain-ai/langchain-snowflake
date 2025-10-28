"""Agent management functionality for Snowflake Cortex Agents.

This module provides CRUD operations for managing Cortex Agents via the REST API.
"""

from typing import Any, Dict, List, Optional

from .._connection import RestApiClient, RestApiRequestBuilder
from .schemas import AgentCreateInput, AgentUpdateInput


class AgentManagement:
    """Mixin class providing agent management functionality."""

    # ============================================================================
    # Request Configuration Methods
    # ============================================================================

    def _build_agent_request_config(
        self, method: str, name: Optional[str] = None, payload: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Build request configuration for agent operations.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            name: Agent name (None for creation/listing, self.name for specific operations)
            payload: Request payload (None for GET/DELETE)

        Returns:
            Request configuration dictionary
        """
        session = self._get_session()

        # Use provided name or default to self.name for specific operations
        target_name = name if name is not None else getattr(self, "name", "")

        return RestApiRequestBuilder.agent_request(
            session=session,
            database=self.database,
            schema=self.schema,
            name=target_name,
            method=method,
            payload=payload,
            request_timeout=self.request_timeout,
            verify_ssl=self.verify_ssl,
        )

    def _process_agent_list_response(self, response_data: Any) -> List[Dict[str, Any]]:
        """Process agent list response.

        Args:
            response_data: Response data from API (can be dict or list)

        Returns:
            List of agent information dictionaries
        """
        # Handle both dict format {"agents": [...]} and direct list format [...]
        if isinstance(response_data, list):
            return response_data
        elif isinstance(response_data, dict):
            return response_data.get("agents", [])
        else:
            return []

    # ============================================================================
    # CRUD Operations (Sync)
    # ============================================================================

    def create_agent(self, agent_config: AgentCreateInput) -> Dict[str, Any]:
        """Create a new Cortex Agent.

        Args:
            agent_config: Agent configuration following Snowflake schema

        Returns:
            Dict containing agent creation response

        Raises:
            SnowflakeRestApiError: If agent creation fails
        """
        payload = agent_config.model_dump(exclude_none=True)
        request_config = self._build_agent_request_config("POST", name="", payload=payload)
        operation_name = f"create agent '{agent_config.name}'"

        return RestApiClient.make_sync_request(request_config, operation_name)

    def describe_agent(self) -> Dict[str, Any]:
        """Get agent description and configuration.

        Returns:
            Dict containing agent details

        Raises:
            SnowflakeRestApiError: If agent retrieval fails
        """
        request_config = self._build_agent_request_config("GET")
        operation_name = f"describe agent '{self.name}'"

        return RestApiClient.make_sync_request(request_config, operation_name)

    def update_agent(self, agent_updates: AgentUpdateInput) -> Dict[str, Any]:
        """Update agent configuration.

        Args:
            agent_updates: Agent updates following Snowflake schema

        Returns:
            Dict containing agent update response

        Raises:
            SnowflakeRestApiError: If agent update fails
        """
        payload = agent_updates.model_dump(exclude_none=True)
        request_config = self._build_agent_request_config("PUT", payload=payload)
        operation_name = f"update agent '{self.name}'"

        return RestApiClient.make_sync_request(request_config, operation_name)

    def delete_agent(self) -> bool:
        """Delete the agent.

        Returns:
            True if deletion was successful

        Raises:
            SnowflakeRestApiError: If agent deletion fails
        """
        request_config = self._build_agent_request_config("DELETE")
        operation_name = f"delete agent '{self.name}'"

        RestApiClient.make_sync_request(request_config, operation_name)
        return True

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents in the schema.

        Returns:
            List of agent information dictionaries

        Raises:
            SnowflakeRestApiError: If agent listing fails
        """
        request_config = self._build_agent_request_config("GET", name="")
        operation_name = f"list agents in {self.database}.{self.schema}"

        response_data = RestApiClient.make_sync_request(request_config, operation_name)
        return self._process_agent_list_response(response_data)

    # ============================================================================
    # CRUD Operations (Async)
    # ============================================================================

    async def create_agent_async(self, agent_config: AgentCreateInput) -> Dict[str, Any]:
        """Create a new Cortex Agent (async).

        Args:
            agent_config: Agent configuration following Snowflake schema

        Returns:
            Dict containing agent creation response

        Raises:
            SnowflakeRestApiError: If agent creation fails
        """
        payload = agent_config.model_dump(exclude_none=True)
        request_config = self._build_agent_request_config("POST", name="", payload=payload)
        operation_name = f"create agent '{agent_config.name}' (async)"

        return await RestApiClient.make_async_request(request_config, operation_name)

    async def describe_agent_async(self) -> Dict[str, Any]:
        """Get agent description and configuration (async).

        Returns:
            Dict containing agent details

        Raises:
            SnowflakeRestApiError: If agent retrieval fails
        """
        request_config = self._build_agent_request_config("GET")
        operation_name = f"describe agent '{self.name}' (async)"

        return await RestApiClient.make_async_request(request_config, operation_name)

    async def update_agent_async(self, agent_updates: AgentUpdateInput) -> Dict[str, Any]:
        """Update agent configuration (async).

        Args:
            agent_updates: Agent updates following Snowflake schema

        Returns:
            Dict containing agent update response

        Raises:
            SnowflakeRestApiError: If agent update fails
        """
        payload = agent_updates.model_dump(exclude_none=True)
        request_config = self._build_agent_request_config("PUT", payload=payload)
        operation_name = f"update agent '{self.name}' (async)"

        return await RestApiClient.make_async_request(request_config, operation_name)

    async def delete_agent_async(self) -> bool:
        """Delete the agent (async).

        Returns:
            True if deletion was successful

        Raises:
            SnowflakeRestApiError: If agent deletion fails
        """
        request_config = self._build_agent_request_config("DELETE")
        operation_name = f"delete agent '{self.name}' (async)"

        await RestApiClient.make_async_request(request_config, operation_name)
        return True

    async def list_agents_async(self) -> List[Dict[str, Any]]:
        """List all agents in the schema (async).

        Returns:
            List of agent information dictionaries

        Raises:
            SnowflakeRestApiError: If agent listing fails
        """
        request_config = self._build_agent_request_config("GET", name="")
        operation_name = f"list agents in {self.database}.{self.schema} (async)"

        response_data = await RestApiClient.make_async_request(request_config, operation_name)
        return self._process_agent_list_response(response_data)
