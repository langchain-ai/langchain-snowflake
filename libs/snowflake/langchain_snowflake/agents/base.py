"""Snowflake Cortex Agent implementation.

This module provides the main SnowflakeCortexAgent class that integrates with
Snowflake's Cortex Agents REST API for managed orchestration and conversation management.
"""

import logging
import time
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Sequence, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import Field

from .._connection import RestApiClient, RestApiRequestBuilder, SnowflakeConnectionMixin
from .._error_handling import SnowflakeToolErrorHandler
from .feedback import FeedbackManagement
from .management import AgentManagement
from .runs import RunManagement
from .threads import ThreadManagement

logger = logging.getLogger(__name__)


class SnowflakeCortexAgent(
    Runnable,
    SnowflakeConnectionMixin,
    AgentManagement,
    ThreadManagement,
    RunManagement,
    FeedbackManagement,
):
    """Snowflake Cortex Agent integration with thread management and feedback.

    This class provides access to Snowflake's Cortex Agents REST API, enabling
    managed orchestration of multiple tools with conversation management.

    Setup:
        Install ``langchain-snowflake`` and configure Snowflake connection.

        .. code-block:: bash

            pip install -U langchain-snowflake

    Key init args:
        name: str
            Name of the Cortex Agent in Snowflake
        database: str
            Database containing the agent
        schema: str
            Schema containing the agent
        session: Optional[Session]
            Active Snowflake session
        auto_create_threads: bool
            Whether to automatically create threads (default: True)

    Instantiate:
        .. code-block:: python

            from langchain_snowflake import SnowflakeCortexAgent

            agent = SnowflakeCortexAgent(
                name="sales_assistant",
                database="sales_db",
                schema="analytics",
                session=session
            )

    Direct Usage:
        .. code-block:: python

            # Basic execution
            response = agent.invoke("What were Q3 sales trends?")

            # With thread management
            thread_id = agent.create_thread(metadata={"customer": "ACME"})
            response = agent.invoke_with_thread("Show ACME sales", thread_id)

            # Check usage
            usage = agent.last_usage
            print(f"Tokens used: {usage['total_tokens']}")

            # Submit feedback
            from langchain_snowflake.agents.schemas import FeedbackInput
            feedback = FeedbackInput(
                request_id=response["run_id"],
                positive=True,
                feedback_message="Good analysis",
                categories=["accuracy"]
            )
            agent.submit_feedback(feedback)

    Async Usage:
        .. code-block:: python

            # Async execution
            response = await agent.ainvoke("What were Q3 sales trends?")

            # Async streaming
            async for chunk in agent.astream("Generate sales report"):
                print(chunk, end="", flush=True)
    """

    # Core agent configuration
    name: str = Field(description="Name of the Cortex Agent in Snowflake")
    database: str = Field(description="Database containing the agent")
    schema: str = Field(description="Schema containing the agent")

    # Thread management configuration
    auto_create_threads: bool = Field(
        default=True, description="Whether to automatically create threads (default: True)"
    )

    # Usage tracking
    track_usage: bool = Field(default=False, description="Whether to track usage metadata (default: False)")

    def __init__(self, **kwargs):
        """Initialize the Snowflake Cortex Agent."""
        # Initialize parent classes (don't pass kwargs to avoid conflicts)
        super().__init__()

        # Set required fields using centralized validation
        from .._validation_utils import SnowflakeValidationUtils

        self.name = SnowflakeValidationUtils.validate_non_empty_string(kwargs.get("name"), "name")
        self.database = SnowflakeValidationUtils.validate_non_empty_string(kwargs.get("database"), "database")
        self.schema = SnowflakeValidationUtils.validate_non_empty_string(kwargs.get("schema"), "schema")

        # Set optional fields with defaults
        self.auto_create_threads = kwargs.get("auto_create_threads", True)
        self.track_usage = kwargs.get("track_usage", False)

        # Set connection fields with defaults (from SnowflakeConnectionMixin)
        self.session = kwargs.get("session", None)
        self.account = kwargs.get("account", None)
        self.user = kwargs.get("user", None)
        self.password = kwargs.get("password", None)
        self.token = kwargs.get("token", None)
        self.private_key_path = kwargs.get("private_key_path", None)
        self.private_key_passphrase = kwargs.get("private_key_passphrase", None)
        self.warehouse = kwargs.get("warehouse", None)
        self.request_timeout = kwargs.get("request_timeout", 30)
        self.verify_ssl = kwargs.get("verify_ssl", True)
        self.respect_session_timeout = kwargs.get("respect_session_timeout", True)

        # Initialize connection mixin session
        self._session = None

        # Initialize tracking attributes
        self._last_usage = None
        self._last_run_id = None

    # ============================================================================
    # SHARED HELPER METHODS
    # ============================================================================

    def _extract_query_from_messages(self, messages: Sequence[BaseMessage]) -> str:
        """Extract query from message sequence.

        Args:
            messages: Sequence of messages

        Returns:
            Query string extracted from the last human message

        Raises:
            ValueError: If no human message found in sequence
        """
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return message.content
            elif isinstance(message, dict) and message.get("role") == "user":
                return message.get("content", "")

        raise ValueError("No human message found in message sequence")

    def _build_agent_execution_config(self, query: str, thread_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Build agent execution request configuration."""
        session = self._get_session()

        # Simple payload (like chat models)
        payload = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}],
        }

        # Add optional parameters
        for param in ["temperature", "max_tokens", "top_p"]:
            if param in kwargs and kwargs[param] is not None:
                payload[param] = kwargs[param]

        # Use existing RestApiRequestBuilder
        request_config = RestApiRequestBuilder.agent_request(
            session=session,
            database=self.database,
            schema=self.schema,
            name=self.name,
            action="run",
            method="POST",
            payload=payload,
            request_timeout=self.request_timeout,
            verify_ssl=self.verify_ssl,
        )

        # Add thread_id as query parameter if provided
        if thread_id:
            url = request_config.get("url", "")
            separator = "&" if "?" in url else "?"
            request_config["url"] = f"{url}{separator}thread_id={thread_id}"

        return request_config

    def _process_agent_response(
        self, response_data: Dict[str, Any], start_time: float, query: str, thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process agent response following chat model patterns."""
        # Simple content extraction (like chat models)
        content = response_data.get("content", "") or response_data.get("output", "") or response_data.get("text", "")

        # Extract run_id from X-Snowflake-Request-Id header (needed for feedback)
        run_id = response_data.get("_snowflake_request_id")

        # Simple usage tracking if enabled
        if self.track_usage:
            execution_time = time.time() - start_time
            self._last_usage = {
                "execution_time": execution_time,
                "name": self.name,
                "timestamp": time.time(),
            }

        result = {
            "output": content,
            "name": self.name,
            "thread_id": thread_id,
            "execution_time": time.time() - start_time,
        }

        # Add run_id if available (needed for feedback functionality)
        if run_id:
            result["run_id"] = run_id

        return result

    def _handle_agent_error(
        self, error: Exception, start_time: float, query: str, thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Simple error handling following package patterns."""
        execution_time = time.time() - start_time
        error_message = SnowflakeToolErrorHandler.handle_tool_error(
            error=error,
            tool_name="cortex_agent",
            operation="agent execution",
            query=query,
            logger_instance=logger,
        )

        return {
            "output": error_message,
            "name": self.name,
            "thread_id": thread_id,
            "error": str(error),
            "execution_time": execution_time,
        }

    # ============================================================================
    # LANGGRAPH COMPATIBILITY METHODS
    # ============================================================================

    def __call__(self, state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """LangGraph-compatible state-based execution.

        This method makes SnowflakeCortexAgent compatible with LangGraph Supervisor
        by handling state dictionaries with message lists.

        Args:
            state: State dictionary containing 'messages' key with message list
            config: Optional configuration for the run

        Returns:
            Updated state dictionary with new messages

        Raises:
            ValueError: If no user message found in state
        """
        # Extract messages from state
        messages = state.get("messages", [])

        # Get the last human message as the query
        query = None
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                query = message.content
                break
            elif isinstance(message, dict) and message.get("role") == "user":
                query = message.get("content")
                break
            elif isinstance(message, str):
                query = message
                break

        if not query:
            raise ValueError("No user message found in state")

        # Execute the agent
        result = self.invoke(query, config=config)

        # Create AI message from result
        ai_message = AIMessage(content=result.get("output", ""), name=self.name)

        # Return updated state
        return {**state, "messages": messages + [ai_message]}

    # ============================================================================
    # CORE AGENT EXECUTION METHODS
    # ============================================================================

    def invoke(
        self,
        input: Union[str, Dict[str, Any], Sequence[BaseMessage]],
        config: Optional[RunnableConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Agent execution with enhanced support for LangGraph state dictionaries and message sequences.

        Args:
            input: String query, dict with 'input'/'query', state dict with 'messages', or message sequence
            config: Optional configuration for the run
            **kwargs: Additional parameters for agent execution

        Returns:
            Dict containing agent response and metadata

        Raises:
            SnowflakeRestApiError: If agent execution fails
        """
        start_time = time.time()

        # Handle different input types
        if isinstance(input, str):
            query = input
            thread_id = kwargs.get("thread_id")
            execution_params = kwargs
        elif isinstance(input, (list, tuple)):  # Message sequence
            query = self._extract_query_from_messages(input)
            thread_id = kwargs.get("thread_id")
            execution_params = kwargs
        elif isinstance(input, dict):
            # Check if it's a state dict with messages
            if "messages" in input:
                query = self._extract_query_from_messages(input["messages"])
                thread_id = input.get("thread_id") or kwargs.get("thread_id")
                execution_params = {**input, **kwargs}
            else:
                # Original dict handling
                query = input.get("input") or input.get("query")
                if not query:
                    raise ValueError("Input dict must contain 'input', 'query', or 'messages' key")
                thread_id = input.get("thread_id")
                execution_params = {**input, **kwargs}

            # Clean up non-execution params
            execution_params.pop("input", None)
            execution_params.pop("query", None)
            execution_params.pop("thread_id", None)
            execution_params.pop("messages", None)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

        try:
            # Use streaming method and collect all chunks
            content_chunks = []
            metadata_chunks: Dict[str, list] = {"thinking": [], "tool_use": [], "tool_result": [], "annotations": []}

            for chunk_json in self._stream_agent_request(query, thread_id, **execution_params):
                # Parse JSON chunk to access all data
                try:
                    import json

                    chunk_data = json.loads(chunk_json)

                    if isinstance(chunk_data, dict):
                        chunk_type = chunk_data.get("type")

                        # Extract text chunks for response
                        if chunk_type == "text":
                            text_content = chunk_data.get("text", "")
                            if text_content:
                                content_chunks.append(text_content)
                            # Store annotations if present
                            if "annotations" in chunk_data:
                                metadata_chunks["annotations"].extend(chunk_data.get("annotations", []))
                        # Store metadata chunks for optional access
                        elif chunk_type == "thinking":
                            metadata_chunks["thinking"].append(chunk_data.get("thinking", {}))
                        elif chunk_type == "tool_use":
                            metadata_chunks["tool_use"].append(chunk_data.get("tool_use", {}))
                        elif chunk_type == "tool_result":
                            metadata_chunks["tool_result"].append(chunk_data.get("tool_result", {}))
                        # Fallback: content field
                        elif "content" in chunk_data and not chunk_type:
                            content_chunks.append(str(chunk_data.get("content", "")))
                except (json.JSONDecodeError, TypeError):
                    # Not JSON - treat as plain text
                    content_chunks.append(str(chunk_json))

            # Combine text chunks
            content = "".join(content_chunks)

            # Process the combined response with optional metadata
            response_data = {"content": content}
            if any(metadata_chunks.values()):
                response_data["metadata"] = metadata_chunks

            return self._process_agent_response(response_data, start_time, query, thread_id)
        except Exception as e:
            return self._handle_agent_error(e, start_time, query, thread_id)

    async def ainvoke(
        self, input: Union[str, Dict[str, Any]], config: Optional[RunnableConfig] = None, **kwargs
    ) -> Dict[str, Any]:
        """Async agent execution with enhanced support for LangGraph state dictionaries and message sequences."""
        start_time = time.time()

        # Direct input handling (like chat models)
        if isinstance(input, str):
            query = input
            thread_id = kwargs.get("thread_id")
            execution_params = kwargs
        else:
            query = input.get("input") or input.get("query")
            if not query:
                raise ValueError("Input dict must contain 'input' or 'query' key")
            thread_id = input.get("thread_id")
            execution_params = {**input, **kwargs}
            # Clean up non-execution params
            execution_params.pop("input", None)
            execution_params.pop("query", None)
            execution_params.pop("thread_id", None)

        try:
            # Use async streaming method and collect all chunks
            content_chunks = []
            metadata_chunks: Dict[str, list] = {"thinking": [], "tool_use": [], "tool_result": [], "annotations": []}

            async for chunk_json in self._stream_agent_request_async(query, thread_id, **execution_params):
                # Parse JSON chunk to access all data
                try:
                    import json

                    chunk_data = json.loads(chunk_json)

                    if isinstance(chunk_data, dict):
                        chunk_type = chunk_data.get("type")

                        # Extract text chunks for response
                        if chunk_type == "text":
                            text_content = chunk_data.get("text", "")
                            if text_content:
                                content_chunks.append(text_content)
                            # Store annotations if present
                            if "annotations" in chunk_data:
                                metadata_chunks["annotations"].extend(chunk_data.get("annotations", []))
                        # Store metadata chunks for optional access
                        elif chunk_type == "thinking":
                            metadata_chunks["thinking"].append(chunk_data.get("thinking", {}))
                        elif chunk_type == "tool_use":
                            metadata_chunks["tool_use"].append(chunk_data.get("tool_use", {}))
                        elif chunk_type == "tool_result":
                            metadata_chunks["tool_result"].append(chunk_data.get("tool_result", {}))
                        # Fallback: content field
                        elif "content" in chunk_data and not chunk_type:
                            content_chunks.append(str(chunk_data.get("content", "")))
                except (json.JSONDecodeError, TypeError):
                    # Not JSON - treat as plain text
                    content_chunks.append(str(chunk_json))

            # Combine text chunks
            content = "".join(content_chunks)

            # Process the combined response with optional metadata
            response_data = {"content": content}
            if any(metadata_chunks.values()):
                response_data["metadata"] = metadata_chunks

            return self._process_agent_response(response_data, start_time, query, thread_id)
        except Exception as e:
            return self._handle_agent_error(e, start_time, query, thread_id)

    # ============================================================================
    # THREAD MANAGEMENT METHODS
    # ============================================================================
    # Thread management methods are now provided by ThreadManagement mixin

    def invoke_with_thread(self, query: str, thread_id: str, **kwargs) -> str:
        """Execute agent with specific thread context.

        Args:
            query: Query to send to the agent
            thread_id: Thread ID for conversation context
            **kwargs: Additional parameters

        Returns:
            Agent response as string
        """
        result = self.invoke(query, thread_id=thread_id, **kwargs)
        return result.get("output", "")

    # ============================================================================
    # STREAMING METHODS
    # ============================================================================

    def stream(
        self,
        input: Union[str, Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream agent responses following LangChain Runnable interface.

        Args:
            input: Either a string query or dict with 'input'/'query' and optional parameters
            config: Optional configuration for the run
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Yields:
            Streaming chunks of the agent response
        """
        # Direct input handling (like chat models)
        if isinstance(input, str):
            query = input
            thread_id = kwargs.get("thread_id")
            execution_params = kwargs
        else:
            query = input.get("input") or input.get("query")
            if not query:
                raise ValueError("Input dict must contain 'input' or 'query' key")
            thread_id = input.get("thread_id")
            execution_params = {**input, **kwargs}
            # Clean up non-execution params
            execution_params.pop("input", None)
            execution_params.pop("query", None)
            execution_params.pop("thread_id", None)

        try:
            for chunk in self._stream_agent_request(query, thread_id, **execution_params):
                yield chunk
        except Exception as e:
            error_message = SnowflakeToolErrorHandler.handle_tool_error(
                error=e, tool_name="cortex_agent", operation="streaming", query=query, logger_instance=logger
            )
            yield error_message

    async def astream(
        self,
        input: Union[str, Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async stream agent responses following LangChain Runnable interface."""
        # Direct input handling (like chat models)
        if isinstance(input, str):
            query = input
            thread_id = kwargs.get("thread_id")
            execution_params = kwargs
        else:
            query = input.get("input") or input.get("query")
            if not query:
                raise ValueError("Input dict must contain 'input' or 'query' key")
            thread_id = input.get("thread_id")
            execution_params = {**input, **kwargs}
            # Clean up non-execution params
            execution_params.pop("input", None)
            execution_params.pop("query", None)
            execution_params.pop("thread_id", None)

        try:
            async for chunk in self._stream_agent_request_async(query, thread_id, **execution_params):
                yield chunk
        except Exception as e:
            error_message = SnowflakeToolErrorHandler.handle_tool_error(
                error=e, tool_name="cortex_agent", operation="async streaming", query=query, logger_instance=logger
            )
            yield f"Error: {error_message}"

    # ============================================================================
    # USAGE TRACKING METHODS
    # ============================================================================

    @property
    def last_usage(self) -> Optional[Dict[str, Any]]:
        """Get the last usage metadata."""
        return self._last_usage

    @property
    def last_run_id(self) -> Optional[str]:
        """Get the last run ID."""
        return self._last_run_id

    # ============================================================================
    # INTERNAL REQUEST METHODS
    # ============================================================================

    def _stream_agent_request(self, query: str, thread_id: Optional[str] = None, **kwargs) -> Iterator[str]:
        """Make streaming agent request using unified RestApiClient."""
        request_config = self._build_agent_execution_config(query, thread_id, stream=True, **kwargs)
        yield from RestApiClient.make_sync_streaming_request(request_config, "agent streaming")

    async def _stream_agent_request_async(
        self, query: str, thread_id: Optional[str] = None, **kwargs
    ) -> AsyncIterator[str]:
        """Make streaming agent request using unified RestApiClient (async).

        Args:
            query: Query to send to the agent
            thread_id: Optional thread ID for conversation context
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Yields:
            String chunks from the streaming response
        """
        request_config = self._build_agent_execution_config(query, thread_id, stream=True, **kwargs)
        async for chunk in RestApiClient.make_async_streaming_request(request_config, "agent streaming"):
            yield chunk
