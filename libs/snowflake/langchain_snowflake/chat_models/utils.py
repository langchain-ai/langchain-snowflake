"""Utility functions for Snowflake chat models."""

import logging
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)


class SnowflakeUtils:
    """Mixin class for Snowflake utility functions."""

    def _format_messages_for_cortex(
        self, messages: List[BaseMessage], use_array_format: bool = False
    ) -> Union[str, List[Dict[str, Any]]]:
        """Format messages for Cortex COMPLETE function.

        Args:
            messages: List of messages to format
            use_array_format: If True, return array format for tool calling/options. If False, return string.

        Returns:
            Either a formatted string prompt or an array of message objects for tool calling.
        """
        if use_array_format:
            # Format as array for tool calling and options support
            formatted_messages = []

            for message in messages:
                if isinstance(message, SystemMessage):
                    formatted_messages.append(
                        {"role": "system", "content": message.content}
                    )
                elif isinstance(message, HumanMessage):
                    formatted_messages.append(
                        {"role": "user", "content": message.content}
                    )
                elif isinstance(message, AIMessage):
                    msg_dict = {"role": "assistant", "content": message.content}

                    # Handle tool calls if present
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        content_list = []
                        if message.content:
                            content_list.append(
                                {"type": "text", "text": message.content}
                            )

                        for tool_call in message.tool_calls:
                            content_list.append(
                                {
                                    "type": "tool_use",
                                    "tool_use": {
                                        "tool_use_id": tool_call.get("id", ""),
                                        "name": tool_call.get("name", ""),
                                        "input": tool_call.get("args", {}),
                                    },
                                }
                            )

                        if content_list:
                            msg_dict["content_list"] = content_list

                    formatted_messages.append(msg_dict)
                else:
                    # Default handling for other message types
                    formatted_messages.append(
                        {"role": "user", "content": str(message.content)}
                    )

            return formatted_messages
        else:
            # Original string concatenation format for backwards compatibility
            formatted_parts = []

            for message in messages:
                if isinstance(message, SystemMessage):
                    formatted_parts.append(f"System: {message.content}")
                elif isinstance(message, HumanMessage):
                    formatted_parts.append(f"Human: {message.content}")
                elif isinstance(message, AIMessage):
                    formatted_parts.append(f"Assistant: {message.content}")
                else:
                    formatted_parts.append(f"{message.content}")

            return "\n\n".join(formatted_parts)

    def _build_cortex_options(self) -> Dict[str, Any]:
        """Build options dictionary for Cortex COMPLETE function."""
        options = {}

        # Check if we have tools
        has_tools = hasattr(self, "_bound_tools") and self._bound_tools

        # Basic model parameters - match documentation format exactly
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.max_tokens is not None:
            options["max_tokens"] = self.max_tokens

        # Only include top_p when no tools are present
        if not has_tools and self.top_p is not None:
            options["top_p"] = self.top_p

        # Tools are added at the top level of options per Snowflake documentation
        if has_tools:
            options["tools"] = self._bound_tools
            # Include tool_choice if it's set (documented as supported)
            if hasattr(self, "_tool_choice") and self._tool_choice:
                options["tool_choice"] = self._tool_choice

        return options

    def _build_cortex_options_for_sql(self) -> Dict[str, Any]:
        """Build options dictionary for SQL function (excludes tools)."""
        options = {}

        # Basic model parameters that SQL function supports
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.max_tokens is not None:
            options["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            options["top_p"] = self.top_p

        return options

    def _create_error_result(self, error_message: str) -> ChatResult:
        """Helper to create an error result."""
        message = AIMessage(
            content=error_message,
            response_metadata={
                "model": self.model,
                "model_name": self.model,
                "finish_reason": "error",
            },
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _has_tools(self) -> bool:
        """Check if tools are bound to this model instance."""
        return hasattr(self, "_bound_tools") and bool(self._bound_tools)

    def _should_use_rest_api(self) -> bool:
        """Determine if REST API should be used instead of SQL function."""
        return getattr(self, "_use_rest_api", False) or self._has_tools()

    def _get_count_tokens_compatible_model(self, model: str) -> str:
        """Map COMPLETE model names to COUNT_TOKENS compatible model names.

        COMPLETE and COUNT_TOKENS support different model sets.
        This maps COMPLETE models to the closest COUNT_TOKENS equivalent.

        Reference:
        - COMPLETE: https://docs.snowflake.com/en/sql-reference/functions/complete-snowflake-cortex
        - COUNT_TOKENS: https://docs.snowflake.com/en/sql-reference/functions/count_tokens-snowflake-cortex
        """
        # Map COMPLETE models to COUNT_TOKENS compatible models
        model_mapping = {
            # Claude models -> llama (closest approximation for token counting)
            "claude-3-5-sonnet": "llama3.1-70b",
            "claude-3-haiku": "llama3.1-8b",
            "claude-3-opus": "llama3.1-70b",
            # Llama models (direct support)
            "llama3.1-70b": "llama3.1-70b",
            "llama3.1-8b": "llama3.1-8b",
            "llama2-70b-chat": "llama2-70b-chat",
            # Mistral models (direct support)
            "mistral-large2": "mistral-large2",
            "mistral-large": "mistral-large2",
        }

        return model_mapping.get(model, "llama3.1-70b")  # Default fallback

    def _estimate_tokens(self, messages: List[Union[BaseMessage, dict]]) -> int:
        """Get accurate token count using official SNOWFLAKE.CORTEX.COUNT_TOKENS function.

        Uses the official Snowflake Cortex COUNT_TOKENS function to get precise token counts
        based on the specific model being used. Falls back to word-count estimation if the
        function is unavailable or fails.

        Reference: https://docs.snowflake.com/en/sql-reference/functions/count_tokens-snowflake-cortex
        """
        try:
            # Get session for SQL execution
            session = self._get_session()

            # Extract text content from all messages
            text_content = []
            for msg in messages:
                if hasattr(msg, "content"):
                    # BaseMessage object
                    content = msg.content
                    if isinstance(content, list):
                        # Handle list content (e.g., multimodal content)
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                text_content.append(str(item["text"]))
                            elif isinstance(item, str):
                                text_content.append(item)
                            # Skip non-text content for token counting
                    elif isinstance(content, str):
                        text_content.append(content)
                    else:
                        text_content.append(str(content))
                elif isinstance(msg, dict) and "content" in msg:
                    # Dict format
                    content = msg["content"]
                    if isinstance(content, list):
                        # Handle list content in dict format
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                text_content.append(str(item["text"]))
                            elif isinstance(item, str):
                                text_content.append(item)
                            # Skip non-text content for token counting
                    else:
                        text_content.append(str(content))

            # Combine all text content
            combined_text = " ".join(text_content)

            if not combined_text.strip():
                return 0

            # Use official SNOWFLAKE.CORTEX.COUNT_TOKENS function
            # Note: COUNT_TOKENS supports different models than COMPLETE
            # Map COMPLETE models to COUNT_TOKENS compatible models
            count_tokens_model = self._get_count_tokens_compatible_model(self.model)

            # Escape single quotes for SQL injection prevention
            escaped_text = combined_text.replace("'", "''")
            escaped_model = count_tokens_model.replace("'", "''")

            sql_query = f"SELECT SNOWFLAKE.CORTEX.COUNT_TOKENS('{escaped_model}', '{escaped_text}') AS token_count"

            result = session.sql(sql_query).collect()
            if result and len(result) > 0:
                return int(result[0]["TOKEN_COUNT"])
            else:
                logger.warning(
                    "COUNT_TOKENS query returned no results, falling back to word count"
                )
                return len(combined_text.split()) * 1.3  # Fallback estimate

        except Exception as e:
            logger.warning(
                f"Error using SNOWFLAKE.CORTEX.COUNT_TOKENS: {e}. Falling back to word count estimation"
            )
            # Fallback to simple word counting
            total_words = 0
            for msg in messages:
                if hasattr(msg, "content"):
                    content = msg.content
                    if isinstance(content, str):
                        total_words += len(content.split())
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                total_words += len(str(item["text"]).split())
                            elif isinstance(item, str):
                                total_words += len(item.split())
                elif isinstance(msg, dict) and "content" in msg:
                    content = msg["content"]
                    if isinstance(content, str):
                        total_words += len(content.split())
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                total_words += len(str(item["text"]).split())
                            elif isinstance(item, str):
                                total_words += len(item.split())

            # Simple estimation: words * 1.3 (tokens typically > words)
            return int(total_words * 1.3)


# ============================================================================
# SHARED UTILITIES FOR CODE DEDUPLICATION
# ============================================================================


class SnowflakeUrlBuilder:
    """Shared utility for building Snowflake REST API URLs."""

    @staticmethod
    def build_rest_api_url(session) -> str:
        """Build the REST API URL for Snowflake Cortex Complete.

        Centralizes the complex URL building logic that was duplicated
        across auth.py and streaming.py modules.

        Args:
            session: Active Snowflake session

        Returns:
            Complete REST API URL for Cortex Complete endpoint
        """
        conn = session._conn._conn
        account = conn.account

        # Build URL with correct hostname format
        # Snowflake account format: account_locator.region_id.cloud_provider
        # For REST API, use the exact hostname from the session connection
        if hasattr(conn, "host") and conn.host:
            # Use the exact host from the connection
            base_url = f"https://{conn.host}"
        elif "." in account:
            # Account already includes region info
            base_url = f"https://{account}.snowflakecomputing.com"
        else:
            # Simple account name, need to add region
            # Extract region from connection if available
            region = getattr(conn, "region", None) or "us-west-2"
            if region and region != "us-west-2":
                base_url = f"https://{account}.{region}.snowflakecomputing.com"
            else:
                base_url = f"https://{account}.snowflakecomputing.com"

        return f"{base_url}/api/v2/cortex/inference:complete"


class SnowflakeMetadataFactory:
    """Shared factory for creating consistent metadata objects."""

    @staticmethod
    def create_usage_metadata(
        usage_data: Optional[Dict[str, Any]] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: Optional[int] = None,
    ):
        """Create usage metadata with consistent structure.

        Args:
            usage_data: Usage data from API response
            input_tokens: Fallback input token count
            output_tokens: Fallback output token count
            total_tokens: Fallback total token count

        Returns:
            UsageMetadata object with appropriate values
        """
        from langchain_core.messages.ai import UsageMetadata

        if usage_data:
            return UsageMetadata(
                input_tokens=usage_data.get("prompt_tokens", input_tokens),
                output_tokens=usage_data.get("completion_tokens", output_tokens),
                total_tokens=usage_data.get(
                    "total_tokens", total_tokens or (input_tokens + output_tokens)
                ),
            )
        else:
            return UsageMetadata(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens or (input_tokens + output_tokens),
            )

    @staticmethod
    def create_response_metadata(
        model: str, finish_reason: str = "stop", **extra_metadata
    ) -> Dict[str, Any]:
        """Create response metadata with consistent structure.

        Args:
            model: Model name
            finish_reason: Reason for completion ("stop", "error", "tool_calls", etc.)
            **extra_metadata: Additional metadata fields

        Returns:
            Response metadata dictionary
        """
        metadata = {
            "model": model,
            "model_name": model,  # LangChain compatibility
            "finish_reason": finish_reason,
        }
        metadata.update(extra_metadata)
        return metadata


class SnowflakePayloadBuilder:
    """Shared utility for building REST API payloads."""

    @staticmethod
    def add_generation_params(
        payload: Dict[str, Any],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Add generation parameters to payload.

        Args:
            payload: Existing payload dictionary
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter

        Returns:
            Updated payload dictionary
        """
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        return payload


class SnowflakeErrorFactory:
    """Shared factory for creating consistent error responses."""

    @staticmethod
    def create_error_result(
        error_message: str, model: str, input_tokens: int = 0, usage_metadata_func=None
    ) -> "ChatResult":
        """Create a consistent error result.

        Args:
            error_message: Error message to include
            model: Model name for metadata
            input_tokens: Input token count for usage tracking
            usage_metadata_func: Function to estimate tokens (optional)

        Returns:
            ChatResult with error message and proper metadata
        """
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        # Create usage metadata
        usage_metadata = SnowflakeMetadataFactory.create_usage_metadata(
            input_tokens=input_tokens, output_tokens=0, total_tokens=input_tokens
        )

        # Create response metadata
        response_metadata = SnowflakeMetadataFactory.create_response_metadata(
            model=model, finish_reason="error"
        )

        # Create error message
        message = AIMessage(
            content=error_message,
            usage_metadata=usage_metadata,
            response_metadata=response_metadata,
        )

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
