"""Utility functions for Snowflake chat models."""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .._error_handling import SnowflakeErrorHandler

logger = logging.getLogger(__name__)


class SnowflakeUtils:
    """Mixin class for Snowflake utility functions."""

    def _format_messages_for_cortex(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Format messages for Snowflake Cortex COMPLETE function.

        Args:
            messages: List of LangChain messages

        Returns:
            List of formatted message dictionaries
        """
        formatted_messages = []

        for message in messages:
            if isinstance(message, SystemMessage):
                formatted_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                formatted_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                formatted_messages.append({"role": "assistant", "content": message.content})
            else:
                # Handle other message types by treating as user messages
                formatted_messages.append({"role": "user", "content": str(message.content)})

        return formatted_messages

    def _build_cortex_complete_query(
        self,
        messages: List[BaseMessage],
        model: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build SQL query for Snowflake Cortex COMPLETE function.

        Args:
            messages: List of LangChain messages
            model: Model name for Cortex
            options: Optional parameters for the model

        Returns:
            SQL query string
        """
        formatted_messages = self._format_messages_for_cortex(messages)

        # Convert messages to JSON string for SQL
        messages_json = json.dumps(formatted_messages)

        # Escape single quotes for SQL
        messages_json = messages_json.replace("'", "''")
        model = model.replace("'", "''")

        # Build base query
        if options:
            options_json = json.dumps(options).replace("'", "''")
            query = (
                f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model}', '{messages_json}', "
                f"PARSE_JSON('{options_json}')) AS response"
            )
        else:
            query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model}', '{messages_json}') AS response"

        return query

    def _parse_cortex_response(self, response_data: Any) -> str:
        """Parse response from Snowflake Cortex COMPLETE function.

        Args:
            response_data: Raw response from Cortex

        Returns:
            Parsed response text
        """
        if isinstance(response_data, str):
            return response_data
        elif isinstance(response_data, dict):
            # Handle structured response
            if "choices" in response_data and response_data["choices"]:
                choice = response_data["choices"][0]
                if "message" in choice:
                    return choice["message"].get("content", "")
                elif "text" in choice:
                    return choice["text"]
                else:
                    return str(choice)
            elif "content" in response_data:
                return response_data["content"]
            elif "text" in response_data:
                return response_data["text"]
            else:
                return str(response_data)
        else:
            return str(response_data)

    def _create_chat_result(
        self,
        response_text: str,
        model: str,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatResult:
        """Create ChatResult from response text.

        Args:
            response_text: Generated response text
            model: Model name used
            input_tokens: Number of input tokens (optional)
            output_tokens: Number of output tokens (optional)
            **kwargs: Additional metadata

        Returns:
            ChatResult object
        """
        # Create AI message
        ai_message = AIMessage(content=response_text)

        # Create generation with metadata
        generation_info = {"model": model, **kwargs}

        # Add token information if available
        if input_tokens is not None:
            generation_info["input_tokens"] = input_tokens
        if output_tokens is not None:
            generation_info["output_tokens"] = output_tokens

        generation = ChatGeneration(message=ai_message, generation_info=generation_info)

        # Create result
        llm_output: Dict[str, Any] = {"model": model}
        if input_tokens is not None or output_tokens is not None:
            token_usage: Dict[str, int] = {}
            llm_output["token_usage"] = token_usage
            if input_tokens is not None:
                token_usage["prompt_tokens"] = input_tokens
            if output_tokens is not None:
                token_usage["completion_tokens"] = output_tokens
                token_usage["total_tokens"] = (input_tokens or 0) + output_tokens

        return ChatResult(generations=[generation], llm_output=llm_output)

    def _estimate_tokens(self, messages: Union[List[BaseMessage], List[Dict[str, str]]]) -> int:
        """Estimate token count for messages.

        This is a rough estimation based on word count.
        For more accurate token counting, use _count_tokens_with_cortex.

        Args:
            messages: List of messages to estimate tokens for

        Returns:
            Estimated token count
        """
        total_words = 0

        if isinstance(messages, list) and messages:
            for msg in messages:
                if isinstance(msg, BaseMessage):
                    if hasattr(msg, "content"):
                        total_words += len(str(msg.content).split())
                elif isinstance(msg, dict):
                    if "content" in msg:
                        total_words += len(str(msg["content"]).split())
                else:
                    total_words += len(str(msg).split())

        # Rough estimation: 1 token ≈ 0.75 words for English text
        return int(total_words * 1.3)

    def _count_tokens_with_cortex(self, messages: List[BaseMessage], model: str, session=None) -> int:
        """Count tokens using Snowflake's CORTEX.COUNT_TOKENS function.

        Args:
            messages: List of messages to count tokens for
            model: Model name for token counting
            session: Snowflake session (uses self._get_session() if not provided)

        Returns:
            Actual token count from Snowflake Cortex
        """
        if session is None:
            session = self._get_session()

        try:
            # Format messages and combine into single text
            formatted_messages = self._format_messages_for_cortex(messages)
            combined_text = ""
            for msg in formatted_messages:
                combined_text += f"{msg['role']}: {msg['content']}\n"

            # Escape text for SQL
            escaped_text = combined_text.replace("'", "''")
            escaped_model = model.replace("'", "''")

            # Use Snowflake's COUNT_TOKENS function
            sql_query = f"SELECT SNOWFLAKE.CORTEX.COUNT_TOKENS('{escaped_model}', '{escaped_text}') AS token_count"

            result = session.sql(sql_query).collect()
            if result and len(result) > 0:
                return int(result[0]["TOKEN_COUNT"])
            else:
                SnowflakeErrorHandler.log_warning_and_fallback(
                    error=Exception("COUNT_TOKENS query returned no results"),
                    operation="token counting",
                    fallback_action="using word count estimation",
                )
                return len(combined_text.split()) * 1.3  # Fallback estimate

        except Exception as e:
            SnowflakeErrorHandler.log_warning_and_fallback(
                error=e, operation="SNOWFLAKE.CORTEX.COUNT_TOKENS", fallback_action="using word count estimation"
            )
            # Fallback to simple word counting
            total_words = 0
            for msg in messages:
                if hasattr(msg, "content"):
                    total_words += len(str(msg.content).split())

            return int(total_words * 1.3)

    def _validate_model_name(self, model: str) -> str:
        """Validate and normalize model name for Snowflake Cortex.

        Args:
            model: Model name to validate

        Returns:
            Validated model name

        Raises:
            ValueError: If model name is invalid
        """
        # Use centralized validation utilities
        from .._validation_utils import SnowflakeValidationUtils

        return SnowflakeValidationUtils.validate_model_name(model)

    def _build_options_dict(self, **kwargs) -> Dict[str, Any]:
        """Build options dictionary for Cortex COMPLETE function.

        Args:
            **kwargs: Keyword arguments to include in options

        Returns:
            Dictionary of options for Cortex
        """
        options = {}

        # Map common LangChain parameters to Cortex options
        if "temperature" in kwargs:
            options["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            options["max_tokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            options["top_p"] = kwargs["top_p"]
        if "stop" in kwargs and kwargs["stop"]:
            options["stop"] = kwargs["stop"]

        # Add any other options that are passed directly
        for key, value in kwargs.items():
            if key not in ["temperature", "max_tokens", "top_p", "stop"] and value is not None:
                options[key] = value

        return options if options else None

    def _extract_usage_metadata(self, response_data: Any) -> Dict[str, Any]:
        """Extract usage metadata from Cortex response.

        Args:
            response_data: Raw response from Cortex

        Returns:
            Dictionary containing usage metadata
        """
        usage_info = {}

        if isinstance(response_data, dict):
            if "usage" in response_data:
                usage = response_data["usage"]
                if "prompt_tokens" in usage:
                    usage_info["input_tokens"] = usage["prompt_tokens"]
                if "completion_tokens" in usage:
                    usage_info["output_tokens"] = usage["completion_tokens"]
                if "total_tokens" in usage:
                    usage_info["total_tokens"] = usage["total_tokens"]

            # Also check for direct token fields
            if "prompt_tokens" in response_data:
                usage_info["input_tokens"] = response_data["prompt_tokens"]
            if "completion_tokens" in response_data:
                usage_info["output_tokens"] = response_data["completion_tokens"]
            if "total_tokens" in response_data:
                usage_info["total_tokens"] = response_data["total_tokens"]

        return usage_info

    def _handle_streaming_response(self, response_stream):
        """Handle streaming response from Cortex.

        Args:
            response_stream: Streaming response object

        Yields:
            Individual response chunks
        """
        try:
            for chunk in response_stream:
                if chunk and hasattr(chunk, "content"):
                    yield chunk.content
                elif isinstance(chunk, str):
                    yield chunk
                elif isinstance(chunk, dict) and "content" in chunk:
                    yield chunk["content"]
        except Exception as e:
            # Handle streaming errors gracefully
            yield f"Error in streaming: {str(e)}"

    def _create_streaming_chat_result(self, content_chunks: List[str], model: str, **kwargs) -> ChatResult:
        """Create ChatResult from streaming content chunks.

        Args:
            content_chunks: List of content chunks from streaming
            model: Model name used
            **kwargs: Additional metadata

        Returns:
            ChatResult object
        """
        # Combine all chunks into final content
        final_content = "".join(content_chunks)

        # Create result using standard method
        return self._create_chat_result(response_text=final_content, model=model, **kwargs)


class SnowflakeMetadataFactory:
    """Factory class for creating LangChain metadata objects."""

    @staticmethod
    def create_usage_metadata(
        usage_data: Optional[Dict[str, Any]] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create usage metadata for LangChain responses.

        Args:
            usage_data: Raw usage data from Snowflake response
            input_tokens: Number of input tokens (optional override)
            output_tokens: Number of output tokens (optional override)

        Returns:
            Dictionary containing usage metadata
        """
        metadata = {}

        if usage_data:
            # Extract from usage_data if available
            if "prompt_tokens" in usage_data:
                metadata["input_tokens"] = usage_data["prompt_tokens"]
            if "completion_tokens" in usage_data:
                metadata["output_tokens"] = usage_data["completion_tokens"]
            if "total_tokens" in usage_data:
                metadata["total_tokens"] = usage_data["total_tokens"]

        # Override with explicit values if provided
        if input_tokens is not None:
            metadata["input_tokens"] = input_tokens
        if output_tokens is not None:
            metadata["output_tokens"] = output_tokens

        # Calculate total if we have both input and output
        if "input_tokens" in metadata and "output_tokens" in metadata:
            metadata["total_tokens"] = metadata["input_tokens"] + metadata["output_tokens"]

        return metadata

    @staticmethod
    def create_response_metadata(
        model: str,
        finish_reason: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create response metadata for LangChain responses.

        Args:
            model: Model name used for the response
            finish_reason: Reason the response finished (e.g., 'stop', 'length', 'error')
            **kwargs: Additional metadata fields

        Returns:
            Dictionary containing response metadata
        """
        metadata = {
            "model_name": model,
            "system_fingerprint": f"snowflake-{model}",
        }

        if finish_reason:
            metadata["finish_reason"] = finish_reason

        # Add any additional metadata
        metadata.update(kwargs)

        return metadata
