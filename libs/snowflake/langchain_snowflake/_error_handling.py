"""Shared error handling utilities for langchain-snowflake package.

This module provides standardized error handling patterns to reduce code duplication
across tools and chat models.
"""

import json
import logging
from typing import Any, Dict, Optional

from langchain_core.outputs import ChatResult

logger = logging.getLogger(__name__)


class SnowflakeErrorHandler:
    """Centralized error handling for Snowflake integrations."""

    @staticmethod
    def log_and_return_json_error(
        error: Exception,
        operation: str,
        additional_context: Optional[Dict[str, Any]] = None,
        logger_instance: Optional[logging.Logger] = None,
    ) -> str:
        """Log error and return standardized JSON error response.

        Args:
            error: The exception that occurred
            operation: Description of the operation that failed
            additional_context: Additional context to include in error response
            logger_instance: Specific logger to use (defaults to module logger)

        Returns:
            JSON string with error information
        """
        log = logger_instance or logger
        log.error(f"Error in {operation}: {error}")

        error_response = {
            "error": f"Failed to {operation}: {str(error)}",
            "operation": operation,
        }

        if additional_context:
            error_response.update(additional_context)

        return json.dumps(error_response)

    @staticmethod
    def log_and_raise(
        error: Exception,
        operation: str,
        logger_instance: Optional[logging.Logger] = None,
    ) -> None:
        """Log error and re-raise it.

        Args:
            error: The exception that occurred
            operation: Description of the operation that failed
            logger_instance: Specific logger to use (defaults to module logger)
        """
        log = logger_instance or logger
        log.error(f"Error in {operation}: {error}")
        raise

    @staticmethod
    def create_chat_error_result(
        error: Exception,
        operation: str,
        model: str,
        input_tokens: int = 0,
        logger_instance: Optional[logging.Logger] = None,
    ) -> ChatResult:
        """Create standardized ChatResult for errors.

        Args:
            error: The exception that occurred
            operation: Description of the operation that failed
            model: Model name for metadata
            input_tokens: Number of input tokens for usage metadata
            logger_instance: Specific logger to use (defaults to module logger)

        Returns:
            ChatResult with error message
        """
        log = logger_instance or logger
        log.error(f"Error in {operation}: {error}")

        # Import here to avoid circular imports
        from langchain_core.messages import AIMessage
        from langchain_core.messages.ai import UsageMetadata
        from langchain_core.outputs import ChatGeneration

        error_message = f"Error: Failed to {operation} - {str(error)}"

        # Create usage metadata
        usage_metadata = UsageMetadata(input_tokens=input_tokens, output_tokens=0, total_tokens=input_tokens)

        # Create AI message with error
        ai_message = AIMessage(content=error_message, usage_metadata=usage_metadata)

        # Create generation and result
        generation = ChatGeneration(message=ai_message)

        return ChatResult(
            generations=[generation],
            llm_output={
                "model": model,
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": 0,
                    "total_tokens": input_tokens,
                },
            },
        )

    @staticmethod
    def log_error(
        operation: str,
        error: Exception,
        logger_instance: Optional[logging.Logger] = None,
    ) -> None:
        """Log an error with standardized format.

        Args:
            operation: Description of the operation that failed
            error: The exception that occurred
            logger_instance: Specific logger to use (defaults to module logger)
        """
        log = logger_instance or logger
        log.error(f"Error in {operation}: {error}")

    @staticmethod
    def log_and_return_str_error(
        operation: str,
        error: Exception,
        fallback_message: str,
        logger_instance: Optional[logging.Logger] = None,
    ) -> str:
        """Log error and return a fallback string message.

        Args:
            operation: Description of the operation that failed
            error: The exception that occurred
            fallback_message: Message to return as fallback
            logger_instance: Specific logger to use (defaults to module logger)

        Returns:
            Fallback message string
        """
        log = logger_instance or logger
        log.error(f"Error in {operation}: {error}")
        return fallback_message

    @staticmethod
    def log_warning_and_fallback(
        error: Exception,
        operation: str,
        fallback_action: str,
        logger_instance: Optional[logging.Logger] = None,
    ) -> None:
        """Log warning for failed operation with fallback.

        Args:
            error: The exception that occurred
            operation: Description of the operation that failed
            fallback_action: Description of fallback being taken
            logger_instance: Specific logger to use (defaults to module logger)
        """
        log = logger_instance or logger
        log.warning(f"{operation} failed: {error}")
        log.info(f"Using fallback: {fallback_action}")


class SnowflakeToolErrorHandler(SnowflakeErrorHandler):
    """Specialized error handler for Snowflake tools."""

    @staticmethod
    def handle_tool_error(
        error: Exception,
        tool_name: str,
        operation: str,
        query: Optional[str] = None,
        logger_instance: Optional[logging.Logger] = None,
    ) -> str:
        """Handle tool execution errors with standardized format.

        Args:
            error: The exception that occurred
            tool_name: Name of the tool that failed
            operation: Description of the operation (e.g., "analyze sentiment")
            query: Original query/input if applicable
            logger_instance: Specific logger to use

        Returns:
            JSON string with tool error information
        """
        log = logger_instance or logger
        log.error(f"Error in {tool_name} during {operation}: {error}")

        error_response = {
            "error": f"Failed to {operation}: {str(error)}",
            "tool": tool_name,
            "operation": operation,
        }

        if query:
            error_response["query"] = query

        return json.dumps(error_response)

    @staticmethod
    def handle_sql_error(
        error: Exception,
        tool_name: str,
        sql_query: str,
        operation: str,
        logger_instance: Optional[logging.Logger] = None,
    ) -> str:
        """Handle SQL execution errors specifically.

        Args:
            error: The SQL exception that occurred
            tool_name: Name of the tool executing SQL
            sql_query: The SQL query that failed (truncated for logging)
            operation: Description of the operation
            logger_instance: Specific logger to use

        Returns:
            JSON string with SQL error information
        """
        log = logger_instance or logger

        # Truncate SQL for logging (first 100 chars)
        sql_preview = sql_query[:100] + "..." if len(sql_query) > 100 else sql_query
        log.error(f"SQL error in {tool_name}: {error}")
        log.debug(f"Failed SQL: {sql_preview}")

        return json.dumps(
            {
                "error": f"Failed to {operation}: {str(error)}",
                "tool": tool_name,
                "operation": operation,
                "error_type": "sql_execution_error",
            }
        )

    @staticmethod
    def handle_rest_api_error(
        error: Exception,
        tool_name: str,
        operation: str,
        endpoint: Optional[str] = None,
        logger_instance: Optional[logging.Logger] = None,
    ) -> str:
        """Handle REST API errors specifically.

        Args:
            error: The REST API exception that occurred
            tool_name: Name of the tool making the API call
            operation: Description of the operation
            endpoint: API endpoint if applicable
            logger_instance: Specific logger to use

        Returns:
            JSON string with REST API error information
        """
        log = logger_instance or logger
        log.error(f"REST API error in {tool_name}: {error}")

        error_response = {
            "error": f"Failed to {operation}: {str(error)}",
            "tool": tool_name,
            "operation": operation,
            "error_type": "rest_api_error",
        }

        if endpoint:
            error_response["endpoint"] = endpoint

        return json.dumps(error_response)
