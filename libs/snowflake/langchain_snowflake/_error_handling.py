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
    def create_error_result_from_message(
        error_message: str, model: str, input_tokens: int = 0, logger_instance: Optional[logging.Logger] = None
    ) -> ChatResult:
        """Create ChatResult from error message string.
        Args:
            error_message: Pre-formatted error message
            model: Model name for metadata
            input_tokens: Number of input tokens for usage metadata
            logger_instance: Specific logger to use (defaults to module logger)

        Returns:
            ChatResult with error message and proper metadata
        """
        log = logger_instance or logger
        log.error(f"Error result created: {error_message}")

        # Import here to avoid circular imports
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration

        # Import metadata factory for compatibility with existing behavior
        from .chat_models.utils import SnowflakeMetadataFactory

        # Create usage metadata using existing factory (maintains compatibility)
        usage_metadata = SnowflakeMetadataFactory.create_usage_metadata(
            input_tokens=input_tokens, output_tokens=0, total_tokens=input_tokens
        )

        # Create response metadata using existing factory (maintains compatibility)
        response_metadata = SnowflakeMetadataFactory.create_response_metadata(model=model, finish_reason="error")

        # Create AI message with error
        ai_message = AIMessage(
            content=error_message,
            usage_metadata=usage_metadata,
            response_metadata=response_metadata,
        )

        # Create generation and result
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

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


class SnowflakeRestApiErrorHandler(SnowflakeErrorHandler):
    """Specialized error handler for REST API responses."""

    @staticmethod
    def safe_parse_json_response(
        response: Any,  # requests.Response
        operation: str,
        logger_instance: Optional[logging.Logger] = None,
    ) -> Dict[str, Any]:
        """Safely parse JSON response with detailed error handling.

        Handles regular JSON responses and detects SSE format (which usually indicates
        a configuration issue where streaming was enabled unexpectedly).

        Args:
            response: HTTP response object (requests.Response)
            operation: Description of the operation for error context
            logger_instance: Specific logger to use

        Returns:
            Parsed JSON data as dictionary, or error info if SSE detected

        Raises:
            ValueError: With detailed error information if parsing fails
        """
        log = logger_instance or logger

        try:
            # Check for empty response
            if not response.content:
                error_msg = f"Empty response from Snowflake API during {operation}. Status: {response.status_code}"
                log.error(error_msg)
                raise ValueError(error_msg)

            # Try to parse JSON first
            try:
                return response.json()
            except json.JSONDecodeError:
                # Check if this is a Server-Sent Events (SSE) response using robust detection
                content = response.text
                if SnowflakeRestApiErrorHandler._is_sse_response(response, content):
                    log.warning(
                        f"Detected SSE format response during {operation} - "
                        "this usually indicates a configuration issue"
                    )
                    return SnowflakeRestApiErrorHandler._parse_sse_response(content, log)
                else:
                    # Re-raise the original JSON error with context
                    raise

        except json.JSONDecodeError as e:
            # Log the actual response content for debugging (truncated)
            content_preview = response.text[:500] if response.text else "No content"
            log.error(f"Failed to parse JSON response during {operation}. Status: {response.status_code}")
            log.error(f"Response content preview: {content_preview}")

            error_msg = f"Invalid JSON response from Snowflake API during {operation}: {str(e)}"
            raise ValueError(error_msg)
        except Exception as e:
            log.error(f"Unexpected error parsing response during {operation}: {e}")
            raise ValueError(f"Failed to parse response during {operation}: {str(e)}")

    @staticmethod
    def _parse_sse_response(content: str, log: logging.Logger) -> Dict[str, Any]:
        """Simple SSE detection - returns helpful error instead of complex parsing.

        Since receiving SSE when expecting JSON usually indicates a configuration issue,
        this method provides a clear error message rather than attempting complex parsing.

        Args:
            content: Raw SSE content string
            log: Logger instance

        Returns:
            Dictionary with error information and suggestions
        """
        log.warning("Received SSE format when JSON was expected. This may indicate a configuration issue.")

        # Provide helpful error response instead of trying to parse SSE
        return {
            "error": "Received Server-Sent Events format instead of JSON",
            "suggestion": "Check if streaming is incorrectly enabled in the request configuration",
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "fix_hint": "Ensure payload['stream'] = False for non-streaming requests",
        }

    @staticmethod
    def _is_sse_response(response: Any, content: str) -> bool:
        """Detect if response is in Server-Sent Events (SSE) format.

        Simple detection based on Content-Type header, which is the most reliable indicator.

        Args:
            response: HTTP response object (requests.Response or aiohttp.ClientResponse)
            content: Response content string

        Returns:
            True if response appears to be SSE format
        """
        # Check Content-Type header (most reliable method)
        if hasattr(response, "headers"):
            content_type = ""
            # Handle both requests.Response and aiohttp.ClientResponse
            if hasattr(response.headers, "get"):
                content_type = response.headers.get("content-type", "").lower()
            elif hasattr(response.headers, "__getitem__"):
                try:
                    content_type = response.headers["content-type"].lower()
                except (KeyError, TypeError):
                    content_type = ""

            if "text/event-stream" in content_type:
                return True

        return False

    @staticmethod
    async def safe_parse_json_response_async(
        response: Any,  # aiohttp.ClientResponse
        operation: str,
        logger_instance: Optional[logging.Logger] = None,
    ) -> Dict[str, Any]:
        """Async version of safe JSON response parsing.

        Handles regular JSON responses and detects SSE format (which usually indicates
        a configuration issue where streaming was enabled unexpectedly).

        Args:
            response: aiohttp ClientResponse object
            operation: Description of the operation for error context
            logger_instance: Specific logger to use

        Returns:
            Parsed JSON data as dictionary, or error info if SSE detected

        Raises:
            ValueError: With detailed error information if parsing fails
        """
        log = logger_instance or logger

        try:
            # Get response content first
            content = await response.text()
            status = response.status

            # Check for empty response
            if not content or not content.strip():
                error_msg = f"Empty response from Snowflake API during {operation}. Status: {status}"
                log.error(error_msg)
                raise ValueError(error_msg)

            # Try to parse JSON first
            try:
                return await response.json()
            except json.JSONDecodeError:
                # Check if this is a Server-Sent Events (SSE) response using robust detection
                if SnowflakeRestApiErrorHandler._is_sse_response(response, content):
                    log.warning(
                        f"Detected SSE format response during {operation} - "
                        "this usually indicates a configuration issue"
                    )
                    return SnowflakeRestApiErrorHandler._parse_sse_response(content, log)
                else:
                    # Re-raise the original JSON error with context
                    raise

        except json.JSONDecodeError as e:
            # Log the actual response content for debugging (truncated)
            content_preview = content[:500] if content else "No content"
            log.error(f"Failed to parse JSON response during {operation}. Status: {status}")
            log.error(f"Response content preview: {content_preview}")

            error_msg = f"Invalid JSON response from Snowflake API during {operation}: {str(e)}"
            raise ValueError(error_msg)
        except Exception as e:
            log.error(f"Unexpected error parsing response during {operation}: {e}")
            raise ValueError(f"Failed to parse response during {operation}: {str(e)}")

    @staticmethod
    def handle_rest_api_response_error(
        error: Exception,
        operation: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        response_content: Optional[str] = None,
        logger_instance: Optional[logging.Logger] = None,
    ) -> str:
        """Handle REST API response parsing errors with rich context.

        Args:
            error: The exception that occurred
            operation: Description of the operation
            endpoint: API endpoint if available
            status_code: HTTP status code if available
            response_content: Response content preview if available
            logger_instance: Specific logger to use

        Returns:
            JSON string with detailed error information
        """
        log = logger_instance or logger

        error_context = {
            "error": f"Failed to {operation}: {str(error)}",
            "operation": operation,
            "error_type": "rest_api_response_error",
        }

        if endpoint:
            error_context["endpoint"] = endpoint
        if status_code:
            error_context["status_code"] = status_code
        if response_content:
            # Truncate content for error response
            error_context["response_preview"] = response_content[:200]

        log.error(f"REST API response error during {operation}: {error}")
        if endpoint:
            log.error(f"Endpoint: {endpoint}")
        if status_code:
            log.error(f"Status code: {status_code}")
        if response_content:
            log.error(f"Response preview: {response_content[:200]}")

        return json.dumps(error_context)
