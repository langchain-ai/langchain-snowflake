"""Tool calling functionality for Snowflake chat models."""

import json
import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from .._error_handling import SnowflakeErrorHandler, SnowflakeRestApiErrorHandler
from .utils import SnowflakeMetadataFactory

logger = logging.getLogger(__name__)


class SnowflakeTools:
    """Mixin class for Snowflake tool calling functionality."""

    def bind_tools(
        self,
        tools: Sequence[Union[BaseTool, Callable, dict]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to the chat model using Snowflake Cortex Complete native tool calling.

        Args:
            tools: A list of tools to bind to the model
            **kwargs: Additional arguments including tool_choice

        Returns:
            A new ChatSnowflake instance with tools bound and REST API enabled
        """
        # Convert LangChain tools to Snowflake tool format
        snowflake_tools = []
        for tool_obj in tools:
            openai_tool = convert_to_openai_tool(tool_obj)

            # Convert OpenAI format to Snowflake format exactly as per official documentation
            input_schema = openai_tool["function"]["parameters"].copy()

            # Ensure properties have descriptions (required for tool calling to work properly)
            if "properties" in input_schema:
                for prop_name, prop_def in input_schema["properties"].items():
                    if "description" not in prop_def:
                        # Add default description if missing
                        prop_def["description"] = f"The {prop_name} parameter"

            tool_spec = {
                "type": "generic",
                "name": openai_tool["function"]["name"],
                "description": openai_tool["function"]["description"],  # Add description field
                "input_schema": input_schema,
            }

            snowflake_tools.append({"tool_spec": tool_spec})

        # Handle tool_choice (processed for LangChain compatibility but not sent to Snowflake)
        tool_choice = kwargs.pop("tool_choice", "auto")
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "tool" and "name" in tool_choice:
                tool_choice = tool_choice["name"]
            else:
                tool_choice = "auto"
        elif tool_choice not in ["auto", "none", "any", "required"]:
            tool_choice = "auto"

        # Warn users if they're trying to use tool_choice other than "auto"
        if tool_choice != "auto":
            SnowflakeErrorHandler.log_info(
                "tool_choice compatibility",
                f"tool_choice='{tool_choice}' is processed for LangChain compatibility but "
                f"is not supported by Snowflake Cortex REST API. The model will auto-select "
                f"tools regardless of this setting. Use 'auto' to suppress this warning.",
                logger,
            )

        # Create a new instance with bound tools, tool_choice, AND REST API enabled
        return self.model_copy(
            update={
                "_bound_tools": snowflake_tools,
                "_original_tools": list(tools),  # Store original tools for execution
                "_tool_choice": tool_choice,
                "_use_rest_api": True,  # This is the key - switch to REST API when tools are bound
                "max_tokens": self.max_tokens,
            }
        )

    def _build_enhanced_system_prompt(self, tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build an enhanced system prompt for better tool calling and reasoning."""
        if not tools or not self._has_tools():
            return """You are a helpful, knowledgeable assistant. Provide accurate, thoughtful responses 
                        based on the information available to you. If you're uncertain about something, 
                        acknowledge that uncertainty rather than guessing."""

        # Extract tool descriptions for the prompt
        tool_descriptions = []
        for tool in tools:
            tool_spec = tool.get("tool_spec", {})
            name = tool_spec.get("name", "unknown")
            description = tool_spec.get("description", "No description available")
            tool_descriptions.append(f"- **{name}**: {description}")

        tool_list = "\n".join(tool_descriptions)

        return f"""You are an intelligent assistant with access to tools. 
                    Your role is to help users by understanding their needs and 
                    using available tools when appropriate.

## Available Tools:
{tool_list}

## Guidelines for Tool Usage:

1. **Analyze First**: Carefully understand what the user is asking for and determine 
if any tools can help provide a better answer.

2. **Choose Wisely**: Select the most appropriate tool(s) based on:
   - The user's specific question or request
   - The capabilities and purpose of each tool
   - The likelihood that the tool will provide relevant, accurate information

3. **Use Tools Effectively**: When calling tools:
   - Provide clear, specific parameters that will yield useful results
   - Call tools in a logical sequence if multiple tools are needed
   - Use tool results to inform your final response

4. **Reasoning Process**: 
   - If multiple tools could be relevant, choose the most specific and appropriate one
   - If a tool doesn't provide the expected information, acknowledge this and either 
   try an alternative approach or explain the limitation
   - Combine tool results with your knowledge to provide comprehensive, accurate answers

5. **Response Quality**:
   - Always provide clear, well-structured responses
   - Cite tool results when they inform your answer
   - If tools are not needed or relevant, provide direct answers based on your knowledge
   - Be transparent about when you're using tools vs. providing direct knowledge

6. **Error Handling**: If a tool fails or returns unexpected results, explain this to the user 
and provide alternative information when possible.

Remember: Use tools to enhance your responses, not replace thoughtful analysis. 
The goal is to provide the most helpful, accurate, and relevant information to the user."""

    def _group_consecutive_tool_messages(
        self, messages: List[BaseMessage]
    ) -> List[Union[BaseMessage, List[ToolMessage]]]:
        """
        Group consecutive ToolMessage objects together.

        Args:
            messages: List of messages to process

        Returns:
            List where consecutive ToolMessage objects are grouped into sublists,
            other messages remain as individual items
        """
        if not getattr(self, "group_tool_messages", True):
            return messages  # Maintain old behavior if disabled

        result = []
        current_tool_group = []

        for message in messages:
            if isinstance(message, ToolMessage):
                current_tool_group.append(message)
            else:
                # If we have accumulated tool messages, add them as a group
                if current_tool_group:
                    result.append(current_tool_group)
                    current_tool_group = []
                result.append(message)

        # Handle any remaining tool messages
        if current_tool_group:
            result.append(current_tool_group)

        return result

    def _process_tool_message_group(self, tool_messages: List[ToolMessage]) -> Dict[str, Any]:
        """
        Process a group of ToolMessage objects into a single user message.

        Args:
            tool_messages: List of consecutive ToolMessage objects

        Returns:
            Dictionary representing a user message with grouped tool results
        """
        tool_results = []

        for tool_msg in tool_messages:
            tool_results.append(
                {
                    "type": "tool_results",
                    "tool_results": {
                        "tool_use_id": tool_msg.tool_call_id,
                        "name": getattr(tool_msg, "name", None) or "unknown",
                        "content": [{"type": "text", "text": str(tool_msg.content)}],
                    },
                }
            )

        return {"role": "user", "content_list": tool_results}

    def _process_single_message(self, message: BaseMessage) -> Dict[str, Any]:
        """
        Process a single message (non-ToolMessage) into API format.

        Args:
            message: Single message to process

        Returns:
            Dictionary representing the message in API format
        """
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}

        elif isinstance(message, AIMessage):
            if hasattr(message, "tool_calls") and message.tool_calls:
                # Build content_list for Snowflake Complete API
                content_list: List[Dict[str, Any]] = []

                # Add text content if present
                if message.content:
                    content_list.append({"type": "text", "text": message.content})

                # Add tool_use blocks
                for tool_call in message.tool_calls:
                    # Handle both dict and ToolCall object formats
                    if isinstance(tool_call, dict):
                        tool_call_id = tool_call.get("id", "") or ""
                        tool_call_name = tool_call["name"]
                        tool_call_args = tool_call["args"]
                    else:
                        # ToolCall object - access as attributes
                        tool_call_id = getattr(tool_call, "id", None) or ""
                        tool_call_name = getattr(tool_call, "name", None)
                        tool_call_args = getattr(tool_call, "args", {})

                    # Ensure we have a valid ID
                    if not tool_call_id:
                        tool_call_id = f"toolu_{id(tool_call)}"

                    # Process input: keep dicts/objects as-is, convert other types appropriately
                    if isinstance(tool_call_args, dict):
                        tool_input = tool_call_args
                    elif isinstance(tool_call_args, (str, list)):
                        tool_input = tool_call_args
                    else:
                        # For other types, try to convert to dict or string
                        tool_input = tool_call_args if isinstance(tool_call_args, (dict, list)) else str(tool_call_args)

                    content_list.append(
                        {
                            "type": "tool_use",
                            "tool_use": {
                                "tool_use_id": str(tool_call_id),
                                "name": tool_call_name,
                                "input": tool_input,
                            },
                        }
                    )

                return {"role": "assistant", "content_list": content_list}
            else:
                return {"role": "assistant", "content": message.content or ""}

        elif isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}

        else:
            # Fallback for unknown message types
            content = str(message.content) if hasattr(message, "content") else str(message)
            return {"role": "user", "content": content}

    def _build_rest_api_payload(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """
        Build REST API payload.

        Args:
            messages: List of LangChain messages

        Returns:
            Dictionary representing the API payload
        """
        api_messages: List[Dict[str, Any]] = []

        # Get bound tools for enhanced system prompt
        bound_tools = None
        if hasattr(self, "_bound_tools") and self._bound_tools:
            bound_tools = [{"tool_spec": tool} for tool in self._bound_tools]

        # Add enhanced system message when tools are available
        if self._has_tools():
            enhanced_prompt = self._build_enhanced_system_prompt(bound_tools)
            api_messages.append({"role": "system", "content": enhanced_prompt})

        # Group messages properly
        grouped_messages = self._group_consecutive_tool_messages(messages)

        # Process each group or individual message
        for item in grouped_messages:
            if isinstance(item, list):  # Tool message group
                user_message = self._process_tool_message_group(item)
                api_messages.append(user_message)
            else:  # Single message
                api_message = self._process_single_message(item)
                api_messages.append(api_message)

        # Build payload
        payload = {
            "model": self.model,
            "messages": api_messages,
            "disable_parallel_tool_use": self.disable_parallel_tool_use,
        }

        # Add tools if bound
        if self._has_tools() and hasattr(self, "_bound_tools"):
            payload["tools"] = self._bound_tools

        return payload

    def _parse_rest_api_response(self, response_or_data, original_messages: List[BaseMessage]) -> ChatResult:
        """Parse REST API response and handle tool calls - handles both requests.Response and Dict[str, Any]."""

        # Use lists and dicts to store mutable data
        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        usage_data: Dict[str, Any] = {}

        # Track tool call input buffers separately
        tool_input_buffers: Dict[str, str] = {}

        # Handle different input types
        if isinstance(response_or_data, dict):
            # Handle RestApiClient response (Dict)
            response_data = response_or_data
            headers = {}
            is_streaming = False
        else:
            # Handle requests.Response
            response_data = response_or_data.json()
            headers = response_or_data.headers
            is_streaming = headers.get("content-type", "").startswith("text/event-stream")

        # Parse streaming response - handle both streaming and non-streaming formats
        try:
            if is_streaming:
                # Handle streaming response
                for line in response_or_data.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            try:
                                data = json.loads(line_str[6:])  # Remove 'data: ' prefix
                                self._parse_streaming_chunk(
                                    data,
                                    content_parts,
                                    tool_calls,
                                    usage_data,
                                    tool_input_buffers,
                                )
                            except json.JSONDecodeError:
                                continue  # Skip malformed lines
            else:
                # Handle regular JSON response
                try:
                    if isinstance(response_or_data, dict):
                        # Direct dict response from RestApiClient
                        data = response_data
                    else:
                        # Parse from requests.Response
                        data = SnowflakeRestApiErrorHandler.safe_parse_json_response(
                            response_or_data, "tool calling REST API request", logger
                        )

                    if isinstance(data, dict):
                        # Direct response format
                        content_str, extracted_tool_calls, extracted_usage = self._parse_json_response(data)
                        content_parts.append(content_str)
                        tool_calls.extend(extracted_tool_calls)
                        usage_data.update(extracted_usage)
                    elif isinstance(data, list):
                        # Array of responses
                        for item in data:
                            self._parse_streaming_chunk(
                                item,
                                content_parts,
                                tool_calls,
                                usage_data,
                                tool_input_buffers,
                            )
                except json.JSONDecodeError:
                    # Fallback: treat as plain text
                    if isinstance(response_or_data, dict):
                        content_parts.append(str(response_data))
                    else:
                        content_parts.append(response_or_data.text)

        except Exception as e:
            # Use centralized REST API error handler for response parsing errors
            SnowflakeRestApiErrorHandler.handle_rest_api_response_error(
                error=e, operation="parse REST API response", logger_instance=logger
            )
            content_parts.append(f"Error parsing response: {str(e)}")

        # Combine all content parts
        full_content = "".join(content_parts)

        # Calculate fallback token counts if not provided in usage_data
        input_tokens = usage_data.get("prompt_tokens", self._estimate_tokens(original_messages))
        output_tokens = usage_data.get("completion_tokens", self._estimate_tokens([{"content": full_content}]))

        message = AIMessage(
            content=full_content,
            tool_calls=tool_calls,
            usage_metadata=SnowflakeMetadataFactory.create_usage_metadata(usage_data, input_tokens, output_tokens),
            response_metadata=SnowflakeMetadataFactory.create_response_metadata(
                self.model, finish_reason="tool_calls" if tool_calls else "stop"
            ),
        )

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _parse_rest_api_response_async(self, response, original_messages: List[BaseMessage]) -> ChatResult:
        """Async version of _parse_rest_api_response."""

        # Use lists and dicts to store mutable data
        content_parts = []
        tool_calls = []
        usage_data = {}

        # Parse response - response is already a dict from RestApiClient.make_async_request()
        try:
            # Response is already parsed as dict by RestApiClient.make_async_request()
            if isinstance(response, dict):
                data = response
            else:
                # Fallback: try to parse if it's not a dict
                data = SnowflakeRestApiErrorHandler.safe_parse_json_response(
                    response, "async tool calling REST API request", logger
                )

            if isinstance(data, dict):
                content_str, extracted_tool_calls, extracted_usage = self._parse_json_response(data)
                content_parts.append(content_str)
                tool_calls.extend(extracted_tool_calls)
                usage_data.update(extracted_usage)
        except Exception as e:
            # Use centralized REST API error handler for async response parsing errors
            SnowflakeRestApiErrorHandler.handle_rest_api_response_error(
                error=e, operation="parse async REST API response", logger_instance=logger
            )
            content_parts.append(f"Error parsing response: {str(e)}")

        # Combine all content parts
        full_content = "".join(content_parts)

        # Calculate fallback token counts if not provided in usage_data
        input_tokens = usage_data.get("prompt_tokens", self._estimate_tokens(original_messages))
        output_tokens = usage_data.get("completion_tokens", self._estimate_tokens([{"content": full_content}]))

        message = AIMessage(
            content=full_content,
            tool_calls=tool_calls,
            usage_metadata=SnowflakeMetadataFactory.create_usage_metadata(usage_data, input_tokens, output_tokens),
            response_metadata=SnowflakeMetadataFactory.create_response_metadata(
                self.model, finish_reason="tool_calls" if tool_calls else "stop"
            ),
        )

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _parse_streaming_chunk(
        self,
        data: dict,
        content_parts: list,
        tool_calls: list,
        usage_data: dict,
        tool_input_buffers: dict,
    ) -> None:
        """Parse a single streaming chunk and update the content and tool calls."""
        # Extract content and tool calls from streaming format
        choices = data.get("choices", [])
        for choice in choices:
            delta = choice.get("delta", {})

            # Handle different delta types
            delta_type = delta.get("type")

            if delta_type == "text":
                # Regular text content - ONLY process from delta.content to avoid duplication
                if "content" in delta and delta["content"]:
                    content_parts.append(delta["content"])
                # DO NOT process content_list for text when delta_type is 'text' to avoid duplication

            elif delta_type == "tool_use":
                # Tool calling - using correct LangChain ToolCall format!

                # Check if this is the start of a new tool call
                if "tool_use_id" in delta and "name" in delta:
                    # Create new tool call using LangChain format
                    tool_call_obj = ToolCall(
                        name=delta["name"],
                        args={},  # Will be updated when input is complete
                        id=delta["tool_use_id"],
                    )
                    tool_calls.append(tool_call_obj)
                    # Initialize input buffer for this tool call
                    tool_input_buffers[delta["tool_use_id"]] = ""

                # Check if this chunk contains input data - ONLY process from delta, not content_list
                if "input" in delta and tool_calls:
                    # Get the most recent tool call ID
                    recent_tool_id = tool_calls[-1]["id"]

                    # Accumulate input in buffer
                    tool_input_buffers[recent_tool_id] += delta["input"]

                    # Try to parse the accumulated input as JSON
                    try:
                        parsed_args = json.loads(tool_input_buffers[recent_tool_id])
                        # Update the tool call with complete args
                        updated_tool_call = ToolCall(
                            name=tool_calls[-1]["name"],
                            args=parsed_args,
                            id=tool_calls[-1]["id"],
                        )
                        tool_calls[-1] = updated_tool_call
                    except json.JSONDecodeError:
                        # Still accumulating JSON, keep the buffer
                        pass

            else:
                # Handle content_list for ONLY cases where delta_type is not already handled
                content_list = delta.get("content_list", [])
                for item in content_list:
                    # Text content (only if delta_type is not 'text' to avoid duplication)
                    if item.get("type") == "text" and "text" in item:
                        content_parts.append(item["text"])

                    # Tool call in Snowflake format with nested tool_use object
                    elif item.get("type") == "tool_use" and "tool_use" in item:
                        tool_use = item["tool_use"]
                        tool_use_id = tool_use.get("tool_use_id")
                        tool_name = tool_use.get("name")

                        if tool_use_id and tool_name:
                            existing_call = None
                            for tc in tool_calls:
                                if tc["id"] == tool_use_id:
                                    existing_call = tc
                                    break

                            if not existing_call:
                                # Create new tool call using LangChain format
                                from langchain_core.messages.tool import tool_call

                                tool_call_obj = tool_call(
                                    name=tool_name,
                                    args=tool_use.get("input") or {},  # Handle null/None values
                                    id=tool_use_id,
                                )
                                tool_calls.append(tool_call_obj)

                    # Tool call metadata in content_list (legacy format)
                    elif "tool_use_id" in item and "name" in item:
                        existing_call = None
                        for tc in tool_calls:
                            if tc["id"] == item["tool_use_id"]:
                                existing_call = tc
                                break

                        if not existing_call:
                            # Create new tool call using LangChain format
                            from langchain_core.messages.tool import tool_call

                            tool_call_obj = tool_call(name=item["name"], args={}, id=item["tool_use_id"])
                            tool_calls.append(tool_call_obj)
                            tool_input_buffers[item["tool_use_id"]] = ""

                    # DO NOT process 'input' from content_list to avoid duplication
                    # Input is only processed from the main delta object above

        # Update usage info
        if "usage" in data:
            usage_data.update(data["usage"])

    def _parse_json_response(self, data: dict) -> tuple[str, list, dict]:
        """Parse a direct JSON response (non-streaming format) with content_list support."""
        full_content = ""
        tool_calls = []
        usage_data = {}

        # Handle direct response format
        if "choices" in data:
            choices = data["choices"]
            for choice in choices:
                if "message" in choice:
                    message = choice["message"]
                    full_content += message.get("content", "")

                    # Check for tool_calls in standard format
                    if "tool_calls" in message:
                        tool_calls.extend(message["tool_calls"])

                    # Check for tool calls in Snowflake's content_list format
                    if "content_list" in message:
                        for item in message["content_list"]:
                            if item.get("type") == "tool_use" and "tool_use" in item:
                                tool_use = item["tool_use"]
                                # Convert to LangChain format
                                tool_calls.append(
                                    {
                                        "id": tool_use.get("tool_use_id", f"call_{len(tool_calls)}"),
                                        "name": tool_use.get("name"),
                                        "args": tool_use.get("input") or {},  # Handle null/None values
                                    }
                                )
                elif "messages" in choice:
                    # Alternative format
                    full_content += choice.get("messages", "")

        # Extract usage data
        if "usage" in data:
            usage_data = data["usage"]

        return full_content, tool_calls, usage_data

    def bind_functions(
        self,
        functions: Sequence[Union[Dict[str, Any], Callable, BaseTool]],
        function_call: Optional[Union[str, Literal["auto", "none"], Dict]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Legacy support for bind_functions (redirects to bind_tools)."""
        SnowflakeErrorHandler.log_info(
            "deprecated method", "bind_functions is deprecated. Use bind_tools instead.", logger
        )
        return self.bind_tools(functions, tool_choice=function_call, **kwargs)

    def _should_use_rest_api(self) -> bool:
        """Determine whether to use REST API or SQL function based on tool usage.

        The ChatSnowflake class supports dual API modes:
        - REST API: Required for tool calling, streaming, and advanced features
        - SQL Function: Simpler approach for basic chat completions

        Returns:
            True if REST API should be used, False for SQL function
        """
        # Use REST API if explicitly enabled via _use_rest_api attribute
        if getattr(self, "_use_rest_api", False):
            return True

        # Use REST API if tools are bound (tools require REST API)
        if self._has_tools():
            return True

        # Default to SQL function for basic chat
        return False

    def _has_tools(self) -> bool:
        """Check if the model has bound tools.

        Returns:
            True if tools are bound to the model, False otherwise
        """
        return hasattr(self, "_bound_tools") and bool(self._bound_tools)
