"""Tool calling functionality for Snowflake chat models."""

import json
import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

import requests
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from .._error_handling import SnowflakeRestApiErrorHandler

logger = logging.getLogger(__name__)


class SnowflakeTools:
    """Mixin class for Snowflake tool calling functionality."""

    def bind_tools(
        self,
        tools: Sequence[Union[BaseTool, Callable, dict]],
        auto_execute: bool = True,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to the chat model using Snowflake Cortex Complete native tool calling.

        Args:
            tools: A list of tools to bind to the model
            auto_execute: Whether to automatically execute planned tools (default: True)
            **kwargs: Additional arguments including tool_choice

        Returns:
            A new ChatSnowflake instance with tools bound and REST API enabled

        Note:
            tool_choice parameter is processed for LangChain compatibility but is not
            supported by Snowflake Cortex REST API. The model will always auto-select
            tools. A warning is logged if tool_choice is set to anything other than "auto".
        """
        # Convert LangChain tools to Snowflake tool format
        snowflake_tools = []
        for tool_obj in tools:
            openai_tool = convert_to_openai_tool(tool_obj)

            # Convert OpenAI format to Snowflake format exactly as per official documentation
            # Key fix: Include descriptions in properties to match official working examples
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
            logger.warning(
                f"tool_choice='{tool_choice}' is processed for LangChain compatibility but "
                f"is not supported by Snowflake Cortex REST API. The model will auto-select "
                f"tools regardless of this setting. Use 'auto' to suppress this warning."
            )

        # Create a new instance with bound tools, tool_choice, AND REST API enabled
        return self.model_copy(
            update={
                "_bound_tools": snowflake_tools,
                "_original_tools": list(tools),  # Store original tools for execution
                "_tool_choice": tool_choice,
                "_use_rest_api": True,  # This is the key - switch to REST API when tools are bound
                "_auto_execute_tools": auto_execute,  # Control automatic tool execution
                "max_tokens": self.max_tokens,  # Use configured max_tokens, not hardcoded value
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

    def _build_rest_api_payload(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Build REST API payload following official Snowflake format."""
        # Convert LangChain messages to REST API format
        api_messages = []

        # Get bound tools for enhanced system prompt
        bound_tools = None
        if hasattr(self, "_bound_tools") and self._bound_tools:
            bound_tools = [{"tool_spec": tool} for tool in self._bound_tools]

        # Add enhanced system message when tools are available
        if self._has_tools():
            enhanced_prompt = self._build_enhanced_system_prompt(bound_tools)
            api_messages.append({"role": "system", "content": enhanced_prompt})

        for message in messages:
            if hasattr(message, "role"):
                role = message.role
            elif isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                role = "user"  # fallback

            api_messages.append({"role": role, "content": message.content})  # type: ignore[dict-item]

        # Build payload exactly as shown in official documentation
        payload = {"model": self.model, "messages": api_messages}

        # Add tools if bound
        if self._has_tools() and hasattr(self, "_bound_tools"):
            payload["tools"] = self._bound_tools
            # DO NOT include tool_choice - Snowflake REST API doesn't support it

        return payload

    def _parse_rest_api_response(self, response: requests.Response, original_messages: List[BaseMessage]) -> ChatResult:
        """Parse REST API response and handle tool calls."""

        # Use lists and dicts to store mutable data
        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        usage_data: Dict[str, Any] = {}

        # Track tool call input buffers separately
        tool_input_buffers: Dict[str, str] = {}

        # Parse streaming response - handle both streaming and non-streaming formats
        try:
            # Check if response has streaming content
            if response.headers.get("content-type", "").startswith("text/event-stream"):
                # Handle streaming response
                for line in response.iter_lines():
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
                    data = SnowflakeRestApiErrorHandler.safe_parse_json_response(
                        response, "tool calling REST API request", logger
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
                    content_parts.append(response.text)

        except Exception as e:
            # Use centralized error handling for response parsing errors
            SnowflakeRestApiErrorHandler.log_error("parse REST API response", e)
            content_parts.append(f"Error parsing response: {str(e)}")

        # Combine all content parts
        full_content = "".join(content_parts)

        # AUTO-EXECUTE TOOLS: Option 1 Implementation
        executed_tool_results = []
        if tool_calls and self._bound_tools and getattr(self, "_auto_execute_tools", True):
            executed_tool_results = self._execute_planned_tools(tool_calls)

            # Enhance content with tool execution results
            if executed_tool_results:
                full_content = self._combine_content_with_tool_results(full_content, executed_tool_results)

        # Create response message using shared factories
        from .utils import SnowflakeMetadataFactory

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

    def _execute_planned_tools(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute the tools that were planned by the LLM.

        Args:
            tool_calls: List of tool calls from LLM response

        Returns:
            List of tool execution results
        """
        results: List[Dict[str, Any]] = []

        for tool_call in tool_calls:
            try:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", f"call_{len(results)}")

                # Find the matching bound tool
                matching_tool = self._find_bound_tool(tool_name)
                if matching_tool:
                    # Execute the tool
                    result = matching_tool.invoke(tool_args)
                    results.append(
                        {
                            "tool_call_id": tool_id,
                            "tool_name": tool_name,
                            "result": result,
                            "status": "success",
                        }
                    )
                else:
                    results.append(
                        {
                            "tool_call_id": tool_id,
                            "tool_name": tool_name,
                            "result": f"Tool '{tool_name}' not found in bound tools",
                            "status": "error",
                        }
                    )

            except Exception as e:
                results.append(
                    {
                        "tool_call_id": tool_call.get("id", f"call_{len(results)}"),
                        "tool_name": tool_call.get("name", "unknown"),
                        "result": f"Tool execution failed: {str(e)}",
                        "status": "error",
                    }
                )
                logger.error(f"Tool execution error for {tool_call.get('name')}: {e}")

        return results

    async def _execute_planned_tools_async(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute the tools that were planned by the LLM asynchronously.

        Args:
            tool_calls: List of tool calls from LLM response

        Returns:
            List of tool execution results
        """
        import asyncio

        # Execute tools concurrently for better performance
        async def execute_single_tool(tool_call):
            try:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", f"call_async_{id(tool_call)}")

                # Find the matching bound tool
                matching_tool = self._find_bound_tool(tool_name)
                if matching_tool:
                    # Use async version if available, otherwise fallback to sync in thread
                    if hasattr(matching_tool, "ainvoke"):
                        result = await matching_tool.ainvoke(tool_args)
                    else:
                        result = await asyncio.to_thread(matching_tool.invoke, tool_args)

                    return {
                        "tool_call_id": tool_id,
                        "tool_name": tool_name,
                        "result": result,
                        "status": "success",
                    }
                else:
                    return {
                        "tool_call_id": tool_id,
                        "tool_name": tool_name,
                        "result": f"Tool '{tool_name}' not found in bound tools",
                        "status": "error",
                    }

            except Exception as e:
                logger.error(f"Async tool execution error for {tool_call.get('name')}: {e}")
                return {
                    "tool_call_id": tool_call.get("id", f"call_async_{id(tool_call)}"),
                    "tool_name": tool_call.get("name", "unknown"),
                    "result": f"Tool execution failed: {str(e)}",
                    "status": "error",
                }

        # Execute all tools concurrently
        if tool_calls:
            results = await asyncio.gather(*[execute_single_tool(tc) for tc in tool_calls])
            return list(results)
        else:
            return []

    def _find_bound_tool(self, tool_name: str):
        """Find a bound tool by name."""
        if not hasattr(self, "_bound_tools") or not self._bound_tools:
            return None

        # _bound_tools contains the original tools passed to bind_tools()
        # They might be stored differently, so we need to check the original tools
        if hasattr(self, "_original_tools"):
            for tool in self._original_tools:
                if hasattr(tool, "name") and tool.name == tool_name:
                    return tool

        # Fallback: check _bound_tools directly
        for tool in self._bound_tools:
            if hasattr(tool, "name") and tool.name == tool_name:
                return tool
            # Also check if it's a dict with tool info
            if isinstance(tool, dict) and tool.get("name") == tool_name:
                # This is a converted tool format, we need the original
                continue

        return None

    def _combine_content_with_tool_results(self, original_content: str, tool_results: List[Dict]) -> str:
        """Combine the original LLM content with tool execution results."""
        if not tool_results:
            return original_content

        # Build enhanced content
        enhanced_parts = [original_content]

        # Add tool results
        if any(result["status"] == "success" for result in tool_results):
            enhanced_parts.append("\n\nTool Execution Results:")
            for result in tool_results:
                if result["status"] == "success":
                    enhanced_parts.append(f"\n• {result['tool_name']}: {result['result']}")
                else:
                    enhanced_parts.append(f"\n• {result['tool_name']}: ❌ {result['result']}")

        return "".join(enhanced_parts)

    async def _parse_rest_api_response_async(self, response, original_messages: List[BaseMessage]) -> ChatResult:
        """Async version of _parse_rest_api_response with async tool execution."""

        # Use lists and dicts to store mutable data
        content_parts = []
        tool_calls = []
        usage_data = {}

        # Parse response (reuse sync logic for parsing)
        try:
            data = SnowflakeRestApiErrorHandler.safe_parse_json_response(
                response, "async tool calling REST API request", logger
            )
            if isinstance(data, dict):
                content_str, extracted_tool_calls, extracted_usage = self._parse_json_response(data)
                content_parts.append(content_str)
                tool_calls.extend(extracted_tool_calls)
                usage_data.update(extracted_usage)
        except Exception as e:
            # Use centralized error handling for async response parsing errors
            SnowflakeRestApiErrorHandler.log_error("parse async REST API response", e)
            content_parts.append(f"Error parsing response: {str(e)}")

        # Combine all content parts
        full_content = "".join(content_parts)

        # AUTO-EXECUTE TOOLS: Async Version
        executed_tool_results = []
        if tool_calls and self._bound_tools and getattr(self, "_auto_execute_tools", True):
            executed_tool_results = await self._execute_planned_tools_async(tool_calls)

            # Enhance content with tool execution results
            if executed_tool_results:
                full_content = self._combine_content_with_tool_results(full_content, executed_tool_results)

        # Create response message using shared factories
        from .utils import SnowflakeMetadataFactory

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
                    from langchain_core.messages.tool import tool_call

                    tool_call_obj = tool_call(
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
                        import json

                        parsed_args = json.loads(tool_input_buffers[recent_tool_id])
                        # Update the tool call with complete args
                        from langchain_core.messages.tool import tool_call

                        updated_tool_call = tool_call(
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

                    # Tool call metadata in content_list (only if delta_type is not 'tool_use')
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
        """Parse a direct JSON response (non-streaming format)."""
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
                    if "tool_calls" in message:
                        tool_calls.extend(message["tool_calls"])
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
        logger.warning("bind_functions is deprecated. Use bind_tools instead.")
        return self.bind_tools(functions, tool_choice=function_call, **kwargs)
