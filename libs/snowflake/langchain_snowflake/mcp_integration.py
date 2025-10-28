"""
MCP Integration for langchain-snowflake

Provides utilities to integrate Model Context Protocol (MCP) tools
with ChatSnowflake's bind_tools() functionality.

Example:
    > from langchain_snowflake import ChatSnowflake, bind_mcp_tools
    > from langchain_mcp_adapters import load_mcp_tools
    >
    > # Load MCP tools
    > mcp_tools = await load_mcp_tools(mcp_session)
    >
    > # Bind to ChatSnowflake
    > agent = bind_mcp_tools(llm, mcp_tools, mcp_session)
    >
    > # Use normally
    > response = await agent.ainvoke("list databases")
"""

import logging
from typing import Any, List, Optional

from langchain.tools import Tool

from ._error_handling import SnowflakeErrorHandler, SnowflakeToolErrorHandler

logger = logging.getLogger(__name__)


class MCPToolWrapper:
    """
    Wrapper to make MCP tools compatible with LangChain's tool interface.

    This wrapper handles the conversion between LangChain's tool calling format
    and MCP's tool execution format, ensuring seamless integration.
    """

    def __init__(self, mcp_tool, mcp_session):
        """Initialize the MCP tool wrapper.

        Args:
            mcp_tool: The MCP tool instance
            mcp_session: The MCP session for tool execution
        """
        self.mcp_tool = mcp_tool
        self.mcp_session = mcp_session

    def run(self, *args, **kwargs) -> str:
        """Execute the MCP tool with LangChain compatibility.

        Args:
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool

        Returns:
            Tool execution result as string
        """
        try:
            SnowflakeErrorHandler.log_debug("MCP tool execution", f"executing {self.mcp_tool.name} with args: {args}")

            # Convert LangChain args to MCP format
            if args and len(args) == 1 and isinstance(args[0], str):
                # Single string argument - common case
                tool_input = args[0]
            elif kwargs:
                # Keyword arguments - convert to MCP format
                tool_input = kwargs
            else:
                # Multiple positional arguments
                tool_input = list(args)

            # Execute MCP tool
            result = self.mcp_session.call_tool(self.mcp_tool.name, tool_input)

            # Convert result to string format expected by LangChain
            if isinstance(result, dict):
                return str(result)
            elif isinstance(result, list):
                return "\n".join(str(item) for item in result)
            else:
                return str(result)

        except Exception as e:
            # Centralized error handling for consistent MCP tool error responses
            return SnowflakeToolErrorHandler.handle_tool_error(
                error=e,
                tool_name=self.mcp_tool.name,
                operation="MCP tool execution",
                logger_instance=logger,
            )


def create_langchain_tool_from_mcp(mcp_tool, mcp_session) -> Tool:
    """
    Convert an MCP tool to a LangChain Tool.

    Args:
        mcp_tool: The MCP tool to convert
        mcp_session: The MCP session for tool execution

    Returns:
        LangChain Tool instance

    Raises:
        ValueError: If tool conversion fails
    """
    try:
        # Create wrapper for MCP tool
        wrapper = MCPToolWrapper(mcp_tool, mcp_session)

        # Extract tool metadata
        name = getattr(mcp_tool, "name", "unknown_mcp_tool")
        description = getattr(mcp_tool, "description", f"MCP tool: {name}")

        # Create LangChain Tool
        langchain_tool = Tool(
            name=name,
            description=description,
            func=wrapper.run,
        )

        return langchain_tool

    except Exception as e:
        SnowflakeToolErrorHandler.handle_tool_error(
            error=e, tool_name="unknown", operation="convert MCP tool to LangChain Tool"
        )
        raise


def filter_compatible_mcp_tools(
    mcp_tools: List[Any], include_patterns: Optional[List[str]] = None, exclude_patterns: Optional[List[str]] = None
) -> List[Any]:
    """
    Filter MCP tools based on compatibility and patterns.

    Args:
        mcp_tools: List of MCP tools to filter
        include_patterns: Patterns to include (tool names matching these patterns)
        exclude_patterns: Patterns to exclude (tool names matching these patterns)

    Returns:
        List of compatible MCP tools
    """
    compatible_tools = []

    for mcp_tool in mcp_tools:
        try:
            # Basic compatibility check
            if not hasattr(mcp_tool, "name"):
                continue

            tool_name = mcp_tool.name.lower()

            # Apply include patterns
            if include_patterns:
                if not any(pattern.lower() in tool_name for pattern in include_patterns):
                    continue

            # Apply exclude patterns
            if exclude_patterns:
                if any(pattern.lower() in tool_name for pattern in exclude_patterns):
                    continue

            # Debug: Check tool attributes
            SnowflakeErrorHandler.log_debug("MCP tool processing", f"processing {mcp_tool.name}")
            SnowflakeErrorHandler.log_debug("MCP tool processing", f"tool type: {type(mcp_tool)}")
            SnowflakeErrorHandler.log_debug("MCP tool processing", f"tool attributes: {dir(mcp_tool)}")

            # Additional compatibility checks can be added here
            # For now, we accept tools with basic attributes

            compatible_tools.append(mcp_tool)
            SnowflakeErrorHandler.log_info("MCP tool wrapping", f"successfully wrapped {mcp_tool.name}")

        except Exception as e:
            # Log error but continue processing other tools
            tool_name = getattr(mcp_tool, "name", "unknown")
            SnowflakeToolErrorHandler.handle_tool_error(
                error=e, tool_name=tool_name, operation="wrap MCP tool for Snowflake compatibility"
            )
            continue

    SnowflakeErrorHandler.log_info(
        "MCP tool creation", f"created {len(compatible_tools)} Snowflake-compatible MCP tools"
    )
    return compatible_tools


def bind_mcp_tools(
    llm,
    mcp_tools: List[Any],
    mcp_session,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    **bind_kwargs,
):
    """
    Bind MCP tools to a ChatSnowflake instance.

    This function filters MCP tools for compatibility, converts them to LangChain Tools,
    and binds them to the provided LLM using bind_tools().

    Args:
        llm: ChatSnowflake instance or compatible LLM
        mcp_tools: List of MCP tools to bind
        mcp_session: MCP session for tool execution
        include_patterns: Optional patterns to include specific tools
        exclude_patterns: Optional patterns to exclude specific tools
        **bind_kwargs: Additional arguments passed to bind_tools()

    Returns:
        LLM instance with bound MCP tools

    Example:
        >>> from langchain_snowflake import ChatSnowflake, bind_mcp_tools
        >>> from langchain_mcp_adapters import load_mcp_tools
        >>>
        >>> # Load MCP tools
        >>> mcp_tools = await load_mcp_tools(mcp_session)
        >>>
        >>> # Create ChatSnowflake instance
        >>> llm = ChatSnowflake(...)
        >>>
        >>> # Bind MCP tools
        >>> agent = bind_mcp_tools(
        ...     llm,
        ...     mcp_tools,
        ...     mcp_session,
        ...     include_patterns=["database", "query"],
        ...     exclude_patterns=["admin"]
        ... )
        >>>
        >>> # Use the agent
        >>> response = await agent.ainvoke("List all databases")
    """
    # Filter compatible tools
    compatible_tools = filter_compatible_mcp_tools(mcp_tools, include_patterns, exclude_patterns)

    if not compatible_tools:
        SnowflakeErrorHandler.log_warning_and_fallback(
            error=Exception("No compatible MCP tools found"),
            operation="MCP tool filtering",
            fallback_action="returning original LLM without tools",
        )
        return llm

    # Convert to LangChain Tools
    langchain_tools = []
    for mcp_tool in compatible_tools:
        try:
            langchain_tool = create_langchain_tool_from_mcp(mcp_tool, mcp_session)
            langchain_tools.append(langchain_tool)
        except Exception as e:
            # Log error but continue with other tools
            tool_name = getattr(mcp_tool, "name", "unknown")
            SnowflakeToolErrorHandler.handle_tool_error(
                error=e, tool_name=tool_name, operation="convert MCP tool to LangChain Tool"
            )
            continue

    # Bind tools to LLM
    if langchain_tools:
        return llm.bind_tools(langchain_tools, **bind_kwargs)
    else:
        SnowflakeErrorHandler.log_warning_and_fallback(
            error=Exception("No LangChain tools created from MCP tools"),
            operation="MCP tool conversion",
            fallback_action="returning original LLM without tools",
        )
        return llm
