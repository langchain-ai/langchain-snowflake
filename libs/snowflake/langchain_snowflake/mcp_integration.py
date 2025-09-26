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

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from langchain.tools import Tool

from ._error_handling import SnowflakeToolErrorHandler

logger = logging.getLogger(__name__)


class MCPToolWrapper:
    """
    A wrapper that stores MCP tool references and creates a compatible Tool.
    """

    def __init__(self, mcp_tool, mcp_session):
        self.mcp_tool = mcp_tool
        self.mcp_session = mcp_session

        # Create the actual Tool instance
        self._tool = self._create_tool()

    def _create_tool(self) -> Tool:
        """Create the LangChain Tool instance"""

        async def execute_mcp_tool(args: Dict[str, Any]) -> str:
            """Execute the MCP tool via the MCP protocol"""
            try:
                logger.debug(f"Executing MCP tool {self.mcp_tool.name} with args: {args}")

                # Execute via MCP session
                result = await self.mcp_session.call_tool(self.mcp_tool.name, args)

                # Format result for ChatSnowflake
                if hasattr(result, "content"):
                    return str(result.content)
                elif isinstance(result, dict):
                    return json.dumps(result, indent=2)
                else:
                    return str(result)

            except Exception as e:
                # Centralized error handling for consistent MCP tool error responses
                return SnowflakeToolErrorHandler.handle_tool_error(
                    error=e,
                    tool_name=self.mcp_tool.name,
                    operation="execute MCP tool",
                    query=str(args),
                )

        def sync_execute(**args):
            """Sync wrapper that runs async MCP tool execution"""
            try:
                # Handle async context
                loop = asyncio.get_running_loop()
                task = loop.create_task(execute_mcp_tool(args))
                return asyncio.run_coroutine_threadsafe(task, loop).result()
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                return asyncio.run(execute_mcp_tool(args))

        # Create the Tool with proper attributes
        return Tool(
            name=self.mcp_tool.name,
            description=f"MCP Tool: {self.mcp_tool.description}",
            func=sync_execute,
            coroutine=execute_mcp_tool,
            args_schema=getattr(self.mcp_tool, "args_schema", None),
        )

    def get_tool(self) -> Tool:
        """Get the wrapped Tool instance"""
        return self._tool

    # Delegate Tool methods/attributes
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped Tool"""
        return getattr(self._tool, name)


def create_mcp_tool_wrapper(mcp_tool, mcp_session) -> Tool:
    """
    Create a LangChain Tool that wraps an MCP tool for ChatSnowflake compatibility.

    This function creates a new Tool instance that bridges the gap between MCP
    protocol execution and Snowflake's expected tool format.

    Args:
        mcp_tool: The original MCP tool from load_mcp_tools()
        mcp_session: Active MCP ClientSession for tool execution

    Returns:
        A LangChain Tool that executes via MCP protocol
    """
    wrapper = MCPToolWrapper(mcp_tool, mcp_session)
    return wrapper.get_tool()


def create_snowflake_compatible_tools(
    mcp_tools: List,
    mcp_session,
    tool_prefix: Optional[str] = None,
    include_tools: Optional[List[str]] = None,
    exclude_tools: Optional[List[str]] = None,
) -> List[Tool]:
    """
    Convert MCP tools to Snowflake-compatible tools for bind_tools()

    Args:
        mcp_tools: List of MCP tools from load_mcp_tools()
        mcp_session: Active MCP ClientSession
        tool_prefix: Optional prefix for tool names (e.g., "mcp_")
        include_tools: If provided, only include tools with these names
        exclude_tools: If provided, exclude tools with these names

    Returns:
        List of LangChain Tools compatible with ChatSnowflake.bind_tools()

    Example:
        >>> tools = create_snowflake_compatible_tools(
        ...     mcp_tools,
        ...     mcp_session,
        ...     include_tools=["cortex_search", "run_sql"]
        ... )
    """
    compatible_tools = []

    for mcp_tool in mcp_tools:
        # Apply filters
        if include_tools and mcp_tool.name not in include_tools:
            continue
        if exclude_tools and mcp_tool.name in exclude_tools:
            continue

        try:
            # Debug: Check tool attributes
            logger.debug(f"Processing MCP tool: {mcp_tool.name}")
            logger.debug(f"Tool type: {type(mcp_tool)}")
            logger.debug(f"Tool attributes: {dir(mcp_tool)}")

            # Optionally add prefix to avoid name conflicts
            if tool_prefix:
                original_name = mcp_tool.name
                mcp_tool.name = f"{tool_prefix}{original_name}"

            wrapper = create_mcp_tool_wrapper(mcp_tool, mcp_session)

            # Verify wrapper was created correctly (it's now a Tool instance)
            if not hasattr(wrapper, "name") or not hasattr(wrapper, "description"):
                raise ValueError("Wrapper missing Tool attributes")
            if not hasattr(wrapper, "func"):
                raise ValueError("Wrapper missing func attribute")

            compatible_tools.append(wrapper)
            logger.info(f"Successfully wrapped MCP tool: {mcp_tool.name}")

        except Exception as e:
            # Use centralized error handling for MCP tool wrapping failures
            tool_name = getattr(mcp_tool, "name", "unknown")
            SnowflakeToolErrorHandler.handle_tool_error(
                error=e, tool_name=tool_name, operation="wrap MCP tool for Snowflake compatibility"
            )
            # Continue processing other tools (don't fail entire batch for one tool)
            continue

    logger.info(f"Created {len(compatible_tools)} Snowflake-compatible MCP tools")
    return compatible_tools


def bind_mcp_tools(
    llm,
    mcp_tools: List,
    mcp_session,
    auto_execute: bool = True,
    tool_prefix: Optional[str] = None,
    include_tools: Optional[List[str]] = None,
    exclude_tools: Optional[List[str]] = None,
    **kwargs,
):
    """
    Convenience function to bind MCP tools to ChatSnowflake

    This is the main entry point for MCP integration. It handles the
    conversion of MCP tools to Snowflake-compatible format and binds
    them to the ChatSnowflake instance.

    Args:
        llm: ChatSnowflake instance
        mcp_tools: List of MCP tools from load_mcp_tools()
        mcp_session: Active MCP ClientSession
        auto_execute: Whether to auto-execute tools (default: True)
        tool_prefix: Optional prefix for tool names
        include_tools: If provided, only include tools with these names
        exclude_tools: If provided, exclude tools with these names
        **kwargs: Additional arguments for bind_tools

    Returns:
        ChatSnowflake instance with MCP tools bound

    Example:
        >>> agent = bind_mcp_tools(
        ...     llm,
        ...     mcp_tools,
        ...     mcp_session,
        ...     auto_execute=True,
        ...     include_tools=["cortex_search", "run_sql"]
        ... )
        >>> response = await agent.ainvoke("search for customer data")
    """
    compatible_tools = create_snowflake_compatible_tools(
        mcp_tools, mcp_session, tool_prefix=tool_prefix, include_tools=include_tools, exclude_tools=exclude_tools
    )

    if not compatible_tools:
        logger.warning("No compatible MCP tools found. Check your filters and MCP connection.")
        return llm

    return llm.bind_tools(compatible_tools, auto_execute=auto_execute, **kwargs)
