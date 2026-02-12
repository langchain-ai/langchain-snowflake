"""Snowflake Cortex Analyst tool for Text2SQL conversions."""

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Tuple, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from snowflake.snowpark import Session

from .._connection import RestApiClient, RestApiRequestBuilder, SnowflakeConnectionMixin
from .._error_handling import SnowflakeErrorHandler, SnowflakeToolErrorHandler
from ._base import SnowflakeCortexAnalystInput

logger = logging.getLogger(__name__)


class SnowflakeCortexAnalyst(BaseTool, SnowflakeConnectionMixin):
    """Snowflake Cortex Analyst integration for Text2SQL.

    Cortex Analyst is Snowflake's AI-powered tool that converts natural language
    questions into SQL queries and executes them against your Snowflake data.

    Setup:
        Install ``langchain-snowflake`` and configure Snowflake connection.

        .. code-block:: bash

            pip install -U langchain-snowflake

    Key init args:
        session: Optional[Session]
            Active Snowflake session
        account: str
            Snowflake account identifier
        user: str
            Snowflake username
        database: str
            Database to query
        schema: str
            Schema to query
        warehouse: Optional[str]
            Warehouse to use for queries
        use_rest_api: bool
            Whether to use REST API (default: True) or SQL function fallback
        semantic_model_file: Optional[str]
            Path to semantic model file on Snowflake stage

    Instantiate:
        .. code-block:: python

            from . import SnowflakeCortexAnalyst

            analyst = SnowflakeCortexAnalyst(
                session=session,
                semantic_model_file="@my_stage/semantic_model.yaml"
            )

    Use within an agent:
        .. code-block:: python

            from langchain.agents import AgentExecutor, create_tool_calling_agent
            from langchain_core.prompts import ChatPromptTemplate
            from .. import ChatSnowflake

            llm = ChatSnowflake(model="claude-3-5-sonnet", session=session)
            tools = [analyst]

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful data analyst assistant."),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])

            agent = create_tool_calling_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools)

            response = agent_executor.invoke({
                "input": "What were our top 5 products by revenue last quarter?"
            })
    """

    # Tool configuration
    name: str = "snowflake_cortex_analyst"
    description: str = (
        "Converts natural language questions into SQL queries using Snowflake Cortex Analyst. "
        "Use this tool when you need to query structured data in Snowflake databases."
    )
    args_schema: Union[Type[BaseModel], Dict[str, Any], None] = SnowflakeCortexAnalystInput

    # Additional fields specific to Cortex Analyst (connection fields inherited from SnowflakeConnectionMixin)
    role: Optional[str] = Field(default=None, description="Snowflake role to use")

    # Analyst configuration
    semantic_model_file: Optional[str] = Field(default=None)
    semantic_view: Optional[str] = Field(default=None)
    use_rest_api: bool = Field(default=True)
    enable_streaming: bool = Field(default=False)

    def __init__(self, **kwargs):
        """Initialize the Cortex Analyst tool with proper session attribute."""
        # Call the parent initializer
        super().__init__(**kwargs)
        # Ensure _session attribute is initialized (from SnowflakeConnectionMixin)
        if not hasattr(self, "_session"):
            self._session = None

    def _build_rest_api_payload(self, query: str, semantic_model: Optional[str] = None) -> Dict[str, Any]:
        """Build REST API payload for analyst request."""
        payload = {"messages": [{"role": "user", "content": [{"type": "text", "text": query}]}]}

        # Add semantic model configuration
        if semantic_model or self.semantic_model_file:
            model_ref = semantic_model or self.semantic_model_file
            if model_ref and model_ref.startswith("@"):
                payload["semantic_model_file"] = model_ref
            else:
                payload["semantic_model"] = model_ref
        elif self.semantic_view:
            payload["semantic_view"] = self.semantic_view

        # Add streaming if enabled
        if self.enable_streaming:
            payload["stream"] = True

        return payload

    # ============================================================================
    # Shared Helper Methods
    # ============================================================================

    def _prepare_analyst_request(self, query: str, semantic_model: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
        """Prepare Cortex Analyst request configuration.

        Args:
            query: Natural language query to convert to SQL
            semantic_model: Optional semantic model to use

        Returns:
            Tuple of (request_config, operation_name)
        """
        session = self._get_session()
        payload = self._build_rest_api_payload(query, semantic_model)

        # Get proxy configuration from session if available
        proxies = None
        if hasattr(session, "_conn") and hasattr(session._conn, "_conn"):
            conn = session._conn._conn
            if hasattr(conn, "proxy_host") and hasattr(conn, "proxy_port"):
                proxy_host = conn.proxy_host
                proxy_port = conn.proxy_port
                if proxy_host and proxy_port:
                    proxy_url = f"http://{proxy_host}:{proxy_port}"
                    proxies = {"http": proxy_url, "https": proxy_url}
                    logger.debug(f"Using proxy from session: {proxy_url}")

        request_config = RestApiRequestBuilder.cortex_analyst_request(
            session=session,
            method="POST",
            payload=payload,
            request_timeout=self.request_timeout,
            verify_ssl=self.verify_ssl,
            stream=self.enable_streaming,
            proxies=proxies,
        )

        operation_name = "Cortex Analyst REST API request"
        return request_config, operation_name

    def _handle_analyst_error(self, error: Exception, operation: str) -> str:
        """Handle Cortex Analyst errors consistently.

        Args:
            error: Exception that occurred
            operation: Operation description

        Returns:
            Error message string
        """
        error_msg = SnowflakeToolErrorHandler.handle_rest_api_error(
            error=error,
            tool_name="cortex_analyst",
            operation=operation,
            logger_instance=logger,
        )
        return error_msg

    # ============================================================================
    # REST API Methods
    # ============================================================================

    def _make_rest_api_request(self, query: str, semantic_model: Optional[str] = None) -> str:
        """Make REST API request to Cortex Analyst service using unified client."""
        try:
            request_config, operation_name = self._prepare_analyst_request(query, semantic_model)
            response_data = RestApiClient.make_sync_request(request_config, operation_name)
            return self._parse_rest_api_response(response_data)

        except Exception as e:
            # Log the error for debugging and re-raise to trigger fallback logic
            self._handle_analyst_error(e, "REST API request")
            raise e

    async def _make_rest_api_request_async(self, query: str, semantic_model: Optional[str] = None) -> str:
        """Make async REST API request to Snowflake Cortex Analyst using unified client."""
        try:
            request_config, operation_name = self._prepare_analyst_request(query, semantic_model)
            response_data = await RestApiClient.make_async_request(request_config, f"async {operation_name}")
            return self._parse_rest_api_response(response_data)

        except Exception as e:
            # Log the error for debugging and re-raise to trigger fallback logic
            self._handle_analyst_error(e, "async REST API request")
            raise e

    def _parse_rest_api_response(self, data: Dict[str, Any]) -> str:
        """Parse REST API response from Cortex Analyst."""
        try:
            message = data.get("message", {})
            content = message.get("content", [])

            result = {
                "request_id": data.get("request_id"),
                "content": [],
                "warnings": data.get("warnings", []),
                "metadata": data.get("response_metadata", {}),
            }

            # Extract different content types
            for item in content:
                content_type = item.get("type")
                if content_type == "text":
                    result["content"].append({"type": "text", "text": item.get("text", "")})
                elif content_type == "sql":
                    result["content"].append(
                        {
                            "type": "sql",
                            "statement": item.get("statement", ""),
                            "confidence": item.get("confidence", {}),
                        }
                    )
                elif content_type == "suggestions":
                    result["content"].append(
                        {
                            "type": "suggestions",
                            "suggestions": item.get("suggestions", []),
                        }
                    )

            return json.dumps(result, indent=2)

        except Exception as e:
            return SnowflakeToolErrorHandler.handle_tool_error(
                error=e,
                tool_name="cortex_analyst",
                operation="parsing REST API response",
                logger_instance=logger,
            )

    def _fallback_sql_function(self, query: str, semantic_model: Optional[str] = None) -> str:
        """Fallback method using Cortex Complete function for text-to-SQL.

        Note: SNOWFLAKE.CORTEX.ANALYST does not exist as a SQL function.
        This method uses CORTEX.COMPLETE for text-to-SQL generation instead.
        """
        session = self._get_session()

        try:
            # Use CORTEX.COMPLETE for text-to-SQL since CORTEX.ANALYST doesn't exist
            SnowflakeErrorHandler.log_info("SQL fallback", "Using Cortex Complete for text-to-SQL fallback", logger)
            return self._fallback_text2sql(session, query)

        except Exception as e:
            # Log the SQL fallback error
            SnowflakeToolErrorHandler.handle_sql_error(
                error=e,
                tool_name="cortex_analyst",
                sql_query=query,
                operation="SQL function fallback",
                logger_instance=logger,
            )
            # Note: SnowflakeToolErrorHandler.handle_sql_error already logs the error
            # Re-raise so the main error handler can decide what to do
            raise e

    async def _fallback_sql_function_async(self, query: str, semantic_model: Optional[str] = None) -> str:
        """Async fallback method using Cortex Complete function for text-to-SQL with native Snowflake async.

        Note: SNOWFLAKE.CORTEX.ANALYST does not exist as a SQL function.
        This method uses CORTEX.COMPLETE for text-to-SQL generation instead.
        """
        session = self._get_session()

        try:
            # Use CORTEX.COMPLETE for text-to-SQL since CORTEX.ANALYST doesn't exist
            SnowflakeErrorHandler.log_info(
                "async SQL fallback", "Using async Cortex Complete for text-to-SQL fallback", logger
            )
            return await self._afallback_text2sql(session, query)

        except Exception as e:
            # Log the async SQL fallback error
            SnowflakeToolErrorHandler.handle_sql_error(
                error=e,
                tool_name="cortex_analyst",
                sql_query=query,
                operation="async SQL function fallback",
                logger_instance=logger,
            )
            # Note: SnowflakeToolErrorHandler.handle_sql_error already logs the error
            # Re-raise so the main error handler can decide what to do
            raise e

    def _run(
        self,
        query: str,
        semantic_model: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute a natural language query using Cortex Analyst.

        Args:
            query: Natural language question about the data
            semantic_model: Optional semantic model to use
            run_manager: Callback manager for the tool run

        Returns:
            JSON string containing SQL query, results, and explanation
        """
        try:
            if self.use_rest_api:
                return self._make_rest_api_request(query, semantic_model)
            else:
                return self._fallback_sql_function(query, semantic_model)
        except Exception as e:
            # Try fallback method if REST API fails
            if self.use_rest_api:
                SnowflakeErrorHandler.log_info("fallback strategy", "Falling back to SQL function", logger)
                try:
                    return self._fallback_sql_function(query, semantic_model)
                except Exception as fallback_error:
                    # Both REST API and SQL fallback failed
                    SnowflakeErrorHandler.log_error(
                        "complete fallback failure",
                        Exception(
                            f"Both REST API and SQL fallback failed. "
                            f"REST API error: {e}, Fallback error: {fallback_error}"
                        ),
                        logger,
                    )
                    return SnowflakeToolErrorHandler.handle_tool_error(
                        error=fallback_error,
                        tool_name="cortex_analyst",
                        operation="execution with fallback",
                        logger_instance=logger,
                    )
            return SnowflakeToolErrorHandler.handle_tool_error(
                error=e,
                tool_name="cortex_analyst",
                operation="execution",
                logger_instance=logger,
            )

    async def _arun(
        self,
        query: str,
        semantic_model: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async execute a natural language query using Cortex Analyst with true async patterns.

        Args:
            query: Natural language question about the data
            semantic_model: Optional semantic model to use
            run_manager: Async callback manager for the tool run

        Returns:
            JSON string containing SQL query, results, and explanation
        """
        try:
            if self.use_rest_api:
                return await self._make_rest_api_request_async(query, semantic_model)
            else:
                return await self._fallback_sql_function_async(query, semantic_model)
        except Exception as e:
            # Try fallback method if REST API fails
            if self.use_rest_api:
                SnowflakeErrorHandler.log_info("async fallback strategy", "Falling back to async SQL function", logger)
                try:
                    return await self._fallback_sql_function_async(query, semantic_model)
                except Exception as fallback_error:
                    # Both async REST API and SQL fallback failed
                    SnowflakeErrorHandler.log_error(
                        "complete async fallback failure",
                        Exception(
                            f"Both async REST API and SQL fallback failed. "
                            f"REST API error: {e}, Fallback error: {fallback_error}"
                        ),
                        logger,
                    )
                    return SnowflakeToolErrorHandler.handle_tool_error(
                        error=fallback_error,
                        tool_name="cortex_analyst",
                        operation="async execution with fallback",
                        logger_instance=logger,
                    )
            return SnowflakeToolErrorHandler.handle_tool_error(
                error=e,
                tool_name="cortex_analyst",
                operation="async execution",
                logger_instance=logger,
            )

    def _fallback_text2sql(self, session: Session, query: str) -> str:
        """Fallback method using COMPLETE function for text2sql."""
        try:
            prompt = f"""
            You are a SQL expert. Convert the following natural language query to SQL.
            Focus on generating clean, efficient SQL queries.
            
            Query: {query}
            
            Return only the SQL query without any explanation.
            """

            sql = f"""
            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                'llama3.1-70b',
                '{prompt.replace("'", "''")}'
            ) as sql_query
            """

            result = session.sql(sql).collect()

            if result:
                generated_sql = result[0]["SQL_QUERY"]
                return json.dumps(
                    {
                        "sql_query": generated_sql,
                        "explanation": "Generated using Cortex COMPLETE as fallback",
                        "method": "fallback_text2sql",
                    }
                )
            else:
                return json.dumps({"error": "No SQL generated", "method": "fallback_text2sql"})

        except Exception as e:
            return SnowflakeToolErrorHandler.handle_sql_error(
                error=e,
                tool_name="cortex_analyst",
                sql_query=query,
                operation="fallback text2sql",
                logger_instance=logger,
            )

    async def _afallback_text2sql(self, session: Session, query: str) -> str:
        """Async fallback method using COMPLETE function for text2sql."""
        try:
            prompt = f"""
            You are a SQL expert. Convert the following natural language query to SQL.
            Focus on generating clean, efficient SQL queries.
            
            Query: {query}
            
            Return only the SQL query without any explanation.
            """

            sql = f"""
            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                'llama3.1-70b',
                '{prompt.replace("'", "''")}'
            ) as sql_query
            """

            # Use native Snowflake async execution
            async_job = session.sql(sql).collect_nowait()
            result = await asyncio.to_thread(async_job.result)

            if result:
                generated_sql = result[0]["SQL_QUERY"]
                return json.dumps(
                    {
                        "sql_query": generated_sql,
                        "explanation": "Generated using Cortex COMPLETE as fallback",
                        "method": "fallback_text2sql",
                    }
                )
            else:
                return json.dumps({"error": "No SQL generated", "method": "fallback_text2sql"})

        except Exception as e:
            return SnowflakeToolErrorHandler.handle_sql_error(
                error=e,
                tool_name="cortex_analyst",
                sql_query=query,
                operation="async fallback text2sql",
                logger_instance=logger,
            )
