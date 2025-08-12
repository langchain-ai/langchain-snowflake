"""Snowflake Cortex Analyst tool for Text2SQL conversions."""

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Type

import aiohttp
import requests
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from snowflake.snowpark import Session

from .._connection import SnowflakeAuthUtils, SnowflakeConnectionMixin
from .._error_handling import SnowflakeToolErrorHandler
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
    args_schema: Type[BaseModel] = SnowflakeCortexAnalystInput

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

    def _build_rest_api_payload(
        self, query: str, semantic_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build REST API payload for analyst request."""
        payload = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}]
        }

        # Add semantic model configuration
        if semantic_model or self.semantic_model_file:
            model_ref = semantic_model or self.semantic_model_file
            if model_ref.startswith("@"):
                payload["semantic_model_file"] = model_ref
            else:
                payload["semantic_model"] = model_ref
        elif self.semantic_view:
            payload["semantic_view"] = self.semantic_view

        # Add streaming if enabled
        if self.enable_streaming:
            payload["stream"] = True

        return payload

    def _make_rest_api_request(
        self, query: str, semantic_model: Optional[str] = None
    ) -> str:
        """Make REST API request to Cortex Analyst service."""
        session = self._get_session()

        # Build URL
        try:
            conn = session._conn._conn
            base_url = f"https://{conn.host}"
        except Exception:
            # Fallback to constructing URL from account
            account_url = self.account or "your-account"
            base_url = f"https://{account_url}.snowflakecomputing.com"

        endpoint = "/api/v2/cortex/analyst/message"
        url = base_url + endpoint

        # Make request using shared auth utilities
        payload = self._build_rest_api_payload(query, semantic_model)

        try:
            # Get effective timeout respecting Snowflake parameter hierarchy
            timeout = self._get_effective_timeout()

            # Use shared auth utilities to get proper headers
            headers = SnowflakeAuthUtils.get_rest_api_headers(
                session=session, account=self.account, user=self.user
            )

            # Make secure request with proper SSL verification and timeout
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout,  # Respect Snowflake parameter hierarchy
                verify=True,  # Always use SSL verification for security
                stream=self.enable_streaming,
            )
            response.raise_for_status()

            data = response.json()
            return self._parse_rest_api_response(data)

        except Exception as e:
            return SnowflakeToolErrorHandler.handle_rest_api_error(
                "Cortex Analyst REST API request", e, query=query
            )

    async def _make_rest_api_request_async(
        self, query: str, semantic_model: Optional[str] = None
    ) -> str:
        """Make async REST API request to Cortex Analyst using aiohttp."""
        session = self._get_session()
        payload = self._build_rest_api_payload(query, semantic_model)

        # Build URL from session
        url = SnowflakeAuthUtils.build_rest_api_url(session) + "/v1/chat/completions"

        try:
            # Get effective timeout respecting Snowflake parameter hierarchy
            timeout = self._get_effective_timeout()

            # Use shared auth utilities to get proper headers
            headers = SnowflakeAuthUtils.get_rest_api_headers(
                session=session, account=self.account, user=self.user
            )

            # Use aiohttp for true async HTTP requests
            async with aiohttp.ClientSession() as client:
                async with client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                    ssl=True,  # Always use SSL verification for security
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return self._parse_rest_api_response(data)

        except Exception as e:
            return SnowflakeToolErrorHandler.handle_rest_api_error(
                "Cortex Analyst async REST API request", e, query=query
            )

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
                    result["content"].append(
                        {"type": "text", "text": item.get("text", "")}
                    )
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
                "parsing REST API response", e, raw_response=data
            )

    def _fallback_sql_function(
        self, query: str, semantic_model: Optional[str] = None
    ) -> str:
        """Fallback method using Cortex Complete function for text-to-SQL.

        Note: SNOWFLAKE.CORTEX.ANALYST does not exist as a SQL function.
        This method uses CORTEX.COMPLETE for text-to-SQL generation instead.
        """
        session = self._get_session()

        try:
            # Use CORTEX.COMPLETE for text-to-SQL since CORTEX.ANALYST doesn't exist
            logger.info("Using Cortex Complete for text-to-SQL fallback")
            return self._fallback_text2sql(session, query)

        except Exception as e:
            return SnowflakeToolErrorHandler.handle_sql_error(
                "SQL function fallback", e, query=query
            )

    async def _fallback_sql_function_async(
        self, query: str, semantic_model: Optional[str] = None
    ) -> str:
        """Async fallback method using Cortex Complete function for text-to-SQL with native Snowflake async.

        Note: SNOWFLAKE.CORTEX.ANALYST does not exist as a SQL function.
        This method uses CORTEX.COMPLETE for text-to-SQL generation instead.
        """
        session = self._get_session()

        try:
            # Use CORTEX.COMPLETE for text-to-SQL since CORTEX.ANALYST doesn't exist
            logger.info("Using async Cortex Complete for text-to-SQL fallback")
            return await self._afallback_text2sql(session, query)

        except Exception as e:
            return SnowflakeToolErrorHandler.handle_sql_error(
                "async SQL function fallback", e, query=query
            )

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
                logger.info("Falling back to SQL function")
                return self._fallback_sql_function(query, semantic_model)
            return SnowflakeToolErrorHandler.handle_tool_error(
                "Cortex Analyst execution", e, query=query
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
                logger.info("Falling back to async SQL function")
                return await self._fallback_sql_function_async(query, semantic_model)
            return SnowflakeToolErrorHandler.handle_tool_error(
                "Cortex Analyst async execution", e, query=query
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
                return json.dumps(
                    {"error": "No SQL generated", "method": "fallback_text2sql"}
                )

        except Exception as e:
            return SnowflakeToolErrorHandler.handle_sql_error(
                "fallback text2sql", e, query=query
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
                return json.dumps(
                    {"error": "No SQL generated", "method": "fallback_text2sql"}
                )

        except Exception as e:
            return SnowflakeToolErrorHandler.handle_sql_error(
                "async fallback text2sql", e, query=query
            )
