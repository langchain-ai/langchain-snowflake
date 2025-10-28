"""SQL query execution tool for Snowflake."""

import json
import logging
from typing import Any, Dict, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .._connection import SnowflakeConnectionMixin, SqlExecutionClient
from .._error_handling import SnowflakeToolErrorHandler
from ._base import SnowflakeQueryInput

logger = logging.getLogger(__name__)


class SnowflakeQueryTool(BaseTool, SnowflakeConnectionMixin):
    """Execute SQL queries on Snowflake.

    This tool allows execution of SQL queries against a Snowflake database
    and returns the results in a formatted manner.

    Setup:
        Install ``langchain-snowflake`` and configure Snowflake connection.

        .. code-block:: bash

            pip install -U langchain-snowflake

    Instantiation:
        .. code-block:: python

            from . import SnowflakeQueryTool

            tool = SnowflakeQueryTool(
                session=session,
                # or connection parameters
                account="your-account",
                user="your-user",
                password="your-password",
                warehouse="your-warehouse"
            )

    Invocation:
        .. code-block:: python

            result = tool.invoke({"query": "SELECT COUNT(*) FROM my_table"})
            print(result)
    """

    name: str = "snowflake_query"
    description: str = "Execute SQL queries on Snowflake database and return results"
    args_schema: Union[Type[BaseModel], Dict[str, Any], None] = SnowflakeQueryInput

    max_rows: int = Field(default=100)

    def __init__(self, **kwargs):
        """Initialize the query tool with proper session attribute."""
        # Extract session from kwargs if provided
        session = kwargs.pop("session", None)

        # Call the parent initializer with remaining kwargs
        super().__init__(**kwargs)

        # Initialize session attribute (from SnowflakeConnectionMixin) using object.__setattr__ to bypass Pydantic
        if session is not None:
            object.__setattr__(self, "_session", session)
        elif not hasattr(self, "_session"):
            object.__setattr__(self, "_session", None)

    def _run(self, query: str, *, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Execute SQL query on Snowflake."""
        session = self._get_session()

        try:
            # Execute the query
            result = session.sql(query).limit(self.max_rows).collect()

            if not result:
                return "Query executed successfully but returned no results."

            # Format results as JSON
            formatted_results = []
            for row in result:
                row_dict = {}
                for field_name in row.as_dict():
                    row_dict[field_name] = str(row[field_name])
                formatted_results.append(row_dict)

            return json.dumps(
                {
                    "results": formatted_results,
                    "row_count": len(formatted_results),
                    "query": query,
                },
                indent=2,
            )

        except Exception as e:
            return SnowflakeToolErrorHandler.handle_sql_error(
                error=e,
                tool_name="SnowflakeQueryTool",
                sql_query=query,
                operation="execute SQL query",
            )

    async def _arun(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async execute SQL query using native async SqlExecutionClient."""
        execution_result = await SqlExecutionClient.execute_async(
            session=self._get_session(), sql=query, operation_name="execute SQL query"
        )

        if not execution_result["success"]:
            return execution_result["error"]

        result = execution_result["result"]
        if not result:
            return "Query executed successfully but returned no results."

        # Format results as JSON
        formatted_results = []
        for row in result:
            row_dict = {}
            for field_name in row.as_dict():
                row_dict[field_name] = str(row[field_name])
            formatted_results.append(row_dict)

        return json.dumps(
            {
                "results": formatted_results,
                "row_count": len(formatted_results),
                "query": query,
            },
            indent=2,
        )
