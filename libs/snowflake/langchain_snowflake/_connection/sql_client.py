"""Unified SQL execution client for Snowflake operations.

This module provides a centralized client for all SQL execution operations.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from snowflake.snowpark import Session

from .._error_handling import SnowflakeToolErrorHandler

logger = logging.getLogger(__name__)


class SqlExecutionClient:
    """Unified SQL execution client for Snowflake operations.

    This client provides a consistent interface for both sync and async
    SQL execution operations with centralized error handling.
    """

    @staticmethod
    def execute_sync(
        session: Session, sql: str, params: Optional[List[Any]] = None, operation_name: str = "SQL execution"
    ) -> Dict[str, Any]:
        """Execute SQL synchronously with centralized error handling.

        Args:
            session: Active Snowflake session
            sql: SQL query to execute
            params: Optional list of parameters for parameterized queries
            operation_name: Description for error logging

        Returns:
            Dictionary with execution results or error information
        """
        try:
            if params:
                result = session.sql(sql, params=params).collect()
            else:
                result = session.sql(sql).collect()
            return {"success": True, "result": result}
        except Exception as e:
            error_msg = SnowflakeToolErrorHandler.handle_tool_error(e, "SqlExecutionClient", operation_name)
            return {"success": False, "error": error_msg}

    @staticmethod
    async def execute_async(
        session: Session, sql: str, params: Optional[List[Any]] = None, operation_name: str = "async SQL execution"
    ) -> Dict[str, Any]:
        """Execute SQL asynchronously using asyncio.to_thread.

        Args:
            session: Active Snowflake session
            sql: SQL query to execute
            params: Optional list of parameters for parameterized queries
            operation_name: Description for error logging

        Returns:
            Dictionary with execution results or error information
        """
        return await asyncio.to_thread(SqlExecutionClient.execute_sync, session, sql, params, operation_name)

    @staticmethod
    def _escape_sql_arg(arg: Any) -> str:
        """Escape a single SQL argument for use in Cortex functions."""
        escaped = str(arg).replace("'", "''")
        return f"'{escaped}'"

    @staticmethod
    def execute_cortex_function(
        session: Session, function_name: str, args: list, operation_name: str = "Cortex function execution"
    ) -> Dict[str, Any]:
        """Execute Cortex functions with proper SQL escaping.

        Args:
            session: Active Snowflake session
            function_name: Name of the Cortex function (e.g., 'SENTIMENT', 'SUMMARIZE')
            args: List of arguments to pass to the function
            operation_name: Description for error logging

        Returns:
            Dictionary with execution results or error information
        """
        # Build SQL with proper escaping
        escaped_args = [SqlExecutionClient._escape_sql_arg(arg) for arg in args]
        sql = f"SELECT SNOWFLAKE.CORTEX.{function_name}({', '.join(escaped_args)}) as result"

        return SqlExecutionClient.execute_sync(session, sql, operation_name=operation_name)

    @staticmethod
    async def execute_cortex_function_async(
        session: Session, function_name: str, args: list, operation_name: str = "async Cortex function execution"
    ) -> Dict[str, Any]:
        """Execute Cortex functions asynchronously with proper SQL escaping.

        Args:
            session: Active Snowflake session
            function_name: Name of the Cortex function (e.g., 'SENTIMENT', 'SUMMARIZE')
            args: List of arguments to pass to the function
            operation_name: Description for error logging

        Returns:
            Dictionary with execution results or error information
        """
        # Build SQL with proper escaping
        escaped_args = [SqlExecutionClient._escape_sql_arg(arg) for arg in args]
        sql = f"SELECT SNOWFLAKE.CORTEX.{function_name}({', '.join(escaped_args)}) as result"

        return await SqlExecutionClient.execute_async(session, sql, operation_name=operation_name)
