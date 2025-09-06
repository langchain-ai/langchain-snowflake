"""Snowflake tools for LangChain integration.

This module provides specialized tools for integrating with Snowflake's Cortex AI functions
and database querying capabilities.

The tools are now organized into logical modules:
- analyst.py: Complex SnowflakeCortexAnalyst for Text2SQL
- cortex_functions.py: Basic Cortex AI tools (sentiment, summarize, translate, complete)
- query.py: SQL query execution tools
- _base.py: Shared schemas and base classes
"""

# Import schemas for convenience (optional - users can import from _base directly)
from ._base import (
    CortexCompleteInput,
    CortexCompleteOutput,
    CortexSentimentInput,
    CortexSentimentOutput,
    CortexSummarizerInput,
    CortexSummarizerOutput,
    CortexTranslatorInput,
    CortexTranslatorOutput,
    # Input schemas
    SnowflakeCortexAnalystInput,
    SnowflakeCortexAnalystOutput,
    SnowflakeQueryInput,
    SnowflakeQueryOutput,
    # Output schemas (documentation only)
    SnowflakeToolResponse,
)
from .analyst import SnowflakeCortexAnalyst
from .cortex_functions import (
    CortexCompleteTool,
    CortexSentimentTool,
    CortexSummarizerTool,
    CortexTranslatorTool,
)
from .query import SnowflakeQueryTool

__all__ = [
    # Main tool classes
    "SnowflakeCortexAnalyst",
    "SnowflakeQueryTool",
    "CortexSentimentTool",
    "CortexSummarizerTool",
    "CortexTranslatorTool",
    "CortexCompleteTool",
    # Output schemas (documentation/type hints only)
    "SnowflakeToolResponse",
    "SnowflakeCortexAnalystOutput",
    "SnowflakeQueryOutput",
    "CortexSentimentOutput",
    "CortexSummarizerOutput",
    "CortexTranslatorOutput",
    "CortexCompleteOutput",
    # Input schemas
    "SnowflakeCortexAnalystInput",
    "SnowflakeQueryInput",
    "CortexSentimentInput",
    "CortexSummarizerInput",
    "CortexTranslatorInput",
    "CortexCompleteInput",
]
