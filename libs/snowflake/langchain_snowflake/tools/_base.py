"""Shared base classes and schemas for Snowflake tools."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ============================================================================
# STANDARDIZED OUTPUT SCHEMAS (DOCUMENTATION & TYPE HINTS ONLY)
# ============================================================================
#
# NOTE: These schemas are provided for documentation and type safety but are NOT
# enforced by LangChain's BaseTool interface. LangChain tools return plain strings
# or simple objects. These schemas serve as:
# 1. Documentation of the expected JSON structure in tool responses
# 2. Type hints for developers using these tools
# 3. Validation schemas for external consumers of tool outputs
#
# The actual tool methods (_run, _arun) return JSON strings that conform to these
# schemas, but LangChain itself does not validate against them.
# ============================================================================


class SnowflakeToolResponse(BaseModel):
    """Base class for all Snowflake tool responses (documentation only)."""

    success: bool = Field(description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")


class SnowflakeCortexAnalystOutput(SnowflakeToolResponse):
    """Output schema for Snowflake Cortex Analyst tool."""

    sql_query: Optional[str] = Field(default=None, description="Generated SQL query")
    results: Optional[List[Dict[str, Any]]] = Field(default=None, description="Query execution results")
    explanation: Optional[str] = Field(default=None, description="Natural language explanation")
    request_id: Optional[str] = Field(default=None, description="Unique request identifier")


class SnowflakeQueryOutput(SnowflakeToolResponse):
    """Output schema for SnowflakeQueryTool."""

    query: str = Field(description="Executed SQL query")
    results: List[Dict[str, Any]] = Field(description="Query results")
    row_count: int = Field(description="Number of rows returned")


class CortexSentimentOutput(SnowflakeToolResponse):
    """Output schema for CortexSentimentTool."""

    text: str = Field(description="Original text analyzed")
    sentiment_score: float = Field(description="Sentiment score (-1 to 1)")
    sentiment_label: Literal["positive", "negative", "neutral"] = Field(description="Sentiment classification")


class CortexSummarizerOutput(SnowflakeToolResponse):
    """Output schema for CortexSummarizerTool."""

    original_text: str = Field(description="Original text (truncated if long)")
    summary: str = Field(description="Generated summary")
    original_length: int = Field(description="Length of original text in characters")
    summary_length: int = Field(description="Length of summary in characters")


class CortexTranslatorOutput(SnowflakeToolResponse):
    """Output schema for CortexTranslatorTool."""

    original_text: str = Field(description="Original text")
    translated_text: str = Field(description="Translated text")
    target_language: str = Field(description="Target language code")
    source_language: Optional[str] = Field(default=None, description="Detected source language")


class CortexCompleteOutput(SnowflakeToolResponse):
    """Output schema for CortexCompleteTool."""

    prompt: str = Field(description="Input prompt")
    completion: str = Field(description="Generated completion")
    model: str = Field(description="Model used for completion")


# ============================================================================
# INPUT SCHEMAS
# ============================================================================


class SnowflakeCortexAnalystInput(BaseModel):
    """Input schema for Snowflake Cortex Analyst tool."""

    query: str = Field(description="Natural language question about the data")
    semantic_model: Optional[str] = Field(default=None, description="Optional semantic model to use for the query")


class SnowflakeQueryInput(BaseModel):
    """Input schema for SnowflakeQueryTool."""

    query: str = Field(description="SQL query to execute on Snowflake")


class CortexSentimentInput(BaseModel):
    """Input schema for CortexSentimentTool."""

    text: str = Field(description="Text to analyze for sentiment")


class CortexSummarizerInput(BaseModel):
    """Input schema for CortexSummarizerTool."""

    text: str = Field(description="Text to summarize")


class CortexTranslatorInput(BaseModel):
    """Input schema for CortexTranslatorTool."""

    text: str = Field(description="Text to translate")
    target_language: str = Field(description="Target language code (e.g., 'es', 'fr', 'de')")
    source_language: Optional[str] = Field(
        default=None,
        description="Source language code (e.g., 'en', 'es', 'fr', 'de'). Defaults to 'en' if not specified.",
    )


class CortexCompleteInput(BaseModel):
    """Input schema for CortexCompleteTool."""

    prompt: str = Field(description="Text prompt for completion")
    model: str = Field(default="llama2-70b-chat", description="Cortex model to use for completion")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens in completion")
