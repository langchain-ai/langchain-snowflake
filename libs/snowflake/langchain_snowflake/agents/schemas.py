"""Pydantic schemas for Snowflake Cortex Agents.

This module defines input and output schemas for Cortex Agent operations,
following the same patterns as existing tools in the package.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentInput(BaseModel):
    """Input schema for Cortex Agent execution.

    This schema defines the parameters for executing a Cortex Agent,
    following LangChain tool input patterns.
    """

    query: str = Field(description="Query to send to the agent")
    thread_id: Optional[int] = Field(default=None, description="Optional thread ID for conversation context")


class AgentOutput(BaseModel):
    """Output schema for Cortex Agent responses.

    This schema documents the expected structure of agent responses,
    following the same documentation patterns as existing tool schemas.

    Note: Like other tool schemas in this package, this is for documentation
    and type hints only. The actual _run methods return JSON strings.
    """

    content: str = Field(description="Agent response content")
    run_id: str = Field(description="Unique run identifier for feedback")
    thread_id: Optional[int] = Field(description="Thread ID if used")
    usage: Dict[str, Any] = Field(description="Usage metadata (tokens, timing)")
    tools_used: List[str] = Field(default_factory=list, description="Tools invoked by agent during execution")
    name: str = Field(description="Name of the agent that processed the request")


class FeedbackInput(BaseModel):
    """Input schema for agent feedback submission following official API specification."""

    request_id: str = Field(description="Request ID associated with the feedback")
    positive: bool = Field(description="Whether the response was good (true) or bad (false)")
    feedback_message: str = Field(description="Detailed feedback message")
    categories: Optional[List[str]] = Field(default=None, description="List of feedback categories")
    thread_id: Optional[int] = Field(default=None, description="ID of the thread")


class AgentUsageMetadata(BaseModel):
    """Schema for agent usage tracking.

    This schema documents the usage metadata structure following
    the same patterns as ChatSnowflake usage tracking.
    """

    execution_time: float = Field(description="Request execution time in seconds")
    input_tokens: int = Field(description="Number of input tokens processed")
    output_tokens: int = Field(description="Number of output tokens generated")
    total_tokens: int = Field(description="Total tokens (input + output)")
    name: str = Field(description="Name of the agent used")
    thread_id: Optional[int] = Field(description="Thread ID if conversation was threaded")
    tools_used: List[str] = Field(default_factory=list, description="List of tools invoked during execution")
    model_used: Optional[str] = Field(default=None, description="Underlying model used by the agent")


# ============================================================================
# AGENT MANAGEMENT SCHEMAS
# ============================================================================


class AgentProfile(BaseModel):
    """Schema for agent profile information."""

    display_name: Optional[str] = Field(default=None, description="Display name for the agent")
    avatar: Optional[str] = Field(default=None, description="Avatar URL or identifier")
    color: Optional[str] = Field(default=None, description="Color theme for the agent")
    description: Optional[str] = Field(default=None, description="Agent description")


class AgentInstructions(BaseModel):
    """Schema for agent instructions and behavior."""

    response: Optional[str] = Field(default=None, description="Instructions for response style")
    orchestration: Optional[str] = Field(default=None, description="Instructions for tool orchestration")
    system: Optional[str] = Field(default=None, description="System-level instructions")
    sample_questions: Optional[List[str]] = Field(default_factory=list, description="Sample questions for the agent")


class BudgetConfig(BaseModel):
    """Schema for agent budget constraints."""

    seconds: Optional[int] = Field(default=None, description="Time budget in seconds")
    tokens: Optional[int] = Field(default=None, description="Token budget")


class OrchestrationConfig(BaseModel):
    """Schema for agent orchestration configuration."""

    budget: Optional[BudgetConfig] = Field(default=None, description="Budget constraints")


class ModelConfig(BaseModel):
    """Schema for agent model configuration."""

    orchestration: Optional[str] = Field(
        default=None, description="Model to use for orchestration (e.g., claude-3-5-sonnet)"
    )


class ToolSpec(BaseModel):
    """Schema for tool specification."""

    type: str = Field(description="Tool type (e.g., cortex_analyst_text_to_sql, cortex_search, generic)")
    name: str = Field(description="Tool name identifier")
    description: str = Field(description="Tool description")
    input_schema: Optional[Dict[str, Any]] = Field(default=None, description="JSON schema for tool input parameters")


class Tool(BaseModel):
    """Schema for agent tool definition."""

    tool_spec: ToolSpec = Field(description="Tool specification")


class ToolResource(BaseModel):
    """Schema for tool resource configuration."""

    type: Optional[str] = Field(default=None, description="Resource type")
    semantic_model_file: Optional[str] = Field(default=None, description="Semantic model file path")
    semantic_view: Optional[str] = Field(default=None, description="Semantic view name")
    search_service: Optional[str] = Field(default=None, description="Search service identifier")
    identifier: Optional[str] = Field(default=None, description="Resource identifier")
    execution_environment: Optional[Dict[str, Any]] = Field(
        default=None, description="Execution environment configuration"
    )


class AgentCreateInput(BaseModel):
    """Input schema for creating Cortex Agents."""

    name: str = Field(description="Agent name")
    comment: Optional[str] = Field(default=None, description="Optional comment about the agent")
    profile: Optional[AgentProfile] = Field(default=None, description="Agent profile information")
    models: Optional[ModelConfig] = Field(default=None, description="Model configuration")
    instructions: Optional[AgentInstructions] = Field(default=None, description="Agent instructions")
    orchestration: Optional[OrchestrationConfig] = Field(default=None, description="Orchestration config")
    tools: Optional[List[Tool]] = Field(default_factory=list, description="Available tools")
    tool_resources: Optional[Dict[str, ToolResource]] = Field(
        default_factory=dict, description="Tool resource configurations"
    )


class AgentUpdateInput(BaseModel):
    """Input schema for updating Cortex Agents."""

    comment: Optional[str] = Field(default=None, description="Updated comment")
    profile: Optional[AgentProfile] = Field(default=None, description="Updated profile")
    models: Optional[ModelConfig] = Field(default=None, description="Updated model config")
    instructions: Optional[AgentInstructions] = Field(default=None, description="Updated instructions")
    orchestration: Optional[OrchestrationConfig] = Field(default=None, description="Updated orchestration")
    tools: Optional[List[Tool]] = Field(default=None, description="Updated tools")
    tool_resources: Optional[Dict[str, ToolResource]] = Field(default=None, description="Updated tool resources")


# ============================================================================
# RUN MANAGEMENT SCHEMAS
# ============================================================================


class RunMessage(BaseModel):
    """Schema for run messages."""

    role: str = Field(description="Message role (system, user, assistant)")
    content: str = Field(description="Message content")


class RunInput(BaseModel):
    """Input schema for creating agent runs."""

    messages: List[RunMessage] = Field(description="Messages for the run")
    thread_id: Optional[int] = Field(default=None, description="Thread ID for conversation context")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, gt=0, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Top-p sampling parameter")
    stream: Optional[bool] = Field(default=False, description="Enable streaming")


class RunOutput(BaseModel):
    """Output schema for agent runs."""

    run_id: str = Field(description="Unique run identifier")
    status: str = Field(description="Run status")
    thread_id: Optional[int] = Field(description="Thread ID if used")
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="Run messages")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Usage information")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    completed_at: Optional[str] = Field(default=None, description="Completion timestamp")


# ============================================================================
# ENHANCED EXECUTION SCHEMAS
# ============================================================================


class EnhancedAgentInput(BaseModel):
    """Enhanced input schema for agent execution with all parameters."""

    query: Optional[str] = Field(default=None, description="Query string (for simple execution)")
    messages: Optional[List[RunMessage]] = Field(default=None, description="Messages list (for advanced execution)")
    thread_id: Optional[int] = Field(default=None, description="Thread ID")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, gt=0, description="Max tokens")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Top-p sampling")
    stream: Optional[bool] = Field(default=False, description="Enable streaming")


# ============================================================================
# THREAD MANAGEMENT SCHEMAS
# ============================================================================


class ThreadUpdateInput(BaseModel):
    """Input schema for updating threads."""

    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Updated thread metadata")
    name: Optional[str] = Field(default=None, description="Updated thread name")


# ============================================================================
# FEEDBACK OUTPUT SCHEMA
# ============================================================================


class FeedbackOutput(BaseModel):
    """Output schema for feedback operations following official API response."""

    status: str = Field(description="Feedback submission status message")


# ============================================================================
# STREAMING SCHEMAS
# ============================================================================


class StreamingMetadata(BaseModel):
    """Schema for streaming response metadata."""

    chunk_index: int = Field(description="Index of the streaming chunk")
    total_chunks: Optional[int] = Field(default=None, description="Total expected chunks")
    tokens_generated: int = Field(description="Tokens generated so far")
    finish_reason: Optional[str] = Field(default=None, description="Reason for completion")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Usage information")


class StreamingChunk(BaseModel):
    """Schema for individual streaming chunks."""

    content: str = Field(description="Chunk content")
    metadata: Optional[StreamingMetadata] = Field(default=None, description="Chunk metadata")
    is_final: bool = Field(default=False, description="Whether this is the final chunk")
