"""Snowflake Cortex Agents integration.

This module provides integration with Snowflake's Cortex Agents REST API,
enabling managed orchestration of multiple tools and conversation management.

Key Features:
- Agent execution with thread management
- Native Snowflake feedback system integration
- Usage tracking following package patterns
- LangChain Runnable interface for agent workflows
- Async support for execution

Architecture:
    The agent is built using focused mixins for maintainability:
    - AgentManagement: Agent CRUD operations
    - ThreadManagement: Thread CRUD operations
    - RunManagement: Agent execution within threads
    - FeedbackManagement: Feedback CRUD operations
    - SnowflakeConnectionMixin: Shared connection handling
"""

from .base import SnowflakeCortexAgent
from .feedback import FeedbackManagement
from .management import AgentManagement
from .runs import RunManagement
from .schemas import (
    # Enhanced schemas
    AgentCreateInput,
    # Basic schemas
    AgentInput,
    AgentInstructions,
    AgentOutput,
    # Component schemas
    AgentProfile,
    AgentUpdateInput,
    BudgetConfig,
    EnhancedAgentInput,
    FeedbackInput,
    FeedbackOutput,
    ModelConfig,
    OrchestrationConfig,
    RunMessage,
    StreamingChunk,
    StreamingMetadata,
    ThreadUpdateInput,
    Tool,
    ToolResource,
    ToolSpec,
)
from .threads import ThreadManagement

__all__ = [
    # Main agent class
    "SnowflakeCortexAgent",
    # Management mixins
    "AgentManagement",
    "ThreadManagement",
    "RunManagement",
    "FeedbackManagement",
    # Basic schemas
    "AgentInput",
    "AgentOutput",
    "FeedbackInput",
    # Enhanced schemas
    "AgentCreateInput",
    "AgentUpdateInput",
    "EnhancedAgentInput",
    "ThreadUpdateInput",
    "FeedbackInput",
    "FeedbackOutput",
    "StreamingChunk",
    "StreamingMetadata",
    # Component schemas
    "AgentProfile",
    "AgentInstructions",
    "BudgetConfig",
    "OrchestrationConfig",
    "ModelConfig",
    "ToolSpec",
    "Tool",
    "ToolResource",
    "RunMessage",
]
