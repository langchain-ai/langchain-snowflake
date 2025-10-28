# 🦜️🔗 Langchain Snowflake

This repository contains 1 package with Snowflake integrations with Langchain/Langgraph:

- [langchain-snowflake](https://pypi.org/project/langchain-snowflake/)

Langchain / Langgraph integration for Snowflake Cortex AI. Native support for chat models, tools, and retrieval with production-ready authentication.

## Features

### Core Capabilities
- **Chat Models** - Native Snowflake Cortex LLM support (Claude, Llama, Mistral, GPT-4)
- **Tool Calling** - Standard LangChain tool binding with Snowflake Cortex functions
- **Structured Output** - Pydantic and JSON schema support
- **Streaming** - Real-time response streaming for better UX

### Snowflake Cortex AI Integration
- **Cortex Search** - Semantic search with relevance scoring
- **Cortex Analyst** - Natural language to SQL via semantic models
- **Cortex Agents** - Managed agent orchestration with thread management
- **Cortex Complete** - Text generation and completion

### Production Ready
- **Multiple Authentication** - Password, PAT, Key Pair, SSO
- **Async/Await Support** - High-performance async operations
- **Error Handling** - Comprehensive error recovery
- **LangSmith Integration** - Built-in observability and tracing
- **50+ Tests** - Comprehensive test coverage

## Installation

```bash
pip install langchain-snowflake
```

## LangSmith Tracing

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your-langsmith-api-key
export LANGCHAIN_PROJECT=snowflake-cortex
```

## Documentation

- [Getting Started](https://github.com/langchain-ai/langchain-snowflake/blob/main/libs/snowflake/docs/getting_started.ipynb)
- [Snowflake Workflows](https://github.com/langchain-ai/langchain-snowflake/blob/main/libs/snowflake/docs/snowflake_workflows.ipynb)
- [Advanced Patterns](https://github.com/langchain-ai/langchain-snowflake/blob/main/libs/snowflake/docs/advanced_patterns.ipynb)
- [MCP Integration](https://github.com/langchain-ai/langchain-snowflake/blob/main/libs/snowflake/docs/mcp_integration.ipynb)
- [Quickstart](https://quickstarts.snowflake.com/guide/build-evaluate-rag-langchain-snowflake/index.html?index=..%2F..index#0)
- [Multi-Agent Demo]() - coming soon

## Contributing

We welcome contributions! See [Development.md](https://github.com/langchain-ai/langchain-snowflake/blob/main/libs/snowflake/DEVELOPMENT.md) for setup and guidelines.

## License

MIT License - see [LICENSE](https://github.com/langchain-ai/langchain-snowflake/blob/main/libs/snowflake/LICENSE) for details

## Support

- Check documentation notebooks in `docs/`
- Report issues on [GitHub Issues](https://github.com/langchain-ai/langchain-snowflake/issues)