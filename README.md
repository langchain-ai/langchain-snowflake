# 🦜️🔗 Langchain Snowflake

This repository contains 1 package with Snowflake integrations with Langchain/Langgraph:

- [langchain-snowflake](https://pypi.org/project/langchain-snowflake/)

Langchain / Langgraph integration for Snowflake Cortex AI. Native support for chat models, tools, and retrieval with production-ready authentication.

## Features

- Chat models with tool calling support
- Cortex Search retrieval with relevance assessment
- Cortex Analyst with semantic Text2SQL support
- Complete toolkit for Cortex AI functions
- Async/await support for high-performance applications
- Agent patterns following Langchain & Langgraph standards
- Multiple authentication methods
- Consolidated error handling
- LangSmith tracing integration
- 50+ comprehensive tests covering core functionality

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
- [Quickstart](https://quickstarts.snowflake.com/guide/build-evaluate-rag-langchain-snowflake/index.html?index=..%2F..index#0)

## Contributing

See [Development.md](https://github.com/langchain-ai/langchain-snowflake/blob/main/libs/snowflake/DEVELOPMENT.md) for setup and guidelines.

## License

MIT License

## Support

- Check documentation notebooks in `docs/`
- Report issues on GitHub
- Join community discussions