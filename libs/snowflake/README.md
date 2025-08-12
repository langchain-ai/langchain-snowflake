# LangChain Snowflake

LangChain integration for Snowflake Cortex AI. Native support for chat models, tools, and retrieval with production-ready authentication.

## Production Ready

- 39 comprehensive tests covering core functionality
- Native tool calling with Cortex Complete
- Structured output support (Pydantic, TypedDict, JSON)
- Complete async/await implementation
- All authentication methods supported

## Features

- Chat models with tool calling support
- Cortex Search retrieval with relevance assessment
- Complete toolkit for 6 Cortex AI functions
- Async support for high-performance applications
- Agent patterns following LangChain standards
- Multiple authentication methods
- LangSmith tracing integration

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

- [Getting Started](docs/getting_started.ipynb)
- [Snowflake Workflows](docs/snowflake_workflows.ipynb)
- [Advanced Patterns](docs/advanced_patterns.ipynb)

## Contributing

See [Development.md](Development.md) for setup and guidelines.

## License

MIT License

## Support

- Check documentation notebooks in `docs/`
- Report issues on GitHub
- Join community discussions