# Development Guide

## Introduction

This guide covers setting up the development environment, running tests, and contributing to the langchain-snowflake package.

## Prerequisites

- Python 3.8+
- Snowflake account with Cortex features enabled
- Git
- Valid Snowflake credentials

## Setup Options

### Option 1: Standard Development Setup

```bash
git clone https://github.com/langchain-ai/langchain-snowflake
cd langchain-snowflake/libs/snowflake
pip install -e .
```

### Option 2: Development with Testing Dependencies

```bash
git clone https://github.com/langchain-ai/langchain-snowflake
cd langchain-snowflake/libs/snowflake
pip install -e ".[dev,test]"
```

## Running Tests

### Unit Tests

```bash
python -m pytest tests/unit_tests/ -v
```

### Integration Tests (requires Snowflake connection)

```bash
python -m pytest tests/integration_tests/ -v
```

### Documentation Tests

Run all notebooks in `docs/` to verify examples work:

```bash
# Test getting started guide
jupyter nbconvert --execute docs/getting_started.ipynb

# Test workflow examples  
jupyter nbconvert --execute docs/snowflake_workflows.ipynb

# Test advanced patterns
jupyter nbconvert --execute docs/advanced_patterns.ipynb
```

## Test All Authentication Methods

Set up environment variables for each method and run integration tests:

```bash
# Password authentication
export SNOWFLAKE_ACCOUNT=your-account
export SNOWFLAKE_USER=your-user
export SNOWFLAKE_PASSWORD=your-password
export SNOWFLAKE_WAREHOUSE=your-warehouse
export SNOWFLAKE_DATABASE=your-database
export SNOWFLAKE_SCHEMA=your-schema

# PAT authentication
export SNOWFLAKE_PAT=your-personal-access-token

# Key pair authentication
export SNOWFLAKE_PRIVATE_KEY_PATH=path/to/private/key
export SNOWFLAKE_PRIVATE_KEY_PASSPHRASE=optional-passphrase

python -m pytest tests/integration_tests/test_auth.py -v
```

## Code Quality

Run all quality checks:

```bash
# Linting and formatting
ruff check . --fix
ruff format .

# Type checking
mypy langchain_snowflake --ignore-missing-imports

# Spell checking
codespell --skip="*.git*,*.ipynb" .
```

## Package Structure

```
langchain_snowflake/
├── __init__.py              # Main exports and session creation
├── chat_models/             # ChatSnowflake implementation
│   ├── base.py             # Core chat model
│   ├── auth.py             # Authentication
│   ├── streaming.py        # Streaming support
│   ├── tools.py            # Tool calling
│   └── utils.py            # Utilities
├── tools/                  # Cortex AI tools
│   ├── analyst.py          # Text2SQL tool
│   ├── cortex_functions.py # Basic Cortex tools
│   ├── query.py            # SQL execution
│   └── _base.py            # Shared schemas
├── retrievers.py           # Cortex Search retriever
├── _connection/            # Connection management
├── _error_handling.py      # Error handling
└── formatters.py           # Document formatting
```

## Dependencies

Core dependencies are defined in `pyproject.toml`:

- `langchain-core`: LangChain integration
- `snowflake-snowpark-python`: Snowflake connectivity
- `aiohttp`: Async HTTP requests
- `pydantic`: Data validation

Optional dependencies:
- `cryptography`: RSA key pair authentication
- `pytest`: Testing framework
- `ruff`: Linting and formatting
- `mypy`: Type checking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks
5. Submit a pull request

All contributions must:
- Include tests for new functionality
- Pass all existing tests
- Follow code quality standards
- Update documentation as needed
