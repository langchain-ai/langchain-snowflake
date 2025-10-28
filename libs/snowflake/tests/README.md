# LangChain Snowflake Tests

This directory contains the test suite for the `langchain-snowflake` package.

## Test Structure

### Unit Tests (`tests/unit_tests/`)

The unit test suite contains **20 core unit tests** covering:

1. **Chat Models** (`test_chat_models.py`) - 8 tests
   - 2 main class tests (initialization)
   - 2 authentication tests (password, token)
   - 2 structured output tests (Pydantic, JSON schema)
   - 2 utility tests (message conversion, validation)

2. **Tools** (`test_tools.py`) - 5 tests
   - 1 CortexCompleteTool test
   - 1 SnowflakeQueryTool test
   - 3 SqlExecutionClient tests (sync execution, Cortex functions, error handling)

3. **Retrievers** (`test_retrievers.py`) - 1 test
   - 1 basic search test

4. **Connection** (`test_connection.py`) - 2 tests
   - 1 SnowflakeConnectionMixin test
   - 1 SnowflakeSessionManager test

5. **Utilities** (`test_utilities.py`) - 3 tests
   - 1 formatters test
   - 1 error handling test
   - 1 input/output schemas test

### LangChain Standard Tests (`test_standard_tests.py`)

- **Standard compliance tests** using `langchain-tests` package
- Automatically skipped if `langchain-tests` is not installed
- Install with: `pip install langchain-tests`

## Running Tests

### Run All Unit Tests

```bash
make test
# or
pytest tests/unit_tests/ --disable-socket --allow-unix-socket
```

### Run Specific Test Files

```bash
pytest tests/unit_tests/test_chat_models.py -v
pytest tests/unit_tests/test_tools.py -v
```

### Run with Coverage

```bash
pytest tests/unit_tests/ --cov=langchain_snowflake --cov-report=html
```

### Run LangChain Standard Tests

```bash
# First install langchain-tests
pip install langchain-tests

# Then run all tests including standard tests
pytest tests/unit_tests/ -v
```

## Test Design

- **Fast execution**: All tests should complete within 3-5 minutes
- **Mocked dependencies**: Unit tests use mocks to avoid external calls
- **Network isolation**: `pytest-socket` prevents accidental network calls
- **Simple and focused**: Each test covers one specific functionality

## Test Configuration

Tests are configured via `pyproject.toml`:

- `pytest-socket` for network isolation
- `pytest-asyncio` for async test support
- Test markers for integration tests (not used in unit tests)
- Automatic test discovery

## Fixtures

Shared test fixtures are defined in `conftest.py`:

- `mock_snowflake_session`: Mock Snowflake session for all tests
- `disable_network_calls`: Automatic network isolation

### Integration Tests (`tests/integration_tests/`)

**NEW:** Integration tests have been reinstated and updated:

1. **Compilation Tests** (`test_compile.py`) - 1 test
   - Import compilation verification for CI

2. **Retriever Tests** (`test_retrievers.py`) - Multiple tests
   - SnowflakeCortexSearchRetriever functionality
   - Authentication and connection testing
   - Search and filtering capabilities
   - Auto-formatting for RAG workflows

**Test Modes:**

- **Compilation-only**: Tests import and basic instantiation (no live credentials needed)
- **Full integration**: Tests with live Snowflake connection (requires credentials)

### Documentation Tests

Documentation notebooks serve as additional integration tests:

- `docs/getting_started.ipynb`
- `docs/snowflake_workflows.ipynb` 
- `docs/advanced_patterns.ipynb`
