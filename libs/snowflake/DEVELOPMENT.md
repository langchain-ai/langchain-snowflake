# Development Guide

## Introduction

This guide covers setting up the development environment, running tests, and contributing to the langchain-snowflake package.

## Prerequisites

- Python 3.10+
- Snowflake account with Cortex features enabled
- Git
- Valid Snowflake credentials

## Setup Options

### Option 1: Poetry Setup (Recommended)

```bash
git clone https://github.com/langchain-ai/langchain-snowflake
cd langchain-snowflake/libs/snowflake
poetry install
```

### Option 2: Standard Development Setup

```bash
git clone https://github.com/langchain-ai/langchain-snowflake
cd langchain-snowflake/libs/snowflake
pip install -e .
```

### Option 3: Development with Testing Dependencies

```bash
git clone https://github.com/langchain-ai/langchain-snowflake
cd langchain-snowflake/libs/snowflake
pip install -e ".[dev,test]"
```

## Running Tests

### Unit Tests

```bash
make test
# or
poetry run pytest tests/unit_tests/ -v
```

### Integration Tests (requires Snowflake connection)

```bash
make compile-test  # Compilation-only tests for CI
# or
poetry run pytest -m compile tests/integration_tests/

### All Quality Checks

```bash
make lint    # Linting, formatting, and type checking
make test    # Unit tests
make compile-test  # Integration compilation tests
```

## Code Quality

Run all quality checks:

```bash
# Linting and formatting
poetry run ruff check . --fix
poetry run ruff format .

# Type checking
poetry run mypy langchain_snowflake

# All quality checks at once
make lint
```
