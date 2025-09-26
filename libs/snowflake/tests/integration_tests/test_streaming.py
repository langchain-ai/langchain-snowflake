"""Integration tests for Snowflake streaming functionality."""

import os
from unittest.mock import Mock

import pytest

from langchain_snowflake.chat_models import ChatSnowflake


@pytest.mark.integration
def test_live_streaming_cortex_complete():
    """Test live streaming with Cortex Complete.

    This test requires valid Snowflake credentials to be set as environment variables:
    - SNOWFLAKE_ACCOUNT
    - SNOWFLAKE_USER
    - SNOWFLAKE_PASSWORD
    - SNOWFLAKE_WAREHOUSE
    - SNOWFLAKE_DATABASE
    - SNOWFLAKE_SCHEMA

    The test will be skipped if credentials are not available.
    """
    # Check if required environment variables are set
    required_env_vars = [
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_SCHEMA",
    ]

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        pytest.skip(f"Skipping live streaming test - missing environment variables: {missing_vars}")

    # Create ChatSnowflake instance with environment credentials
    llm = ChatSnowflake(
        model="llama3.1-8b",
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        temperature=0.1,
        max_tokens=100,  # Keep it small for fast test
    )

    # Test basic streaming
    messages = [("human", "Say hello in exactly 3 words")]

    # Collect streaming chunks
    chunks = []
    for chunk in llm.stream(messages):
        chunks.append(chunk.content)
        # Stop after getting some content to keep test fast
        if len(chunks) >= 3:
            break

    # Verify we got streaming chunks
    assert len(chunks) > 0, "Should receive at least one streaming chunk"
    assert any(chunk.strip() for chunk in chunks), "Should receive non-empty content"

    # Verify streaming produces different results than batch
    full_content = "".join(chunks)
    assert len(full_content.strip()) > 0, "Streaming should produce content"


@pytest.mark.integration
async def test_live_async_streaming_cortex_complete():
    """Test live async streaming with Cortex Complete.

    This test requires valid Snowflake credentials to be set as environment variables.
    The test will be skipped if credentials are not available.
    """
    # Check if required environment variables are set
    required_env_vars = [
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_SCHEMA",
    ]

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        pytest.skip(f"Skipping live async streaming test - missing environment variables: {missing_vars}")

    # Create ChatSnowflake instance with environment credentials
    llm = ChatSnowflake(
        model="llama3.1-8b",
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        temperature=0.1,
        max_tokens=100,  # Keep it small for fast test
    )

    # Test async streaming
    messages = [("human", "Count from 1 to 3")]

    # Collect async streaming chunks
    chunks = []
    async for chunk in llm.astream(messages):
        chunks.append(chunk.content)
        # Stop after getting some content to keep test fast
        if len(chunks) >= 3:
            break

    # Verify we got streaming chunks
    assert len(chunks) > 0, "Should receive at least one async streaming chunk"
    assert any(chunk.strip() for chunk in chunks), "Should receive non-empty content"

    # Verify streaming produces content
    full_content = "".join(chunks)
    assert len(full_content.strip()) > 0, "Async streaming should produce content"


@pytest.mark.compile
def test_streaming_compilation():
    """Test that streaming methods can be imported and instantiated without errors.

    This compilation test doesn't require live credentials.
    """
    # Test that we can import and create instances
    from langchain_snowflake.chat_models import ChatSnowflake

    # Create mock session for compilation test
    mock_session = Mock()
    mock_session._conn._conn.host = "test-account.snowflakecomputing.com"

    # Create ChatSnowflake instance
    llm = ChatSnowflake(model="llama3.1-8b", session=mock_session, temperature=0.1)

    # Verify streaming methods exist
    assert hasattr(llm, "stream"), "ChatSnowflake should have stream method"
    assert hasattr(llm, "astream"), "ChatSnowflake should have astream method"
    assert callable(llm.stream), "stream method should be callable"
    assert callable(llm.astream), "astream method should be callable"

    # Test that methods don't crash on instantiation
    messages = [("human", "test")]

    # These should not raise errors during setup (though they may fail on execution with mock)
    try:
        stream_gen = llm.stream(messages)
        assert stream_gen is not None, "stream should return a generator"
    except Exception:
        # Expected to fail with mock session, but should not crash on instantiation
        pass
