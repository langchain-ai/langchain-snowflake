"""Test Snowflake Cortex Search retriever.

You need to create a Cortex Search service in Snowflake with the specified
columns below to run the integration tests.

Set the following environment variables before the tests:
export SNOWFLAKE_ACCOUNT=<snowflake_account>
export SNOWFLAKE_USER=<snowflake_user>  # Note: USER not USERNAME
export SNOWFLAKE_PASSWORD=<snowflake_password>
export SNOWFLAKE_DATABASE=<snowflake_database>
export SNOWFLAKE_SCHEMA=<snowflake_schema>
export SNOWFLAKE_WAREHOUSE=<snowflake_warehouse>
"""

import os
from typing import List

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError
from snowflake.snowpark import Session

from langchain_snowflake import SnowflakeCortexSearchRetriever, create_session_from_env


@pytest.fixture(scope="session")
def snowflake_session():
    """Create Snowflake session from environment variables for integration tests."""
    required_vars = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        pytest.skip(f"Integration tests require environment variables: {missing_vars}")

    try:
        return create_session_from_env()
    except Exception as e:
        pytest.skip(f"Failed to create Snowflake session: {e}")


# Compilation test for CI - always passes
@pytest.mark.compile
def test_retriever_imports_compile() -> None:
    """Test that retriever imports compile correctly."""
    assert SnowflakeCortexSearchRetriever is not None
    assert Document is not None
    assert Session is not None


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invoke(snowflake_session) -> None:
    """Test the invoke() method."""

    search_columns = ["TRANSCRIPT_TEXT", "CUSTOMER_NAME", "DEAL_STAGE", "SALES_REP"]

    # Updated to match current API
    kwargs = {
        "session": snowflake_session,
        "service_name": (
            f"{os.environ.get('SNOWFLAKE_DATABASE', 'TESTDB')}."
            f"{os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')}."
            "sales_conversation_search"
        ),
        "search_columns": search_columns,
        "filter_dict": {"@eq": {"CUSTOMER_NAME": "TechCorp Inc"}},
        "k": 10,
    }

    retriever = SnowflakeCortexSearchRetriever(**kwargs)

    documents = retriever.invoke("Tell me about the discovery call with TechCorp Inc")
    assert len(documents) > 0

    for doc in documents:
        check_document(doc, search_columns)
        # Validate the filter was passed through correctly (if customer name in metadata exists)
        if "CUSTOMER_NAME" in doc.metadata:
            assert doc.metadata["CUSTOMER_NAME"] == "TechCorp Inc"


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invoke_no_columns_or_filter(snowflake_session) -> None:
    """Test the invoke() method with no columns or filter."""

    kwargs = {
        "session": snowflake_session,
        "service_name": (
            f"{os.environ.get('SNOWFLAKE_DATABASE', 'TESTDB')}."
            f"{os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')}."
            "sales_conversation_search"
        ),
        "k": 10,
    }

    retriever = SnowflakeCortexSearchRetriever(**kwargs)

    documents = retriever.invoke("Who is the Sales Rep for SmallBiz Solutions")
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content or len(doc.metadata) > 0  # Should have content or metadata


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_constructor_no_service_name() -> None:
    """Test the constructor with no service name provided."""

    search_columns = ["TRANSCRIPT_TEXT", "CUSTOMER_NAME", "DEAL_STAGE", "SALES_REP"]

    kwargs = {
        "search_columns": search_columns,
        "k": 10,
    }

    with pytest.raises(ValidationError):
        SnowflakeCortexSearchRetriever(**kwargs)


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invalid_service_name(snowflake_session) -> None:
    """Test the constructor with invalid service name format."""

    search_columns = ["TRANSCRIPT_TEXT", "CUSTOMER_NAME", "DEAL_STAGE", "SALES_REP"]

    kwargs = {
        "session": snowflake_session,
        "service_name": "invalid_service_name",  # Should be database.schema.service
        "search_columns": search_columns,
        "k": 10,
    }

    # This should create the retriever but fail when trying to parse service name
    retriever = SnowflakeCortexSearchRetriever(**kwargs)

    # The error will be logged as a warning and return empty results (graceful degradation)
    documents = retriever.invoke("test query")
    assert len(documents) == 0  # Should return empty list due to invalid service name


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invoke_different_k_values(snowflake_session) -> None:
    """Test the invoke() method with different k values."""

    search_columns = ["TRANSCRIPT_TEXT", "CUSTOMER_NAME", "DEAL_STAGE", "SALES_REP"]

    # Test with k=1
    kwargs_1 = {
        "session": snowflake_session,
        "service_name": (
            f"{os.environ.get('SNOWFLAKE_DATABASE', 'TESTDB')}."
            f"{os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')}."
            "sales_conversation_search"
        ),
        "search_columns": search_columns,
        "k": 1,
    }

    retriever_1 = SnowflakeCortexSearchRetriever(**kwargs_1)
    documents_1 = retriever_1.invoke("What is the deal value for SecureBank Ltd?")
    assert len(documents_1) <= 1  # Should return at most 1 document

    # Test with k=5
    kwargs_5 = {
        "session": snowflake_session,
        "service_name": (
            f"{os.environ.get('SNOWFLAKE_DATABASE', 'TESTDB')}."
            f"{os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')}."
            "sales_conversation_search"
        ),
        "search_columns": search_columns,
        "k": 5,
    }

    retriever_5 = SnowflakeCortexSearchRetriever(**kwargs_5)
    documents_5 = retriever_5.invoke("What is the deal value for SecureBank Ltd?")
    assert len(documents_5) <= 5  # Should return at most 5 documents


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_with_filter(snowflake_session) -> None:
    """Test the invoke() method with filter."""

    search_columns = ["TRANSCRIPT_TEXT", "CUSTOMER_NAME", "DEAL_STAGE", "SALES_REP"]

    kwargs = {
        "session": snowflake_session,
        "service_name": (
            f"{os.environ.get('SNOWFLAKE_DATABASE', 'TESTDB')}."
            f"{os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')}."
            "sales_conversation_search"
        ),
        "search_columns": search_columns,
        "k": 10,
        "filter_dict": {"@eq": {"CUSTOMER_NAME": "TechCorp Inc"}},
    }

    retriever = SnowflakeCortexSearchRetriever(**kwargs)

    documents = retriever.invoke("Tell me about the discovery call with TechCorp Inc")
    assert len(documents) > 0

    # All documents should have TechCorp Inc customer name due to filter
    for doc in documents:
        if "CUSTOMER_NAME" in doc.metadata:
            assert doc.metadata["CUSTOMER_NAME"] == "TechCorp Inc"


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_different_columns(snowflake_session) -> None:
    """Test the invoke() method with different search columns."""

    # Test with minimal columns
    minimal_columns = ["DEAL_VALUE"]
    kwargs_minimal = {
        "session": snowflake_session,
        "service_name": (
            f"{os.environ.get('SNOWFLAKE_DATABASE', 'TESTDB')}."
            f"{os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')}."
            "sales_conversation_search"
        ),
        "search_columns": minimal_columns,
        "k": 5,
    }

    retriever_minimal = SnowflakeCortexSearchRetriever(**kwargs_minimal)
    documents_minimal = retriever_minimal.invoke("What is the deal value for SecureBank Ltd?")

    assert len(documents_minimal) > 0
    for doc in documents_minimal:
        check_document(doc, minimal_columns)

    # Test with more columns
    full_columns = ["TRANSCRIPT_TEXT", "CUSTOMER_NAME", "DEAL_STAGE", "SALES_REP", "DEAL_VALUE"]
    kwargs_full = {
        "session": snowflake_session,
        "service_name": (
            f"{os.environ.get('SNOWFLAKE_DATABASE', 'TESTDB')}."
            f"{os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')}."
            "sales_conversation_search"
        ),
        "search_columns": full_columns,
        "k": 5,
    }

    retriever_full = SnowflakeCortexSearchRetriever(**kwargs_full)
    documents_full = retriever_full.invoke("What is the deal value for SecureBank Ltd?")

    assert len(documents_full) > 0
    for doc in documents_full:
        check_document(doc, full_columns)


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_session_auth(snowflake_session) -> None:
    """Test authentication with a provided `snowflake.snowpark.Session object`."""

    search_columns = ["TRANSCRIPT_TEXT", "CUSTOMER_NAME", "DEAL_STAGE", "SALES_REP"]

    kwargs = {
        "session": snowflake_session,
        "service_name": (
            f"{os.environ.get('SNOWFLAKE_DATABASE', 'TESTDB')}."
            f"{os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')}."
            "sales_conversation_search"
        ),
        "search_columns": search_columns,
        "k": 10,
    }

    retriever = SnowflakeCortexSearchRetriever(**kwargs)

    documents = retriever.invoke("What is the product line for FastTrack Ltd?")
    assert len(documents) > 0

    for doc in documents:
        check_document(doc, search_columns)


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_auto_format_rag(snowflake_session) -> None:
    """Test auto formatting for RAG functionality."""

    search_columns = ["TRANSCRIPT_TEXT", "CUSTOMER_NAME", "DEAL_STAGE", "SALES_REP"]

    # Test with auto_format_for_rag=True (default)
    kwargs_auto = {
        "session": snowflake_session,
        "service_name": (
            f"{os.environ.get('SNOWFLAKE_DATABASE', 'TESTDB')}."
            f"{os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')}."
            "sales_conversation_search"
        ),
        "search_columns": search_columns,
        "k": 5,
        "auto_format_for_rag": True,
    }

    retriever_auto = SnowflakeCortexSearchRetriever(**kwargs_auto)
    documents_auto = retriever_auto.invoke("What is the deal stage for FastTrack Ltd?")
    assert len(documents_auto) > 0

    # Test with auto_format_for_rag=False
    kwargs_manual = {
        "session": snowflake_session,
        "service_name": (
            f"{os.environ.get('SNOWFLAKE_DATABASE', 'TESTDB')}."
            f"{os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')}."
            "sales_conversation_search"
        ),
        "search_columns": search_columns,
        "k": 5,
        "auto_format_for_rag": False,
    }

    retriever_manual = SnowflakeCortexSearchRetriever(**kwargs_manual)
    documents_manual = retriever_manual.invoke("What is the deal stage for FastTrack Ltd?")
    assert len(documents_manual) > 0

    # Both should return documents, but formatting might be different
    for doc in documents_auto:
        check_document(doc, search_columns)

    for doc in documents_manual:
        check_document(doc, search_columns)


def check_document(doc: Document, search_columns: List[str]) -> None:
    """Check the document returned by the retriever."""
    assert isinstance(doc, Document)
    assert doc.page_content or len(doc.metadata) > 0  # Should have content or metadata

    # If search_columns are specified, check that we have some of the expected metadata
    if search_columns:
        # At least some metadata should be present (not all columns may be returned)
        assert len(doc.metadata) > 0


def check_documents(documents: List[Document], search_columns: List[str]) -> None:
    """Check the documents returned by the retriever."""
    for doc in documents:
        check_document(doc, search_columns)
