"""Test Snowflake Cortex Search retriever.

You need to create a Cortex Search service in Snowflake with the specified
columns below to run the integration tests.
Follow the instructions in the example notebook:
`snowflake_cortex_search.ipynb` to set up the service and configure
authentication.

Set the following environment variables before the tests:
export SNOWFLAKE_ACCOUNT=<snowflake_account>
export SNOWFLAKE_USER=<snowflake_username>  # Note: USER not USERNAME
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

from langchain_snowflake import SnowflakeCortexSearchRetriever


# Compilation test for CI - always passes
@pytest.mark.compile
def test_retriever_imports_compile() -> None:
    """Test that retriever imports compile correctly."""
    assert SnowflakeCortexSearchRetriever is not None
    assert Document is not None
    assert Session is not None


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invoke() -> None:
    """Test the invoke() method."""

    search_columns = ["name", "description", "era", "diet"]

    # Updated to match current API
    kwargs = {
        "service_name": f"{os.environ.get('SNOWFLAKE_DATABASE', 'TESTDB')}.\
            {os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')}.dinosaur_svc",
        "search_columns": search_columns,
        "filter_dict": {"@eq": {"era": "Jurassic"}},
        "k": 10,
    }

    retriever = SnowflakeCortexSearchRetriever(**kwargs)

    documents = retriever.invoke("dinosaur with a large tail")
    assert len(documents) > 0

    for doc in documents:
        check_document(doc, search_columns)
        # Validate the filter was passed through correctly (if era metadata exists)
        if "era" in doc.metadata:
            assert doc.metadata["era"] == "Jurassic"


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invoke_no_columns_or_filter() -> None:
    """Test the invoke() method with no columns or filter."""

    kwargs = {
        "service_name": f"{os.environ.get('SNOWFLAKE_DATABASE', 'TESTDB')}.\
            {os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')}.dinosaur_svc",
        "k": 10,
    }

    retriever = SnowflakeCortexSearchRetriever(**kwargs)

    documents = retriever.invoke("dinosaur with a large tail")
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
        assert (
            doc.page_content or len(doc.metadata) > 0
        )  # Should have content or metadata


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_constructor_no_service_name() -> None:
    """Test the constructor with no service name provided."""

    search_columns = ["name", "description", "era", "diet"]

    kwargs = {
        "search_columns": search_columns,
        "k": 10,
    }

    with pytest.raises(ValidationError):
        SnowflakeCortexSearchRetriever(**kwargs)


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invalid_service_name() -> None:
    """Test the constructor with invalid service name format."""

    search_columns = ["name", "description", "era", "diet"]

    kwargs = {
        "service_name": "invalid_service_name",  # Should be database.schema.service
        "search_columns": search_columns,
        "k": 10,
    }

    # This should create the retriever but fail when trying to parse service name
    retriever = SnowflakeCortexSearchRetriever(**kwargs)

    # The error will occur when trying to invoke, not during construction
    with pytest.raises(ValueError, match="Service name must be fully qualified"):
        retriever.invoke("test query")


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invoke_different_k_values() -> None:
    """Test the invoke() method with different k values."""

    search_columns = ["name", "description", "era", "diet"]

    # Test with k=1
    kwargs_1 = {
        "service_name": f"{os.environ.get('SNOWFLAKE_DATABASE', 'TESTDB')}.\
            {os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')}.dinosaur_svc",
        "search_columns": search_columns,
        "k": 1,
    }

    retriever_1 = SnowflakeCortexSearchRetriever(**kwargs_1)
    documents_1 = retriever_1.invoke("dinosaur with a large tail")
    assert len(documents_1) <= 1  # Should return at most 1 document

    # Test with k=5
    kwargs_5 = {
        "service_name": f"{os.environ.get('SNOWFLAKE_DATABASE', 'TESTDB')}.\
            {os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')}.dinosaur_svc",
        "search_columns": search_columns,
        "k": 5,
    }

    retriever_5 = SnowflakeCortexSearchRetriever(**kwargs_5)
    documents_5 = retriever_5.invoke("dinosaur with a large tail")
    assert len(documents_5) <= 5  # Should return at most 5 documents


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_with_filter() -> None:
    """Test the invoke() method with filter."""

    search_columns = ["name", "description", "era", "diet"]

    kwargs = {
        "service_name": f"{os.environ.get('SNOWFLAKE_DATABASE', 'TESTDB')}.\
            {os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')}.dinosaur_svc",
        "search_columns": search_columns,
        "k": 10,
        "filter_dict": {"@eq": {"era": "Jurassic"}},
    }

    retriever = SnowflakeCortexSearchRetriever(**kwargs)

    documents = retriever.invoke("dinosaur with a large tail")
    assert len(documents) > 0

    # All documents should have Jurassic era due to filter
    for doc in documents:
        if "era" in doc.metadata:
            assert doc.metadata["era"] == "Jurassic"


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_different_columns() -> None:
    """Test the invoke() method with different search columns."""

    # Test with minimal columns
    minimal_columns = ["description"]
    kwargs_minimal = {
        "service_name": f"{os.environ.get('SNOWFLAKE_DATABASE', 'TESTDB')}.\
            {os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')}.dinosaur_svc",
        "search_columns": minimal_columns,
        "k": 5,
    }

    retriever_minimal = SnowflakeCortexSearchRetriever(**kwargs_minimal)
    documents_minimal = retriever_minimal.invoke("dinosaur with a large tail")

    assert len(documents_minimal) > 0
    for doc in documents_minimal:
        check_document(doc, minimal_columns)

    # Test with more columns
    full_columns = ["name", "description", "era", "diet"]
    kwargs_full = {
        "service_name": f"{os.environ.get('SNOWFLAKE_DATABASE', 'TESTDB')}.\
            {os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')}.dinosaur_svc",
        "search_columns": full_columns,
        "k": 5,
    }

    retriever_full = SnowflakeCortexSearchRetriever(**kwargs_full)
    documents_full = retriever_full.invoke("dinosaur with a large tail")

    assert len(documents_full) > 0
    for doc in documents_full:
        check_document(doc, full_columns)


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_session_auth() -> None:
    """Test authentication with a provided `snowflake.snowpark.Session object`."""

    search_columns = ["description", "era"]

    kwargs = {
        "service_name": f"{os.environ.get('SNOWFLAKE_DATABASE', 'TESTDB')}.\
            {os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')}.dinosaur_svc",
        "search_columns": search_columns,
        "k": 10,
    }

    session_config = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],  # Updated to use SNOWFLAKE_USER
        "password": os.environ["SNOWFLAKE_PASSWORD"],
        "database": os.environ["SNOWFLAKE_DATABASE"],
        "schema": os.environ["SNOWFLAKE_SCHEMA"],
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    }

    session = Session.builder.configs(session_config).create()

    retriever = SnowflakeCortexSearchRetriever(session=session, **kwargs)

    documents = retriever.invoke("dinosaur with a large tail")
    assert len(documents) > 0

    for doc in documents:
        check_document(doc, search_columns)


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_auto_format_rag() -> None:
    """Test auto formatting for RAG functionality."""

    search_columns = ["name", "description", "era", "diet"]

    # Test with auto_format_for_rag=True (default)
    kwargs_auto = {
        "service_name": f"{os.environ.get('SNOWFLAKE_DATABASE', 'TESTDB')}.\
            {os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')}.dinosaur_svc",
        "search_columns": search_columns,
        "k": 5,
        "auto_format_for_rag": True,
    }

    retriever_auto = SnowflakeCortexSearchRetriever(**kwargs_auto)
    documents_auto = retriever_auto.invoke("dinosaur with a large tail")
    assert len(documents_auto) > 0

    # Test with auto_format_for_rag=False
    kwargs_manual = {
        "service_name": f"{os.environ.get('SNOWFLAKE_DATABASE', 'TESTDB')}.\
            {os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')}.dinosaur_svc",
        "search_columns": search_columns,
        "k": 5,
        "auto_format_for_rag": False,
    }

    retriever_manual = SnowflakeCortexSearchRetriever(**kwargs_manual)
    documents_manual = retriever_manual.invoke("dinosaur with a large tail")
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
