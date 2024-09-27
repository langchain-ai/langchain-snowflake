"""Test Snowflake Cortex Search retriever.

You need to create a Cortex Search service in Snowflake with the specified
columns below to run the integration tests.
Follow the instructions in the example notebook:
`snowflake_cortex_search.ipynb` to set up the service and configure
authentication.

Set the following environment variables before the tests:
export SNOWFLAKE_ACCOUNT=<snowflake_account>
export SNOWFLAKE_USERNAME=<snowflake_username>
export SNOWFLAKE_PASSWORD=<snowflake_password>
export SNOWFLAKE_DATABASE=<snowflake_database>
export SNOWFLAKE_SCHEMA=<snowflake_schema>
export SNOWFLAKE_ROLE=<snowflake_role>
"""

import os
from unittest import mock

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError
from snowflake.snowpark import Session
from langchain_snowflake import CortexSearchRetriever, CortexSearchRetrieverError


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invoke() -> None:
    """Test the invoke() method."""

    columns = ["name", "description", "era", "diet"]
    search_column = "description"

    kwargs = {
        "search_service": "dinosaur_svc",
        "columns": columns,
        "search_column": search_column,
        "filter": {"@eq": {"era": "Jurassic"}},
        "limit": 10,
    }

    retriever = CortexSearchRetriever(**kwargs)

    documents = retriever.invoke("dinosaur with a large tail")
    assert len(documents) > 0

    for doc in documents:
        check_document(doc, columns, search_column)
        # Validate the filter was passed through correctly
        assert doc.metadata["era"] == "Jurassic"


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invoke_no_columns_or_filter() -> None:
    """Test the invoke() method with no columns or filter."""

    kwargs = {
        "search_service": "dinosaur_svc",
        "search_column": "description",
        "limit": 10,
    }

    retriever = CortexSearchRetriever(**kwargs)

    documents = retriever.invoke("dinosaur with a large tail")
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
        assert doc.page_content


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_constructor_no_search_column() -> None:
    """Test the constructor with no search column name provided."""

    columns = ["name", "description", "era", "diet"]

    kwargs = {
        "search_service": "dinosaur_svc",
        "columns": columns,
        "limit": 10,
    }

    with pytest.raises(ValidationError):
        CortexSearchRetriever(**kwargs)


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_retriever_no_service_name() -> None:
    """Test the constructor with no search service name provided."""

    columns = ["name", "description", "era", "diet", "height_meters"]

    kwargs = {
        "columns": columns,
        "limit": 10,
        "search_column": "description",
    }

    with pytest.raises(ValidationError):
        CortexSearchRetriever(**kwargs)


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invoke_invalid_filter() -> None:
    """Test the invoke() method with an invalid filter object."""

    columns = ["name", "description", "era", "diet"]

    kwargs = {
        "columns": columns,
        "search_service": "dinosaur_svc",
        "limit": 10,
        "search_column": "description",
        "filter": {"@eq": ["era", "Jurassic"]},
    }

    retriever = CortexSearchRetriever(**kwargs)

    with pytest.raises(CortexSearchRetrieverError):
        retriever.invoke("dinosaur with a large tail")


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invoke_limit() -> None:
    """Test the invoke() method with an overridden limit."""

    columns = ["name", "description", "era", "diet"]
    search_column = "description"

    kwargs = {
        "search_service": "dinosaur_svc",
        "columns": columns,
        "search_column": search_column,
        "limit": 1,
    }

    retriever = CortexSearchRetriever(**kwargs)

    new_limit = 2

    documents = retriever.invoke("dinosaur with a large tail", limit=new_limit)
    assert len(documents) == new_limit


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invoke_filter() -> None:
    """Test the invoke() method with an overridden filter."""

    columns = ["name", "description", "era", "diet"]
    search_column = "description"

    kwargs = {
        "search_service": "dinosaur_svc",
        "columns": columns,
        "search_column": search_column,
        "limit": 10,
        "filter": {"@eq": {"era": "Jurassic"}},
    }

    retriever = CortexSearchRetriever(**kwargs)

    documents = retriever.invoke("dinosaur with a large tail", filter=None)
    assert len(documents) == 10

    observed_eras = set()
    for doc in documents:
        observed_eras.add(doc.metadata["era"])

    # Since we overrode the default filter with None, we should see more than one era.
    assert len(observed_eras) > 1


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_invoke_columns() -> None:
    """Test the invoke() method with overridden columns."""

    columns = ["description", "era"]
    search_column = "description"

    kwargs = {
        "search_service": "dinosaur_svc",
        "columns": columns,
        "search_column": search_column,
        "limit": 10,
    }

    retriever = CortexSearchRetriever(**kwargs)

    override_columns = ["description"]
    documents = retriever.invoke("dinosaur with a large tail", columns=override_columns)

    assert len(documents) == 10

    for doc in documents:
        check_document(doc, override_columns, search_column)
        assert "era" not in doc.metadata


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_session_auth() -> None:
    """Test authentication with a provided `snowlfake.snowpark.Session object`."""

    columns = ["description", "era"]
    search_column = "description"

    kwargs = {
        "search_service": "dinosaur_svc",
        "columns": columns,
        "search_column": search_column,
        "limit": 10,
    }

    session_config = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USERNAME"],
        "password": os.environ["SNOWFLAKE_PASSWORD"],
        "database": os.environ["SNOWFLAKE_DATABASE"],
        "schema": os.environ["SNOWFLAKE_SCHEMA"],
        "role": os.environ["SNOWFLAKE_ROLE"],
    }

    session = Session.builder.configs(session_config).create()

    retriever = CortexSearchRetriever(sp_session=session, **kwargs)

    documents = retriever.invoke("dinosaur with a large tail")
    assert len(documents) > 0

    for doc in documents:
        check_document(doc, columns, search_column)


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_session_auth_validation_error() -> None:
    """Test validation errors when both a `snowlfake.snowpark.Session object` and 
    another authentication paramter are provided."""

    columns = ["name", "description", "era", "diet"]
    search_column = "description"
    kwargs = {
        "search_service": "dinosaur_svc",
        "columns": columns,
        "search_column": search_column,
        "limit": 10,
    }

    session_config = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USERNAME"],
        "password": os.environ["SNOWFLAKE_PASSWORD"],
        "database": os.environ["SNOWFLAKE_DATABASE"],
        "schema": os.environ["SNOWFLAKE_SCHEMA"],
        "role": os.environ["SNOWFLAKE_ROLE"],
    }

    session = Session.builder.configs(session_config).create()

    for param in ["account", "user", "password", "role", "authenticator"]:
        with pytest.raises(CortexSearchRetrieverError):
            kwargs[param] = "fake_value"
            CortexSearchRetriever(
                sp_session=session, **kwargs
            )
            del kwargs[param]


@pytest.mark.requires("snowflake.core")
def test_snowflake_cortex_search_session_auth_overrides() -> None:
    """Test overrides to the provided `snowlfake.snowpark.Session object`."""

    columns = ["name", "description", "era", "diet"]
    search_column = "description"
    kwargs = {
        "search_service": "dinosaur_svc",
        "columns": columns,
        "search_column": search_column,
        "limit": 10,
    }

    session_config = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USERNAME"],
        "password": os.environ["SNOWFLAKE_PASSWORD"],
        "database": os.environ["SNOWFLAKE_DATABASE"],
        "schema": os.environ["SNOWFLAKE_SCHEMA"],
        "role": os.environ["SNOWFLAKE_ROLE"],
    }

    for param in ["database", "schema"]:
        session_config_copy = session_config.copy()
        session_config_copy[param] = None
        session = Session.builder.configs(session_config_copy).create()

        retriever = CortexSearchRetriever(sp_session=session, **kwargs)

        documents = retriever.invoke("dinosaur with a large tail")
        assert len(documents) > 0

        check_documents(documents, columns, search_column)


@pytest.mark.skip(
    """This test requires a Snowflake account with externalbrowser authentication
    enabled."""
)
@pytest.mark.requires("snowflake.core")
@mock.patch.dict(os.environ, {"SNOWFLAKE_PASSWORD": ""})
def test_snowflake_cortex_search_constructor_externalbrowser_authenticator() -> None:
    """Test the constructor with external browser authenticator."""

    columns = ["name", "description", "era", "diet"]
    search_column = "description"
    kwargs = {
        "search_service": "dinosaur_svc",
        "columns": columns,
        "search_column": search_column,
        "limit": 10,
        "authenticator": "externalbrowser",
    }

    retriever = CortexSearchRetriever(**kwargs)

    documents = retriever.invoke("dinosaur with a large tail")
    assert len(documents) > 0
    check_documents(documents, columns, search_column)


def check_document(doc, columns, search_column) -> None:
    """Check the document returned by the retriever."""
    assert isinstance(doc, Document)
    assert doc.page_content
    for column in columns:
        if column == search_column:
            continue
        assert column in doc.metadata


def check_documents(documents, columns, search_column) -> None:
    """Check the documents returned by the retriever."""
    for doc in documents:
        check_document(doc, columns, search_column)
