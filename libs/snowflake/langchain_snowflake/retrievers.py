"""Snowflake retrievers using Cortex Search."""

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import aiohttp
import requests
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from snowflake.snowpark import Session

from ._connection import SnowflakeAuthUtils, SnowflakeConnectionMixin
from ._error_handling import SnowflakeErrorHandler
from .formatters import format_cortex_search_documents

logger = logging.getLogger(__name__)


class SnowflakeCortexSearchRetriever(BaseRetriever, SnowflakeConnectionMixin):
    """Snowflake Cortex Search retriever using REST API exclusively.

    This retriever integrates with Snowflake's Cortex Search service exclusively
    through the REST API. Cortex Search is a managed service that provides
    enterprise-grade semantic search capabilities.

    Note: This retriever uses Cortex Search, which is different from Search Preview.
    Cortex Search only supports REST API access, not SQL functions.

    Setup:
        Install ``langchain-snowflake`` and configure Snowflake connection.

        .. code-block:: bash

            pip install -U langchain-snowflake

    Key init args:
        service_name: str
            Fully qualified name of the Cortex Search service
        session: Optional[Session]
            Active Snowflake session
        k: int
            Number of documents to retrieve (default: 4)
        search_columns: Optional[List[str]]
            Columns to return in search results
        filter_dict: Optional[Dict[str, Any]]
            Filter criteria for search results

    Instantiate:
        .. code-block:: python

            from . import SnowflakeCortexSearchRetriever

            # Using existing session (recommended)
            retriever = SnowflakeCortexSearchRetriever(
                service_name="mydb.myschema.my_search_service",
                session=session,
                k=5
            )

            # Using connection parameters
            retriever = SnowflakeCortexSearchRetriever(
                service_name="mydb.myschema.my_search_service",
                account="your-account",
                user="your-user",
                password="your-password",
                warehouse="your-warehouse",
                k=3
            )

    Usage:
        .. code-block:: python

            query = "What is machine learning?"
            docs = retriever.invoke(query)
            for doc in docs:
                print(doc.page_content)

    Use within a chain:
        .. code-block:: python

            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from . import ChatSnowflake

            prompt = ChatPromptTemplate.from_template(
                \"\"\"Answer the question based only on the context provided.

            Context: {context}

            Question: {question}\"\"\"
            )

            llm = ChatSnowflake(model="llama3.1-70b", session=session)

            # With auto_format_for_rag=True (default), no format_docs needed!
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            # Or with manual control:
            # from .formatters import format_cortex_search_documents
            # retriever_manual = SnowflakeCortexSearchRetriever(..., auto_format_for_rag=False)
            # chain = (
            #     {"context": retriever_manual | format_cortex_search_documents, "question": RunnablePassthrough()}
            #     | prompt | llm | StrOutputParser()
            # )

            response = chain.invoke("What is the capital of France?")
    """

    # Retriever-specific fields (connection fields inherited from SnowflakeConnectionMixin)
    service_name: str = Field(description="Fully qualified Cortex Search service name")

    # Search parameters
    k: int = Field(default=4, description="Number of documents to retrieve")
    search_columns: Optional[List[str]] = Field(default=None, description="Columns to include in results")
    filter_dict: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")

    # RAG optimization parameters
    auto_format_for_rag: bool = Field(
        default=True,
        description="Automatically format documents for RAG by extracting from TRANSCRIPT_TEXT metadata",
    )

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the retriever with proper session attribute."""
        # Call the parent initializer
        super().__init__(**kwargs)
        # Ensure _session attribute is initialized (from SnowflakeConnectionMixin)
        if not hasattr(self, "_session"):
            self._session = None

    # _get_session() method inherited from SnowflakeConnectionMixin

    def format_documents(self, docs: List[Document]) -> List[Document]:
        """Format documents for RAG usage by extracting content from TRANSCRIPT_TEXT metadata.

        This method converts Snowflake Cortex Search documents to be optimized for RAG chains.
        The formatted content is placed in the page_content field for compatibility with
        standard LangChain patterns.

        Args:
            docs: List of Document objects from Cortex Search

        Returns:
            List of Document objects with content properly formatted for RAG
        """
        if not docs:
            return docs

        formatted_docs = []
        for doc in docs:
            # Extract content using the standalone utility
            formatted_content = format_cortex_search_documents([doc])

            # Create new document with formatted content in page_content
            # Keep original metadata for reference
            formatted_doc = Document(
                page_content=formatted_content,
                metadata=(doc.metadata.copy() if hasattr(doc, "metadata") and doc.metadata else {}),
            )

            # Add formatting metadata
            if hasattr(formatted_doc, "metadata"):
                formatted_doc.metadata["_formatted_for_rag"] = True
                formatted_doc.metadata["_original_page_content"] = getattr(doc, "page_content", "")

            formatted_docs.append(formatted_doc)

        logger.debug(f"Formatted {len(docs)} documents for RAG usage")
        return formatted_docs

    def _parse_service_name(self) -> tuple[str, str, str]:
        """Parse the fully qualified service name into database, schema, and service components."""
        parts = self.service_name.split(".")
        if len(parts) != 3:
            raise ValueError(f"Service name must be fully qualified (database.schema.service): {self.service_name}")
        return parts[0], parts[1], parts[2]

    # _get_rest_api_headers() functionality replaced by SnowflakeAuthUtils.get_rest_api_headers()

    def _build_rest_api_payload(self, query: str) -> Dict[str, Any]:
        """Build REST API payload for search request."""
        payload = {"query": query, "limit": self.k}

        if self.search_columns:
            payload["columns"] = self.search_columns

        if self.filter_dict:
            payload["filter"] = self.filter_dict

        return payload

    def _build_cortex_search_url(self, session: Session, database: str, schema: str, service: str) -> str:
        """Build the correct Cortex Search REST API URL.

        Based on Snowflake documentation:
        https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/query-cortex-search-service#rest-api

        Format: https://<ACCOUNT_URL>/api/v2/databases/<DB_NAME>/schemas/<SCHEMA_NAME>/cortex-search-services/<SERVICE_NAME>:query
        """
        try:
            conn = session._conn._conn
            account = conn.account

            # Build base URL with correct hostname format
            if hasattr(conn, "host") and conn.host:
                base_url = f"https://{conn.host}"
            elif "." in account:
                base_url = f"https://{account}.snowflakecomputing.com"
            else:
                region = getattr(conn, "region", None) or "us-west-2"
                if region and region != "us-west-2":
                    base_url = f"https://{account}.{region}.snowflakecomputing.com"
                else:
                    base_url = f"https://{account}.snowflakecomputing.com"

            # Build Cortex Search specific endpoint (NOT Cortex Complete)
            endpoint = (
                f"/api/v2/databases/{quote(database)}/schemas/{quote(schema)}"
                f"/cortex-search-services/{quote(service)}:query"
            )
            url = base_url + endpoint

            logger.debug(f"Built Cortex Search URL: {url}")
            return url

        except Exception as e:
            logger.error(f"Error building Cortex Search URL: {e}")
            raise ValueError(f"Failed to build Cortex Search URL: {e}")

    def _make_rest_api_request(self, query: str) -> List[Document]:
        """Make REST API request to Cortex Search service using correct URL format."""
        session = self._get_session()
        database, schema, service = self._parse_service_name()

        # Build correct Cortex Search URL (different from Cortex Complete)
        url = self._build_cortex_search_url(session, database, schema, service)

        # Get headers using shared utilities
        headers = SnowflakeAuthUtils.get_rest_api_headers(
            session=session,
            account=getattr(self, "account", None),
            user=getattr(self, "user", None),
        )

        # Build payload
        payload = self._build_rest_api_payload(query)

        try:
            # Use configured timeout with SSL verification
            timeout = getattr(self, "request_timeout", 30)
            verify_ssl = getattr(self, "verify_ssl", True)

            response = requests.post(url, json=payload, headers=headers, timeout=timeout, verify=verify_ssl)
            response.raise_for_status()

            data = response.json()
            return self._parse_rest_api_response(data)

        except requests.exceptions.RequestException as e:
            return SnowflakeErrorHandler.log_and_raise(
                error=e,
                operation="Cortex Search REST API request",
                logger_instance=logger,
            )

    async def _make_rest_api_request_async(self, query: str) -> List[Document]:
        """Make async REST API request to Cortex Search service using aiohttp."""
        session = self._get_session()
        database, schema, service = self._parse_service_name()

        # Build correct Cortex Search URL (different from Cortex Complete)
        url = self._build_cortex_search_url(session, database, schema, service)

        # Get headers using shared utilities
        headers = SnowflakeAuthUtils.get_rest_api_headers(
            session=session,
            account=getattr(self, "account", None),
            user=getattr(self, "user", None),
        )

        # Build payload
        payload = self._build_rest_api_payload(query)

        try:
            # Use configured timeout with SSL verification
            timeout = getattr(self, "request_timeout", 30)
            verify_ssl = getattr(self, "verify_ssl", True)

            # Use aiohttp for true async HTTP
            async with aiohttp.ClientSession() as client:
                async with client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                    ssl=verify_ssl,
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return self._parse_rest_api_response(data)

        except Exception as e:
            logger.error(f"Async Cortex Search REST API request failed: {e}")
            raise

    def _parse_rest_api_response(self, data: Dict[str, Any]) -> List[Document]:
        """Parse REST API response into Document objects."""
        documents = []

        results = data.get("results", [])
        for result in results:
            # Extract content and metadata
            content = result.get("content", "")
            metadata = {k: v for k, v in result.items() if k != "content"}

            documents.append(Document(page_content=content, metadata=metadata))

        return documents[: self.k]  # Ensure we don't exceed requested limit

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Retrieve documents relevant to the query using Cortex Search REST API."""
        try:
            docs = self._make_rest_api_request(query)

            # Apply auto-formatting if enabled
            if self.auto_format_for_rag:
                docs = self.format_documents(docs)

            return docs
        except Exception as e:
            SnowflakeErrorHandler.log_warning_and_fallback(
                error=e,
                operation="Cortex Search REST API",
                fallback_action="returning empty results",
                logger_instance=logger,
            )
            return []

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Asynchronously retrieve documents relevant to the query using Cortex Search REST API."""
        try:
            # Use native async REST API with aiohttp
            docs = await self._make_rest_api_request_async(query)

            # Apply auto-formatting if enabled
            if self.auto_format_for_rag:
                docs = self.format_documents(docs)

            return docs
        except Exception as e:
            SnowflakeErrorHandler.log_warning_and_fallback(
                error=e,
                operation="Cortex Search async REST API",
                fallback_action="returning empty results",
                logger_instance=logger,
            )
            return []
