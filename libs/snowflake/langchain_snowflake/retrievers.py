"""Snowflake retrievers using Cortex Search."""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from ._connection import RestApiClient, RestApiRequestBuilder, SnowflakeConnectionMixin
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
        content_field: str
            Metadata field containing the actual content (default: "TRANSCRIPT_TEXT")
        join_separator: str
            String to join multiple documents (default: "\\n\\n")
        fallback_to_page_content: bool
            Fall back to page_content when metadata field is empty (default: True)

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

            # Using custom content field (e.g., for datasets that store content in "CHUNK")
            retriever_custom = SnowflakeCortexSearchRetriever(
                service_name="mydb.myschema.my_search_service",
                session=session,
                k=5,
                content_field="CHUNK",  # Extract from metadata["CHUNK"] instead of "TRANSCRIPT_TEXT"
                join_separator="\\n---\\n",  # Custom separator
                fallback_to_page_content=True
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
        description="Automatically format documents for RAG by extracting from metadata",
    )

    # Document formatting parameters (configurable alternatives to hardcoded defaults)
    content_field: str = Field(
        default="TRANSCRIPT_TEXT",
        description="Metadata field containing the actual content to extract into page_content",
    )
    join_separator: str = Field(
        default="\n\n",
        description="String to join multiple documents when formatting",
    )
    fallback_to_page_content: bool = Field(
        default=True,
        description="If True, fall back to page_content when metadata field is empty",
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
        """Format documents for RAG usage by extracting content from configurable metadata field.

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
            # Extract content using the standalone utility with configurable parameters
            formatted_content = format_cortex_search_documents(
                [doc],
                content_field=self.content_field,
                join_separator=self.join_separator,
                fallback_to_page_content=self.fallback_to_page_content,
            )

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
                formatted_doc.metadata["_content_field_used"] = self.content_field

            formatted_docs.append(formatted_doc)

        SnowflakeErrorHandler.log_debug(
            "document formatting",
            f"formatted {len(docs)} documents for RAG usage using content_field='{self.content_field}'",
        )
        return formatted_docs

    def _parse_service_name(self) -> tuple[str, str, str]:
        """Parse the fully qualified service name into database, schema, and service components."""
        # Use centralized validation utilities
        from ._validation_utils import SnowflakeValidationUtils

        return SnowflakeValidationUtils.validate_service_name(self.service_name)

    # _get_rest_api_headers() functionality replaced by SnowflakeAuthUtils.get_rest_api_headers()

    def _build_rest_api_payload(self, query: str) -> Dict[str, Any]:
        """Build REST API payload for search request."""
        payload = {"query": query, "limit": self.k}

        if self.search_columns:
            payload["columns"] = self.search_columns

        if self.filter_dict:
            payload["filter"] = self.filter_dict

        return payload

    def _make_rest_api_request(self, query: str) -> List[Document]:
        """Make REST API request to Cortex Search service using unified client."""
        session = self._get_session()
        database, schema, service = self._parse_service_name()
        payload = self._build_rest_api_payload(query)

        try:
            # Build request using unified client
            request_config = RestApiRequestBuilder.cortex_search_request(
                session=session,
                database=database,
                schema=schema,
                service=service,
                method="POST",
                payload=payload,
                request_timeout=self.request_timeout,
                verify_ssl=self.verify_ssl,
            )

            response_data = RestApiClient.make_sync_request(request_config, "Cortex Search REST API request")
            return self._parse_rest_api_response(response_data)

        except Exception as e:
            SnowflakeErrorHandler.log_and_raise(
                error=e,
                operation="Cortex Search REST API request",
                logger_instance=logger,
            )

    async def _make_rest_api_request_async(self, query: str) -> List[Document]:
        """Make async REST API request to Cortex Search service using unified client."""
        session = self._get_session()
        database, schema, service = self._parse_service_name()
        payload = self._build_rest_api_payload(query)

        try:
            # Build request using unified client
            request_config = RestApiRequestBuilder.cortex_search_request(
                session=session,
                database=database,
                schema=schema,
                service=service,
                method="POST",
                payload=payload,
                request_timeout=self.request_timeout,
                verify_ssl=self.verify_ssl,
            )

            response_data = await RestApiClient.make_async_request(
                request_config, "async Cortex Search REST API request"
            )
            return self._parse_rest_api_response(response_data)

        except Exception as e:
            SnowflakeErrorHandler.log_and_raise(
                error=e,
                operation="async Cortex Search REST API request",
                logger_instance=logger,
            )

    def _parse_rest_api_response(self, data: Dict[str, Any]) -> List[Document]:
        """Parse REST API response into Document objects."""
        documents = []

        results = data.get("results", [])
        if not isinstance(results, list):
            logger.warning(f"Expected list in 'results', got {type(results).__name__}")
            return []

        for result in results:
            # Ensure result is a dict before accessing it
            if not isinstance(result, dict):
                logger.warning(f"Expected dict in results, got {type(result).__name__}")
                continue

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
