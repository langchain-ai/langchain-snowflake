"""Document formatters for Snowflake-specific data structures.

This module provides utilities for formatting documents retrieved from various
Snowflake services for optimal use in RAG (Retrieval-Augmented Generation) chains.
"""

import logging
from typing import List

from langchain_core.documents import Document

from ._error_handling import SnowflakeErrorHandler

logger = logging.getLogger(__name__)


def format_cortex_search_documents(
    docs: List[Document],
    content_field: str = "TRANSCRIPT_TEXT",
    join_separator: str = "\n\n",
    fallback_to_page_content: bool = True,
) -> str:
    """Format documents from Snowflake Cortex Search for RAG usage.

    This function extracts content from Cortex Search documents and formats them
    into a single string suitable for use as context in RAG applications.

    Args:
        docs: List of Document objects from Cortex Search
        content_field: Metadata field containing the main content
        join_separator: String used to join multiple documents
        fallback_to_page_content: Whether to use page_content if content_field is missing

    Returns:
        Formatted string containing all document content

    Example:
        >>> from langchain_snowflake import SnowflakeCortexSearchRetriever, format_cortex_search_documents
        >>> retriever = SnowflakeCortexSearchRetriever(...)
        >>> docs = retriever.get_relevant_documents("query")
        >>> context = format_cortex_search_documents(docs, content_field="CONTENT")
    """
    if not docs:
        SnowflakeErrorHandler.log_debug(
            "document formatting", "no documents provided to format_cortex_search_documents"
        )
        return ""

    content_parts = []

    for i, doc in enumerate(docs):
        # Try to extract content from specified field
        if content_field in doc.metadata:
            content = doc.metadata[content_field]
            if content and str(content).strip():
                content_parts.append(str(content).strip())
                SnowflakeErrorHandler.log_debug(
                    "document formatting",
                    f"extracted content from {content_field} for document {i + 1}: {len(str(content))} chars",
                )
                continue

        # Fallback to page_content if enabled and available
        if fallback_to_page_content and doc.page_content and str(doc.page_content).strip():
            content = doc.page_content
            if content:
                content_parts.append(str(content).strip())
                SnowflakeErrorHandler.log_debug(
                    "document formatting", f"used page_content fallback for document {i + 1}: {len(str(content))} chars"
                )
                continue

        # Log when no content is found
        SnowflakeErrorHandler.log_warning_and_fallback(
            error=Exception(f"No content found for document {i + 1}"),
            operation="document content extraction",
            fallback_action=f"available metadata keys: {list(doc.metadata.keys())}",
        )

    result = join_separator.join(content_parts)
    SnowflakeErrorHandler.log_debug(
        "document formatting", f"formatted {len(docs)} documents into {len(result)} characters of context"
    )

    return result
