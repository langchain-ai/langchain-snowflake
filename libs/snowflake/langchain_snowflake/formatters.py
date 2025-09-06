"""Document formatters for Snowflake-specific data structures.

This module provides utilities for formatting documents retrieved from various
Snowflake services for optimal use in RAG (Retrieval-Augmented Generation) chains.
"""

import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def format_cortex_search_documents(
    docs: List[Document],
    content_field: str = "TRANSCRIPT_TEXT",
    join_separator: str = "\n\n",
    fallback_to_page_content: bool = True,
) -> str:
    """Format Snowflake Cortex Search documents for RAG usage.

    This utility extracts content from Snowflake Cortex Search documents which
    typically store the actual content in metadata fields rather than page_content.

    Args:
        docs: List of Document objects from Snowflake Cortex Search
        content_field: Metadata field containing the actual content (default: "TRANSCRIPT_TEXT")
        join_separator: String to join multiple documents (default: "\\n\\n")
        fallback_to_page_content: If True, fall back to page_content when metadata field is empty

    Returns:
        Formatted string ready for use in RAG chains

    Example:
        >>> from .formatters import format_cortex_search_documents
        >>> formatted_context = format_cortex_search_documents(docs)
        >>> chain = retriever | format_cortex_search_documents | prompt | llm
    """
    if not docs:
        logger.debug("No documents provided to format_cortex_search_documents")
        return ""

    content_parts = []

    for i, doc in enumerate(docs):
        content = None

        # Try to extract from specified metadata field
        if hasattr(doc, "metadata") and doc.metadata and content_field in doc.metadata:
            content = doc.metadata[content_field]
            if content and str(content).strip():
                content_parts.append(str(content).strip())
                logger.debug(f"Extracted content from {content_field} for document {i + 1}: {len(str(content))} chars")
                continue

        # Fallback to page_content if enabled and available
        if fallback_to_page_content and hasattr(doc, "page_content") and doc.page_content:
            content = doc.page_content
            if content and str(content).strip():
                content_parts.append(str(content).strip())
                logger.debug(f"Used page_content fallback for document {i + 1}: {len(str(content))} chars")
                continue

        # Log when no content is found
        logger.warning(
            f"""No content found for document {i + 1}. Available metadata keys: 
                    {list(doc.metadata.keys()) if hasattr(doc, 'metadata') and doc.metadata else 'None'}"""
        )

    result = join_separator.join(content_parts)
    logger.debug(f"Formatted {len(docs)} documents into {len(result)} characters of context")

    return result
