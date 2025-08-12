"""Basic Snowflake Cortex AI function tools for sentiment, summarization, translation, and completion."""

import asyncio
import json
import logging
from typing import Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from .._connection import SnowflakeConnectionMixin
from .._error_handling import SnowflakeToolErrorHandler
from ._base import (
    CortexCompleteInput,
    CortexSentimentInput,
    CortexSummarizerInput,
    CortexTranslatorInput,
)

logger = logging.getLogger(__name__)


class CortexSentimentTool(BaseTool, SnowflakeConnectionMixin):
    """Analyze sentiment using Snowflake Cortex SENTIMENT function."""

    name: str = "cortex_sentiment"
    description: str = (
        "Analyze sentiment of text using Snowflake Cortex SENTIMENT function"
    )
    args_schema: Type[BaseModel] = CortexSentimentInput

    def __init__(self, **kwargs):
        """Initialize the sentiment tool with proper session attribute."""
        # Call the parent initializer
        super().__init__(**kwargs)
        # Ensure _session attribute is initialized (from SnowflakeConnectionMixin)
        if not hasattr(self, "_session"):
            self._session = None

    def _run(
        self, text: str, *, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Analyze sentiment of the given text."""
        session = self._get_session()

        try:
            sql = f"""
            SELECT SNOWFLAKE.CORTEX.SENTIMENT(
                '{text.replace("'", "''")}'
            ) as sentiment_score
            """

            result = session.sql(sql).collect()

            if result:
                sentiment_score = result[0]["SENTIMENT_SCORE"]

                # Interpret the sentiment score
                if sentiment_score > 0.1:
                    sentiment_label = "positive"
                elif sentiment_score < -0.1:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"

                return json.dumps(
                    {
                        "text": text,
                        "sentiment_score": float(sentiment_score),
                        "sentiment_label": sentiment_label,
                    }
                )
            else:
                return json.dumps({"error": "No sentiment analysis result"})

        except Exception as e:
            return SnowflakeToolErrorHandler.handle_tool_error(
                e, "CortexSentimentTool", "analyze sentiment"
            )

    async def _arun(
        self, text: str, *, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Async analyze sentiment of the given text using native Snowflake async."""
        session = self._get_session()

        try:
            escaped_text = text.replace("'", "''")
            sql = f"""
            SELECT SNOWFLAKE.CORTEX.SENTIMENT(
                '{escaped_text}'
            ) as sentiment_score
            """

            # Use native Snowflake async execution
            async_job = session.sql(sql).collect_nowait()

            # Wait for completion and get results using thread pool only for the result retrieval
            result = await asyncio.to_thread(async_job.result)

            if result:
                sentiment_score = result[0]["SENTIMENT_SCORE"]

                # Interpret the sentiment score
                if sentiment_score > 0.1:
                    sentiment_label = "positive"
                elif sentiment_score < -0.1:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"

                return json.dumps(
                    {
                        "text": text,
                        "sentiment_score": float(sentiment_score),
                        "sentiment_label": sentiment_label,
                    }
                )
            else:
                return json.dumps({"error": "No sentiment analysis result"})

        except Exception as e:
            return SnowflakeToolErrorHandler.handle_tool_error(
                e, "CortexSentimentTool", "analyze sentiment async"
            )


class CortexSummarizerTool(BaseTool, SnowflakeConnectionMixin):
    """Summarize text using Snowflake Cortex SUMMARIZE function."""

    name: str = "cortex_summarize"
    description: str = "Summarize text using Snowflake Cortex SUMMARIZE function"
    args_schema: Type[BaseModel] = CortexSummarizerInput

    def __init__(self, **kwargs):
        """Initialize the summarizer tool with proper session attribute."""
        # Call the parent initializer
        super().__init__(**kwargs)
        # Ensure _session attribute is initialized (from SnowflakeConnectionMixin)
        if not hasattr(self, "_session"):
            self._session = None

    def _run(
        self, text: str, *, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Summarize the given text."""
        session = self._get_session()

        try:
            sql = f"""
            SELECT SNOWFLAKE.CORTEX.SUMMARIZE(
                '{text.replace("'", "''")}'
            ) as summary
            """

            result = session.sql(sql).collect()

            if result:
                summary = result[0]["SUMMARY"]
                return json.dumps(
                    {
                        "original_text": text[:100] + "..."
                        if len(text) > 100
                        else text,
                        "summary": summary,
                        "original_length": len(text),
                        "summary_length": len(summary) if summary else 0,
                    }
                )
            else:
                return json.dumps({"error": "No summary generated"})

        except Exception as e:
            return SnowflakeToolErrorHandler.handle_tool_error(
                e, "CortexSummarizerTool", "summarize text"
            )

    async def _arun(
        self, text: str, *, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Async summarize the given text using native Snowflake async."""
        session = self._get_session()

        try:
            escaped_text = text.replace("'", "''")
            sql = f"""
            SELECT SNOWFLAKE.CORTEX.SUMMARIZE(
                '{escaped_text}'
            ) as summary
            """

            # Use native Snowflake async execution
            async_job = session.sql(sql).collect_nowait()

            # Wait for completion and get results using thread pool only for the result retrieval
            result = await asyncio.to_thread(async_job.result)

            if result:
                summary = result[0]["SUMMARY"]
                return json.dumps(
                    {
                        "original_text": text[:100] + "..."
                        if len(text) > 100
                        else text,
                        "summary": summary,
                        "original_length": len(text),
                        "summary_length": len(summary) if summary else 0,
                    }
                )
            else:
                return json.dumps({"error": "No summary generated"})

        except Exception as e:
            return SnowflakeToolErrorHandler.handle_tool_error(
                e, "CortexSummarizerTool", "summarize text async"
            )


class CortexTranslatorTool(BaseTool, SnowflakeConnectionMixin):
    """Translate text using Snowflake Cortex TRANSLATE function."""

    name: str = "cortex_translate"
    description: str = "Translate text using Snowflake Cortex TRANSLATE function"
    args_schema: Type[BaseModel] = CortexTranslatorInput

    def __init__(self, **kwargs):
        """Initialize the translator tool with proper session attribute."""
        # Call the parent initializer
        super().__init__(**kwargs)
        # Ensure _session attribute is initialized (from SnowflakeConnectionMixin)
        if not hasattr(self, "_session"):
            self._session = None

    def _run(
        self,
        text: str,
        target_language: str = "es",  # Default to Spanish to prevent missing argument errors
        source_language: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Translate the given text to target language."""
        session = self._get_session()

        try:
            # Handle both string and dict inputs
            if isinstance(text, dict):
                text_content = text.get("text", str(text))
            else:
                text_content = str(text)

            # CORTEX.TRANSLATE expects (text, source_language, target_language)
            # If source_language is not provided, we default to 'en' (English)
            # Note: 'auto' is not supported by Snowflake CORTEX.TRANSLATE
            source_lang = source_language or "en"
            escaped_text = text_content.replace("'", "''")
            sql = f"""
            SELECT SNOWFLAKE.CORTEX.TRANSLATE(
                '{escaped_text}',
                '{source_lang}',
                '{target_language}'
            ) as translated_text
            """

            result = session.sql(sql).collect()

            if result:
                translated_text = result[0]["TRANSLATED_TEXT"]
                return json.dumps(
                    {
                        "original_text": text_content,
                        "translated_text": translated_text,
                        "target_language": target_language,
                    }
                )
            else:
                return json.dumps({"error": "No translation generated"})

        except Exception as e:
            return SnowflakeToolErrorHandler.handle_tool_error(
                e, "CortexTranslatorTool", "translate text"
            )

    async def _arun(
        self,
        text: str,
        target_language: str = "es",  # Default to Spanish to prevent missing argument errors
        source_language: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async translate the given text to target language using native Snowflake async."""
        session = self._get_session()

        try:
            # Handle both string and dict inputs
            if isinstance(text, dict):
                text_content = text.get("text", str(text))
            else:
                text_content = str(text)

            escaped_text = text_content.replace("'", "''")
            # CORTEX.TRANSLATE expects (text, source_language, target_language)
            # If source_language is not provided, we default to 'en' (English)
            # Note: 'auto' is not supported by Snowflake CORTEX.TRANSLATE
            source_lang = source_language or "en"
            sql = f"""
            SELECT SNOWFLAKE.CORTEX.TRANSLATE(
                '{escaped_text}',
                '{source_lang}',
                '{target_language}'
            ) as translated_text
            """

            # Use native Snowflake async execution
            async_job = session.sql(sql).collect_nowait()

            # Wait for completion and get results using thread pool only for the result retrieval
            result = await asyncio.to_thread(async_job.result)

            if result:
                translated_text = result[0]["TRANSLATED_TEXT"]
                return json.dumps(
                    {
                        "original_text": text_content,
                        "translated_text": translated_text,
                        "target_language": target_language,
                    }
                )
            else:
                return json.dumps({"error": "No translation generated"})

        except Exception as e:
            return SnowflakeToolErrorHandler.handle_tool_error(
                e, "CortexTranslatorTool", "translate text async"
            )


class CortexCompleteTool(BaseTool, SnowflakeConnectionMixin):
    """Generate text completions using Snowflake Cortex COMPLETE function."""

    name: str = "cortex_complete"
    description: str = (
        "Generate text completions using Snowflake Cortex COMPLETE function"
    )
    args_schema: Type[BaseModel] = CortexCompleteInput

    def __init__(self, **kwargs):
        """Initialize the complete tool with proper session attribute."""
        # Call the parent initializer
        super().__init__(**kwargs)
        # Ensure _session attribute is initialized (from SnowflakeConnectionMixin)
        if not hasattr(self, "_session"):
            self._session = None

    def _run(
        self,
        prompt: str,
        model: str = "llama3.1-70b",
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Generate text completion for the given prompt."""
        session = self._get_session()

        try:
            sql = f"""
            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                '{model}',
                '{prompt.replace("'", "''")}'
            ) as completion
            """

            result = session.sql(sql).collect()

            if result:
                completion = result[0]["COMPLETION"]
                return json.dumps(
                    {"prompt": prompt, "completion": completion, "model": model}
                )
            else:
                return json.dumps({"error": "No completion generated"})

        except Exception as e:
            return SnowflakeToolErrorHandler.handle_tool_error(
                e, "CortexCompleteTool", "generate completion"
            )

    async def _arun(
        self,
        prompt: str,
        model: str = "llama3.1-70b",
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async generate text completion for the given prompt using native Snowflake async."""
        session = self._get_session()

        try:
            escaped_prompt = prompt.replace("'", "''")
            sql = f"""
            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                '{model}',
                '{escaped_prompt}'
            ) as completion
            """

            # Use native Snowflake async execution
            async_job = session.sql(sql).collect_nowait()

            # Wait for completion and get results using thread pool only for the result retrieval
            result = await asyncio.to_thread(async_job.result)

            if result:
                completion = result[0]["COMPLETION"]
                return json.dumps(
                    {"prompt": prompt, "completion": completion, "model": model}
                )
            else:
                return json.dumps({"error": "No completion generated"})

        except Exception as e:
            return SnowflakeToolErrorHandler.handle_tool_error(
                e, "CortexCompleteTool", "generate completion async"
            )
