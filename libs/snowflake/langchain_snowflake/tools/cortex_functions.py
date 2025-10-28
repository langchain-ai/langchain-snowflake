"""Basic Snowflake Cortex AI function tools for sentiment, summarization, translation, and completion."""

import json
import logging
from typing import Any, Dict, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from .._connection import SnowflakeConnectionMixin, SqlExecutionClient
from ._base import (
    CortexCompleteInput,
    CortexSentimentInput,
    CortexSummarizerInput,
    CortexTranslatorInput,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Cortex Function Tools
# ============================================================================


class CortexSentimentTool(BaseTool, SnowflakeConnectionMixin):
    """Analyze sentiment using Snowflake Cortex SENTIMENT function."""

    name: str = "cortex_sentiment"
    description: str = "Analyze sentiment of text using Snowflake Cortex SENTIMENT function"
    args_schema: Union[Type[BaseModel], Dict[str, Any], None] = CortexSentimentInput

    def __init__(self, **kwargs):
        """Initialize the sentiment tool with proper session attribute."""
        # Extract session from kwargs if provided
        session = kwargs.pop("session", None)

        # Call the parent initializer with remaining kwargs
        super().__init__(**kwargs)

        # Initialize session attribute (from SnowflakeConnectionMixin) using object.__setattr__ to bypass Pydantic
        if session is not None:
            object.__setattr__(self, "_session", session)
        elif not hasattr(self, "_session"):
            object.__setattr__(self, "_session", None)

    def _run(self, text: str, *, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Analyze sentiment of the given text."""
        execution_result = SqlExecutionClient.execute_cortex_function(
            session=self._get_session(), function_name="SENTIMENT", args=[text], operation_name="analyze sentiment"
        )

        if not execution_result["success"]:
            return execution_result["error"]

        result = execution_result["result"]
        if result:
            sentiment_score = result[0]["RESULT"]

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

    async def _arun(self, text: str, *, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Async analyze sentiment using native async SqlExecutionClient."""
        execution_result = await SqlExecutionClient.execute_cortex_function_async(
            session=self._get_session(), function_name="SENTIMENT", args=[text], operation_name="analyze sentiment"
        )

        if not execution_result["success"]:
            return execution_result["error"]

        result = execution_result["result"]
        if result:
            sentiment_score = result[0]["RESULT"]

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


class CortexSummarizerTool(BaseTool, SnowflakeConnectionMixin):
    """Summarize text using Snowflake Cortex SUMMARIZE function."""

    name: str = "cortex_summarize"
    description: str = "Summarize text using Snowflake Cortex SUMMARIZE function"
    args_schema: Union[Type[BaseModel], Dict[str, Any], None] = CortexSummarizerInput

    def __init__(self, **kwargs):
        """Initialize the summarizer tool with proper session attribute."""
        # Extract session from kwargs if provided
        session = kwargs.pop("session", None)

        # Call the parent initializer with remaining kwargs
        super().__init__(**kwargs)

        # Initialize session attribute (from SnowflakeConnectionMixin) using object.__setattr__ to bypass Pydantic
        if session is not None:
            object.__setattr__(self, "_session", session)
        elif not hasattr(self, "_session"):
            object.__setattr__(self, "_session", None)

    def _run(self, text: str, *, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Summarize the given text."""
        execution_result = SqlExecutionClient.execute_cortex_function(
            session=self._get_session(), function_name="SUMMARIZE", args=[text], operation_name="summarize text"
        )

        if not execution_result["success"]:
            return execution_result["error"]

        result = execution_result["result"]
        if result:
            summary = result[0]["RESULT"]
            return json.dumps(
                {
                    "original_text": (text[:100] + "..." if len(text) > 100 else text),
                    "summary": summary,
                    "original_length": len(text),
                    "summary_length": len(summary) if summary else 0,
                }
            )
        else:
            return json.dumps({"error": "No summary generated"})

    async def _arun(self, text: str, *, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Async summarize text using native async SqlExecutionClient."""
        execution_result = await SqlExecutionClient.execute_cortex_function_async(
            session=self._get_session(), function_name="SUMMARIZE", args=[text], operation_name="summarize text"
        )

        if not execution_result["success"]:
            return execution_result["error"]

        result = execution_result["result"]
        if result:
            summary = result[0]["RESULT"]
            return json.dumps(
                {
                    "original_text": (text[:100] + "..." if len(text) > 100 else text),
                    "summary": summary,
                    "original_length": len(text),
                    "summary_length": len(summary) if summary else 0,
                }
            )
        else:
            return json.dumps({"error": "No summary generated"})


class CortexTranslatorTool(BaseTool, SnowflakeConnectionMixin):
    """Translate text using Snowflake Cortex TRANSLATE function."""

    name: str = "cortex_translate"
    description: str = "Translate text using Snowflake Cortex TRANSLATE function"
    args_schema: Union[Type[BaseModel], Dict[str, Any], None] = CortexTranslatorInput

    def __init__(self, **kwargs):
        """Initialize the translator tool with proper session attribute."""
        # Extract session from kwargs if provided
        session = kwargs.pop("session", None)

        # Call the parent initializer with remaining kwargs
        super().__init__(**kwargs)

        # Initialize session attribute (from SnowflakeConnectionMixin) using object.__setattr__ to bypass Pydantic
        if session is not None:
            object.__setattr__(self, "_session", session)
        elif not hasattr(self, "_session"):
            object.__setattr__(self, "_session", None)

    def _run(
        self,
        text: str,
        target_language: str = "es",  # Default to Spanish to prevent missing argument errors
        source_language: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Translate the given text to target language."""
        # Handle both string and dict inputs
        if isinstance(text, dict):
            text_content = text.get("text", str(text))
        else:
            text_content = str(text)

        # CORTEX.TRANSLATE expects (text, source_language, target_language)
        # If source_language is not provided, we default to 'en' (English)
        # Note: 'auto' is not supported by Snowflake CORTEX.TRANSLATE
        source_lang = source_language or "en"

        execution_result = SqlExecutionClient.execute_cortex_function(
            session=self._get_session(),
            function_name="TRANSLATE",
            args=[text_content, source_lang, target_language],
            operation_name="translate text",
        )

        if not execution_result["success"]:
            return execution_result["error"]

        result = execution_result["result"]
        if result:
            translated_text = result[0]["RESULT"]
            return json.dumps(
                {
                    "original_text": text_content,
                    "translated_text": translated_text,
                    "target_language": target_language,
                }
            )
        else:
            return json.dumps({"error": "No translation generated"})

    async def _arun(
        self,
        text: str,
        target_language: str = "es",  # Default to Spanish to prevent missing argument errors
        source_language: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async translate text using native async SqlExecutionClient."""
        # Handle both string and dict inputs
        if isinstance(text, dict):
            text_content = text.get("text", str(text))
        else:
            text_content = str(text)

        # CORTEX.TRANSLATE expects (text, source_language, target_language)
        # If source_language is not provided, we default to 'en' (English)
        # Note: 'auto' is not supported by Snowflake CORTEX.TRANSLATE
        source_lang = source_language or "en"

        execution_result = await SqlExecutionClient.execute_cortex_function_async(
            session=self._get_session(),
            function_name="TRANSLATE",
            args=[text_content, source_lang, target_language],
            operation_name="translate text",
        )

        if not execution_result["success"]:
            return execution_result["error"]

        result = execution_result["result"]
        if result:
            translated_text = result[0]["RESULT"]
            return json.dumps(
                {
                    "original_text": text_content,
                    "translated_text": translated_text,
                    "target_language": target_language,
                }
            )
        else:
            return json.dumps({"error": "No translation generated"})


class CortexCompleteTool(BaseTool, SnowflakeConnectionMixin):
    """Generate text completions using Snowflake Cortex COMPLETE function."""

    name: str = "cortex_complete"
    description: str = "Generate text completions using Snowflake Cortex COMPLETE function"
    args_schema: Union[Type[BaseModel], Dict[str, Any], None] = CortexCompleteInput

    def __init__(self, **kwargs):
        """Initialize the complete tool with proper session attribute."""
        # Extract session from kwargs if provided
        session = kwargs.pop("session", None)

        # Call the parent initializer with remaining kwargs
        super().__init__(**kwargs)

        # Initialize session attribute (from SnowflakeConnectionMixin) using object.__setattr__ to bypass Pydantic
        if session is not None:
            object.__setattr__(self, "_session", session)
        elif not hasattr(self, "_session"):
            object.__setattr__(self, "_session", None)

    def _run(
        self,
        prompt: str,
        model: str = "llama3.1-70b",
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Generate text completion for the given prompt."""
        execution_result = SqlExecutionClient.execute_cortex_function(
            session=self._get_session(),
            function_name="COMPLETE",
            args=[model, prompt],
            operation_name="generate completion",
        )

        if not execution_result["success"]:
            return execution_result["error"]

        result = execution_result["result"]
        if result:
            completion = result[0]["RESULT"]
            return json.dumps({"prompt": prompt, "completion": completion, "model": model})
        else:
            return json.dumps({"error": "No completion generated"})

    async def _arun(
        self,
        prompt: str,
        model: str = "llama3.1-70b",
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async generate completion using native async SqlExecutionClient."""
        execution_result = await SqlExecutionClient.execute_cortex_function_async(
            session=self._get_session(),
            function_name="COMPLETE",
            args=[model, prompt],
            operation_name="generate completion",
        )

        if not execution_result["success"]:
            return execution_result["error"]

        result = execution_result["result"]
        if result:
            completion = result[0]["RESULT"]
            return json.dumps({"prompt": prompt, "completion": completion, "model": model})
        else:
            return json.dumps({"error": "No completion generated"})
