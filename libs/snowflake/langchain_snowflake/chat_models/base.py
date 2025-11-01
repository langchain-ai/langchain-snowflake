"""Core ChatSnowflake class implementation."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, SecretStr
from snowflake.snowpark import Session

from .._connection.rest_client import RestApiClient, RestApiRequestBuilder
from .._connection.sql_client import SqlExecutionClient
from .._error_handling import SnowflakeErrorHandler
from .auth import SnowflakeAuth
from .streaming import SnowflakeStreaming
from .structured_output import SnowflakeStructuredOutput
from .tools import SnowflakeTools
from .utils import SnowflakeMetadataFactory, SnowflakeUtils

logger = logging.getLogger(__name__)


class ChatSnowflake(
    SnowflakeAuth,
    SnowflakeStreaming,
    SnowflakeTools,
    SnowflakeStructuredOutput,
    SnowflakeUtils,
    BaseChatModel,
):
    """Snowflake chat model integration using Cortex LLM functions.

    This class provides access to Snowflake's Cortex Complete function with
    models like llama3.1-70b, mistral-large2, claude-3-5-sonnet, and more.

    Setup:
        Install ``langchain-snowflake`` and configure Snowflake connection.

        .. code-block:: bash

            pip install -U langchain-snowflake

    Key init args — completion params:
        model: str
            Name of Snowflake Cortex model to use (e.g., 'llama3.1-70b', 'mistral-large2')
        temperature: float
            Sampling temperature (0.0 to 1.0)
        max_tokens: Optional[int]
            Max number of tokens to generate (default: 4096)

    Key init args — client params:
        session: Optional[Session]
            Active Snowflake session. If not provided, will create from connection params.
        account: Optional[str]
            Snowflake account identifier
        user: Optional[str]
            Snowflake username
        password: Optional[SecretStr]
            Snowflake password
        warehouse: Optional[str]
            Snowflake warehouse to use
        database: Optional[str]
            Snowflake database to use
        schema: Optional[str]
            Snowflake schema to use
        request_timeout: int
            Request timeout in seconds for API calls (default: 300)
        verify_ssl: bool
            Whether to verify SSL certificates (default: True)

    Instantiate:
        .. code-block:: python

            from .. import ChatSnowflake

            # Using existing session
            llm = ChatSnowflake(
                model="llama3.1-70b",
                session=session,
                temperature=0.1,
                max_tokens=1000
            )

            # Using connection parameters with network configuration
            llm = ChatSnowflake(
                model="mistral-large2",
                account="your-account",
                user="your-user",
                password="your-password",
                warehouse="your-warehouse",
                temperature=0.0,
                request_timeout=600,  # 10 minutes for long-running operations
                verify_ssl=True       # Always verify SSL in production
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful assistant."),
                ("human", "What is the capital of France?"),
            ]
            response = llm.invoke(messages)
            print(response.content)

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk.content, end="", flush=True)

    Async:
        .. code-block:: python

            response = await llm.ainvoke(messages)
            async for chunk in llm.astream(messages):
                print(chunk.content, end="", flush=True)

    Tool calling:
        .. code-block:: python


            @tool
            def get_weather(city: str) -> str:
                '''Get weather for a city.'''
                return f"The weather in {city} is 72°F and sunny."

            llm_with_tools = llm.bind_tools([get_weather])
            messages = [("human", "What's the weather in Paris?")]
            response = llm_with_tools.invoke(messages)

    Structured output:
        .. code-block:: python

            from typing import Literal
            from pydantic import BaseModel

            class Sentiment(BaseModel):
                sentiment: Literal["positive", "negative", "neutral"]
                confidence: float

            structured_llm = llm.with_structured_output(Sentiment)
            result = structured_llm.invoke("I love this product!")
            print(result.sentiment, result.confidence)

    Response metadata:
        .. code-block:: python

            response = llm.invoke(messages)
            print(response.response_metadata)
            # {'model': 'llama3.1-70b', 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
    """

    # Model configuration
    model: str = Field(default="llama3.1-70b")
    """Name of Snowflake Cortex model to use."""

    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    """Sampling temperature (0.0 to 1.0)."""

    max_tokens: int = Field(default=4096, ge=1)
    """Maximum number of tokens to generate."""

    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    """Nucleus sampling parameter."""

    # Session and connection
    session: Optional[Session] = Field(default=None)
    """Active Snowflake session."""

    warehouse: Optional[str] = Field(default=None)
    """Snowflake warehouse to use."""

    database: Optional[str] = Field(default=None)
    """Snowflake database to use."""

    schema: Optional[str] = Field(default=None)
    """Snowflake schema to use."""

    # Authentication
    account: Optional[str] = Field(default=None)
    """Snowflake account identifier."""

    user: Optional[str] = Field(default=None)
    """Snowflake username."""

    password: Optional[SecretStr] = Field(default=None)
    """Snowflake password."""

    token: Optional[str] = Field(default=None)
    """Snowflake Personal Access Token (PAT) - preferred for REST API authentication."""

    private_key_path: Optional[str] = Field(default=None)
    """Path to RSA private key file for key pair authentication - alternative for REST API."""

    private_key_passphrase: Optional[str] = Field(default=None)
    """Passphrase for RSA private key."""

    # Retry configuration
    max_retries: int = Field(default=3)
    """Maximum number of retries for API calls."""

    # Network configuration
    request_timeout: int = Field(default=300)
    """Request timeout in seconds for API calls. Increased from 60s to 300s for complex operations."""

    verify_ssl: bool = Field(default=True)
    """Whether to verify SSL certificates for HTTPS requests. Set to False only for testing environments."""

    # LangChain compatibility fields
    ls_structured_output_format: Optional[str] = Field(default=None)
    """Structured output format for LangChain compatibility."""

    # Tool execution control
    disable_parallel_tool_use: bool = Field(default=False)
    """Whether to disable parallel tool use (force sequential execution)."""

    group_tool_messages: bool = Field(default=True)
    """Whether to group consecutive ToolMessage objects into single user message."""

    @property
    def _ls_structured_output_format(self) -> Optional[str]:
        """Return the structured output format for LangChain compatibility."""
        return self.ls_structured_output_format

    @_ls_structured_output_format.setter
    def _ls_structured_output_format(self, value: Optional[str]) -> None:
        """Set the structured output format for LangChain compatibility."""
        self.ls_structured_output_format = value

    def __init__(
        self,
        model: str = "llama3.1-70b",
        session: Any = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        warehouse: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        account: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[SecretStr] = None,
        token: Optional[str] = None,
        private_key_path: Optional[str] = None,
        private_key_passphrase: Optional[str] = None,
        request_timeout: int = 300,
        verify_ssl: bool = True,
        disable_parallel_tool_use: bool = False,
        group_tool_messages: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.session = session
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self.account = account
        self.user = user
        self.password = password
        self.token = token
        self.private_key_path = private_key_path
        self.private_key_passphrase = private_key_passphrase
        self.request_timeout = request_timeout
        self.verify_ssl = verify_ssl
        self.disable_parallel_tool_use = disable_parallel_tool_use
        self.group_tool_messages = group_tool_messages

        # Tool-related attributes for bind_tools compatibility
        self._bound_tools: List[Dict[str, Any]] = []
        self._tool_choice: Optional[str] = None
        self._use_rest_api: bool = False  # New attribute to control REST API usage

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "snowflake-cortex"

    def _build_cortex_options_for_sql(self) -> Optional[Dict[str, Any]]:
        """Build options dictionary for SQL-based Cortex COMPLETE function.

        This method builds options specifically for SQL function calls,
        excluding tools and other REST API-specific features.

        Returns:
            Dictionary of options for Cortex SQL function, or None if no options

        Raises:
            ValueError: If parameter validation fails
        """
        try:
            # Use centralized options building from SnowflakeUtils
            # Only include basic generation parameters (no tools, no streaming for SQL)
            options = self._build_options_dict(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )

            return options

        except Exception as e:
            # Use centralized error handling
            from .._error_handling import SnowflakeErrorHandler

            SnowflakeErrorHandler.log_and_raise(e, "build Cortex options for SQL")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response using either REST API (for tools) or SQL function (for basic chat)."""
        try:
            # Route based on tool usage
            if self._should_use_rest_api():
                return self._generate_via_rest_api(messages, stop, run_manager, **kwargs)
            else:
                return self._generate_via_sql(messages, stop, run_manager, **kwargs)
        except Exception as e:
            input_tokens = self._estimate_tokens(messages)
            return SnowflakeErrorHandler.create_chat_error_result(
                error=e,
                operation="generate response",
                model=self.model,
                input_tokens=input_tokens,
                logger_instance=logger,
            )

    def _generate_via_sql(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response using SQL function SNOWFLAKE.CORTEX.COMPLETE (existing implementation)."""

        session = self._get_session()

        # Format messages for Cortex
        formatted_prompt = self._format_messages_for_cortex(messages)

        # Build options (without tools since SQL doesn't support them)
        options = self._build_cortex_options_for_sql()

        try:
            if options:
                # formatted_prompt is always a List[Dict[str, str]] from _format_messages_for_cortex
                prompt_data: List[Dict[str, str]] = formatted_prompt

                # Use parameterized queries to avoid JSON escaping issues entirely
                sql = """SELECT SNOWFLAKE.CORTEX.COMPLETE(?, PARSE_JSON(?), PARSE_JSON(?)) as response"""
                execution_result = SqlExecutionClient.execute_sync(
                    session=session,
                    sql=sql,
                    params=[self.model, json.dumps(prompt_data), json.dumps(options)],
                    operation_name="call Cortex COMPLETE",
                )

                if not execution_result["success"]:
                    SnowflakeErrorHandler.log_and_raise(
                        ValueError(execution_result["error"]), "call Cortex COMPLETE via SQL"
                    )
                result = execution_result["result"]
            else:
                # Use simple string format with proper escaping
                if isinstance(formatted_prompt, str):
                    # Use parameterized query for simple format too
                    sql = "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) as response"
                    execution_result = SqlExecutionClient.execute_sync(
                        session=session,
                        sql=sql,
                        params=[self.model, formatted_prompt],
                        operation_name="call Cortex COMPLETE",
                    )

                    if not execution_result["success"]:
                        SnowflakeErrorHandler.log_and_raise(
                            ValueError(execution_result["error"]), "call Cortex COMPLETE via SQL"
                        )
                    result = execution_result["result"]
                else:
                    # Convert to string for simple format
                    prompt_text = " ".join(
                        [msg.get("content", "") for msg in formatted_prompt if isinstance(msg, dict)]
                    )
                    sql = "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) as response"
                    execution_result = SqlExecutionClient.execute_sync(
                        session=session,
                        sql=sql,
                        params=[self.model, prompt_text],
                        operation_name="call Cortex COMPLETE",
                    )

                    if not execution_result["success"]:
                        SnowflakeErrorHandler.log_and_raise(
                            ValueError(execution_result["error"]), "call Cortex COMPLETE via SQL"
                        )
                    result = execution_result["result"]

            if not result:
                try:
                    raise ValueError("No response from Cortex Complete")
                except Exception as e:
                    SnowflakeErrorHandler.log_and_raise(e, "validate Cortex Complete response")

            response_text = result[0].as_dict()["RESPONSE"]

            # Parse response based on format
            if options and response_text.strip().startswith("{"):
                try:
                    response_data = json.loads(response_text)
                    content = response_data.get("choices", [{}])[0].get("messages", "")
                    usage_data = response_data.get("usage", {})

                    message = AIMessage(
                        content=content,
                        usage_metadata=SnowflakeMetadataFactory.create_usage_metadata(usage_data),
                        response_metadata=SnowflakeMetadataFactory.create_response_metadata(self.model),
                    )
                except json.JSONDecodeError:
                    # Fallback to treating as plain text using shared factories
                    input_tokens = self._estimate_tokens(messages)
                    output_tokens = self._estimate_tokens([{"content": response_text}])

                    message = AIMessage(
                        content=response_text,
                        usage_metadata=SnowflakeMetadataFactory.create_usage_metadata(
                            input_tokens=input_tokens, output_tokens=output_tokens
                        ),
                        response_metadata=SnowflakeMetadataFactory.create_response_metadata(self.model),
                    )
            else:
                # Simple string response using shared factories
                input_tokens = self._estimate_tokens(messages)
                output_tokens = self._estimate_tokens([{"content": response_text}])

                message = AIMessage(
                    content=response_text,
                    usage_metadata=SnowflakeMetadataFactory.create_usage_metadata(
                        input_tokens=input_tokens, output_tokens=output_tokens
                    ),
                    response_metadata=SnowflakeMetadataFactory.create_response_metadata(self.model),
                )

            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

        except Exception as e:
            # Use centralized error handling for SQL execution errors
            SnowflakeErrorHandler.log_and_raise(e, "call Cortex COMPLETE via SQL")
            raise

    async def _generate_via_sql_async(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate response by delegating to sync method with asyncio.to_thread."""
        return await asyncio.to_thread(self._generate_via_sql, messages, stop, run_manager, **kwargs)

    def _make_rest_api_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make REST API request to Snowflake Cortex using RestApiClient."""
        session = self._get_session()

        request_config = RestApiRequestBuilder.cortex_complete_request(
            session=session,
            method="POST",
            payload=payload,
            request_timeout=self.request_timeout,
            verify_ssl=self.verify_ssl,
        )

        return RestApiClient.make_sync_request(request_config, "Cortex Complete")

    async def _make_rest_api_request_async(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make async REST API request using RestApiClient."""

        session = self._get_session()

        try:
            request_config = RestApiRequestBuilder.cortex_complete_request(
                session=session,
                method="POST",
                payload=payload,
                request_timeout=self.request_timeout,
                verify_ssl=self.verify_ssl,
            )

            response_data = await RestApiClient.make_async_request(request_config, "async Cortex Complete")

            SnowflakeErrorHandler.log_debug("async REST API", "Async REST API request completed successfully", logger)
            return response_data

        except Exception as e:
            SnowflakeErrorHandler.log_error("async REST API request", e, logger)
            raise

    def _generate_via_rest_api(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response using REST API /api/v2/cortex/inference:complete (for tool calling)."""

        try:
            # Get session info for authentication
            self._get_session()

            # Build the REST API payload
            payload = self._build_rest_api_payload(messages)

            # Use non-streaming for invoke() - cleaner JSON responses
            payload["stream"] = False

            # Make the REST API call using RestApiClient
            response_data = self._make_rest_api_request(payload)

            # Parse the response and handle tool calls
            return self._parse_rest_api_response(response_data, messages)

        except Exception as e:
            # Use centralized error handling with better context
            input_tokens = self._estimate_tokens(messages)
            return SnowflakeErrorHandler.create_chat_error_result(
                error=e,
                operation="generate response via REST API",
                model=self.model,
                input_tokens=input_tokens,
            )

    async def _generate_via_rest_api_async(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate response using REST API with native aiohttp."""

        try:
            # Build the REST API payload
            payload = self._build_rest_api_payload(messages)

            # Use non-streaming for ainvoke() - cleaner JSON responses
            payload["stream"] = False

            # Make the async REST API call using RestApiClient
            response_data = await self._make_rest_api_request_async(payload)

            # Parse the response and handle tool calls - USE ASYNC VERSION for async tool execution
            return await self._parse_rest_api_response_async(response_data, messages)

        except Exception as e:
            # Use centralized error handling with better context
            input_tokens = self._estimate_tokens(messages)
            return SnowflakeErrorHandler.create_chat_error_result(
                error=e,
                operation="generate response via async REST API",
                model=self.model,
                input_tokens=input_tokens,
            )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate chat completion using native async patterns."""
        try:
            # Determine whether to use SQL or REST API based on tool requirements
            if self._should_use_rest_api():
                # Use native async REST API with aiohttp
                return await self._generate_via_rest_api_async(messages, stop, run_manager, **kwargs)
            else:
                # Use native Snowflake async for SQL execution
                return await self._generate_via_sql_async(messages, stop, run_manager, **kwargs)

        except Exception as e:
            # Use centralized error handling with better context
            input_tokens = self._estimate_tokens(messages)
            return SnowflakeErrorHandler.create_chat_error_result(
                error=e,
                operation="generate async response",
                model=self.model,
                input_tokens=input_tokens,
            )

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens in the given text."""
        return self._estimate_tokens([{"role": "user", "content": text}])

    def get_token_ids(self, text: str) -> List[int]:
        """Get the token IDs for the given text."""
        # This is a placeholder - Snowflake doesn't provide token IDs
        # Return estimated token count as single list
        return list(range(self.get_num_tokens(text)))
