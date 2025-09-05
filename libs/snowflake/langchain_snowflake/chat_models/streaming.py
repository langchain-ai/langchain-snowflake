"""Streaming functionality for Snowflake chat models."""

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import aiohttp
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk

logger = logging.getLogger(__name__)


class SnowflakeStreaming:
    """Mixin class for Snowflake streaming functionality."""

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completions from Snowflake Cortex.

        Args:
            messages: List of messages to send to the model
            stop: List of stop sequences (not supported by Cortex)
            run_manager: Callback manager for the run
            **kwargs: Additional keyword arguments

        Yields:
            ChatGenerationChunk: Streaming chunks of the response

        Note: Uses Cortex COMPLETE's native streaming via REST API when tools
        are bound, otherwise falls back to simulated streaming for SQL function.
        """
        try:
            if self._should_use_rest_api():
                # Use native streaming via REST API
                for chunk in self._stream_via_rest_api(messages, run_manager, **kwargs):
                    yield chunk
            else:
                # Simulate streaming by chunking the complete response
                result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

                if result.generations:
                    content = result.generations[0].message.content

                    # Split content into chunks for streaming effect
                    chunk_size = max(1, len(content) // 20)  # Aim for ~20 chunks

                    for i in range(0, len(content), chunk_size):
                        chunk_content = content[i : i + chunk_size]

                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(
                                content=chunk_content,
                                usage_metadata=(result.generations[0].message.usage_metadata if i == 0 else None),
                                response_metadata=(result.generations[0].message.response_metadata if i == 0 else {}),
                            )
                        )

                        yield chunk

                        if run_manager:
                            run_manager.on_llm_new_token(chunk_content)

        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            # Fallback to a single error chunk
            yield ChatGenerationChunk(message=AIMessageChunk(content=f"Error: {e}"))

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream response by delegating to sync _stream method.

        This eliminates code duplication by using the sync implementation
        with asyncio.to_thread() for non-blocking execution.
        """
        # Determine streaming method based on tool requirements
        try:
            if self._should_use_rest_api():
                # Use native async REST API streaming with aiohttp
                async for chunk in self._astream_via_rest_api(messages, run_manager, **kwargs):
                    yield chunk
            else:
                # For SQL-based streaming, we currently delegate to sync method since
                # Snowflake doesn't support native SQL streaming, only batch results
                def sync_stream():
                    return list(self._stream(messages, stop, run_manager, **kwargs))

                chunks = await asyncio.to_thread(sync_stream)
                for chunk in chunks:
                    yield chunk

        except Exception as e:
            logger.error(f"Error in async streaming: {e}")
            # Yield an error chunk
            from langchain_core.outputs import ChatGenerationChunk

            error_chunk = ChatGenerationChunk(message=AIMessageChunk(content=f"Error: {str(e)}"))
            yield error_chunk

    def _stream_via_rest_api(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completions via REST API with native streaming support."""
        try:
            # Build REST API payload with streaming enabled
            payload = self._build_rest_api_payload(messages)
            payload["stream"] = True  # Enable native streaming

            # Add generation parameters using shared utility
            from .utils import SnowflakePayloadBuilder

            payload = SnowflakePayloadBuilder.add_generation_params(
                payload, self.temperature, self.max_tokens, self.top_p
            )

            # Make streaming request using shared utilities
            from .._connection import SnowflakeAuthUtils

            session = self._get_session()
            response = SnowflakeAuthUtils.make_rest_api_request(
                session=session,
                payload=payload,
                account=getattr(self, "account", None),
                user=getattr(self, "user", None),
                token=getattr(self, "token", None),
                private_key_path=getattr(self, "private_key_path", None),
                private_key_passphrase=getattr(self, "private_key_passphrase", None),
                request_timeout=getattr(self, "request_timeout", 30),
                verify_ssl=getattr(self, "verify_ssl", True),
                stream=True,  # Enable streaming response
            )
            response.raise_for_status()

            # Parse Server-Sent Events stream
            content_buffer: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            tool_input_buffers: Dict[str, str] = {}
            usage_data: Dict[str, Any] = {}

            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        try:
                            data = json.loads(line_str[6:])

                            # Parse the streaming chunk
                            chunk_parts: List[str] = []
                            self._parse_streaming_chunk(
                                data,
                                chunk_parts,
                                tool_calls,
                                usage_data,
                                tool_input_buffers,
                            )

                            # Yield chunk if there's content
                            if chunk_parts:
                                chunk_content = "".join(chunk_parts)
                                content_buffer.append(chunk_content)

                                # Create usage metadata for first chunk only using shared factory
                                usage_metadata = None
                                if len(content_buffer) == 1 and usage_data:
                                    from .utils import SnowflakeMetadataFactory

                                    usage_metadata = SnowflakeMetadataFactory.create_usage_metadata(usage_data)

                                chunk = ChatGenerationChunk(
                                    message=AIMessageChunk(
                                        content=chunk_content,
                                        tool_calls=tool_calls if tool_calls else None,
                                        usage_metadata=usage_metadata,
                                        response_metadata=(
                                            {"model_name": self.model} if len(content_buffer) == 1 else {}
                                        ),
                                    )
                                )

                                yield chunk

                                if run_manager:
                                    run_manager.on_llm_new_token(chunk_content)

                        except json.JSONDecodeError:
                            continue  # Skip malformed JSON

        except Exception as e:
            logger.error(f"Error during REST API streaming: {e}")
            # Fallback to regular generation
            result = self._generate_via_rest_api(messages)
            if result.generations:
                content = result.generations[0].message.content
                yield ChatGenerationChunk(message=AIMessageChunk(content=content))

    async def _astream_via_rest_api(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream chat completions via REST API using aiohttp for true async."""
        try:
            # Build REST API payload with streaming enabled
            payload = self._build_rest_api_payload(messages)
            payload["stream"] = True  # Enable native streaming

            # Add generation parameters using shared utility
            from .utils import SnowflakePayloadBuilder

            payload = SnowflakePayloadBuilder.add_generation_params(
                payload, self.temperature, self.max_tokens, self.top_p
            )

            # Get session and build URL/headers manually for streaming
            session = self._get_session()
            from .._connection import SnowflakeAuthUtils

            url = SnowflakeAuthUtils.build_rest_api_url(session) + "/api/v2/cortex/inference:complete"
            headers = SnowflakeAuthUtils.get_rest_api_headers(
                session=session,
                account=getattr(self, "account", None),
                user=getattr(self, "user", None),
            )

            # Get timeout and SSL config
            request_timeout = getattr(self, "request_timeout", 30)
            verify_ssl = getattr(self, "verify_ssl", True)

            # Use aiohttp for true async streaming
            async with aiohttp.ClientSession() as client:
                async with client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=request_timeout),
                    ssl=verify_ssl,
                ) as response:
                    response.raise_for_status()

                    # Parse Server-Sent Events stream asynchronously
                    content_buffer = []

                    async for line_bytes in response.content:
                        line = line_bytes.decode("utf-8").strip()

                        if line.startswith("data: "):
                            data_part = line[6:]  # Remove 'data: ' prefix

                            if data_part == "[DONE]":
                                break

                            try:
                                chunk_data = json.loads(data_part)

                                # Process chunk using shared utility

                                if "choices" in chunk_data and chunk_data["choices"]:
                                    choice = chunk_data["choices"][0]

                                    # Handle content chunks
                                    delta = choice.get("delta", {})
                                    if "content" in delta:
                                        content = delta["content"]
                                        content_buffer.append(content)

                                        # Create chunk using shared factory
                                        chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))
                                        yield chunk

                                # Handle usage data if present
                                if "usage" in chunk_data:
                                    chunk_data["usage"]

                            except json.JSONDecodeError:
                                continue  # Skip malformed JSON

        except Exception as e:
            logger.error(f"Error during async REST API streaming: {e}")
            # Fallback to async regular generation
            result = await self._generate_via_rest_api_async(messages)
            if result.generations:
                content = result.generations[0].message.content
                yield ChatGenerationChunk(message=AIMessageChunk(content=content))
