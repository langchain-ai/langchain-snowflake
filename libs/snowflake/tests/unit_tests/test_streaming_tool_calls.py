"""Unit tests for fix #40: streaming correctly yields tool_use deltas as ToolCallChunks."""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessageChunk

from langchain_snowflake.chat_models import ChatSnowflake


class MockSession:
    """Minimal mock session."""

    def sql(self, query):
        mock_result = MagicMock()
        mock_result.collect.return_value = []
        return mock_result


@pytest.fixture
def llm():
    return ChatSnowflake(model="claude-3-5-sonnet", session=MockSession())


def _make_chunk(**delta_fields):
    """Build a JSON-encoded Cortex streaming chunk."""
    return json.dumps({"choices": [{"delta": delta_fields}]})


class TestSyncStreamingToolCalls:
    """Tests for _stream_via_rest_api (fix #40)."""

    def _run_stream(self, llm, raw_chunks):
        """Patch the REST client and collect yielded ChatGenerationChunks."""
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="Call a tool")]

        with (
            patch(
                "langchain_snowflake.chat_models.streaming.RestApiClient.make_sync_streaming_request",
                return_value=iter(raw_chunks),
            ),
            patch(
                "langchain_snowflake.chat_models.streaming.RestApiRequestBuilder.cortex_complete_request",
                return_value=MagicMock(),
            ),
            patch.object(llm, "_get_session", return_value=MockSession()),
            patch.object(llm, "_build_rest_api_payload", return_value={"model": "claude-3-5-sonnet", "messages": []}),
        ):
            return list(llm._stream_via_rest_api(messages))

    def test_text_delta_yields_ai_message_chunk(self, llm):
        raw = [_make_chunk(type="text", content="Hello world")]
        chunks = self._run_stream(llm, raw)
        assert len(chunks) == 1
        msg = chunks[0].message
        assert isinstance(msg, AIMessageChunk)
        assert msg.content == "Hello world"
        assert msg.tool_call_chunks == []

    def test_tool_use_delta_yields_tool_call_chunk(self, llm):
        raw = [
            _make_chunk(type="tool_use", tool_use_id="toolu_01", name="get_weather", input=""),
            _make_chunk(type="tool_use", input='{"city": "NYC"}'),
        ]
        chunks = self._run_stream(llm, raw)
        assert len(chunks) == 2

        first_msg = chunks[0].message
        assert isinstance(first_msg, AIMessageChunk)
        assert len(first_msg.tool_call_chunks) == 1
        tc = first_msg.tool_call_chunks[0]
        assert tc["id"] == "toolu_01"
        assert tc["name"] == "get_weather"
        assert tc["index"] == 0

        second_msg = chunks[1].message
        assert len(second_msg.tool_call_chunks) == 1
        tc2 = second_msg.tool_call_chunks[0]
        assert tc2["args"] == '{"city": "NYC"}'
        assert tc2["index"] == 0

    def test_mixed_text_and_tool_use_deltas(self, llm):
        raw = [
            _make_chunk(type="text", content="I'll check that."),
            _make_chunk(type="tool_use", tool_use_id="toolu_02", name="get_stock", input=""),
        ]
        chunks = self._run_stream(llm, raw)
        assert len(chunks) == 2
        assert chunks[0].message.content == "I'll check that."
        assert chunks[1].message.tool_call_chunks[0]["name"] == "get_stock"

    def test_empty_text_delta_is_skipped(self, llm):
        raw = [_make_chunk(type="text", content="")]
        chunks = self._run_stream(llm, raw)
        assert len(chunks) == 0

    def test_invalid_json_chunk_is_skipped(self, llm):
        raw = ["not-json", _make_chunk(type="text", content="ok")]
        chunks = self._run_stream(llm, raw)
        assert len(chunks) == 1
        assert chunks[0].message.content == "ok"

    def test_none_chunks_are_skipped(self, llm):
        raw = [None, _make_chunk(type="text", content="real")]
        chunks = self._run_stream(llm, raw)
        assert len(chunks) == 1

    def test_tool_call_index_increments_per_new_id(self, llm):
        raw = [
            _make_chunk(type="tool_use", tool_use_id="id_a", name="fn_a", input=""),
            _make_chunk(type="tool_use", input="partial_a"),
            _make_chunk(type="tool_use", tool_use_id="id_b", name="fn_b", input=""),
        ]
        chunks = self._run_stream(llm, raw)
        assert chunks[0].message.tool_call_chunks[0]["index"] == 0
        assert chunks[1].message.tool_call_chunks[0]["index"] == 0
        assert chunks[2].message.tool_call_chunks[0]["index"] == 1


class TestAsyncStreamingToolCalls:
    """Tests for _astream_via_rest_api (fix #40)."""

    @pytest.mark.asyncio
    async def test_async_text_delta_yields_chunk(self, llm):
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="Hello")]
        raw = [_make_chunk(type="text", content="Async response")]

        async def async_gen(_config, _label):
            for item in raw:
                yield item

        with (
            patch(
                "langchain_snowflake.chat_models.streaming.RestApiClient.make_async_streaming_request",
                new=async_gen,
            ),
            patch(
                "langchain_snowflake.chat_models.streaming.RestApiRequestBuilder.cortex_complete_request",
                return_value=MagicMock(),
            ),
            patch.object(llm, "_get_session", return_value=MockSession()),
            patch.object(llm, "_build_rest_api_payload", return_value={"model": "claude-3-5-sonnet", "messages": []}),
        ):
            chunks = [c async for c in llm._astream_via_rest_api(messages)]

        assert len(chunks) == 1
        assert chunks[0].message.content == "Async response"

    @pytest.mark.asyncio
    async def test_async_tool_use_delta_yields_tool_call_chunk(self, llm):
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content="Use tool")]
        raw = [_make_chunk(type="tool_use", tool_use_id="toolu_99", name="my_fn", input="")]

        async def async_gen(_config, _label):
            for item in raw:
                yield item

        with (
            patch(
                "langchain_snowflake.chat_models.streaming.RestApiClient.make_async_streaming_request",
                new=async_gen,
            ),
            patch(
                "langchain_snowflake.chat_models.streaming.RestApiRequestBuilder.cortex_complete_request",
                return_value=MagicMock(),
            ),
            patch.object(llm, "_get_session", return_value=MockSession()),
            patch.object(llm, "_build_rest_api_payload", return_value={"model": "claude-3-5-sonnet", "messages": []}),
        ):
            chunks = [c async for c in llm._astream_via_rest_api(messages)]

        assert len(chunks) == 1
        tc = chunks[0].message.tool_call_chunks[0]
        assert tc["id"] == "toolu_99"
        assert tc["name"] == "my_fn"
