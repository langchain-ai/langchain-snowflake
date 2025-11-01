"""Tests for tool message grouping functionality."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_snowflake.chat_models import ChatSnowflake


class TestToolMessageGrouping:
    """Test tool message grouping functionality."""

    @pytest.fixture
    def chat_snowflake(self):
        """Create ChatSnowflake instance for testing."""
        return ChatSnowflake(
            model="claude-3-5-sonnet",
            session=None,  # Mock session
            group_tool_messages=True,
        )

    def test_single_tool_message(self, chat_snowflake):
        """Test that single ToolMessage works correctly."""
        messages = [
            HumanMessage(content="Check weather"),
            AIMessage(content="I'll check", tool_calls=[{"id": "call_1", "name": "get_weather", "args": {}}]),
            ToolMessage(content="Weather: 72°F", tool_call_id="call_1"),
        ]

        payload = chat_snowflake._build_rest_api_payload(messages)

        # Should have 3 messages: human, assistant, user (with tool result)
        # No system message because no tools are bound
        assert len(payload["messages"]) == 3
        assert payload["messages"][-1]["role"] == "user"
        assert len(payload["messages"][-1]["content_list"]) == 1

    def test_multiple_consecutive_tool_messages(self, chat_snowflake):
        """Test that multiple consecutive ToolMessage objects are grouped."""
        messages = [
            HumanMessage(content="Check weather and stock"),
            AIMessage(
                content="I'll check both",
                tool_calls=[
                    {"id": "call_1", "name": "get_weather", "args": {}},
                    {"id": "call_2", "name": "get_stock", "args": {}},
                ],
            ),
            ToolMessage(content="Weather: 72°F", tool_call_id="call_1"),
            ToolMessage(content="Stock: $150.25", tool_call_id="call_2"),
        ]

        payload = chat_snowflake._build_rest_api_payload(messages)

        # Should have 3 messages: human, assistant, user (with grouped tool results)
        assert len(payload["messages"]) == 3
        user_message = payload["messages"][-1]
        assert user_message["role"] == "user"
        assert len(user_message["content_list"]) == 2  # Both tool results grouped

    def test_mixed_message_sequence(self, chat_snowflake):
        """Test tool messages interspersed with other messages."""
        messages = [
            HumanMessage(content="Check weather"),
            AIMessage(content="I'll check", tool_calls=[{"id": "call_1", "name": "get_weather", "args": {}}]),
            ToolMessage(content="Weather: 72°F", tool_call_id="call_1"),
            HumanMessage(content="Now check stock"),
            AIMessage(content="I'll check stock", tool_calls=[{"id": "call_2", "name": "get_stock", "args": {}}]),
            ToolMessage(content="Stock: $150.25", tool_call_id="call_2"),
        ]

        payload = chat_snowflake._build_rest_api_payload(messages)

        # Should have 6 messages: human, assistant, user (tool result), human, assistant, user (tool result)
        assert len(payload["messages"]) == 6
        # Each tool result should be in its own user message
        assert payload["messages"][2]["role"] == "user"
        assert payload["messages"][5]["role"] == "user"

    def test_group_tool_messages_disabled(self):
        """Test that grouping can be disabled."""
        chat_snowflake = ChatSnowflake(model="claude-3-5-sonnet", session=None, group_tool_messages=False)

        messages = [
            HumanMessage(content="Check weather"),
            AIMessage(content="I'll check", tool_calls=[{"id": "call_1", "name": "get_weather", "args": {}}]),
            ToolMessage(content="Weather: 72°F", tool_call_id="call_1"),
        ]

        payload = chat_snowflake._build_rest_api_payload(messages)

        # Should maintain old behavior (separate messages)
        assert len(payload["messages"]) == 3

    def test_empty_tool_results(self, chat_snowflake):
        """Test edge case of empty tool results."""
        messages = [
            HumanMessage(content="Check weather"),
            AIMessage(content="I'll check", tool_calls=[{"id": "call_1", "name": "get_weather", "args": {}}]),
            ToolMessage(content="", tool_call_id="call_1"),
        ]

        payload = chat_snowflake._build_rest_api_payload(messages)

        # Should handle empty content gracefully
        assert len(payload["messages"]) == 3
        user_message = payload["messages"][-1]
        assert user_message["role"] == "user"
        assert len(user_message["content_list"]) == 1

    def test_tool_message_with_name_attribute(self, chat_snowflake):
        """Test ToolMessage with name attribute."""
        messages = [
            HumanMessage(content="Check weather"),
            AIMessage(content="I'll check", tool_calls=[{"id": "call_1", "name": "get_weather", "args": {}}]),
            ToolMessage(content="Weather: 72°F", tool_call_id="call_1", name="get_weather"),
        ]

        payload = chat_snowflake._build_rest_api_payload(messages)

        user_message = payload["messages"][-1]
        tool_result = user_message["content_list"][0]["tool_results"]
        assert tool_result["name"] == "get_weather"

    def test_tool_message_without_name_attribute(self, chat_snowflake):
        """Test ToolMessage without name attribute."""
        messages = [
            HumanMessage(content="Check weather"),
            AIMessage(content="I'll check", tool_calls=[{"id": "call_1", "name": "get_weather", "args": {}}]),
            ToolMessage(content="Weather: 72°F", tool_call_id="call_1"),
        ]

        payload = chat_snowflake._build_rest_api_payload(messages)

        user_message = payload["messages"][-1]
        tool_result = user_message["content_list"][0]["tool_results"]
        assert tool_result["name"] == "unknown"

    def test_disable_parallel_tool_use_parameter(self, chat_snowflake):
        """Test that disable_parallel_tool_use parameter is included in payload."""
        messages = [
            HumanMessage(content="Check weather"),
            AIMessage(content="I'll check", tool_calls=[{"id": "call_1", "name": "get_weather", "args": {}}]),
            ToolMessage(content="Weather: 72°F", tool_call_id="call_1"),
        ]

        payload = chat_snowflake._build_rest_api_payload(messages)

        # Should include disable_parallel_tool_use parameter
        assert "disable_parallel_tool_use" in payload
        assert not payload["disable_parallel_tool_use"]  # Default value

    def test_disable_parallel_tool_use_enabled(self):
        """Test with disable_parallel_tool_use enabled."""
        chat_snowflake = ChatSnowflake(model="claude-3-5-sonnet", session=None, disable_parallel_tool_use=True)

        messages = [
            HumanMessage(content="Check weather"),
            AIMessage(content="I'll check", tool_calls=[{"id": "call_1", "name": "get_weather", "args": {}}]),
            ToolMessage(content="Weather: 72°F", tool_call_id="call_1"),
        ]

        payload = chat_snowflake._build_rest_api_payload(messages)

        # Should include disable_parallel_tool_use parameter set to True
        assert payload["disable_parallel_tool_use"]

    def test_three_consecutive_tool_messages(self, chat_snowflake):
        """Test grouping of three consecutive tool messages."""
        messages = [
            HumanMessage(content="Check weather, stock, and news"),
            AIMessage(
                content="I'll check all three",
                tool_calls=[
                    {"id": "call_1", "name": "get_weather", "args": {}},
                    {"id": "call_2", "name": "get_stock", "args": {}},
                    {"id": "call_3", "name": "get_news", "args": {}},
                ],
            ),
            ToolMessage(content="Weather: 72°F", tool_call_id="call_1"),
            ToolMessage(content="Stock: $150.25", tool_call_id="call_2"),
            ToolMessage(content="News: Market up 2%", tool_call_id="call_3"),
        ]

        payload = chat_snowflake._build_rest_api_payload(messages)

        # Should have 3 messages: human, assistant, user (with all 3 tool results grouped)
        assert len(payload["messages"]) == 3
        user_message = payload["messages"][-1]
        assert user_message["role"] == "user"
        assert len(user_message["content_list"]) == 3  # All three tool results grouped

    def test_ai_message_without_tool_calls(self, chat_snowflake):
        """Test AI message without tool calls."""
        messages = [HumanMessage(content="Hello"), AIMessage(content="Hi there!")]

        payload = chat_snowflake._build_rest_api_payload(messages)

        # Should have 2 messages: human, assistant
        assert len(payload["messages"]) == 2
        assert payload["messages"][-1]["role"] == "assistant"
        assert payload["messages"][-1]["content"] == "Hi there!"

    def test_ai_message_with_tool_calls_and_text(self, chat_snowflake):
        """Test AI message with both text content and tool calls."""
        messages = [
            HumanMessage(content="Check weather"),
            AIMessage(
                content="I'll check the weather for you",
                tool_calls=[{"id": "call_1", "name": "get_weather", "args": {"location": "San Francisco"}}],
            ),
            ToolMessage(content="Weather: 72°F", tool_call_id="call_1"),
        ]

        payload = chat_snowflake._build_rest_api_payload(messages)

        # Should have 3 messages: human, assistant, user
        assert len(payload["messages"]) == 3

        # Assistant message should have both text and tool_use in content_list array
        assistant_message = payload["messages"][1]
        assert assistant_message["role"] == "assistant"
        assert "content_list" in assistant_message
        assert len(assistant_message["content_list"]) == 2  # Text + tool_use
        assert assistant_message["content_list"][0]["type"] == "text"
        assert assistant_message["content_list"][0]["text"] == "I'll check the weather for you"
        assert assistant_message["content_list"][1]["type"] == "tool_use"
        assert assistant_message["content_list"][1]["tool_use"]["name"] == "get_weather"
