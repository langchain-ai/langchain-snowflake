from langchain_snowflake import ChatSnowflake


def test_with_structured_output_preserves_runtime_configuration():
    llm = ChatSnowflake(
        model="test",
        max_retries=0,
        request_timeout=17,
        verify_ssl=False,
        disable_parallel_tool_use=True,
        group_tool_messages=False,
    )

    structured = llm.with_structured_output(
        {
            "type": "object",
            "properties": {"x": {"type": "string"}},
        }
    )

    assert structured.max_retries == llm.max_retries
    assert structured.request_timeout == llm.request_timeout
    assert structured.verify_ssl == llm.verify_ssl
    assert (
        structured.disable_parallel_tool_use
        == llm.disable_parallel_tool_use
    )
    assert structured.group_tool_messages == llm.group_tool_messages
