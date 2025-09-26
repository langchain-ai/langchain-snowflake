from langchain_snowflake import __all__

EXPECTED_ALL = [
    # Chat Models
    "ChatSnowflake",
    # Retrievers
    "SnowflakeCortexSearchRetriever",
    # Tools
    "SnowflakeQueryTool",
    "CortexSentimentTool",
    "CortexSummarizerTool",
    "CortexTranslatorTool",
    "CortexCompleteTool",
    "SnowflakeCortexAnalyst",
    # MCP Integration
    "MCPToolWrapper",
    "create_mcp_tool_wrapper",
    "create_snowflake_compatible_tools",
    "bind_mcp_tools",
    # Document Formatters
    "format_cortex_search_documents",
    # Authentication utilities
    "create_session_from_env",
    "create_session_from_connection_string",
    "create_session_from_pat",
    "create_session_from_key_pair",
    "get_default_session",
    # Version
    "__version__",
]


def test_all_imports() -> None:
    """Test that all expected exports are available in __all__."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)


def test_individual_imports() -> None:
    """Test that each component can be imported individually."""
    # Chat Models
    from langchain_snowflake import ChatSnowflake

    assert ChatSnowflake is not None

    # Retrievers
    from langchain_snowflake import SnowflakeCortexSearchRetriever

    assert SnowflakeCortexSearchRetriever is not None

    # Tools
    from langchain_snowflake import (
        CortexCompleteTool,
        CortexSentimentTool,
        CortexSummarizerTool,
        CortexTranslatorTool,
        SnowflakeCortexAnalyst,
        SnowflakeQueryTool,
    )

    assert all(
        [
            SnowflakeQueryTool is not None,
            CortexSentimentTool is not None,
            CortexSummarizerTool is not None,
            CortexTranslatorTool is not None,
            CortexCompleteTool is not None,
            SnowflakeCortexAnalyst is not None,
        ]
    )

    # MCP Integration
    from langchain_snowflake import (
        MCPToolWrapper,
        bind_mcp_tools,
        create_mcp_tool_wrapper,
        create_snowflake_compatible_tools,
    )

    assert all(
        [
            MCPToolWrapper is not None,
            create_mcp_tool_wrapper is not None,
            create_snowflake_compatible_tools is not None,
            bind_mcp_tools is not None,
        ]
    )

    # Document Formatters
    from langchain_snowflake import format_cortex_search_documents

    assert format_cortex_search_documents is not None

    # Authentication utilities
    from langchain_snowflake import (
        create_session_from_connection_string,
        create_session_from_env,
        create_session_from_key_pair,
        create_session_from_pat,
        get_default_session,
    )

    assert all(
        [
            create_session_from_env is not None,
            create_session_from_connection_string is not None,
            create_session_from_pat is not None,
            create_session_from_key_pair is not None,
            get_default_session is not None,
        ]
    )


def test_no_unexpected_exports() -> None:
    """Test that we don't have any unexpected exports."""
    from langchain_snowflake import __all__

    # Should only export what's in EXPECTED_ALL
    unexpected = set(__all__) - set(EXPECTED_ALL)
    assert len(unexpected) == 0, f"Unexpected exports found: {unexpected}"

    # Should export everything in EXPECTED_ALL
    missing = set(EXPECTED_ALL) - set(__all__)
    assert len(missing) == 0, f"Missing expected exports: {missing}"


def test_version_availability() -> None:
    """Test that version is accessible."""
    from langchain_snowflake import __version__

    assert isinstance(__version__, str)
    # Version should be non-empty (unless in development)
    # assert len(__version__) > 0  # Uncomment if you want to enforce non-empty version
