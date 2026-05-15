"""Test to reproduce the empty payload bug in RestApiClient.prepare_request().

Bug: When create_thread() is called without metadata, payload={} is passed to
prepare_request(). The check `if payload:` treats {} as falsy, so the POST
request is sent without a JSON body. Snowflake returns 400 Bad Request.

Reference: https://sailpoint customer report, langchain-snowflake v0.2.3
"""

from unittest.mock import MagicMock, patch

from langchain_snowflake._connection.rest_client import RestApiClient


class TestEmptyPayloadBug:
    """Reproduce the empty payload bug in rest_client.py:79."""

    @patch.object(
        RestApiClient, "_build_simple_url", return_value="https://test.snowflakecomputing.com/api/v2/cortex/threads"
    )
    @patch(
        "langchain_snowflake._connection.rest_client.SnowflakeAuthUtils.get_rest_api_headers",
        return_value={"Authorization": "Bearer test", "Content-Type": "application/json"},
    )
    def test_empty_dict_payload_is_included_in_request(self, mock_headers, mock_url):
        """Empty dict payload {} must be included in the request config.

        Previously this was a bug: `if payload:` treated {} as falsy, causing
        create_thread() without metadata to send a POST with no body (400 error).
        Fixed by changing to `if payload is not None:`.
        """
        mock_session = MagicMock()

        request_config = RestApiClient.prepare_request(
            session=mock_session,
            endpoint="/cortex/threads",
            method="POST",
            payload={},
        )

        assert "json" in request_config, "Empty dict payload {} must be included in request config."
        assert request_config["json"] == {}


class TestCreateThreadEndToEnd:
    """End-to-end test simulating the customer's exact call: create_thread() with no args."""

    @patch.object(RestApiClient, "make_sync_request", return_value={"thread_id": "12345"})
    @patch.object(
        RestApiClient, "_build_simple_url", return_value="https://test.snowflakecomputing.com/api/v2/cortex/threads"
    )
    @patch(
        "langchain_snowflake._connection.rest_client.SnowflakeAuthUtils.get_rest_api_headers",
        return_value={"Authorization": "Bearer test", "Content-Type": "application/json"},
    )
    @patch("langchain_snowflake._connection.session_manager.SnowflakeSessionManager.get_or_create_session")
    def test_create_thread_no_metadata(self, mock_session_mgr, mock_headers, mock_url, mock_request):
        """create_thread() with no arguments must send a POST with an empty JSON body."""
        from langchain_snowflake.agents.base import SnowflakeCortexAgent

        mock_session = MagicMock()
        mock_session.get_current_account.return_value = "test-account"
        mock_session_mgr.return_value = mock_session

        agent = SnowflakeCortexAgent(
            name="test_agent",
            database="test_db",
            schema="test_schema",
            session=mock_session,
        )

        thread_id = agent.create_thread()

        assert thread_id == "12345"
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        request_config = call_args[0][0]
        assert "json" in request_config, (
            "create_thread() without metadata must include an empty JSON body in the request"
        )
        assert request_config["json"] == {}

    @patch.object(RestApiClient, "make_sync_request", return_value={"thread_id": "67890"})
    @patch.object(
        RestApiClient, "_build_simple_url", return_value="https://test.snowflakecomputing.com/api/v2/cortex/threads"
    )
    @patch(
        "langchain_snowflake._connection.rest_client.SnowflakeAuthUtils.get_rest_api_headers",
        return_value={"Authorization": "Bearer test", "Content-Type": "application/json"},
    )
    @patch("langchain_snowflake._connection.session_manager.SnowflakeSessionManager.get_or_create_session")
    def test_create_thread_with_metadata(self, mock_session_mgr, mock_headers, mock_url, mock_request):
        """create_thread() with metadata must include it in the JSON body."""
        from langchain_snowflake.agents.base import SnowflakeCortexAgent

        mock_session = MagicMock()
        mock_session.get_current_account.return_value = "test-account"
        mock_session_mgr.return_value = mock_session

        agent = SnowflakeCortexAgent(
            name="test_agent",
            database="test_db",
            schema="test_schema",
            session=mock_session,
        )

        thread_id = agent.create_thread(metadata={"app": "sailpoint"})

        assert thread_id == "67890"
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        request_config = call_args[0][0]
        assert "json" in request_config
        assert request_config["json"] == {"metadata": {"app": "sailpoint"}}

    @patch.object(
        RestApiClient, "_build_simple_url", return_value="https://test.snowflakecomputing.com/api/v2/cortex/threads"
    )
    @patch(
        "langchain_snowflake._connection.rest_client.SnowflakeAuthUtils.get_rest_api_headers",
        return_value={"Authorization": "Bearer test", "Content-Type": "application/json"},
    )
    def test_none_payload_is_excluded_from_request(self, mock_headers, mock_url):
        """None payload should correctly be excluded — this is expected behavior."""
        mock_session = MagicMock()

        request_config = RestApiClient.prepare_request(
            session=mock_session,
            endpoint="/cortex/threads",
            method="POST",
            payload=None,
        )

        assert "json" not in request_config

    @patch.object(
        RestApiClient, "_build_simple_url", return_value="https://test.snowflakecomputing.com/api/v2/cortex/threads"
    )
    @patch(
        "langchain_snowflake._connection.rest_client.SnowflakeAuthUtils.get_rest_api_headers",
        return_value={"Authorization": "Bearer test", "Content-Type": "application/json"},
    )
    def test_nonempty_payload_is_included(self, mock_headers, mock_url):
        """Non-empty payload should be included — this works correctly today."""
        mock_session = MagicMock()

        request_config = RestApiClient.prepare_request(
            session=mock_session,
            endpoint="/cortex/threads",
            method="POST",
            payload={"metadata": {"app": "sailpoint"}},
        )

        assert "json" in request_config
        assert request_config["json"] == {"metadata": {"app": "sailpoint"}}

    @patch.object(
        RestApiClient, "_build_simple_url", return_value="https://test.snowflakecomputing.com/api/v2/cortex/threads"
    )
    @patch(
        "langchain_snowflake._connection.rest_client.SnowflakeAuthUtils.get_rest_api_headers",
        return_value={"Authorization": "Bearer test", "Content-Type": "application/json"},
    )
    def test_empty_dict_payload_should_be_included_after_fix(self, mock_headers, mock_url):
        """EXPECTED after fix: Empty dict {} should be included in request config.

        This test will FAIL on the current buggy code and PASS after the fix.
        """
        mock_session = MagicMock()

        request_config = RestApiClient.prepare_request(
            session=mock_session,
            endpoint="/cortex/threads",
            method="POST",
            payload={},
        )

        assert "json" in request_config, (
            "FAILS: Empty dict payload {} was dropped from request config. "
            "Fix rest_client.py:79 from `if payload:` to `if payload is not None:`"
        )
        assert request_config["json"] == {}
