"""Feedback management functionality for Snowflake Cortex Agents.

This module provides feedback submission via the REST API.
"""

import logging
from typing import Any, Dict, Optional

from .._connection import RestApiClient, RestApiRequestBuilder
from .schemas import FeedbackInput, FeedbackOutput

logger = logging.getLogger(__name__)


class FeedbackManagement:
    """Mixin class providing feedback management functionality."""

    # ============================================================================
    # Request Configuration Methods
    # ============================================================================

    def _build_feedback_request_config(
        self,
        method: str,
        payload: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Build request configuration for feedback operations.

        Args:
            method: HTTP method
            payload: Request payload

        Returns:
            Request configuration dictionary
        """
        session = self._get_session()

        return RestApiRequestBuilder.feedback_request(
            session=session,
            database=self.database,
            schema=self.schema,
            name=self.name,
            method=method,
            payload=payload,
            request_timeout=self.request_timeout,
            verify_ssl=self.verify_ssl,
        )

    # ============================================================================
    # Feedback Operations (Sync)
    # ============================================================================

    def submit_feedback(self, feedback: FeedbackInput) -> FeedbackOutput:
        """Submit feedback for a run using Snowflake's native feedback system.

        Args:
            feedback: Feedback data

        Returns:
            FeedbackOutput containing status message

        Raises:
            SnowflakeRestApiError: If feedback submission fails
        """
        payload = feedback.model_dump(exclude_none=True)
        request_config = self._build_feedback_request_config("POST", payload=payload)
        operation_name = f"submit feedback for request '{feedback.request_id}'"

        # Use RestApiClient
        response_data = RestApiClient.make_sync_request(request_config, operation_name)
        return FeedbackOutput(**response_data)

    # ============================================================================
    # Feedback Operations (Async)
    # ============================================================================

    async def submit_feedback_async(self, feedback: FeedbackInput) -> FeedbackOutput:
        """Submit feedback for a run using Snowflake's native feedback system.

        Args:
            feedback: Feedback data

        Returns:
            FeedbackOutput containing status message

        Raises:
            SnowflakeRestApiError: If feedback submission fails
        """
        payload = feedback.model_dump(exclude_none=True)
        request_config = self._build_feedback_request_config("POST", payload=payload)
        operation_name = f"submit feedback for request '{feedback.request_id}'"

        # Use RestApiClient
        response_data = await RestApiClient.make_async_request(request_config, operation_name)
        return FeedbackOutput(**response_data)
