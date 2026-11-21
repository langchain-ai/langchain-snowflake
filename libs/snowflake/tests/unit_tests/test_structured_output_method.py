"""`with_structured_output` must reject methods it does not implement."""

from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from langchain_snowflake.chat_models import ChatSnowflake


class Joke(BaseModel):
    setup: str
    punchline: str


class MockSession:
    def sql(self, query):  # noqa: ANN001, ANN201
        result = Mock()
        result.collect.return_value = [["Test response"]]
        return result


@pytest.fixture
def llm():  # noqa: ANN201
    with patch("langchain_snowflake.chat_models.base.Session"):
        return ChatSnowflake(model="llama3.1-70b", session=MockSession())


@pytest.mark.parametrize("method", ["json_mode", "json_schema", "typo"])
def test_unsupported_method_raises(llm, method: str) -> None:  # noqa: ANN001
    """An unimplemented method must fail loudly, not fall back silently."""
    with pytest.raises(ValueError, match="Unsupported method"):
        llm.with_structured_output(Joke, method=method)


def test_function_calling_is_accepted(llm) -> None:  # noqa: ANN001
    assert llm.with_structured_output(Joke, method="function_calling") is not None


def test_default_method_is_accepted(llm) -> None:  # noqa: ANN001
    assert llm.with_structured_output(Joke) is not None
