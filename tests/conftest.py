import pytest
from helpers import FakeLLM


@pytest.fixture
def fake_llm():
    return FakeLLM()
