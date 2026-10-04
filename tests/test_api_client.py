"""
Tests for APIClient request shapes (SDK client mocked; no network).
"""

import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.utils.api_client import APIClient


def _response(
    blocks, stop_reason="end_turn", model="claude-sonnet-5-5", stop_details=None
):
    return SimpleNamespace(
        content=blocks,
        model=model,
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=SimpleNamespace(input_tokens=3, output_tokens=5),
    )


def _client(model=None):
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}, clear=True):
        client = APIClient(model=model)
    client.client = Mock()
    return client


def test_default_model_is_sonnet_5_5():
    assert _client().model == "claude-sonnet-5-5"


def test_sonnet_5_5_request_shape():
    client = _client()
    client.client.beta.messages.create.return_value = _response(
        [
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text='{"ok": true}'),
        ]
    )

    result = client.create_message(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=256,
    )

    kwargs = client.client.beta.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-5-5"
    assert "temperature" not in kwargs and "extra_body" not in kwargs
    assert kwargs["output_config"] == {"effort": "low"}
    assert kwargs["max_tokens"] >= 16000  # thinking counts toward max_tokens
    assert kwargs["fallbacks"] == "default"
    assert kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    # thinking blocks are skipped; only text is returned
    assert result["content"] == '{"ok": true}'


def test_refusal_raises():
    client = _client()
    client.client.beta.messages.create.return_value = _response(
        [],
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="cyber"),
    )

    with pytest.raises(RuntimeError, match="declined.*cyber"):
        client.create_message(messages=[{"role": "user", "content": "hi"}])


def test_older_model_keeps_temperature():
    client = _client(model="claude-sonnet-4-5-20250929")
    client.client.messages.create.return_value = _response(
        [SimpleNamespace(type="text", text="ok")],
        model="claude-sonnet-4-5-20250929",
    )

    client.create_message(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=256,
        temperature=0,
    )

    kwargs = client.client.messages.create.call_args.kwargs
    assert kwargs["extra_body"] == {"temperature": 0}
    assert kwargs["max_tokens"] == 256
    assert "output_config" not in kwargs
