"""
Tests for AnthropicLLM and the JSON HTTP helper (no network, no API key needed).

Most tests swap post_json for a fake that saves the request; the last ones use
the fake HTTP server from tests/fake_ollama.py to test real HTTP and errors.
"""

import json
import logging

import pytest

from rag_titan.agents import MathAgent
from rag_titan.llm import anthropic_llm
from rag_titan.llm.anthropic_llm import AnthropicLLM, merge_same_role
from rag_titan.llm.http_json import post_json

KEY = "sk-ant-test-secret"


def text_reply(text, stop_reason="end_turn"):
    return {"content": [{"type": "text", "text": text}], "stop_reason": stop_reason}


def tool_reply(call_id, name, arguments):
    return {"content": [{"type": "tool_use", "id": call_id, "name": name, "input": arguments}],
            "stop_reason": "tool_use"}


@pytest.fixture
def fake_api(monkeypatch):
    """Replace post_json: save each request, answer with the next prepared reply."""
    calls, replies = [], []

    def fake_post(url, payload, headers=None, timeout=None, service="The API"):
        # deep copy, so later changes to the history do not change saved requests
        calls.append({"url": url, "payload": json.loads(json.dumps(payload)),
                      "headers": headers})
        return replies.pop(0)

    monkeypatch.setattr(anthropic_llm, "post_json", fake_post)
    return calls, replies


def test_request_has_system_max_tokens_and_key_header(fake_api):
    calls, replies = fake_api
    replies.append(text_reply("Blue."))

    answer = AnthropicLLM(api_key=KEY, temperature=0).ask("Answer in one word.", "Sky colour?")

    request = calls[0]
    assert answer == "Blue."
    assert request["url"] == "https://api.anthropic.com/v1/messages"
    assert request["headers"] == {"x-api-key": KEY, "anthropic-version": "2023-06-01"}
    assert request["payload"] == {
        "model": "claude-haiku-5-5", "max_tokens": 2048, "temperature": 0,
        "system": "Answer in one word.",
        "messages": [{"role": "user", "content": "Sky colour?"}],
    }


def test_reply_format_and_history_message(fake_api):
    _, replies = fake_api
    replies.append(text_reply("Hi", stop_reason="end_turn") | {"id": "msg_1", "usage": {}})

    reply = AnthropicLLM("claude-sonnet-5-5", api_key=KEY).chat(
        [{"role": "user", "content": "hello"}])

    assert reply["content"] == "Hi" and reply["tool_calls"] == []
    # only role and content: the API rejects id, usage, stop_reason in the history
    assert reply["message"] == {"role": "assistant",
                                "content": [{"type": "text", "text": "Hi"}]}


def test_agent_tool_round_trip(fake_api):
    calls, replies = fake_api
    replies += [tool_reply("toolu_1", "add", {"a": 2, "b": 3}),
                tool_reply("toolu_2", "multiply", {"a": 5, "b": 4}),
                text_reply("The answer is 20.")]

    answer = MathAgent(AnthropicLLM(api_key=KEY)).run("What is (2 + 3) * 4?")

    assert answer == "The answer is 20."
    tools = calls[0]["payload"]["tools"]
    assert {tool["name"] for tool in tools} == {"add", "subtract", "multiply", "divide"}
    assert tools[0]["input_schema"]["properties"]["a"]["type"] == "string"

    last = calls[2]["payload"]["messages"]
    assert [message["role"] for message in last] == ["user", "assistant", "user",
                                                     "assistant", "user"]
    assert last[2]["content"] == [{"type": "tool_result", "tool_use_id": "toolu_1",
                                   "content": "5.0"}]
    assert last[4]["content"][0]["tool_use_id"] == "toolu_2"


def test_back_to_back_messages_are_merged():
    merged = merge_same_role([
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "a"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "b"}]},
    ])

    assert [message["role"] for message in merged] == ["user", "assistant", "user"]
    assert [block["tool_use_id"] for block in merged[2]["content"]] == ["a", "b"]


def test_missing_key_and_bad_settings(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicLLM().ask("system", "hi")
    with pytest.raises(ValueError, match="max_tokens"):
        AnthropicLLM(max_tokens=0)
    with pytest.raises(ValueError, match="must be a dict"):
        AnthropicLLM(api_key=KEY).chat(["hi"])

    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    assert AnthropicLLM().api_key == KEY
    assert KEY not in repr(AnthropicLLM())


def test_cut_off_reply_logs_a_warning(fake_api, caplog):
    _, replies = fake_api
    replies.append(text_reply("A long answer that", stop_reason="max_tokens"))

    with caplog.at_level(logging.WARNING):
        answer = AnthropicLLM(api_key=KEY, max_tokens=5).ask("system", "explain")

    assert answer == "A long answer that"
    assert "max_tokens=5" in caplog.text


def test_real_http_request_and_reply(fake_ollama):
    fake_ollama.reply = (200, json.dumps(text_reply("Paris")))

    answer = AnthropicLLM(api_key=KEY, base_url=fake_ollama.address + "/v1/").ask(
        "Be brief.", "Capital of France?")

    assert answer == "Paris"
    assert fake_ollama.requests[0]["path"] == "/v1/messages"
    assert fake_ollama.requests[0]["body"]["system"] == "Be brief."


def test_http_error_shows_the_reason_but_not_the_key(fake_ollama):
    fake_ollama.reply = (401, json.dumps({"type": "error", "error": {
        "type": "authentication_error", "message": "invalid x-api-key"}}))

    with pytest.raises(RuntimeError) as caught:
        AnthropicLLM(api_key=KEY, base_url=fake_ollama.address).ask("system", "hi")

    message = str(caught.value)
    assert "HTTP 401: invalid x-api-key" in message and "claude-haiku-5-5" in message
    assert KEY not in message and caught.value.__cause__ is None


def test_post_json_errors_are_plain(fake_ollama):
    fake_ollama.reply = (500, "server broke")
    with pytest.raises(RuntimeError, match="Test returned HTTP 500: server broke"):
        post_json(fake_ollama.address, {}, service="Test")

    fake_ollama.reply = (200, "not json")
    with pytest.raises(RuntimeError, match="not JSON"):
        post_json(fake_ollama.address, {}, service="Test")

    with pytest.raises(RuntimeError, match="Could not reach Test"):
        post_json("http://127.0.0.1:9", {}, service="Test", timeout=2)
