"""
Chat model that runs on Anthropic's Claude (https://www.anthropic.com).

Get an API key at https://console.anthropic.com and put it in the
ANTHROPIC_API_KEY environment variable (or your .env file). Only the standard
library is used, so nothing extra needs to be installed.

Example:
    from rag_titan.llm import AnthropicLLM

    llm = AnthropicLLM("claude-haiku-5-5")
    print(llm.ask("Answer in one word.", "What colour is the sky?"))
"""

import os

from rag_titan.llm.base_llm import BaseLLM
from rag_titan.llm.http_json import post_json
from rag_titan.llm.tool_schema import tools_to_schemas
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)

API_URL = "https://api.anthropic.com/v1"
API_VERSION = "2023-06-01"          # the Messages API version header Anthropic requires


class AnthropicLLM(BaseLLM):
    """
    Talks to Anthropic's Messages API. Supports tools, so agents work too.

    Args:
        model: Claude model name, for example "claude-haiku-5-5" (fast, cheap)
            or "claude-sonnet-5-5".
        api_key: your key. None reads the ANTHROPIC_API_KEY environment variable.
        base_url: API address; change it only for a proxy or gateway.
        max_tokens: longest reply allowed, in tokens (Anthropic requires one).
        temperature: optional randomness from 0 to 1; None uses the model's default.
    """

    def __init__(self, model="claude-haiku-5-5", api_key=None, base_url=API_URL,
                 max_tokens=2048, temperature=None):
        super().__init__(model)
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError(f"max_tokens must be a whole number above 0, got {max_tokens!r}")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature

    def __repr__(self):
        # never show the key, even when someone prints the object
        return f"AnthropicLLM(model={self.model!r})"

    def chat(self, messages, tools=None):
        """Send messages to Claude. See BaseLLM.chat() for the reply format."""
        if not self.api_key:
            raise ValueError("AnthropicLLM needs an API key: set the ANTHROPIC_API_KEY "
                             "environment variable (or .env), or pass api_key=...")
        logger.debug("Calling Anthropic model '%s'", self.model)

        system_prompt, chat_messages = self._separate_system(messages)
        payload = {"model": self.model, "max_tokens": self.max_tokens,
                   "messages": merge_same_role(chat_messages)}
        if system_prompt:
            payload["system"] = system_prompt
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = self._convert_tools(tools)

        headers = {"x-api-key": self.api_key, "anthropic-version": API_VERSION}
        try:
            reply = post_json(f"{self.base_url}/messages", payload, headers, service="Anthropic")
        except RuntimeError as error:
            raise RuntimeError(
                f"Anthropic chat failed for model '{self.model}': {error}"
            ) from None
        return self._format_reply(reply)

    def tool_result_message(self, tool_call, result):
        """Build the user message with a tool_result block that Claude expects."""
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_call.get("id") or "toolu_0",
                         "content": str(result)}],
        }

    def _separate_system(self, messages):
        """Move system messages out: Anthropic takes them as one top-level "system" text."""
        system_parts, chat_messages = [], []

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(f"Each chat message must be a dict, got {type(message).__name__}")
            if message.get("role") == "system":
                system_parts.append(message.get("content") or "")
            else:
                # keep only the two fields the API accepts
                chat_messages.append({"role": message.get("role"),
                                      "content": message.get("content") or ""})

        return "\n\n".join(part for part in system_parts if part).strip(), chat_messages

    def _convert_tools(self, tools):
        """Turn rag-titan tool schemas into Anthropic's {name, description, input_schema}."""
        converted = []
        for tool in tools_to_schemas(tools):
            function = tool.get("function") or {}
            converted.append({
                "name": function.get("name"),
                "description": function.get("description", ""),
                "input_schema": function.get("parameters", {"type": "object", "properties": {}}),
            })
        return converted

    def _format_reply(self, reply):
        """Pull the text and tool_use blocks out of Claude's reply."""
        blocks = reply.get("content") or []
        text_parts, tool_calls = [], []

        for block in blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append({"id": block.get("id"), "name": block.get("name"),
                                   "arguments": block.get("input") or {}})

        if reply.get("stop_reason") == "max_tokens":
            logger.warning("Claude's reply was cut off at max_tokens=%d; pass a larger "
                           "max_tokens for longer answers", self.max_tokens)

        return {
            "content": "".join(text_parts),
            "tool_calls": tool_calls,
            # only role and content go back into the history: the API rejects other fields
            "message": {"role": "assistant", "content": blocks},
        }


def merge_same_role(messages):
    """
    Join back-to-back messages from the same role into one message.

    An agent sends one tool result per message, but Claude wants all results
    for one turn together, so they are combined into a single list of blocks.

    Args:
        messages: list of {"role": ..., "content": text or list of blocks}.
    """
    merged = []
    for message in messages:
        if merged and merged[-1]["role"] == message["role"]:
            merged[-1]["content"] = _as_blocks(merged[-1]["content"]) + _as_blocks(
                message["content"])
        else:
            merged.append(dict(message))
    return merged


def _as_blocks(content):
    """Turn plain text into [{"type": "text", ...}]; block lists are copied as they are."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return list(content)
