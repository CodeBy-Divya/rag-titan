"""
Anthropic Claude chat model integration (https://anthropic.com).
"""

import os

from rag_titan.llm.base_llm import BaseLLM
from rag_titan.llm.http_client import post_json
from rag_titan.llm.tool_schema import tools_to_schemas
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)


class AnthropicLLM(BaseLLM):
    """
    Chat model using Anthropic's Messages API.

    Args:
        model: model identifier, e.g. "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022".
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY environment variable).
        base_url: API endpoint URL (defaults to "https://api.anthropic.com/v1").
        max_tokens: maximum response tokens (defaults to 2048).
        temperature: optional sampling temperature.
    """

    def __init__(self, model="claude-3-5-haiku-20241022", api_key=None,
                 base_url="https://api.anthropic.com/v1", max_tokens=2048,
                 temperature=None):
        super().__init__(model)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, messages, tools=None):
        """Send messages to Anthropic and return standard reply dict."""
        self._check_api_key()
        logger.debug("Calling Anthropic model '%s'", self.model)

        url = f"{self.base_url}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        system_prompt, chat_messages = self._separate_system(messages)
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": chat_messages,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        anthropic_tools = self._convert_tools(tools)
        if anthropic_tools:
            payload["tools"] = anthropic_tools

        reply = post_json(url, payload, headers=headers)
        return self._format_reply(reply)

    def tool_result_message(self, tool_call, result):
        """Build the user tool_result message block expected by Anthropic."""
        call_id = tool_call.get("id") or "toolu_0"
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": str(result),
                }
            ],
        }

    def _check_api_key(self):
        """Ensure an API key is available."""
        if not self.api_key:
            raise ValueError(
                "AnthropicLLM requires an API key. Set the ANTHROPIC_API_KEY "
                "environment variable or pass api_key=..."
            )

    def _separate_system(self, messages):
        """Extract system messages into a top-level instruction string."""
        system_parts = []
        filtered_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg.get("content", ""))
            else:
                filtered_messages.append(msg)

        system_text = "\n\n".join(part for part in system_parts if part).strip()
        return system_text, filtered_messages

    def _convert_tools(self, tools):
        """Convert standard OpenAI-style tool schemas into Anthropic format."""
        schemas = tools_to_schemas(tools)
        anthropic_tools = []
        for tool in schemas:
            func = tool.get("function") or {}
            anthropic_tools.append({
                "name": func.get("name"),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })
        return anthropic_tools

    def _format_reply(self, reply):
        """Extract text content and tool_use blocks from Anthropic's response."""
        content_blocks = reply.get("content") or []
        text_parts = []
        tool_calls = []

        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "arguments": block.get("input") or {},
                })

        return {
            "content": "".join(text_parts),
            "tool_calls": tool_calls,
            "message": reply,
        }
