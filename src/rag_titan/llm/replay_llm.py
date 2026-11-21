"""
Record real LLM replies once, then replay them without the model.

    recorder = RecordingLLM(OllamaLLM("gpt-oss:20b-cloud"))
    ...run a RAG type with recorder...
    replies = recorder.replies                 # save as JSON

    llm = ReplayLLM(replies)                   # later: same answers, no model, no network

The browser playground uses this to show real model answers for its examples.
"""

import hashlib
import json

from rag_titan.llm.base_llm import BaseLLM
from rag_titan.utils.not_recorded import NotRecorded


def prompt_key(messages):
    """Return a short id for a conversation: only roles and texts count."""
    plain = [{"role": message["role"], "content": message["content"]} for message in messages]
    text = json.dumps(plain, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


class RecordingLLM(BaseLLM):
    """
    Passes every new prompt to a real LLM and remembers the reply.

    A prompt asked again gets the remembered reply, like ReplayLLM would give.
    Models do not always answer the same, so this keeps the recording exactly
    what a replay will do.

    Args:
        llm: any BaseLLM, for example an OllamaLLM.
        known: optional replies from an earlier recording ({prompt key: reply}).
            A prompt found there reuses its reply instead of asking the model,
            so recording again (for a new RAG type, say) does not change the
            runs that were recorded before. `replies` holds only the prompts
            this recording asked, so replies nobody uses any more are left out.
    """

    def __init__(self, llm, known=None):
        super().__init__(llm.model)
        self.llm = llm
        self.known = dict(known or {})
        self.replies = {}

    def chat(self, messages, tools=None):
        """Ask the real LLM (only the first time) and save its reply under the prompt's key."""
        if tools:
            raise ValueError("RecordingLLM records plain replies only, not tool calls")

        key = prompt_key(messages)
        if key not in self.replies:
            if key in self.known:
                self.replies[key] = self.known[key]
            else:
                self.replies[key] = self.llm.chat(messages)["content"]

        return {"content": self.replies[key], "tool_calls": [], "message": {"role": "assistant"}}

    def tool_result_message(self, tool_call, result):
        """Not used: tool calls are not recorded."""
        return self.llm.tool_result_message(tool_call, result)


class ReplayLLM(BaseLLM):
    """
    Answers from recorded replies.

    Args:
        replies: dict of {prompt key: reply text}, e.g. RecordingLLM.replies.
        model: name to show, e.g. the model the replies came from.
    """

    def __init__(self, replies, model="replay"):
        if not isinstance(replies, dict):
            raise ValueError("replies must be a dict of {prompt key: reply text}")

        super().__init__(model)
        self.replies = replies

    def chat(self, messages, tools=None):
        """Return the recorded reply for exactly these messages."""
        key = prompt_key(messages)
        if key not in self.replies:
            raise NotRecorded(
                f"No recorded reply for this prompt (key {key}). The question, the "
                "documents or the settings differ from the recording; record it again."
            )

        return {"content": self.replies[key], "tool_calls": [], "message": {"role": "assistant"}}

    def tool_result_message(self, tool_call, result):
        """Build a simple tool message (tool calls are not replayed)."""
        return {"role": "tool", "content": str(result)}
