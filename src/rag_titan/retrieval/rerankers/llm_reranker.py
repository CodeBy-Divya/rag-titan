"""
LLM reranking: your chat model rates the chunks a search found.

One call per question rates every candidate from 0 to 10 (a "listwise" prompt:
cheaper than one call per chunk, and the model can compare the chunks):

    {"1": 10, "2": 0, "3": 5}   ->   rerank_score 1.0, 0.0, 0.5

Works with every LLM provider (Ollama, Claude) and needs nothing extra. The
chunks are fenced off and the prompt says to ignore instructions inside them,
because document text is data, never orders (handbook release checklist:
"prompt-injection defense at the retrieved-content boundary").
"""

import json
import re

from rag_titan.prompts import LLM_RERANK_PROMPT
from rag_titan.retrieval.rerankers.base_reranker import BaseReranker
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)

FALLBACK_NOTE = "the model's reply could not be read, so the search order was kept"
NOT_RATED_NOTE = "the model gave no score for this chunk, so it counts as 0"

# "1": 7  or  1 = 7  or  passage 1: 7  (small models do not always write clean JSON)
LOOSE_SCORE = re.compile(r'"?(?:passage\s*)?(\d+)"?\s*[:=]\s*(\d+(?:\.\d+)?)', re.IGNORECASE)
# an opening or closing passage tag inside a chunk would break the fence
FENCE_TAG = re.compile(r"</?\s*passage", re.IGNORECASE)


class LLMReranker(BaseReranker):
    """
    Re-orders chunks by asking the chat model how useful each one is.

    Args:
        llm: any BaseLLM, for example the one that writes the answers.
        max_chars: each chunk is cut to this many characters in the prompt,
            so a long chunk cannot crowd out the others.
    """

    def __init__(self, llm, max_chars=800):
        if llm is None or not hasattr(llm, "ask"):
            raise ValueError("The llm reranker needs a chat model: LLMReranker(llm)")
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
            raise ValueError(f"max_chars must be a whole number above 0, got {max_chars!r}")
        self.llm = llm
        self.max_chars = max_chars

    def rerank(self, query, matches, top_n=3):
        """
        Rate every match in one call and keep the best top_n.

        Returns:
            Copies of the best matches. "llm_score" is the model's 0-10 rating,
            "rerank_score" the same from 0 to 1; equal ratings keep the better
            search score first. When the reply cannot be read, the search order
            is kept and every match says why in "rerank_note".
        """
        if not matches:
            return []

        reply = self.llm.ask(LLM_RERANK_PROMPT, self.build_input(query, matches))
        scores = parse_scores(reply, len(matches))

        if not scores:
            logger.warning("LLM reranker: %s. Reply: %r", FALLBACK_NOTE, reply[:200])
            return [{**match, "rerank_score": match["score"], "rerank_note": FALLBACK_NOTE}
                    for match in matches[:top_n]]

        reranked = []
        for number, match in enumerate(matches, start=1):
            rating = scores.get(number)
            item = {**match, "llm_score": rating, "rerank_score": (rating or 0.0) / 10}
            if rating is None:
                logger.warning("LLM reranker: no score for %s", match.get("id"))
                item["rerank_note"] = NOT_RATED_NOTE
            reranked.append(item)

        reranked.sort(key=lambda m: (m["rerank_score"], m.get("score", 0.0)), reverse=True)
        return reranked[:top_n]

    def score(self, query, match):
        """Rating of one chunk from 0 to 1 (rerank() rates all chunks in one call)."""
        return self.rerank(query, [match], top_n=1)[0]["rerank_score"]

    def build_input(self, query, matches):
        """The user message: the question, then each chunk inside a numbered fence."""
        blocks = []
        for number, match in enumerate(matches, start=1):
            text = FENCE_TAG.sub("passage", match["text"][:self.max_chars])
            blocks.append(f"<passage {number}>\n{text}\n</passage {number}>")
        return f"QUESTION:\n{query}\n\nPASSAGES:\n" + "\n\n".join(blocks)


def parse_scores(reply, count):
    """
    Read the model's ratings: {passage number: score from 0 to 10}.

    Accepts clean JSON ({"1": 7}), a JSON list ([7, 0, 5]) or loose lines
    ("1: 7"). Numbers outside 1..count are ignored and scores are kept
    within 0..10. Returns {} when nothing can be read.
    """
    text = reply or ""
    found = {}
    # an object first: "Passage [1] is best: {"1": 9}" must not read [1] as the scores
    match = re.search(r"\{.*?\}", text, re.DOTALL) or re.search(r"\[.*?\]", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else None
    except ValueError:
        data = None

    if isinstance(data, list):
        data = {str(number): value for number, value in enumerate(data, start=1)}
    if isinstance(data, dict):
        for key, value in data.items():
            number_value = isinstance(value, (int, float)) and not isinstance(value, bool)
            if str(key).strip().isdigit() and number_value:
                found[int(key)] = value
    if not found:
        found = {int(number): float(value) for number, value in LOOSE_SCORE.findall(text)}

    return {number: min(max(float(value), 0.0), 10.0)
            for number, value in found.items() if 1 <= number <= count}
