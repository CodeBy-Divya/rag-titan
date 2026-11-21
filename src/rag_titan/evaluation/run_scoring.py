"""
Score one question of an end-to-end run: what was answered, and whether the
chunks given to the LLM held the evidence (used by evaluate_rag() and the playground).
"""

import time

from rag_titan.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from rag_titan.prompts import NO_ANSWER

# replies that mean "no answer was given"
NOT_ANSWERED = {NO_ANSWER, "Please enter a question.", "Please index your documents first."}


def run_question(rag, question):
    """Run one question, time it and score the chunks the LLM was given."""
    start = time.perf_counter()
    result = rag.run(question["question"])
    ms = (time.perf_counter() - start) * 1000

    return {
        "id": question["id"],
        "task_type": question.get("task_type"),
        "answer": result["answer"],
        "answered": result["answer"].strip() not in NOT_ANSWERED,
        **score_sources(question, result["sources"]),
        "ms": ms,
        "steps": result["steps"],
    }


def score_sources(question, sources):
    """
    Score the chunks given to the LLM against the question's evidence.

    Args:
        question: a question dict with "required_evidence_ids".
        sources: the chunks the LLM was given (or would be given).

    Returns:
        {"recall", "reciprocal_rank", "ndcg", "sources"}; "sources" lists the
        file of every chunk.
    """
    required = question["required_evidence_ids"]
    k = max(len(sources), 1)   # score every source; k must be at least 1
    return {
        "recall": recall_at_k(sources, required, k),
        "reciprocal_rank": reciprocal_rank(sources, required, k),
        "ndcg": ndcg_at_k(sources, required, k),
        "sources": [chunk["source"] for chunk in sources],
    }
