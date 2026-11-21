"""
Base class for every RAG type.

To add one, inherit from BaseRAG and:
    1. describe its steps in FLOW (the docs playground draws this diagram)
    2. implement run(), recording every step with a Trace (search with
       trace.search(self.retriever, ...), so hybrid search shows its own steps)
answer() then works for free.
"""

from rag_titan.rag_types.flow import expand_search

# kinds of FLOW nodes, they decide the colour in the flow diagram
NODE_KINDS = ["search", "llm", "output"]


class BaseRAG:
    """
    Common interface for RAG types.

    FLOW lists the steps and the arrows between them:
        {"nodes": [{"id": "retrieve", "label": "Retrieve", "kind": "search"}, ...],
         "edges": [["retrieve", "rerank"], ...]}
    Every step that run() records must be a node, and two steps in a row
    must be joined by an edge.
    """

    FLOW = {"nodes": [], "edges": []}

    def run(self, query):
        """
        Answer the question and show how.

        Returns:
            A dict with "answer" (the text), "sources" (the chunks used for
            the answer) and "steps" (one {"step", "input", "output", "ms"}
            dict per step, in order).
        """
        raise NotImplementedError

    def flow(self):
        """
        The FLOW to draw for this RAG type with its retriever: the search node
        is named after the retriever, or replaced by its steps (see flow.py).
        """
        return expand_search(self.FLOW, getattr(self, "retriever", None))

    def answer(self, query):
        """Return only the answer text."""
        return self.run(query)["answer"]


def check_count(name, value):
    """
    Return `value` when it is a whole number above 0, else raise a clear ValueError.

    Args:
        name: the option's name, used in the message (for example "top_k").
        value: the value to check. True/False are refused: Python counts
            True as 1, but top_k=True is surely a mistake.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a whole number above 0, got {value!r}")
    return value


def check_search_sizes(top_k, top_n):
    """
    Check the two sizes every RAG type has: chunks searched (top_k) and kept (top_n).

    The reranker keeps the best top_n of the top_k chunks found, so top_n
    larger than top_k can only be a mistake.
    """
    check_count("top_k", top_k)
    check_count("top_n", top_n)
    if top_n > top_k:
        raise ValueError(f"top_n ({top_n}) cannot be more than top_k ({top_k}): the reranker "
                         "keeps the best top_n of the top_k chunks found")
