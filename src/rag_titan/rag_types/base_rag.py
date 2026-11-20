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
