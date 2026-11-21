"""
Parent-Child RAG (RAG handbook 2026, section 09):

    Question -> Retrieve (small children) -> Rerank -> Expand to parents -> Generate

Small chunks match a question precisely but often hold only half of the answer.
This RAG searches the small children, then gives the model the whole section
(the parent) each good child sits in. The index must be built with parents:
RAGTitan(rag_type="parent_child") and `rag-titan index --rag-type parent_child`
do that for you (see chunking/parent_child.py).

The handbook's warning, "watch token inflation", is handled by parent_budget:
parents are added best first until the budget (in characters) is used up.
"""

from rag_titan.rag_types.base_rag import BaseRAG, check_count, check_search_sizes
from rag_titan.rag_types.trace import Trace
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)

NO_PARENTS = ("This index has no parent sections, so Parent-Child RAG cannot expand its "
              "chunks. Index the files again for it: RAGTitan(rag_type='parent_child')"
              ".index(files), or rag-titan index --rag-type parent_child")


class ParentChildRAG(BaseRAG):
    """
    Search small chunks, answer from the sections they belong to.

    Args:
        retriever: any BaseRetriever over an index built with parents.
        reranker: a BaseReranker; it orders the children before expanding.
        generator: a BaseGenerator.
        top_k: how many children to retrieve (small chunks: take more).
        top_n: how many children to keep after reranking.
        parent_budget: most characters of parent text given to the model.
    """

    FLOW = {
        "nodes": [
            {"id": "retrieve", "label": "Retrieve children", "kind": "search"},
            {"id": "rerank", "label": "Rerank", "kind": "search"},
            {"id": "expand", "label": "Expand to parents", "kind": "search"},
            {"id": "generate", "label": "Generate answer", "kind": "output"},
        ],
        "edges": [["retrieve", "rerank"], ["rerank", "expand"], ["expand", "generate"]],
    }

    # small children: search more of them (TOP_K / TOP_N change these)
    DEFAULT_TOP_K, DEFAULT_TOP_N = 8, 4

    def __init__(self, retriever, reranker, generator, top_k=DEFAULT_TOP_K,
                 top_n=DEFAULT_TOP_N, parent_budget=3000):
        check_search_sizes(top_k, top_n)
        check_count("parent_budget", parent_budget)
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.top_k = top_k
        self.top_n = top_n
        self.parent_budget = parent_budget

    def run(self, query):
        """Answer the question from the parents of the best children, recording each step."""
        query = query.strip()
        trace = Trace()

        if not query:
            return trace.result("Please enter a question.")

        if not self.retriever.has_documents():
            return trace.result("Please index your documents first.")

        children = trace.search(self.retriever, query=query, top_k=self.top_k)
        best = trace.record("rerank", self.reranker.rerank,
                            query=query, matches=children, top_n=self.top_n)
        parents = trace.record("expand", self.expand, chunks=best)

        answer = trace.record("generate", self.generator.generate, query=query, chunks=parents)
        return trace.result(answer, parents)

    def expand(self, chunks):
        """
        Swap each child for its parent: best first, each parent once, within the budget.

        Args:
            chunks: reranked children, best first.

        Returns:
            One dict per parent, in the order of its best child. It keeps the best
            child's "id" (so citations like [chunk_4] still work) and scores, and
            has the parent's text and place, plus "children" (the ids of every
            kept child in it) and "child_text" (the best child's own text).
        """
        parents = {}
        used = 0

        for chunk in chunks:
            if "parent_text" not in chunk:
                raise ValueError(NO_PARENTS)

            parent_id = chunk["parent_id"]
            if parent_id in parents:
                # a second child of the same section: the section is already in
                parents[parent_id]["children"].append(chunk["id"])
                continue

            size = len(chunk["parent_text"])
            if parents and used + size > self.parent_budget:
                # the first parent is always kept, so the model never gets nothing
                logger.info("Skipped %s: %d more characters would pass the budget of %d",
                            parent_id, size, self.parent_budget)
                continue

            parents[parent_id] = as_parent(chunk)
            used += size

        return list(parents.values())


def as_parent(child):
    """Turn a child chunk into its parent: the parent's text and place, the child's scores."""
    parent = {key: value for key, value in child.items()
              if not key.startswith("parent_") and key != "page_numbers"}
    parent.update({
        "text": child["parent_text"],
        "start_offset": child.get("parent_start_offset"),
        "end_offset": child.get("parent_end_offset"),
        "parent_id": child["parent_id"],
        "children": [child["id"]],
        "child_text": child["text"],
    })
    if "parent_page_numbers" in child:
        parent["page_numbers"] = child["parent_page_numbers"]
    return parent
