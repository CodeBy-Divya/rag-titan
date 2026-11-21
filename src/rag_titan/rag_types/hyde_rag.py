"""
HyDE RAG, inspired by "Precise Zero-Shot Dense Retrieval without Relevance
Labels" (Gao et al. 2022; RAG handbook 2026, section 08):

    Question -> Write a hypothetical answer -> Search with it (and with the question)
             -> Fuse the lists (RRF) -> Rerank -> Generate

A question and its answer are worded differently ("How many days off?" vs
"Employees receive 20 paid leave days"). The model first writes a short made-up
answer passage; searching with it finds chunks that read like the answer. The
made-up text is only used to search: the reranker and the answer always use the
original question, and only real chunks reach the model.

The handbook's warning, "hallucinated hypotheticals can mislead", is why
with_question is on by default: the question is searched too and the two lists
are fused, so a wrong guess cannot take over the results alone.

Simplified from the paper: one made-up passage instead of several, and the two
searches are fused by rank (RRF) instead of averaging their embeddings.
"""

from rag_titan.prompts import HYDE_PROMPT
from rag_titan.rag_types.base_rag import BaseRAG, check_search_sizes
from rag_titan.rag_types.trace import Trace
from rag_titan.retrieval.fusion import search_and_fuse
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)


class HyDERAG(BaseRAG):
    """
    Searches with a made-up answer passage, then answers from the real chunks.

    Args:
        llm: any BaseLLM; writes the made-up passage.
        retriever: any BaseRetriever (dense, bm25 or hybrid).
        reranker: a BaseReranker; it ranks against the original question.
        generator: a BaseGenerator.
        top_k: how many chunks the (fused) search returns.
        top_n: how many chunks the model reads.
        with_question: also search with the question and fuse both lists
            (default True); False searches with the made-up passage only.
    """

    DEFAULT_TOP_K, DEFAULT_TOP_N = 5, 2

    FLOW = {
        "nodes": [
            {"id": "hypothesize", "label": "Write hypothetical answer", "kind": "llm"},
            {"id": "retrieve", "label": "Search", "kind": "search"},
            {"id": "fuse_queries", "label": "Fuse lists (RRF)", "kind": "search"},
            {"id": "rerank", "label": "Rerank", "kind": "search"},
            {"id": "generate", "label": "Generate answer", "kind": "output"},
        ],
        # retrieve -> retrieve: the second search (the question); without it,
        # a single search goes straight to rerank
        "edges": [["hypothesize", "retrieve"], ["retrieve", "retrieve"],
                  ["retrieve", "fuse_queries"], ["fuse_queries", "rerank"],
                  ["retrieve", "rerank"], ["rerank", "generate"]],
    }

    def __init__(self, llm, retriever, reranker, generator, top_k=DEFAULT_TOP_K,
                 top_n=DEFAULT_TOP_N, with_question=True):
        check_search_sizes(top_k, top_n)
        if not isinstance(with_question, bool):
            raise ValueError(f"with_question must be True or False, got {with_question!r}")
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.top_k = top_k
        self.top_n = top_n
        self.with_question = with_question

    def run(self, query, trace=None):
        """Answer the question from chunks found with a made-up answer, recording each step."""
        query = query.strip()
        trace = trace if trace is not None else Trace()

        if not query:
            return trace.result("Please enter a question.")

        if not self.retriever.has_documents():
            return trace.result("Please index your documents first.")

        passage = trace.record("hypothesize", self.hypothesize, query=query)
        matches = search_and_fuse(self.retriever, self.search_queries(query, passage),
                                  self.top_k, trace)
        best = trace.record("rerank", self.reranker.rerank,
                            query=query, matches=matches, top_n=self.top_n)

        answer = trace.record("generate", self.generator.generate, query=query, chunks=best)
        return trace.result(answer, best)

    def hypothesize(self, query):
        """Ask the model for a short made-up passage that would answer the question."""
        return (self.llm.ask(HYDE_PROMPT, query) or "").strip()

    def search_queries(self, query, passage):
        """
        What to search with: the passage first, then the question (with_question).

        An empty passage (the model gave nothing) falls back to the question
        alone, so the run still finds real chunks.
        """
        if not passage:
            logger.warning("HyDE: the model wrote no passage, searching with the question")
            return [query]
        return [passage, query] if self.with_question else [passage]
