"""
Shared base of the RAG types that change *what is searched for* (HyDE, RAG-Fusion):

    Question -> make search queries (one LLM call) -> search each one
             -> fuse the lists (RRF) -> Rerank -> Generate

A subclass only says how it makes its queries (search_queries()); the search,
the fusing, the reranking and the answer are the same for all of them. The
reranker and the answer always use the original question.
"""

from rag_titan.rag_types.base_rag import BaseRAG, check_search_sizes
from rag_titan.rag_types.trace import Trace
from rag_titan.retrieval.fusion import search_and_fuse


class QueryTransformRAG(BaseRAG):
    """
    Searches with several queries made from the question, then answers once.

    Args:
        llm: any BaseLLM; makes the queries.
        retriever: any BaseRetriever (dense, bm25 or hybrid).
        reranker: a BaseReranker; it ranks against the original question.
        generator: a BaseGenerator.
        top_k: how many chunks the fused search returns.
        top_n: how many chunks the model reads.
    """

    DEFAULT_TOP_K, DEFAULT_TOP_N = 5, 2

    def __init__(self, llm, retriever, reranker, generator, top_k=DEFAULT_TOP_K,
                 top_n=DEFAULT_TOP_N):
        check_search_sizes(top_k, top_n)
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.top_k = top_k
        self.top_n = top_n

    def run(self, query, trace=None):
        """Answer the question from the fused searches, recording each step."""
        query = query.strip()
        trace = trace if trace is not None else Trace()

        if not query:
            return trace.result("Please enter a question.")

        if not self.retriever.has_documents():
            return trace.result("Please index your documents first.")

        matches = search_and_fuse(self.retriever, self.search_queries(query, trace),
                                  self.top_k, trace)
        best = trace.record("rerank", self.reranker.rerank,
                            query=query, matches=matches, top_n=self.top_n)

        answer = trace.record("generate", self.generator.generate, query=query, chunks=best)
        return trace.result(answer, best)

    def search_queries(self, query, trace):
        """
        Return the texts to search with, recording the LLM step that made them.

        Never empty: when the model gives nothing usable, return [query].
        """
        raise NotImplementedError
