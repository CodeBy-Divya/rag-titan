"""
Adaptive RAG:

Question -> Choose route -> Retrieve -> Rerank -> Check evidence
                                                  |
                                  Enough evidence -> Answer
                                                  |
                              Not enough -> Rewrite and retry
"""

from rag_titan.prompts import DIRECT_REPLY_PROMPT, NO_ANSWER
from rag_titan.rag_types.adaptive.evidence_checker import EvidenceChecker
from rag_titan.rag_types.adaptive.query_rewriter import QueryRewriter
from rag_titan.rag_types.adaptive.router import Router
from rag_titan.rag_types.base_rag import BaseRAG, check_count, check_search_sizes
from rag_titan.rag_types.trace import Trace
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)

# a complex question starts with a search this many times bigger than a simple one
COMPLEX_FACTOR = 2


class AdaptiveRAG(BaseRAG):
    """
    A RAG that changes its effort based on the question.

    Greetings get a direct reply, simple questions a small search,
    complex questions a bigger search. When the evidence is weak,
    the question is rewritten and the search is tried again.

    Args:
        llm: any BaseLLM, used for routing, checking and rewriting.
        retriever: any BaseRetriever, for example a DenseRetriever.
        reranker: a BaseReranker.
        generator: a BaseGenerator.
        max_attempts: how many searches to try before giving up.
        top_k: chunks searched for a simple question; a complex question
            searches 2x as many, and every retry adds top_k more.
        top_n: chunks kept for a simple question (2x for complex, +top_n per retry).
    """

    FLOW = {
        "nodes": [
            {"id": "route", "label": "Choose route", "kind": "llm"},
            {"id": "direct_reply", "label": "Direct reply", "kind": "output"},
            {"id": "retrieve", "label": "Retrieve", "kind": "search"},
            {"id": "rerank", "label": "Rerank", "kind": "search"},
            {"id": "check", "label": "Check evidence", "kind": "llm"},
            {"id": "rewrite", "label": "Rewrite query", "kind": "llm"},
            {"id": "generate", "label": "Generate answer", "kind": "output"},
            {"id": "give_up", "label": "No answer", "kind": "output"},
        ],
        "edges": [
            ["route", "direct_reply"], ["route", "retrieve"],
            ["retrieve", "rerank"], ["rerank", "check"],
            ["check", "generate"], ["check", "rewrite"], ["check", "give_up"],
            ["rewrite", "retrieve"],
        ],
    }

    # sizes of a simple question's first search (TOP_K / TOP_N change these)
    DEFAULT_TOP_K, DEFAULT_TOP_N = 4, 2
    DEFAULT_MAX_ATTEMPTS = 2

    def __init__(self, llm, retriever, reranker, generator, max_attempts=DEFAULT_MAX_ATTEMPTS,
                 top_k=DEFAULT_TOP_K, top_n=DEFAULT_TOP_N):
        check_count("max_attempts", max_attempts)
        check_search_sizes(top_k, top_n)
        self.top_k = top_k
        self.top_n = top_n
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.max_attempts = max_attempts

        self.router = Router(llm)
        self.evidence_checker = EvidenceChecker(llm)
        self.query_rewriter = QueryRewriter(llm)

    def run(self, query, trace=None):
        """Answer the question from the indexed documents and record each step."""
        query = query.strip()
        trace = trace if trace is not None else Trace()

        if not query:
            return trace.result("Please enter a question.")

        route = trace.record("route", self.router.route, query=query)
        logger.info("Selected route: %s", route)

        if route == "direct":
            reply = trace.record("direct_reply", self.llm.ask,
                                 instructions=DIRECT_REPLY_PROMPT, user_input=query)
            return trace.result(reply)

        if not self.retriever.has_documents():
            return trace.result("Please index your documents first.")

        top_k, top_n = self.search_sizes(route)
        return self._search_and_answer(query, top_k, top_n, trace)

    def search_sizes(self, route):
        """(top_k, top_n) of the first search: bigger for a "complex" question."""
        factor = COMPLEX_FACTOR if route == "complex" else 1
        return self.top_k * factor, self.top_n * factor

    def _search_and_answer(self, query, top_k, top_n, trace):
        """Search, check the evidence, and retry with a rewritten query."""
        # the original question never changes, only the search query does
        search_query = query

        for attempt in range(1, self.max_attempts + 1):
            logger.info("Search attempt %d: '%s'", attempt, search_query)

            matches = trace.search(self.retriever, query=search_query, top_k=top_k)
            chunks = trace.record("rerank", self.reranker.rerank,
                                  query=query, matches=matches, top_n=top_n)
            enough = trace.record("check", self.evidence_checker.has_enough,
                                  query=query, chunks=chunks)

            if enough:
                logger.info("Enough evidence found")
                answer = trace.record("generate", self.generator.generate,
                                      query=query, chunks=chunks)
                return trace.result(answer, chunks)

            if attempt == self.max_attempts:
                break

            logger.info("Not enough evidence, rewriting and widening the search")
            search_query = trace.record("rewrite", self.query_rewriter.rewrite, query=query)
            top_k += self.top_k
            top_n += self.top_n

        trace.add("give_up", {"attempts": self.max_attempts}, NO_ANSWER)
        return trace.result(NO_ANSWER)
