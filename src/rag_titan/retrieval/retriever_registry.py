"""
Pick the retriever and the reranker named in the settings.

Adding one: create its class, add its name to RETRIEVERS or RERANKERS and one
`if name == ...` line below.
"""

from rag_titan.retrieval.bm25_retriever import BM25Retriever
from rag_titan.retrieval.dense_retriever import DenseRetriever
from rag_titan.retrieval.hybrid_retriever import HybridRetriever
from rag_titan.retrieval.rerankers.keyword_reranker import KeywordReranker
from rag_titan.retrieval.rerankers.mmr_reranker import MMRReranker
from rag_titan.retrieval.rerankers.no_reranker import NoReranker

# every name build_retriever() / build_reranker() knows, used by the CLI and errors
RETRIEVERS = ["dense", "bm25", "hybrid"]
RERANKERS = ["keyword", "none", "mmr"]


def build_retriever(settings, embedder, store):
    """
    Create the retriever named in settings["retriever"].

    "dense" finds chunks by meaning, "bm25" by exact words, "hybrid" uses both
    and fuses the rankings (RRF). All of them search the same store.

    Args:
        settings: the dict from get_settings().
        embedder: the BaseEmbedder that indexed the store.
        store: the BaseStore holding the chunks.
    """
    name = _name(settings, "retriever")
    dedupe = settings["dedupe_overlap"]

    dense = DenseRetriever(embedder, store, dedupe_overlap=dedupe)
    if name == "dense":
        return dense
    if name == "bm25":
        return BM25Retriever(store, dedupe_overlap=dedupe)
    if name == "hybrid":
        return HybridRetriever(dense, BM25Retriever(store, dedupe_overlap=dedupe))

    raise ValueError(f"Unknown retriever: {settings['retriever']!r}. "
                     f"Use one of: {', '.join(RETRIEVERS)}")


def build_reranker(settings, llm=None, embedder=None):
    """
    Create the reranker named in settings["reranker"].

    Args:
        settings: the dict from get_settings().
        llm: not used yet; kept for future LLM-based rerankers.
        embedder: needed by "mmr", which compares chunks with each other.
    """
    name = _name(settings, "reranker")

    if name == "keyword":
        return KeywordReranker()
    if name == "none":
        return NoReranker()
    if name == "mmr":
        if embedder is None:
            raise ValueError("The mmr reranker needs the embedder: build_reranker(settings, "
                             "embedder=...)")
        return MMRReranker(embedder)

    raise ValueError(f"Unknown reranker: {settings['reranker']!r}. "
                     f"Use one of: {', '.join(RERANKERS)}")


def _name(settings, key):
    """The setting's value as a clean lowercase name ("Hybrid " -> "hybrid")."""
    return str(settings[key]).strip().lower()
