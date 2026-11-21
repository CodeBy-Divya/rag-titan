"""
Reciprocal Rank Fusion (RRF): merge several ranked result lists into one
(Cormack, Clarke and Buettcher 2009; RAG handbook 2026, section 5.3).

    RRF(d) = sum over the lists r that found d:  1 / (k + rank_r(d))

Only ranks count, never raw scores, so lists from different searches (vector
and keyword, or one search per rewording of a question) can be merged. A chunk
found near the top of many lists wins.

Used by hybrid search (vector + keyword lists) and by the RAG types that search
several times: HyDE (a made-up answer and the question) and RAG-Fusion
(several rewordings of the question), through search_and_fuse().
"""

from rag_titan.retrieval.dedupe import remove_duplicates

# the constant of the original paper: a large k keeps one top rank from deciding everything
RRF_K = 60


def rrf_fuse(lists, k=RRF_K):
    """
    Merge ranked lists with RRF.

    Args:
        lists: {name: [chunks, best first]}, e.g. {"dense": [...], "bm25": [...]}.
            Each chunk has "id" and "score" (0 to 1).
        k: the RRF constant.

    Returns:
        A list of (chunk, score_trace) pairs, best fused score first. The trace
        holds, per list that found the chunk, its score and its rank
        ({"dense": 0.71, "dense_rank": 2, ...}), plus the summed "rrf".
        Equal RRF scores are common (first and second place in either order),
        so they are broken by the 0-1 relevance summed over the lists, then by
        the order the chunks were first seen.
    """
    chunks, traces, relevance, first_seen = {}, {}, {}, {}
    for name, results in lists.items():
        for rank, result in enumerate(results, start=1):
            key = result["id"]
            first_seen.setdefault(key, len(first_seen))
            chunks.setdefault(key, _plain_chunk(result))
            trace = traces.setdefault(key, {"rrf": 0.0})
            trace[name] = result.get("score_trace", {}).get(name, result["score"])
            trace[f"{name}_rank"] = rank
            trace["rrf"] += 1 / (k + rank)
            relevance[key] = relevance.get(key, 0.0) + result["score"]

    # rounding makes "equal" robust to float noise in the sums of 1 / (k + rank)
    order = sorted(chunks, key=lambda key: (-round(traces[key]["rrf"], 12), -relevance[key],
                                            first_seen[key]))
    return [(chunks[key], traces[key]) for key in order]


def fuse_ranked(lists, top_k, k=RRF_K, dedupe_overlap=0.5):
    """
    Fuse lists, drop near-copies, and keep the top_k, scored from 0 to 1.

    "score" is the RRF score divided by the best possible one (first in every
    list), so a chunk that tops every list scores 1.0. Near-copies are removed
    after fusing: removed before, each list could keep a different copy of the
    same passage, and the passage would lose the votes of the other lists.

    Returns:
        Chunk dicts with "score" and "score_trace" (see rrf_fuse), best first.
    """
    best_possible = len(lists) / (k + 1)
    matches = [{**chunk, "score": trace["rrf"] / best_possible, "score_trace": trace}
               for chunk, trace in rrf_fuse(lists, k)]
    if dedupe_overlap is not None:
        matches = remove_duplicates(matches, dedupe_overlap)
    return matches[:top_k]


def search_and_fuse(retriever, queries, top_k, trace, pool_size=None, k=RRF_K,
                    dedupe_overlap=0.5):
    """
    Search once per query, then fuse the result lists: the core of HyDE and RAG-Fusion.

    Every search is recorded in `trace` (a "retrieve" step, or the retriever's
    own steps, e.g. hybrid's), then one "fuse_queries" step merges them. Each
    result's score_trace shows its rank for every query ("query_1_rank": 2, ...).

    Args:
        retriever: any BaseRetriever.
        queries: the texts to search with, e.g. [question, made-up answer].
        top_k: how many fused chunks to return.
        trace: the Trace of the RAG run.
        pool_size: results per search before fusing; None uses max(20, 4 * top_k).
        k: the RRF constant (see fuse_ranked()).
        dedupe_overlap: drop near-copies after fusing (see fuse_ranked()).

    Returns:
        The top_k fused chunks. With one query: that search's own top_k, no fusing.
    """
    queries = [query for query in queries if isinstance(query, str) and query.strip()]
    if not queries:
        raise ValueError("search_and_fuse needs at least one query with text")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"top_k must be a whole number of at least 1, got {top_k!r}")
    if len(queries) == 1:
        return trace.search(retriever, query=queries[0], top_k=top_k)

    pool = pool_size or max(20, 4 * top_k)
    lists = {f"query_{number}": trace.search(retriever, query=query, top_k=pool)
             for number, query in enumerate(queries, start=1)}
    return trace.record("fuse_queries", fuse_ranked, lists=lists, top_k=top_k, k=k,
                        dedupe_overlap=dedupe_overlap)


def _plain_chunk(result):
    """The chunk without the scores one retriever added."""
    return {key: value for key, value in result.items()
            if key not in ("score", "retriever", "score_trace")}
