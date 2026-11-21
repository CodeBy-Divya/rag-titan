"""
Maximal Marginal Relevance: pick chunks that are relevant but not repeats of
each other (RAG handbook 2026, sections 3.05 and 5.5).

    next = argmax over chunks d not picked yet:
           lambda * relevance(d) - (1 - lambda) * max similarity(d, s) for s picked

lambda = 1 keeps the retriever's order; lower values swap a near-copy of an
already picked chunk for a chunk that adds something new, so the few chunks
given to the LLM cover more of the question.
"""

import numpy as np

from rag_titan.retrieval.rerankers.base_reranker import BaseReranker

# texts embedded before are remembered (chunks come back often), up to this many
CACHE_SIZE = 10_000


class MMRReranker(BaseReranker):
    """
    Re-orders chunks for relevance and variety.

    Args:
        embedder: the BaseEmbedder used to compare chunks with each other.
        lambda_: weight of relevance from 0 to 1 (0.7: mostly relevance,
            some variety).
    """

    def __init__(self, embedder, lambda_=0.7):
        if not isinstance(lambda_, (int, float)) or not 0 <= lambda_ <= 1:
            raise ValueError(f"lambda_ must be a number from 0 to 1, got {lambda_!r}")
        self.embedder, self.lambda_ = embedder, lambda_
        self._vectors = {}

    def rerank(self, query, matches, top_n=3):
        """
        Pick top_n matches one by one with MMR.

        Returns:
            The picked matches in the order they were picked. "rerank_score"
            is each one's MMR value when it was picked.
        """
        if not matches:
            return []
        similar = self._similarities([match["text"] for match in matches])
        relevance = [match["score"] for match in matches]
        picked, left, values = [], list(range(len(matches))), []

        while left and len(picked) < top_n:
            def mmr(i):
                repeat = max((similar[i][j] for j in picked), default=0.0)
                return self.lambda_ * relevance[i] - (1 - self.lambda_) * repeat

            best = max(left, key=mmr)        # the first of equal values wins
            # a plain float: numpy's float32 cannot be saved as JSON (the playground's results)
            values.append(float(mmr(best)))
            picked.append(best)
            left.remove(best)

        return [{**matches[i], "rerank_score": value} for i, value in zip(picked, values)]

    def score(self, query, match):
        """The retriever's relevance (MMR itself needs all matches, see rerank)."""
        return match["score"]

    def _similarities(self, texts):
        """Return the cosine similarity of every pair of texts as a matrix."""
        missing = [text for text in dict.fromkeys(texts) if text not in self._vectors]
        if missing:
            if len(self._vectors) + len(missing) > CACHE_SIZE:
                self._vectors.clear()
            for text, vector in zip(missing, self.embedder.embed_many(missing)):
                self._vectors[text] = np.asarray(vector, dtype=np.float32)

        matrix = np.stack([self._vectors[text] for text in texts])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        unit = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
        return unit @ unit.T
