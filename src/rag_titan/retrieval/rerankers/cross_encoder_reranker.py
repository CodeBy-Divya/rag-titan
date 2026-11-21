"""
Cross-encoder reranking (RAG handbook 2026, section 04: "joint query-document
pair relevance model").

A vector search embeds the question and each chunk separately, then compares
the two vectors. A cross-encoder reads the question and the chunk together, in
one pass, so it sees how they relate word by word. That is more precise, but
slower, so it only re-orders the few chunks a search already found:

    search: 8 candidates  ->  cross-encoder scores 8 (question, chunk) pairs  ->  best top_n

Needs the optional extra:  pip install "rag-titan[rerank]"  (sentence-transformers).
The model is downloaded from Hugging Face on first use, then runs on your machine.
"""

import math

from rag_titan.retrieval.rerankers.base_reranker import BaseReranker
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)

# small (about 90 MB), fast on a laptop CPU, trained on question -> passage search
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INSTALL_HINT = ('The cross_encoder reranker needs sentence-transformers: '
                'pip install "rag-titan[rerank]"')


class CrossEncoderReranker(BaseReranker):
    """
    Re-orders chunks with a cross-encoder model.

    Args:
        model_name: a Hugging Face cross-encoder, for example
            "cross-encoder/ms-marco-MiniLM-L-6-v2" or "BAAI/bge-reranker-base".
        model: optional, a ready model with predict(pairs) -> scores (tests
            pass a fake one; then sentence-transformers is not needed).
        batch_size: how many pairs the model scores at once.
    """

    def __init__(self, model_name=DEFAULT_RERANK_MODEL, model=None, batch_size=32):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be the name of a cross-encoder model")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a whole number above 0, got {batch_size!r}")
        if model is None:
            _check_installed()          # fail now, not in the middle of the first question

        self.model_name = model_name.strip()
        self.batch_size = batch_size
        self._model = model

    def rerank(self, query, matches, top_n=3):
        """
        Score every (question, chunk) pair in one batch and keep the best top_n.

        Returns:
            Copies of the best matches, best first. "rerank_score" is the
            model's relevance from 0 to 1; the search's "score" is kept.
        """
        if not matches:
            return []

        pairs = [(query, match["text"]) for match in matches]
        raw = self._load().predict(pairs, batch_size=self.batch_size)
        relevance = to_unit_scores([float(value) for value in raw])

        reranked = [{**match, "rerank_score": value} for match, value in zip(matches, relevance)]
        # sort() keeps the search order for equal scores
        reranked.sort(key=lambda match: match["rerank_score"], reverse=True)
        return reranked[:top_n]

    def score(self, query, match):
        """Relevance of one chunk from 0 to 1 (rerank() scores all chunks at once)."""
        return self.rerank(query, [match], top_n=1)[0]["rerank_score"]

    def _load(self):
        """Load the model the first time it is needed (it can take a few seconds)."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model


def to_unit_scores(values):
    """
    Turn model scores into 0-1 relevance, like every other score in the package.

    Some cross-encoders return probabilities (already 0 to 1), others raw logits
    (any number, e.g. -11.2 or 8.4). Logits go through the sigmoid 1 / (1 + e^-x),
    which keeps their order: a higher logit always gives a higher relevance.
    """
    if all(0.0 <= value <= 1.0 for value in values):
        return values
    return [1.0 / (1.0 + math.exp(-max(min(value, 50.0), -50.0))) for value in values]


def _check_installed():
    """Raise a clear ImportError when the [rerank] extra is missing."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError as error:
        raise ImportError(INSTALL_HINT) from error
