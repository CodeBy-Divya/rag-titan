"""
BM25: the classic keyword search score (RAG handbook 2026, section 5.2).

    score(q, d) = sum over words t of q:
                  IDF(t) * f(t,d) * (k1 + 1) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))

    f(t,d)  how often word t appears in chunk d
    |d|     number of words in d; avgdl: the average over all chunks
    IDF(t)  ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5)), N chunks, n(t) of them contain t
    k1      how fast repeating a word stops helping (1.2-2.0 is usual)
    b       how much long chunks are penalised (0 = not at all, 1 = fully)

Exact words (ids, names, error codes) score high even when they are rare, which
is where vector search is weak; synonyms and paraphrases score nothing.
"""

import math
import re
from collections import Counter

# runs of letters and digits in any language: "INV-2041" -> "inv", "2041"
WORD = re.compile(r"\w+")


def tokenize(text):
    """Return the lowercase words of a text, in order (repeats kept)."""
    return WORD.findall(text.lower())


class BM25Index:
    """
    Keyword index over a fixed list of texts.

    Args:
        texts: the texts to search, for example chunk texts.
        k1: word-repeat saturation, usually 1.2 to 2.0.
        b: length normalisation, from 0 (none) to 1 (full).
    """

    def __init__(self, texts, k1=1.5, b=0.75):
        if not isinstance(k1, (int, float)) or k1 < 0:
            raise ValueError(f"k1 must be a number of at least 0, got {k1!r}")
        if not isinstance(b, (int, float)) or not 0 <= b <= 1:
            raise ValueError(f"b must be a number from 0 to 1, got {b!r}")

        self.k1, self.b = k1, b
        self.lengths = []
        self.postings = {}          # word -> {text number: times it appears}

        for number, text in enumerate(texts):
            counts = Counter(tokenize(text))
            self.lengths.append(sum(counts.values()))
            for word, count in counts.items():
                self.postings.setdefault(word, {})[number] = count

        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0

    def __len__(self):
        return len(self.lengths)

    def idf(self, word):
        """How rare a word is: high for rare words, near 0 for words in every text."""
        found_in = len(self.postings.get(word, {}))
        return math.log(1 + (len(self) - found_in + 0.5) / (found_in + 0.5))

    def scores(self, query):
        """Return {text number: BM25 score} for every text sharing a word with the query."""
        totals = {}
        for word in set(tokenize(query)):
            idf = self.idf(word)
            for number, count in self.postings.get(word, {}).items():
                # long texts need more repeats for the same score (b), and every
                # extra repeat helps less than the one before (k1)
                length_part = 1 - self.b + self.b * self.lengths[number] / self.average_length
                gain = count * (self.k1 + 1) / (count + self.k1 * length_part)
                totals[number] = totals.get(number, 0.0) + idf * gain
        return totals

    def search(self, query, top_k=5):
        """
        Return up to top_k (text number, score) pairs, best first.

        Texts without any query word are left out; equal scores keep text order.
        """
        totals = self.scores(query)
        best = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
        return best[:top_k]
