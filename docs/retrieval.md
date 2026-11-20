# Retrieval: dense, BM25 and hybrid

The retriever finds the chunks the model will read. rag-titan has three, chosen
with one setting, and every RAG type works with all of them.

| `RETRIEVER` | Finds chunks by | Strong at | Weak at |
|-------------|-----------------|-----------|---------|
| `dense` (default) | meaning: the question's vector vs each chunk's vector | other words for the same thing ("notebook" for "computer") | exact ids, codes and rare names |
| `bm25` | exact words, scored with BM25 | ids, error codes, names, numbers | questions in other words: finds nothing |
| `hybrid` | both searches, their rankings fused (RRF) | a mix of both kinds of question | costs both searches (BM25 itself is about 0.1 ms) |

**Start with `hybrid`** when your questions mix natural language with ids,
product names or codes; the RAG handbook calls it "a strong first default".
Then [measure it](#compare-them-on-your-own-questions) on your own questions.

## Use it

=== "Python"

    ```python
    from rag_titan import RAGTitan

    rag = RAGTitan(settings={"retriever": "hybrid"})
    rag.index(["handbook.pdf"])
    print(rag.ask("What does error E-4022 mean?"))
    ```

=== ".env"

    ```bash
    RETRIEVER=hybrid        # dense, bm25 or hybrid
    RERANKER=mmr            # optional: keyword (default), none or mmr
    ```

=== "Command line"

    ```bash
    rag-titan ask --files handbook.pdf --question "What does error E-4022 mean?" \
        --retriever hybrid --trace
    ```

The retriever only changes how the index is searched, never what is stored, so
you can switch it at any time without indexing again (also with a
[saved vector store](vector-stores.md)).

## A complete example

[`examples/compare_retrievers.py`](https://github.com/aion7891/rag-titan/blob/main/examples/compare_retrievers.py)
indexes a small IT help desk ([`examples/retrieval_docs/`](https://github.com/aion7891/rag-titan/tree/main/examples/retrieval_docs):
VPN and account error codes, lost devices, new starters, printing) and asks six
questions: three about exact error codes (*What does error E-4022 mean?*) and
three in other words than the documents (*Somebody took my notebook from the
train. Who do I tell?*, where the policy says "company computer ... goes missing").

```python
"""
Compare the dense, bm25 and hybrid retrievers on a small IT help desk, then look
inside one hybrid search and answer with it.

The questions in examples/retrieval_questions.json are of two kinds: three ask
about exact error codes (E-4012, ...), three use other words than the documents
("notebook" for "computer", "copier" for "printer").

Before running (Ollama installed and running):
    ollama pull gemma3:4b
    ollama pull nomic-embed-text

Run (from the repo root):
    python examples/compare_retrievers.py
"""

from pathlib import Path

from rag_titan import RAGTitan
from rag_titan.evaluation import evaluate_retriever, format_report, load_questions

HERE = Path(__file__).parent
files = sorted((HERE / "retrieval_docs").glob("*.md"))
questions = load_questions(HERE / "retrieval_questions.json")

# 1. Score each retriever: is the right file the first result? (no LLM needed)
results = []
for retriever in ["dense", "bm25", "hybrid"]:
    rag = RAGTitan(settings={"retriever": retriever, "chunker": "paragraph"})
    rag.index(files)
    results.append(evaluate_retriever(rag.parts["retriever"], questions, top_k=1,
                                      name=retriever))
print(format_report(results))
for result in results:
    missed = [row["id"] for row in result["rows"] if row["recall"] < 1]
    print(f"  {result['name']:7} missed: {', '.join(missed) or 'nothing'}")

# 2. Look inside one hybrid search: each chunk's rank in both searches
question = "What does error E-4022 mean?"
print(f"\nHybrid search for: {question}")
for chunk in rag.parts["retriever"].retrieve(question, top_k=2):
    trace = chunk["score_trace"]
    print(f"  {chunk['source']:18} score {chunk['score']:.2f}  "
          f"vector #{trace.get('dense_rank', '-')}  keyword #{trace.get('bm25_rank', '-')}")

# 3. Answer with the hybrid RAG (this step uses the LLM)
print(f"\nAnswer: {rag.ask(question)}")
```

What it printed with `nomic-embed-text` and `gemma3:4b`:

```text
Retrieval report: 6 questions, top_k=1

retriever  Recall@1       MRR    nDCG@1    p50 ms    p95 ms
-----------------------------------------------------------
dense         0.833     0.833     0.833       6.8       8.1
bm25          0.500     0.500     0.500       0.0       0.1
hybrid        1.000     1.000     1.000       6.5       7.0
  dense   missed: code_1
  bm25    missed: meaning_1, meaning_2, meaning_3
  hybrid  missed: nothing

Hybrid search for: What does error E-4022 mean?
  account_errors.md  score 0.99  vector #2  keyword #1
  vpn_errors.md      score 0.98  vector #1  keyword #4

Answer: Error E-4022: your password is older than 90 days. Choose a new password in the self-service portal before you can sign in again. [chunk_2]
```

What happened:

- **dense** missed `code_1`: *E-4022* and *E-4012* look alike as vectors, so it
  ranked the VPN errors first.
- **bm25** found every error code but nothing at all for the three reworded
  questions: none of their real words ("notebook", "train", "copier") is in the
  documents.
- **hybrid** got all six first. For *E-4022* the keyword search put
  `account_errors.md` first and the vector search second; the VPN errors were
  first for vectors but only fourth for keywords, so the account errors win.

Six questions show the idea, they do not prove a winner: on your own documents
the numbers will differ, so [compare them on your own questions](#compare-them-on-your-own-questions).

## How each one works

### BM25: exact words

For each word of the question that a chunk contains (RAG handbook 2026, §5.2):

```text
score(q, d) = Σ IDF(t) · f(t,d) · (k1 + 1) / ( f(t,d) + k1 · (1 − b + b · |d| / avgdl) )

IDF(t) = ln(1 + (N − n(t) + 0.5) / (n(t) + 0.5))
```

| Symbol | Meaning | rag-titan |
|--------|---------|-----------|
| `f(t,d)` | how often word `t` is in chunk `d` | |
| `IDF(t)` | rare words count more; `N` chunks, `n(t)` of them contain `t` | never negative |
| `k1` | every repeat of a word helps less than the one before | `1.5` |
| `b` | long chunks need more repeats for the same score | `0.75` |

Words are lowercase runs of letters and digits in any language, so *INV-2041*
becomes `inv` and `2041`. Common English words (*the, my, how, who, ...*) are
skipped, as the English analyzers of Lucene and Elasticsearch do: they are in
every text, so matching them is noise. For other languages pass your own list,
`BM25Retriever(store, stopwords={"der", "die", "das", ...})`, or `stopwords=()`
to score every word.

The keyword index is built from the chunks in your vector store on the first
search and rebuilt by itself when the store changes, so it always matches what
is indexed. Measured with 100,000 chunks: building it takes about 3 to 4
seconds once, a search about 1 ms (rare words) to 100 ms (common words).

### Hybrid: Reciprocal Rank Fusion

Both searches return up to `max(20, 4 × top_k)` chunks; each chunk then gets
(RAG handbook 2026, §5.3, with `k = 60` from the original paper):

```text
RRF(d) = Σ over both lists  1 / (k + rank(d))
```

Only ranks are used, so a BM25 score of 7 and a cosine of 0.7 never have to be
compared, and a chunk found by both searches beats a chunk found by one. Every
result says how it was made:

```python
{"rrf": 0.0325, "dense": 0.651, "dense_rank": 2, "bm25": 3.616, "bm25_rank": 1}
```

Two details matter in practice:

- **Near-copies are removed after fusing.** Overlapping chunks repeat text. If
  each search dropped its own near-copies first, vector search could keep one
  copy of a passage and keyword search the other, and the passage would lose
  the vote of both searches. rag-titan fuses first, then removes near-copies.
- **Ties go to the stronger evidence.** First place in one list plus second in
  the other gives exactly the same RRF score either way round; the chunk with
  more relevance in total wins.

### MMR: relevant but not repetitive

`RERANKER=mmr` picks the chunks for the model one by one (RAG handbook 2026, §5.5):

```text
next = argmax  λ · relevance(d) − (1 − λ) · max similarity(d, chunk already picked)
```

With `λ = 0.7` it mostly follows relevance, but swaps a near-copy of an already
picked chunk for one that adds something new, so the few chunks the model reads
cover more of the question. `λ = 1` keeps the retriever's order.

## Scores from 0 to 1

Every retriever gives each chunk a `score` from 0 to 1, so rerankers work the
same way with all of them. The original score is kept in `score_trace`:

| Retriever | `score` | `score_trace` |
|-----------|---------|---------------|
| dense | cosine similarity | `{"dense": 0.71}` |
| bm25 | BM25 score ÷ the best one for this question | `{"bm25": 6.3}` |
| hybrid | RRF ÷ the best possible (first in both lists) | ranks, scores and `rrf` |

## Compare them on your own questions

The fast way, without the model, scores whether the right files are found:

```bash
rag-titan evaluate --files docs/*.md --questions questions.json \
    --retriever dense bm25 hybrid --retrieval-only --top-k 3
```

Without `--retrieval-only` every retriever also answers, combined with any RAG
types and chunkers (`--rag-type simple adaptive --chunker word paragraph`). Setups
that only differ in retriever share one index, so the files are embedded once.
The questions file format is in [Evaluating RAG](evaluation.md). In the
[RAG playground](rag-playground.md) you can tick the retrievers and watch hybrid
search run as three steps: *Vector search → Keyword search → Fuse ranks (RRF)*.

## What is standard and what is simplified

BM25, RRF and MMR follow the standard formulas above (RAG handbook 2026, §5).
rag-titan keeps them small and readable:

- one tokenizer for every language and a short English stopword list; no
  stemming, so *invoice* and *invoices* are different words;
- the keyword index lives in memory and is built from the store's chunks (fine
  up to about 100,000 chunks); the native hybrid search of Qdrant, Weaviate or
  pgvector is not used yet;
- fixed `k = 60`, `k1 = 1.5`, `b = 0.75` and `λ = 0.7`, all changeable in Python
  (`HybridRetriever(..., k=...)`, `BM25Retriever(..., k1=..., b=...)`,
  `MMRReranker(embedder, lambda_=...)`).
