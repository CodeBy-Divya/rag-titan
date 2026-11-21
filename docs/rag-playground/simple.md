---
hide:
  - toc
---

# Simple RAG playground

**Simple RAG** searches once and answers once:

```
Question -> Retrieve top_k chunks -> Rerank, keep top_n -> Generate the answer
```

Move any knob below and the question runs again. The diagram lights up the steps
in the order they ran; **click a step** to see what went in and what came out.

<div id="rag-type-app" data-rag-type="simple"><p><em>The playground needs JavaScript.</em></p></div>

## The knobs

| Knob | Setting | What it changes |
|---|---|---|
| Chunker | `CHUNKER`, `CHUNK_SIZE` | how the documents are cut: words, sentences or paragraphs |
| Retriever | `RETRIEVER` | `dense` finds meaning, `bm25` exact words, `hybrid` both (fused) |
| Reranker | `RERANKER` | `keyword` rewards shared words, `none` keeps the search order, `mmr` avoids near-copies, `llm` lets the model rate each chunk |
| top_k | `TOP_K` | how many chunks the search finds (default 5) |
| top_n | `TOP_N` | how many of them reach the model (default 2, at most top_k) |

The code under the result builds the same setup in Python, ready to copy.

## How the answers are made here

The search, the reranking and the scores run **live** in your browser, for any
value, with the real rag-titan code. The model's answers are **real replies**
(`gemma3:4b`), recorded once and replayed. A reply exists only for the exact
prompt that was recorded, so:

- a slider position marked **● answers recorded** replays the real answer;
- at any other position the run stops at *Generate answer*, drawn dashed, and
  shows **exactly what the model would read**, with its evidence score. Run that
  setup in Python (the code below the result) for the real answer.

The `cross_encoder` reranker needs PyTorch, which cannot run in a browser: try it in
Python with `pip install "rag-titan[rerank]"`. See also the
[Compare page](../rag-playground.md) to run several RAG types side by side.
