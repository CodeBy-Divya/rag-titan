---
hide:
  - toc
---

# Adaptive RAG playground

**Adaptive RAG** decides how much work each question needs, and checks its own
evidence before answering:

```
Question -> Choose route -> Retrieve -> Rerank -> Check evidence
                                                   |-- enough      -> Answer
                                                   |-- not enough  -> Rewrite the question, search wider, check again
                                                   |-- out of tries -> "I don't know"
```

Move any knob below and the question runs again. The diagram lights up the steps
in the order they ran, with the loop in orange; **click a step** to see what went
in and what came out.

<div id="rag-type-app" data-rag-type="adaptive"><p><em>The playground needs JavaScript.</em></p></div>

## The knobs

| Knob | Setting | What it changes |
|---|---|---|
| Chunker | `CHUNKER`, `CHUNK_SIZE` | how the documents are cut |
| Retriever | `RETRIEVER` | `dense`, `bm25` or `hybrid` |
| Reranker | `RERANKER` | `keyword`, `none`, `mmr` or `llm` |
| top_k | `TOP_K` | chunks searched for a *simple* question (default 4). A *complex* question searches 2 × top_k, and every retry adds top_k more |
| top_n | `TOP_N` | chunks kept the same way (default 2; 2 × for complex, + top_n per retry) |
| max attempts | `ADAPTIVE_MAX_ATTEMPTS` | searches before it gives up and says it does not know (default 2) |

Adaptive RAG asks the model up to five times per question (route, check, rewrite,
check again, answer), so more of its steps depend on recorded replies than in
Simple RAG: a run past the recording can stop at *Check evidence* too. The search
steps before it are always live.

## How the answers are made here

The search, the reranking and the scores run **live** in your browser with the
real rag-titan code; the model's replies (`gemma3:4b`) were recorded once and are
replayed. Positions marked **● answers recorded** replay real replies; elsewhere
the run stops at the first model step whose reply was not recorded, drawn
dashed. The code below the result runs the same setup in Python with your own
model. The [Compare page](../rag-playground.md) runs several RAG types side by side.
