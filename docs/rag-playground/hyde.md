---
hide:
  - toc
---

# HyDE RAG playground

**HyDE RAG** asks the model for a short **made-up answer** first, then searches
with it, because an answer reads more like the documents than a question does:

```
Question -> Write a hypothetical answer -> Search with it -> Search with the question
         -> Fuse the two lists (RRF) -> Rerank -> Generate the answer
```

The made-up text is only used to search: the reranker and the answer use your
question, and only real chunks reach the model. Move any knob below and the
question runs again; **click "Write hypothetical answer"** to read what the model
made up, and **"Fuse lists"** to see each chunk's rank in both searches.

<div id="rag-type-app" data-rag-type="hyde"><p><em>The playground needs JavaScript.</em></p></div>

## Things to try

Every result below is a real recorded run you can repeat with the knobs.

1. **Read what the model made up.** Pick *What is the maximum hotel cost per
   night?*, chunker `word`, retriever `hybrid`, and click **Write hypothetical
   answer**: the model wrote that the maximum is **$350** at "designated hotels".
   The policy says 150 euros. The made-up text is only used to search; it is never
   given to the model as a source.
2. **When fusing hurts.** Same setup, with *search the question too* **on** (the
   default): the 150-euro chunk of `expenses.md` is found only by the question's
   search, so chunks found by **both** searches rank above it. It comes 5th of 5
   in *Fuse lists*, the reranker keeps two others, and the model says it does not
   know. Turn the knob **off**: the made-up answer's own search puts that chunk
   first and the answer is *"Hotel costs are refunded up to 150 euros per night."*
   Reciprocal Rank Fusion rewards agreement between the lists, even over a clear
   first place in one of them.
3. **The big picture.** Over all 54 recorded setups (3 chunkers × 3 retrievers ×
   6 questions) the chunks given to the model held all the evidence in 45 for
   Simple RAG, 44 for HyDE with the question searched too, and 48 with the made-up
   answer alone. These short policies use the same words as the questions, so
   there is little for HyDE to fix, and six questions are too few to call a
   winner. HyDE is meant for questions worded far from the documents: measure it
   on yours with `rag-titan evaluate --rag-type simple hyde`.

## The knobs

| Knob | Setting | What it changes |
|---|---|---|
| Chunker | `CHUNKER`, `CHUNK_SIZE` | how the documents are cut |
| Retriever | `RETRIEVER` | `dense`, `bm25` or `hybrid`; each search uses it |
| Reranker | `RERANKER` | orders the fused chunks against your question |
| top_k | `TOP_K` | chunks the fused search returns (default 5) |
| top_n | `TOP_N` | chunks the model reads (default 2) |
| search the question too | `HYDE_WITH_QUESTION` | on (default): the question's search is fused in, so a wrong guess cannot take over; off: the made-up answer alone, like the paper |

## How the answers are made here

The searches, the fusing, the reranking and the scores run **live** in your
browser with the real rag-titan code; the model's replies (`gemma3:4b`) were
recorded once and are replayed. Replies were recorded with the defaults and with
one knob at a time on the values listed under it; the line under the knobs says
**● Answers recorded** when they replay. Elsewhere the run stops at the first model
step whose reply was not recorded, drawn dashed, and shows exactly what the model
would read. The code below the result runs the same setup in Python. The
[Compare page](../rag-playground.md) runs several RAG types side by side.
