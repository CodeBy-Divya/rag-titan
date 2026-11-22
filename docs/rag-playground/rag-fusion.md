---
hide:
  - toc
---

# RAG-Fusion playground

**RAG-Fusion** asks the model for a few **rewordings** of your question, searches
with each of them and with the question, and **fuses** the result lists, so a
chunk found by several wordings comes first:

```
Question -> Write N search queries -> Search the question -> Search each query
         -> Fuse the lists (RRF) -> Rerank -> Generate the answer
```

Move any knob below and the question runs again; **click "Write search queries"**
to read the rewordings, **"Search"** to see what each one found, and **"Fuse
lists"** to see each chunk's rank for every query.

<div id="rag-type-app" data-rag-type="rag_fusion"><p><em>The playground needs JavaScript.</em></p></div>

## Things to try

Every result below is a real recorded run you can repeat with the knobs.

1. **More wordings, more files.** Pick *What do I get for working from home:
   equipment and money for furniture?*, chunker `paragraph`, retriever **`bm25`**.
   The model writes *"work from home benefits"*, *"remote work allowances"* and
   *"home office stipends"*. Click **Fuse lists**: the `expenses.md` chunk with the
   furniture money is in the top 3 of all four searches, so it is kept, and the
   answer has the 300 euros. Simple RAG with the same knobs gives the model two
   `remote_work.md` chunks only. With keyword search RAG-Fusion found all the
   evidence in 18 of the 18 recorded setups, Simple RAG in 15.
2. **Consistent beats first.** Pick *When do I need a note from the doctor?*,
   chunker `word`, retriever `hybrid`. In **Fuse lists** the right `leave_policy.md`
   chunk is first for three of the four searches, but an `expenses.md` chunk is
   third or fourth in **all four**, and that adds up to more: the model gets two wrong
   chunks and says it does not know (Simple RAG answers this one). Set **queries**
   to **5**: with more votes the answer comes back. Reciprocal Rank Fusion rewards
   being found by every search more than being first in most of them.
3. **The big picture.** Over all 54 recorded setups the chunks given to the model
   held all the evidence in 45 for Simple RAG and in 46, 47 and 48 for RAG-Fusion
   with 1, 3 and 5 queries. Each extra query is one more search and the model's
   rewordings can drift, so measure it on your own questions:
   `rag-titan evaluate --rag-type simple rag_fusion --fusion-queries 5`.

## The knobs

| Knob | Setting | What it changes |
|---|---|---|
| Chunker | `CHUNKER`, `CHUNK_SIZE` | how the documents are cut |
| Retriever | `RETRIEVER` | `dense`, `bm25` or `hybrid`; every search uses it |
| Reranker | `RERANKER` | orders the fused chunks against your question |
| top_k | `TOP_K` | chunks the fused search returns (default 5) |
| top_n | `TOP_N` | chunks the model reads (default 2) |
| queries | `FUSION_QUERIES` | rewordings the model writes (default 3), so queries + 1 searches |

## How the answers are made here

The searches, the fusing, the reranking and the scores run **live** in your
browser with the real rag-titan code; the model's replies (`gemma3:4b`) were
recorded once and are replayed. Replies were recorded with the defaults and with
one knob at a time on the values listed under it; the line under the knobs says
**● Answers recorded** when they replay. Elsewhere the run stops at the first model
step whose reply was not recorded, drawn dashed. The code below the result runs
the same setup in Python. The [Compare page](../rag-playground.md) runs several
RAG types side by side.
