---
hide:
  - toc
---

# Parent-Child RAG playground

**Parent-Child RAG** searches small chunks, which match a question precisely,
then gives the model the **whole section** each good chunk sits in:

```
Question -> Retrieve small children (top_k) -> Rerank, keep top_n
         -> Expand each child to its parent section (each section once, within the budget)
         -> Generate the answer from the sections
```

Move any knob below and the question runs again. **Click "Expand to parents"** to
see which children were found, which section each one belongs to, and the text the
model is given.

<div id="rag-type-app" data-rag-type="parent_child"><p><em>The playground needs JavaScript.</em></p></div>

## The knobs

| Knob | Setting | What it changes |
|---|---|---|
| Child chunker | `CHUNKER`, `CHUNK_SIZE` | how the small, searched children are cut |
| Parent size | `PARENT_SIZE` | the largest parent section in characters, used when indexing (default 1500) |
| Retriever | `RETRIEVER` | `dense`, `bm25` or `hybrid` |
| Reranker | `RERANKER` | orders the children before they are expanded |
| top_k | `TOP_K` | children searched (default 8: children are small, so take more) |
| top_n | `TOP_N` | children kept (default 4); children of the same section become one parent |
| parent budget | `PARENT_BUDGET` | most characters of section text given to the model (default 3000). The first section is always kept |

The budget is the handbook's warning made into a knob: whole sections can make the
prompt long ("token inflation"), so sections are added best first until the
budget is used up.

## How the answers are made here

Search, reranking and the expansion to parents run **live** in your browser with
the real rag-titan code; the model's answers (`gemma3:4b`) were recorded once and
are replayed. Positions marked **● answers recorded** replay real answers;
elsewhere the run stops at *Generate answer*, drawn dashed, and shows exactly the
sections the model would read. The code below the result builds the same setup in
Python. The [Compare page](../rag-playground.md) runs several RAG types side by side.
