# Vector stores

The vector store keeps your chunks and their vectors, and finds the chunks
closest to a question. By default everything lives in memory and is gone when
Python exits, so every run indexes your files again. A **saved** store keeps
the index: index once, ask as often as you like.

## Which one?

| `VECTOR_STORE` | Install                          | Where the index lives              | Best for |
|----------------|----------------------------------|------------------------------------|----------|
| `memory`       | nothing                          | in memory, gone when Python exits  | trying things, tests, small files |
| `sqlite`       | nothing                          | one file, `rag_titan.db`           | most projects, up to about 100,000 chunks |
| `chroma`       | `pip install "rag-titan[chroma]"`| a folder, or a Chroma server       | teams that already use Chroma |
| `qdrant`       | `pip install "rag-titan[qdrant]"`| a folder, a Qdrant server or Qdrant Cloud | large indexes, several programs at once |

Not sure? Start with **`sqlite`**: nothing to install, one file to copy or
delete. Measured with 100,000 chunks of 768 numbers: the first search reads
the file in about 0.4 s, every search after that takes about 20 ms. Move to
Qdrant when you outgrow it, or when several programs must use the index at once.

Every store gives the same results for the same index: each search result is
the chunk with a cosine `score` (higher is closer), so rerankers, citations and
evaluation work the same with all of them.

## Set it up

=== "SQLite"

    Nothing to install.

    ```bash
    # .env
    VECTOR_STORE=sqlite
    # VECTOR_STORE_PATH=rag_titan.db      # the default
    ```

    ```python
    from rag_titan import RAGTitan

    rag = RAGTitan(settings={"vector_store": "sqlite"})
    rag = RAGTitan(settings={"vector_store": "sqlite", "vector_store_path": "hr/index.db"})
    ```

=== "Chroma"

    ```bash
    pip install "rag-titan[chroma]"
    ```

    ```bash
    # .env
    VECTOR_STORE=chroma
    # VECTOR_STORE_PATH=rag_titan_chroma          # local folder (the default)
    # VECTOR_STORE_URL=http://localhost:8000      # or a Chroma server
    # COLLECTION=rag_titan                        # one collection per index
    ```

    ```python
    rag = RAGTitan(settings={"vector_store": "chroma", "collection": "hr_docs"})
    ```

    Collection names need 3 or more letters, digits, `.`, `_` or `-`.

=== "Qdrant"

    ```bash
    pip install "rag-titan[qdrant]"
    ```

    ```bash
    # .env
    VECTOR_STORE=qdrant
    # VECTOR_STORE_PATH=rag_titan_qdrant          # local folder (the default)
    # VECTOR_STORE_URL=http://localhost:6333      # or a server / Qdrant Cloud
    # QDRANT_API_KEY=...                          # Qdrant Cloud only
    # COLLECTION=rag_titan
    ```

    ```python
    rag = RAGTitan(settings={"vector_store": "qdrant",
                             "vector_store_url": "https://xyz.cloud.qdrant.io:6333"})
    ```

    A local folder is used by one program at a time; for several programs at
    once, run a Qdrant server (`docker run -p 6333:6333 qdrant/qdrant`).

=== "Memory"

    The default: nothing to set up, nothing saved.

    ```bash
    VECTOR_STORE=memory
    ```

## Index once, ask later

```python
from rag_titan import RAGTitan

# today
rag = RAGTitan(settings={"vector_store": "sqlite"})
rag.index(["handbook.pdf", "leave_policy.docx"])

# tomorrow, in a new Python session: no index() call
rag = RAGTitan(settings={"vector_store": "sqlite"})
print(rag.ask("How many leave days do I get?"))
```

The same from the command line:

```bash
rag-titan index --files handbook.pdf leave_policy.docx --vector-store sqlite
rag-titan ask --question "How many leave days do I get?" --vector-store sqlite
```

Set `VECTOR_STORE=sqlite` in `.env` and you can leave out `--vector-store`.

## A complete example

Write a normal Python script: import `rag_titan`, create the RAG with a saved
store, index, ask. The first run indexes; every later run finds the index in
`hr_index.db` and skips straight to the questions.

### With `RAGTitan` (simplest)

[`examples/persistent_rag.py`](https://github.com/aion7891/rag-titan/blob/main/examples/persistent_rag.py):

```python
"""
A RAG that keeps its index in a file: the first run indexes, every later run
just asks. Run it twice to see the second run skip the indexing.

Before running (Ollama installed and running):
    ollama pull gemma3:4b
    ollama pull nomic-embed-text

Run (from the repo root):
    python examples/persistent_rag.py
"""

from pathlib import Path

from rag_titan import RAGTitan

DOCS = Path(__file__).parent / "sample_docs"

# 1. Create the RAG with a saved vector store: the index lives in hr_index.db
rag = RAGTitan(
    rag_type="simple",
    settings={
        "vector_store": "sqlite",            # or "chroma" / "qdrant" (pip install their extra)
        "vector_store_path": "hr_index.db",  # created on the first run
    },
)

# 2. Index the documents, only when the saved index is still empty
if rag.count() == 0:
    files = [DOCS / "leave_policy.md", DOCS / "expenses.md"]
    print(f"Indexing {len(files)} files...")
    rag.index(files)
else:
    print(f"Using the saved index: {rag.count()} chunks, no indexing needed")

# 3. Add a file later. It runs on every run here, to show that a file indexed
#    before is replaced (new chunk ids), not stored twice
rag.index([DOCS / "remote_work.md"], reset=False)
print(f"The index holds {rag.count()} chunks")

# 4. Ask questions
for question in ["How many paid leave days do I get?", "Can I work from home on Fridays?"]:
    result = rag.run(question)
    print(f"\nQ: {question}\nA: {result['answer']}")
    for chunk in result["sources"]:
        print(f"   from {chunk['source']} ({chunk['id']})")
```

What it printed with `gemma3:4b` and `nomic-embed-text`, run twice:

```text
$ python examples/persistent_rag.py
Indexing 2 files...
The index holds 3 chunks

Q: How many paid leave days do I get?
A: Every full-time employee gets 20 paid leave days per calendar year. Part-time employees get leave days in proportion to their working hours. [chunk_0, page 1]
   from leave_policy.md (chunk_0)
   from remote_work.md (chunk_2)

Q: Can I work from home on Fridays?
A: According to [chunk_2], employees may work from home up to three days per week, after agreeing the days with their manager.
   from remote_work.md (chunk_2)
   from leave_policy.md (chunk_0)

$ python examples/persistent_rag.py
Using the saved index: 3 chunks, no indexing needed
The index holds 3 chunks

Q: How many paid leave days do I get?
A: Every full-time employee gets 20 paid leave days per calendar year [chunk_0]. Part-time employees get leave days in proportion to their working hours [chunk_0].
   from leave_policy.md (chunk_0)
   from remote_work.md (chunk_3)

Q: Can I work from home on Fridays?
A: Based on the available documents, “Employees may work from home up to three days per week, after agreeing the days with their manager.” [chunk_3]
   from remote_work.md (chunk_3)
   from leave_policy.md (chunk_0)
```

The second run did not index the policies again. Step 3 added
`remote_work.md` once more, so its old chunk (`chunk_2`) was replaced by
`chunk_3`: the index still holds 3 chunks. To use Chroma or Qdrant, change
only `"vector_store"`.

### Built from the parts

The same RAG with every class picked by hand, for when you want to swap one
of them (see [Building blocks](building-blocks.md)).
[`examples/persistent_rag_parts.py`](https://github.com/aion7891/rag-titan/blob/main/examples/persistent_rag_parts.py):

```python
"""
The same saved-index RAG, built from its parts: you pick every class yourself.

Before running (Ollama installed and running):
    ollama pull gemma3:4b
    ollama pull nomic-embed-text

Run (from the repo root):
    python examples/persistent_rag_parts.py
"""

from pathlib import Path

from rag_titan import SimpleRAG
from rag_titan.chunking import ParagraphChunker
from rag_titan.embeddings import OllamaEmbedder
from rag_titan.generation import GroundedGenerator
from rag_titan.llm import OllamaLLM
from rag_titan.pipelines import IndexingPipeline
from rag_titan.retrieval import DenseRetriever
from rag_titan.retrieval.rerankers import KeywordReranker
from rag_titan.vector_stores import SQLiteStore

DOCS = Path(__file__).parent / "sample_docs"

# 1. The parts: the store is the only one that keeps anything between runs
llm = OllamaLLM("gemma3:4b")
embedder = OllamaEmbedder("nomic-embed-text")
store = SQLiteStore("hr_index_parts.db")      # or ChromaStore(...) / QdrantStore(...)

# 2. Indexing: load -> chunk -> embed -> store (first run only)
if store.count() == 0:
    pipeline = IndexingPipeline(ParagraphChunker(1000, 0), embedder, store)
    chunk_count = pipeline.run(sorted(DOCS.glob("*.md")))
    print(f"Indexed {chunk_count} chunks")
else:
    print(f"Using the saved index: {store.count()} chunks")

# 3. Answering: retrieve -> rerank -> generate (the embedder must be the one that indexed)
rag = SimpleRAG(DenseRetriever(embedder, store), KeywordReranker(), GroundedGenerator(llm))
print(rag.answer("How many paid leave days do I get?"))
```

```text
$ python examples/persistent_rag_parts.py
Indexed 3 chunks
Every full-time employee gets 20 paid leave days per calendar year [chunk_1]. Part-time employees get leave days in proportion to their working hours [chunk_1].

$ python examples/persistent_rag_parts.py
Using the saved index: 3 chunks
Every full-time employee gets 20 paid leave days per calendar year. Part-time employees get leave days in proportion to their working hours. [chunk_1]
```

## Add, replace and start over

| You want to                      | Python                                  | Command line |
|----------------------------------|-----------------------------------------|--------------|
| start over with these files      | `rag.index(files)`                      | `rag-titan index --files ...` |
| add files, keep the rest         | `rag.index(files, reset=False)`         | `rag-titan index --files ... --add` |
| update a file that changed       | `rag.index(["policy.pdf"], reset=False)`| `rag-titan index --files policy.pdf --add` |

When you add a file that is already in the index, its old chunks are removed
first, so nothing is stored twice. Chunk ids go on counting (`chunk_57`,
`chunk_58`, ...) and are never given out again, so citations always point to
the right chunk.

Indexing a file that has no text skips it with a warning. Asking before
anything is indexed answers "Please index your documents first."

## "This index was built with ..."

Vectors from different embedding models cannot be compared. The store
remembers which model built the index, so a mix-up stops with a clear message
instead of returning wrong chunks:

```text
ValueError: This index was built with nomic-embed-text (768 dims), but you are
using hash (512 dims). Index again with reset=True, or switch EMBED_PROVIDER /
EMBED_MODEL back.
```

Fix it by setting `EMBED_PROVIDER` / `EMBED_MODEL` back to what built the
index, or build it again with the new model (`rag.index(files)` or
`rag-titan index --files ...`). Changing the **chat** model, the RAG type, the
chunker or `CHUNK_SIZE` never needs a new index (new chunk settings apply to
files you index next).

## Evaluation leaves your index alone

`rag-titan evaluate`, `compare_setups()`, `evaluate_stages()` and
`collect_dataset()` index the files they are given, so they always work on a
temporary in-memory copy and never change a saved index. `rag.evaluate()` and
`rag.evaluate_retrieval()` only search, so they score your saved index as it is.

## Your own store

Any database can be added: inherit `BaseStore`. See
[Extending rag-titan](extending.md#example-a-new-vector-store).
