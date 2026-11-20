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
