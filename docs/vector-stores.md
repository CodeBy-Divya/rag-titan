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

[`examples/persistent_rag.py`](https://github.com/aion7891/rag-titan/blob/main/examples/persistent_rag.py)
is a small RAG with a saved index. Each command is a separate run of Python,
like coming back tomorrow: only `index` and `add` read files, `ask` answers
straight from `hr_index.db`.

```python
"""
A RAG that keeps its index on disk: index once, ask later, add files as they change.

Every command below is a separate run of Python, like a new session tomorrow:
the index is saved in hr_index.db (one SQLite file) and read back from there.

Before running (Ollama installed and running):
    ollama pull gemma3:4b
    ollama pull nomic-embed-text

Run (from the repo root):
    python examples/persistent_rag.py index                 # 1. index the sample policies
    python examples/persistent_rag.py ask "How many paid leave days do I get?"
    python examples/persistent_rag.py add examples/sample_docs/remote_work.md
    python examples/persistent_rag.py info                  # what the index holds
"""

import sys
from pathlib import Path

from rag_titan import RAGTitan

SAMPLES = Path(__file__).parent / "sample_docs"

# the only change from an in-memory RAG: a saved store and where to keep it
SETTINGS = {
    "vector_store": "sqlite",            # or "chroma" / "qdrant" (pip install their extra)
    "vector_store_path": "hr_index.db",  # created on the first index
}


def index():
    """Build the index from scratch (anything indexed before is replaced)."""
    rag = RAGTitan(settings=SETTINGS)
    files = sorted(str(path) for path in SAMPLES.glob("*.md"))
    chunk_count = rag.index(files)
    print(f"Indexed {chunk_count} chunks from {len(files)} files into hr_index.db")


def ask(question):
    """Answer from the saved index: nothing is indexed here."""
    rag = RAGTitan(settings=SETTINGS)
    result = rag.run(question)
    print(f"Answer: {result['answer']}")
    for chunk in result["sources"]:
        print(f"  from {chunk['source']} ({chunk['id']}, score {chunk['score']:.2f})")


def add(path):
    """Add one file, or replace its old chunks if it was indexed before."""
    rag = RAGTitan(settings=SETTINGS)
    chunk_count = rag.index([path], reset=False)
    print(f"Indexed {chunk_count} chunks from {Path(path).name}; "
          f"the index now holds {rag.pipeline.store.count()} chunks")


def info():
    """Show what the saved index holds and how it was built."""
    store = RAGTitan(settings=SETTINGS).pipeline.store
    print(f"{store.count()} chunks, built with {store.get_info()}")


def main(argv):
    """Run one command: index, ask QUESTION, add FILE or info."""
    commands = {"index": (index, 0), "ask": (ask, 1), "add": (add, 1), "info": (info, 0)}
    if not argv or argv[0] not in commands or len(argv) - 1 != commands[argv[0]][1]:
        raise SystemExit(__doc__)
    command, _ = commands[argv[0]]
    command(*argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:])
```

What it printed with `gemma3:4b` and `nomic-embed-text`:

```text
$ python examples/persistent_rag.py index
Indexed 3 chunks from 3 files into hr_index.db

$ python examples/persistent_rag.py ask "How many paid leave days do I get?"
Answer: Every full-time employee gets 20 paid leave days per calendar year [chunk_1]. Part-time employees get leave days in proportion to their working hours [chunk_1].
  from leave_policy.md (chunk_1, score 0.76)
  from remote_work.md (chunk_2, score 0.61)

$ python examples/persistent_rag.py add examples/sample_docs/remote_work.md
Indexed 1 chunks from remote_work.md; the index now holds 3 chunks

$ python examples/persistent_rag.py info
3 chunks, built with {'embed_model': 'nomic-embed-text', 'dimensions': 768, 'chunker': 'WordChunker', 'next_chunk': 4}

$ python examples/persistent_rag.py ask "Can I work from home on Fridays?"
Answer: Based on the available documents, “Employees may work from home up to three days per week, after agreeing the days with their manager.” [chunk_3]
  from remote_work.md (chunk_3, score 0.65)
  from leave_policy.md (chunk_1, score 0.51)
```

Adding `remote_work.md` again replaced its old chunk (`chunk_2`) instead of
storing it twice: the index still holds 3 chunks, and the new chunk got the
next free id, `chunk_3`. To use Chroma or Qdrant instead, change only
`"vector_store"` in `SETTINGS`.

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
