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
store = rag.pipeline.store
if store.count() == 0:
    files = [DOCS / "leave_policy.md", DOCS / "expenses.md"]
    print(f"Indexing {len(files)} files...")
    rag.index(files)
else:
    print(f"Using the saved index: {store.count()} chunks, no indexing needed")

# 3. Add a file later. It runs on every run here, to show that a file indexed
#    before is replaced (new chunk ids), not stored twice
rag.index([DOCS / "remote_work.md"], reset=False)
print(f"The index holds {store.count()} chunks")

# 4. Ask questions
for question in ["How many paid leave days do I get?", "Can I work from home on Fridays?"]:
    result = rag.run(question)
    print(f"\nQ: {question}\nA: {result['answer']}")
    for chunk in result["sources"]:
        print(f"   from {chunk['source']} ({chunk['id']})")
