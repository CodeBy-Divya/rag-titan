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
