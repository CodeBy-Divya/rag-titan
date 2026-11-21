"""Tests for ParentChildRAG: search children, answer from parents (handbook section 09)."""

import pytest
from conftest import FakeLLM

from rag_titan import ParentChildRAG, build_rag, build_rag_parts, get_settings
from rag_titan.chunking import ParentChildChunker, WordChunker
from rag_titan.embeddings import HashEmbedder
from rag_titan.evaluation import end_to_end
from rag_titan.generation import GroundedGenerator
from rag_titan.pipelines import IndexingPipeline
from rag_titan.rag_types.trace import describe_steps
from rag_titan.retrieval import DenseRetriever
from rag_titan.retrieval.rerankers.no_reranker import NoReranker
from rag_titan.vector_stores.memory_store import MemoryStore
from rag_titan.vector_stores.sqlite_store import SQLiteStore

LEAVE = (
    "# Leave policy\n\n"
    "## Annual leave\n\n"
    "Every full-time employee gets 20 paid leave days per calendar year. "
    "Up to 5 unused days move to the next year.\n\n"
    "## Sick leave\n\n"
    "Tell your manager before 10 am on the first sick day. "
    "From the third sick day you need a note from the doctor.\n"
)


def child(number, parent, size=100):
    """A reranked child chunk of `parent` with a parent text of `size` characters."""
    return {"id": f"chunk_{number}", "source": "leave.md", "text": f"child {number}",
            "score": 1 - number / 10, "rerank_score": 1 - number / 10,
            "start_offset": 40 + number, "end_offset": 50 + number, "page_numbers": [1],
            "parent_id": f"leave.md#parent_{parent}", "parent_text": f"P{parent} " + "x" * size,
            "parent_start_offset": 10 * parent, "parent_end_offset": 10 * parent + size,
            "parent_page_numbers": [1, 2]}


def rag_with(tmp_path, llm=None, store=None, **options):
    """A ParentChildRAG over LEAVE, indexed with small word children."""
    (tmp_path / "leave.md").write_text(LEAVE, encoding="utf-8")
    embedder, store = HashEmbedder(64), store or MemoryStore()
    IndexingPipeline(ParentChildChunker(WordChunker(8, 2)), embedder, store).run(
        [tmp_path / "leave.md"])
    llm = llm or FakeLLM(["You need a doctor's note from the third day [chunk_4]."])
    return ParentChildRAG(DenseRetriever(embedder, store), NoReranker(),
                          GroundedGenerator(llm), **options)


def expander(budget=3000):
    return ParentChildRAG(None, None, None, parent_budget=budget)


def test_expand_uses_each_parent_once_best_child_first():
    parents = expander().expand([child(0, 1), child(1, 0), child(2, 1)])

    assert [p["parent_id"] for p in parents] == ["leave.md#parent_1", "leave.md#parent_0"]
    assert parents[0]["children"] == ["chunk_0", "chunk_2"]
    assert parents[0]["id"] == "chunk_0"                 # citations stay chunk ids
    assert parents[0]["text"].startswith("P1 ") and parents[0]["child_text"] == "child 0"
    assert parents[0]["rerank_score"] == 1.0


def test_expand_gives_the_parent_its_own_place():
    parent = expander().expand([child(3, 2)])[0]

    assert (parent["start_offset"], parent["end_offset"]) == (20, 120)
    assert parent["page_numbers"] == [1, 2]
    assert not any(key.startswith("parent_") and key != "parent_id" for key in parent)


def test_parent_budget_limits_the_text_but_keeps_the_first_parent():
    chunks = [child(0, 0, size=900), child(1, 1, size=900), child(2, 2, size=50)]

    assert [p["parent_id"][-1] for p in expander(1000).expand(chunks)] == ["0", "2"]
    assert len(expander(100).expand(chunks)) == 1      # bigger than the budget, still kept


def test_index_without_parents_gives_a_clear_error():
    plain = {k: v for k, v in child(0, 0).items() if not k.startswith("parent_")}
    with pytest.raises(ValueError, match="rag-titan index --rag-type parent_child"):
        expander().expand([plain])


def test_run_searches_children_and_answers_from_parents(tmp_path):
    llm = FakeLLM(["You need a doctor's note from the third day [chunk_4]."])
    result = rag_with(tmp_path, llm=llm).run("When do I need a note from the doctor?")

    steps = [step["step"] for step in result["steps"]]
    assert steps == ["retrieve", "rerank", "expand", "generate"]
    children, parents = result["steps"][2]["input"]["chunks"], result["sources"]
    assert len(parents) < len(children)               # siblings became one parent
    assert parents[0]["text"].startswith("Leave policy > Sick leave")
    # the model reads whole sections, not the small children
    prompt = llm.calls[0][1]["content"]
    assert "Tell your manager before 10 am" in prompt and "from the doctor" in prompt
    # the log names the parents and how many children each one took in
    assert "-> 2 chunks: leave.md#parent_1 (3 children, 137 chars)" in describe_steps(
        result["steps"])


def test_run_handles_empty_question_and_empty_index(tmp_path):
    assert rag_with(tmp_path).answer("   ") == "Please enter a question."
    empty = ParentChildRAG(DenseRetriever(HashEmbedder(64), MemoryStore()), NoReranker(), None)
    assert empty.answer("leave?") == "Please index your documents first."


@pytest.mark.parametrize("name", ["top_k", "top_n", "parent_budget"])
@pytest.mark.parametrize("value", [0, -1, 2.5, "8", True])
def test_bad_numbers_raise(name, value):
    with pytest.raises(ValueError, match=name):
        ParentChildRAG(None, None, None, **{name: value})


def test_factory_indexes_parent_child_with_parents():
    settings = get_settings({"embed_provider": "hash", "parent_size": 700})
    chunker = build_rag_parts(settings, "parent_child")["pipeline"].chunker
    assert isinstance(chunker, ParentChildChunker) and chunker.parent_size == 700
    assert isinstance(chunker.child_chunker, WordChunker)

    assert isinstance(build_rag_parts(settings, "simple")["pipeline"].chunker, WordChunker)
    assert isinstance(build_rag("parent_child", build_rag_parts(settings, "parent_child")),
                      ParentChildRAG)


def test_compare_setups_gives_parent_child_its_own_index(tmp_path):
    (tmp_path / "leave.md").write_text(LEAVE, encoding="utf-8")
    files, indexes = [str(tmp_path / "leave.md")], {}

    simple = end_to_end._indexed_parts(files, {"embed_provider": "hash"}, indexes, "simple")
    parent = end_to_end._indexed_parts(files, {"embed_provider": "hash"}, indexes, "parent_child")

    assert len(indexes) == 2
    assert all("parent_id" not in c for c in simple["pipeline"].store.all_chunks())
    assert all("parent_id" in c for c in parent["pipeline"].store.all_chunks())


def test_add_replaces_children_and_parents_together(tmp_path):
    store = SQLiteStore(str(tmp_path / "index.db"))
    rag_with(tmp_path, store=store)
    (tmp_path / "leave.md").write_text(LEAVE.replace("20 paid", "25 paid"), encoding="utf-8")
    pipeline = IndexingPipeline(ParentChildChunker(WordChunker(8, 2)), HashEmbedder(64), store)
    pipeline.run([tmp_path / "leave.md"], reset=False)

    chunks = store.all_chunks()
    assert len({c["id"] for c in chunks}) == len(chunks)
    annual = [c for c in chunks if c["parent_text"].startswith("Leave policy > Annual")]
    assert annual and all("25 paid" in c["parent_text"] for c in annual)
    assert not any("20 paid" in c["parent_text"] for c in chunks)
