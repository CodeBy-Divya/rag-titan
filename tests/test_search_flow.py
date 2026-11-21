"""
Tests that every RAG type draws and records the right search steps for every
retriever (dense, bm25, hybrid): the diagram and the trace must agree.
"""

import pytest
from conftest import PromptLLM

from rag_titan.chunking import ParentChildChunker, WordChunker
from rag_titan.embeddings import HashEmbedder
from rag_titan.factory import PARENT_RAG_TYPES, RAG_TYPES, RETRIEVERS, build_rag, build_retriever
from rag_titan.generation import GroundedGenerator
from rag_titan.pipelines import IndexingPipeline
from rag_titan.rag_types.base_rag import NODE_KINDS
from rag_titan.rag_types.flow import expand_search
from rag_titan.rag_types.trace import describe_steps
from rag_titan.retrieval.rerankers import KeywordReranker
from rag_titan.vector_stores import MemoryStore


def parts_with(retriever, tmp_path, rag_type="simple"):
    (tmp_path / "hr.md").write_text("You get 20 leave days. Invoice INV-2041 was paid.",
                                    encoding="utf-8")
    embedder, store, llm = HashEmbedder(64), MemoryStore(), PromptLLM()
    chunker = WordChunker(5, 0)
    if rag_type in PARENT_RAG_TYPES:      # the same rule as build_rag_parts()
        chunker = ParentChildChunker(chunker)
    IndexingPipeline(chunker, embedder, store).run([tmp_path / "hr.md"])
    settings = {"retriever": retriever, "dedupe_overlap": 0.5}
    return {"llm": llm, "retriever": build_retriever(settings, embedder, store),
            "reranker": KeywordReranker(), "generator": GroundedGenerator(llm)}


@pytest.mark.parametrize("retriever", RETRIEVERS)
@pytest.mark.parametrize("rag_type", RAG_TYPES)
def test_trace_follows_the_drawn_flow(rag_type, retriever, tmp_path):
    rag = build_rag(rag_type, parts_with(retriever, tmp_path, rag_type))
    flow = rag.flow()
    ids = [node["id"] for node in flow["nodes"]]
    edges = {tuple(edge) for edge in flow["edges"]}

    steps = [step["step"] for step in rag.run("how many leave days?")["steps"]]

    assert len(ids) == len(set(ids)) and all(n["kind"] in NODE_KINDS for n in flow["nodes"])
    assert all(a in ids and b in ids for a, b in edges)
    assert set(steps) <= set(ids)
    for pair in zip(steps, steps[1:]):
        assert pair in edges, f"{rag_type} + {retriever}: no edge {pair}"


def test_hybrid_steps_in_the_trace_log(tmp_path):
    rag = build_rag("simple", parts_with("hybrid", tmp_path))
    result = rag.run("INV-2041 leave days")

    assert [s["step"] for s in result["steps"]] == ["vector_search", "keyword_search", "fuse",
                                                    "rerank", "generate"]
    log = describe_steps(result["steps"])
    assert "fuse" in log and "dense=" in log and "keyword=" in log


def test_expand_search_labels_and_replaces():
    flow = {"nodes": [{"id": "retrieve", "label": "Retrieve", "kind": "search"},
                      {"id": "generate", "label": "Answer", "kind": "output"}],
            "edges": [["retrieve", "generate"]]}

    class Labelled:
        label = "Keyword search"

    assert expand_search(flow, Labelled())["nodes"][0]["label"] == "Keyword search"
    assert flow["nodes"][0]["label"] == "Retrieve"                 # the original is not changed
    assert expand_search(flow, None) == flow
    assert expand_search({"nodes": [], "edges": []}, Labelled()) == {"nodes": [], "edges": []}

    class Steps:
        SEARCH_FLOW = {"nodes": [{"id": "a", "label": "A", "kind": "search"},
                                 {"id": "b", "label": "B", "kind": "search"}],
                       "edges": [["a", "b"]]}

    expanded = expand_search(flow, Steps())
    assert [n["id"] for n in expanded["nodes"]] == ["a", "b", "generate"]
    assert sorted(expanded["edges"]) == [["a", "b"], ["b", "generate"]]
