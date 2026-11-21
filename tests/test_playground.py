"""Tests for the playground presets: record, replay, check, and the committed recording."""

import json
from pathlib import Path

import pytest
from conftest import FakeEmbedder

from rag_titan.llm import BaseLLM, RecordingLLM
from rag_titan.playground import (
    PRESET_CHUNKING,
    PRESET_KEYS,
    Playground,
    build_parts,
    check_presets,
    index_documents,
    preset_settings,
    record_presets,
    replay_parts,
)

PRESETS = Path(__file__).parent.parent / "docs" / "assets" / "playground" / "presets.json"

DOCUMENTS = [
    {"source": "hr.md", "text": "You get 20 leave days every year. Salary is paid monthly."},
    {"source": "it.md", "text": "Remote work needs a laptop. The office opens at nine."},
]
QUESTIONS = [
    {"id": "q1", "question": "how many leave days", "required_evidence_ids": ["hr.md"]},
    {"id": "q2", "question": "remote work laptop", "required_evidence_ids": ["it.md"]},
]


class MoodyLLM(BaseLLM):
    """A model that never answers the same twice, like a real one with sampling."""

    def __init__(self):
        super().__init__("moody")
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        replies = ["simple", "complex", "yes", "no", "leave days per year"]
        reply = replies[self.calls % len(replies)] + f" #{self.calls}"
        if self.calls % 3 == 0:
            reply = ["simple", "yes", "no"][self.calls % 2]
        return {"content": reply, "tool_calls": [], "message": {}}


def test_preset_settings_keep_only_what_a_preset_needs():
    settings = preset_settings(PRESET_CHUNKING["paragraph"])
    assert list(settings) == PRESET_KEYS
    assert settings["chunker"] == "paragraph" and settings["chunk_size"] == 300


def test_index_documents_needs_no_files():
    parts = build_parts(preset_settings({"chunk_size": 5, "chunk_overlap": 0}),
                        FakeEmbedder(), MoodyLLM())
    assert index_documents(parts, DOCUMENTS) == parts["store"].count() > 2


def test_recording_llm_asks_each_prompt_once():
    llm = MoodyLLM()
    recorder = RecordingLLM(llm)
    first = recorder.ask("route", "how many leave days")
    again = recorder.ask("route", "how many leave days")

    assert first == again and llm.calls == 1 and len(recorder.replies) == 1


def test_a_recording_replays_the_same_even_with_a_moody_model():
    presets = record_presets(DOCUMENTS, QUESTIONS, PRESET_CHUNKING, MoodyLLM(), FakeEmbedder())

    assert set(presets["expected"]) == set(PRESET_CHUNKING)
    assert set(presets["expected"]["word"]) == {"dense", "bm25", "hybrid"}
    assert set(presets["expected"]["word"]["hybrid"]) == {"simple", "adaptive", "parent_child",
                                                          "hyde"}
    assert set(presets["expected"]["word"]["hybrid"]["parent_child"]) == {"default"}
    assert check_presets(json.loads(json.dumps(presets))) == []   # survives a JSON round trip


def test_check_finds_a_changed_run():
    presets = record_presets(DOCUMENTS, QUESTIONS, PRESET_CHUNKING, MoodyLLM(), FakeEmbedder())
    presets["expected"]["word"]["hybrid"]["simple"]["default"]["q1"]["answer"] = "something else"

    problems = check_presets(presets)
    assert len(problems) == 1 and problems[0].startswith("word / hybrid / simple / default / q1")


@pytest.mark.skipif(not PRESETS.exists(), reason="no recorded presets yet")
def test_the_committed_recording_still_replays():
    """Fails when prompts, RAG types or chunkers changed: run
    `python tools/playground/record_traces.py` to record again."""
    presets = json.loads(PRESETS.read_text(encoding="utf-8"))

    assert check_presets(presets) == []
    parts = replay_parts(presets, "paragraph")
    assert parts["retriever"].has_documents()


@pytest.mark.skipif(not PRESETS.exists(), reason="no recorded presets yet")
def test_playground_session_gives_the_page_what_it_needs():
    playground = Playground(json.loads(PRESETS.read_text(encoding="utf-8")))
    info = playground.info()

    assert info["rag_types"] == ["simple", "adaptive", "parent_child"]
    assert info["retrievers"] == ["dense", "bm25", "hybrid"]
    assert info["flows"]["adaptive"]["dense"]["nodes"][0]["id"] == "route"
    hybrid_ids = [node["id"] for node in info["flows"]["simple"]["hybrid"]["nodes"]]
    assert hybrid_ids == ["vector_search", "keyword_search", "fuse", "rerank", "generate"]
    assert set(info["chunkings"]) == set(PRESET_CHUNKING)

    run = playground.run("paragraph", "simple", "q_001")
    assert [step["step"] for step in run["steps"]] == ["retrieve", "rerank", "generate"]
    assert run["answered"] and run["recall"] == 1.0
    assert json.loads(json.dumps(run)) == run          # plain JSON for the browser
    hybrid = playground.run("paragraph", "simple", "q_001", retriever="hybrid")
    assert [step["step"] for step in hybrid["steps"]][:3] == ["vector_search", "keyword_search",
                                                              "fuse"]
    assert "rrf" in hybrid["steps"][2]["output"][0]["score_trace"]

    report = playground.evaluate(["word"], ["simple", "adaptive"])
    assert [r["name"] for r in report["results"]] == ["simple + word", "adaptive + word"]
    both = playground.evaluate(["word"], ["simple"], ["dense", "hybrid"])
    assert [r["name"] for r in both["results"]] == ["simple + word + dense",
                                                    "simple + word + hybrid"]
    assert "steps" not in report["results"][0]["rows"][0]
    assert report["report"].startswith("End-to-end report: 6 questions")


def test_playground_session_bad_names():
    playground = Playground(record_presets(DOCUMENTS, QUESTIONS, PRESET_CHUNKING,
                                           MoodyLLM(), FakeEmbedder()))
    with pytest.raises(ValueError, match="Unknown chunking"):
        playground.run("lines", "simple", "q1")
    with pytest.raises(ValueError, match="Unknown question"):
        playground.run("word", "simple", "q9")
    with pytest.raises(ValueError, match="Unknown retriever"):
        playground.run("word", "simple", "q1", retriever="sparse")


@pytest.mark.skipif(not PRESETS.exists(), reason="no recorded presets yet")
def test_every_page_setting_gives_plain_json():
    """The browser gets results as JSON: numpy numbers (e.g. MMR scores) would break it."""
    playground = Playground(json.loads(PRESETS.read_text(encoding="utf-8")))
    for rag_type in playground.presets["rag_types"]:
        for reranker in ("keyword", "none", "mmr", "llm"):
            run = playground.run("word", rag_type, "q_006", "hybrid", {"reranker": reranker})
            assert json.loads(json.dumps(run)) == run
