"""
Record the browser playground's examples with real models, or check a recording.

It indexes the sample policies in examples/sample_docs/ with every preset chunking,
runs every RAG type with every retriever on every question in examples/questions.json,
with its defaults and with each knob value of PRESET_GRID (one knob at a time), and
saves the model's replies, the embedding vectors and the expected runs in
docs/assets/playground/presets.json.

Usage (from the repo root, Ollama running):
    python tools/playground/record_traces.py                     # record with gemma3:4b
    python tools/playground/record_traces.py --model qwen3:4b    # another chat model
    python tools/playground/record_traces.py --check             # replay, no Ollama needed
    python tools/playground/record_traces.py --fresh             # ask the model for every reply

Recording again keeps the replies and vectors already in presets.json for every
prompt it meets again, so adding a RAG type or a knob value never changes the runs
(and the "Things to try" claims) recorded before. Use --fresh after changing the
model, or to record everything anew.

Record again whenever prompts, RAG types, chunkers or the sample files change:
--check (and the tests) fail until you do.
"""

import argparse
import json
import sys
from pathlib import Path

from rag_titan.embeddings import OllamaEmbedder
from rag_titan.evaluation import load_questions
from rag_titan.llm import OllamaLLM
from rag_titan.loaders import load_document
from rag_titan.playground import PRESET_CHUNKING, PRESET_GRID, check_presets, record_presets

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "examples" / "sample_docs"
QUESTIONS = ROOT / "examples" / "questions.json"
OUT = ROOT / "docs" / "assets" / "playground" / "presets.json"


def record(model, embed_model, fresh=False):
    """Run everything with the real models and write presets.json."""
    documents = [load_document(path) for path in sorted(SAMPLES.glob("*.md"))]
    for document in documents:
        document.pop("file_type", None)   # the playground only needs source and text

    earlier = None
    if OUT.exists() and not fresh:
        earlier = json.loads(OUT.read_text(encoding="utf-8"))
        if (earlier.get("model"), earlier.get("embed_model")) != (model, embed_model):
            earlier = None          # other models: their replies do not fit
    presets = record_presets(documents, load_questions(QUESTIONS), PRESET_CHUNKING,
                             OllamaLLM(model), OllamaEmbedder(embed_model), grid=PRESET_GRID,
                             earlier=earlier)

    # compact JSON: the file is downloaded by every playground visitor
    OUT.write_text(json.dumps(presets, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":")), encoding="utf-8")
    runs = sum(len(runs) for by_retriever in presets["expected"].values()
               for by_type in by_retriever.values() for by_tuning in by_type.values()
               for runs in by_tuning.values())
    print(f"Recorded {runs} runs, {len(presets['replies'])} replies and "
          f"{len(presets['vectors'])} vectors -> {OUT.relative_to(ROOT)} "
          f"({OUT.stat().st_size / 1024:.0f} KB)")


def check():
    """Replay the recording and fail when any run differs."""
    problems = check_presets(json.loads(OUT.read_text(encoding="utf-8")))
    for problem in problems:
        print(problem)
    if problems:
        raise SystemExit(f"{len(problems)} run(s) differ: record again")
    print("Every recorded run replays the same.")


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--model", default="gemma3:4b", help="Ollama chat model")
    parser.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model")
    parser.add_argument("--check", action="store_true", help="replay instead of recording")
    parser.add_argument("--fresh", action="store_true",
                        help="ask the model for every reply, ignoring presets.json")
    args = parser.parse_args(argv)

    if args.check:
        check()
    else:
        record(args.model, args.embed_model, args.fresh)


if __name__ == "__main__":
    main(sys.argv[1:])
