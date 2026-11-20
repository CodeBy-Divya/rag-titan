"""
What the browser playground calls: one object, plain JSON-ready dicts in and out.

    playground = Playground(json.loads(presets_text))
    playground.info()                                    # RAG types, FLOWs, questions, ...
    playground.run("paragraph", "adaptive", "q_002")     # answer, steps, scores
    playground.run("paragraph", "simple", "q_002", retriever="hybrid")
    playground.evaluate(["word", "paragraph"], ["simple", "adaptive"], ["dense", "hybrid"])
"""

from rag_titan.evaluation.end_to_end import evaluate_rag, format_rag_report
from rag_titan.factory import build_rag
from rag_titan.playground.presets import replay_parts, with_retriever


class Playground:
    """
    Runs the recorded presets: real code, recorded model replies and vectors.

    Args:
        presets: the dict saved by tools/playground/record_traces.py.
    """

    def __init__(self, presets):
        self.presets = presets
        self.parts = {}       # one index per chunking, built the first time it is used
        self.searchers = {}   # those parts with each retriever: {(chunking, retriever): parts}

    def info(self):
        """Everything the page needs to draw its controls and diagrams."""
        first = next(iter(self.presets["settings"]))
        retrievers = self.presets.get("retrievers", ["dense"])
        return {
            "model": self.presets["model"],
            "embed_model": self.presets["embed_model"],
            "rag_types": list(self.presets["rag_types"]),
            "retrievers": list(retrievers),
            # flows[rag_type][retriever]: hybrid search draws its own steps
            "flows": {name: {r: self._rag(first, name, r).flow() for r in retrievers}
                      for name in self.presets["rag_types"]},
            "chunkings": self.presets["settings"],
            "questions": self.presets["questions"],
            "documents": self.presets["documents"],
        }

    def run(self, chunking, rag_type, question_id, retriever="dense"):
        """
        Answer one preset question with one setup (chunking, RAG type, retriever).

        Returns:
            A dict with "answer", "steps" (every step with input and output),
            "sources" (files given to the LLM), "recall", "answered" and "ms".
        """
        question = self._question(question_id)
        row = evaluate_rag(self._rag(chunking, rag_type, retriever), [question])["rows"][0]
        return {key: row[key] for key in
                ("answer", "steps", "sources", "recall", "ndcg", "answered", "ms")}

    def evaluate(self, chunkings, rag_types, retrievers=None):
        """Every RAG type x chunking x retriever on every preset question, like compare_setups()."""
        results = []
        setups = [(c, t, r) for c in chunkings for r in retrievers or ["dense"] for t in rag_types]
        for chunking, rag_type, retriever in setups:
            name = f"{rag_type} + {chunking}" + (f" + {retriever}" if retrievers else "")
            result = evaluate_rag(self._rag(chunking, rag_type, retriever),
                                  self.presets["questions"], name=name)
            # the steps of every question are too much for a report: keep the scores
            result["rows"] = [{key: value for key, value in row.items() if key != "steps"}
                              for row in result["rows"]]
            results.append(result)
        return {"results": results, "report": format_rag_report(results)}

    def _rag(self, chunking, rag_type, retriever="dense"):
        """Build a RAG type on the (cached) index of one chunking, with one retriever."""
        if chunking not in self.presets["settings"]:
            raise ValueError(f"Unknown chunking {chunking!r}: use one of "
                             f"{list(self.presets['settings'])}")
        if chunking not in self.parts:
            self.parts[chunking] = replay_parts(self.presets, chunking)
        key = (chunking, retriever)
        if key not in self.searchers:
            self.searchers[key] = with_retriever(self.parts[chunking], retriever)
        return build_rag(rag_type, self.searchers[key])

    def _question(self, question_id):
        """Return the preset question with this id."""
        for question in self.presets["questions"]:
            if question["id"] == question_id:
                return question
        raise ValueError(f"Unknown question id {question_id!r}")
