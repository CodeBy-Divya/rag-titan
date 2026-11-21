"""
`rag-titan evaluate`: compare RAG types, chunkers and retrievers on test
questions, or score the logged data of any RAG system.

    rag-titan evaluate --files *.md --questions questions.json \\
        --rag-type simple adaptive --chunker word paragraph --retriever dense hybrid
    rag-titan evaluate --files *.md --questions questions.json --reranker keyword llm
    rag-titan evaluate --files *.md --questions questions.json \\
        --retriever dense bm25 hybrid --retrieval-only          # fast, no LLM
    rag-titan evaluate --input data.json --output report.json --baseline last_report.json
"""

import itertools
from pathlib import Path

from rag_titan.app import RAGTitan
from rag_titan.cli_options import add_tuning_options, tuning_settings
from rag_titan.evaluation import (
    compare_setups,
    evaluate_file,
    evaluate_retriever,
    format_rag_report,
    format_report,
    load_questions,
)
from rag_titan.evaluation.reports import summary_text
from rag_titan.factory import CHUNKERS, RAG_TYPES, RERANKERS, RETRIEVERS


def add_evaluate_command(commands):
    """Describe `rag-titan evaluate` and its options."""
    evaluate = commands.add_parser("evaluate", help="score RAG types on test questions")
    evaluate.add_argument("--files", nargs="+", help="PDF, DOCX, MD or TXT files")
    evaluate.add_argument("--log-level", default="WARNING", help="DEBUG, INFO or WARNING")
    evaluate.add_argument("--questions", help="questions JSON file (with --files)")
    evaluate.add_argument("--rag-type", nargs="+", default=["simple"], choices=RAG_TYPES,
                          help="one or more RAG types to compare")
    evaluate.add_argument("--chunker", nargs="+", choices=CHUNKERS,
                          help="one or more chunkers to compare (default: from settings)")
    evaluate.add_argument("--retriever", nargs="+", choices=RETRIEVERS,
                          help="one or more retrievers to compare (default: from settings)")
    evaluate.add_argument("--reranker", nargs="+", choices=RERANKERS,
                          help="one or more rerankers to compare (default: from settings)")
    evaluate.add_argument("--chunk-size", type=int, help="chunk size for every setup")
    evaluate.add_argument("--chunk-overlap", type=int, help="chunk overlap for every setup")
    evaluate.add_argument("--retrieval-only", action="store_true",
                          help="score search only (fast, no LLM); "
                               "--rag-type and --reranker are ignored")
    add_tuning_options(evaluate)       # --top-k, --top-n, ... for every setup
    add_data_options(evaluate)


def add_data_options(evaluate):
    """Options of `rag-titan evaluate --input`: score the logged data of any RAG system."""
    evaluate.add_argument("--input", help="JSON data file with chunks, queries and gold labels "
                                          "(instead of --files and --questions)")
    evaluate.add_argument("--output", default="rag_report.json",
                          help="where to save the JSON report (a .md report is saved next to it)")
    evaluate.add_argument("--baseline", help="an earlier JSON report, to list what got worse")
    evaluate.add_argument("--k", type=int, default=5, help="results scored per query (--input)")
    evaluate.add_argument("--fail-on-regression", action="store_true",
                          help="exit with code 1 when a metric got worse than the baseline")


def setup_settings(args, chunker, retriever, reranker=None):
    """The settings one setup changes: chunker, retriever, reranker, sizes and tuning."""
    settings = {**tuning_settings(args), **({"chunker": chunker} if chunker else {})}
    if retriever:
        settings["retriever"] = retriever
    if reranker:
        settings["reranker"] = reranker
    if args.chunk_size is not None:
        settings["chunk_size"] = args.chunk_size
    if args.chunk_overlap is not None:
        settings["chunk_overlap"] = args.chunk_overlap
    return settings


def combinations(args):
    """Every (chunker, retriever) pair asked for; None means "from the settings"."""
    return itertools.product(args.chunker or [None], getattr(args, "retriever", None) or [None])


def build_setups(args):
    """
    One setup per RAG type x chunker x retriever x reranker,
    e.g. "adaptive + paragraph + hybrid + llm".
    """
    setups = []
    rerankers = getattr(args, "reranker", None) or [None]
    for (chunker, retriever), reranker, rag_type in itertools.product(
            combinations(args), rerankers, args.rag_type):
        name = " + ".join(part for part in (rag_type, chunker, retriever, reranker) if part)
        setups.append({"name": name, "rag_type": rag_type,
                       "settings": setup_settings(args, chunker, retriever, reranker)})
    return setups


def run_evaluate(args):
    """Print the report for `rag-titan evaluate`, then the questions that went wrong."""
    if args.input:
        run_evaluate_data(args)
        return
    if not args.files or not args.questions:
        raise SystemExit("rag-titan evaluate: give --input data.json, "
                         "or --files and --questions")
    questions = load_questions(args.questions)

    if args.retrieval_only:
        results = run_retrieval_only(args, questions)
        print(format_report(results))
    else:
        results = compare_setups(args.files, questions, build_setups(args))
        print(format_rag_report(results))

    print_weak_rows(results)


def run_retrieval_only(args, questions):
    """Score search alone for every chunker x retriever (no LLM)."""
    results = []
    for chunker, retriever in combinations(args):
        # a temporary index: evaluating must never replace a saved one
        settings = {**setup_settings(args, chunker, retriever), "vector_store": "memory"}
        rag = RAGTitan(settings=settings)
        rag.index(args.files)
        name = " + ".join(part for part in (chunker, retriever) if part)
        results.append(evaluate_retriever(rag.parts["retriever"], questions, args.top_k or 5,
                                          name=name or rag.settings["retriever"]))
    return results


def run_evaluate_data(args):
    """Evaluate a data file, save the JSON and Markdown reports and print a summary."""
    if args.fail_on_regression and not args.baseline:
        raise SystemExit("rag-titan evaluate: --fail-on-regression needs --baseline")

    report = evaluate_file(args.input, args.output, k=args.k, baseline=args.baseline)
    print(f"Reports saved: {args.output} and {Path(args.output).with_suffix('.md')}")
    print(summary_text(report))

    if args.fail_on_regression and report["regressions"]:
        raise SystemExit(1)


def print_weak_rows(results):
    """List every question that missed evidence or was not answered."""
    weak = [(result["name"], row) for result in results for row in result["rows"]
            if row["recall"] < 1 or not row.get("answered", True)]
    if not weak:
        print("\nEvery question found all its evidence.")
        return

    print("\nQuestions to look at:")
    for name, row in weak:
        answered = "" if row.get("answered", True) else "  not answered"
        print(f"  {name}: {row['id']}  recall {row['recall']:.2f}{answered}  "
              f"found {sorted(set(row['sources'])) or 'nothing'}")
