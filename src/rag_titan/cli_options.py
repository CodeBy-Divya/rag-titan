"""
Tuning options shared by `rag-titan ask`, `chat`, `index` and `evaluate`.
Each one wins over its setting in .env, for that command only:

    --top-k 8          TOP_K                  chunks searched
    --top-n 3          TOP_N                  chunks kept after reranking
    --max-attempts 3   ADAPTIVE_MAX_ATTEMPTS  adaptive: searches before giving up
    --parent-size 800  PARENT_SIZE            parent_child: largest section (indexing)
    --parent-budget    PARENT_BUDGET          parent_child: characters given to the model
    --no-hyde-with-question  HYDE_WITH_QUESTION   hyde: search the made-up answer only
    --fusion-queries 4 FUSION_QUERIES         rag_fusion: rewordings to search with
"""

import argparse

# option -> the setting it changes
TUNING = {"top_k": "top_k", "top_n": "top_n", "max_attempts": "adaptive_max_attempts",
          "parent_size": "parent_size", "parent_budget": "parent_budget",
          "hyde_with_question": "hyde_with_question", "fusion_queries": "fusion_queries"}


def positive_int(text):
    """argparse type: a whole number above 0, with a short message otherwise."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a whole number, got {text!r}")
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be above 0, got {value}")
    return value


def add_tuning_options(command, search=True):
    """
    Add the tuning options to one command.

    Args:
        command: the argparse parser of the command.
        search: False for `index`, which only builds the index and so only
            takes --parent-size.
    """
    command.add_argument("--parent-size", type=positive_int,
                         help="parent_child: largest parent section in characters, used when "
                              "indexing (default: PARENT_SIZE, 1500)")
    if not search:
        return
    command.add_argument("--top-k", type=positive_int,
                         help="chunks to search (default: TOP_K, else simple 5, adaptive 4, "
                              "parent_child 8)")
    command.add_argument("--top-n", type=positive_int,
                         help="chunks to keep after reranking (default: TOP_N, else simple 2, "
                              "adaptive 2, parent_child 4)")
    command.add_argument("--max-attempts", type=positive_int,
                         help="adaptive: searches before giving up (default: "
                              "ADAPTIVE_MAX_ATTEMPTS, 2)")
    command.add_argument("--parent-budget", type=positive_int,
                         help="parent_child: most characters of parent text given to the model "
                              "(default: PARENT_BUDGET, 3000)")
    command.add_argument("--hyde-with-question", action=argparse.BooleanOptionalAction,
                         help="hyde: also search with the question and fuse the two "
                              "searches (default: HYDE_WITH_QUESTION, on)")
    command.add_argument("--fusion-queries", type=positive_int,
                         help="rag_fusion: rewordings of the question to search with, "
                              "1 to 10 (default: FUSION_QUERIES, 3)")


def tuning_settings(args):
    """The settings the tuning options given on the command line change."""
    return {setting: getattr(args, option) for option, setting in TUNING.items()
            if getattr(args, option, None) is not None}
