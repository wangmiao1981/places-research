# Copyright 2025 Miao Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
analyze.py — CLI orchestrator that wires all analysis stages with resume support.

Usage::

    python src/analyze.py --input grid_search_results.json
    python src/analyze.py --input data.json --output-dir ./output --resume
    python src/analyze.py --input data.json --stage sentiment
    python src/analyze.py --input data.json --dry-run
    python src/analyze.py --input data.json --max-tokens 50000
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from user_interview import UserInterviewer
from stats_analyzer import StatsAnalyzer

# These modules are from Issues #6-9 and may not be present when running
# on branches that haven't merged them yet.  They are imported lazily in
# _run_stage() so the test suite can patch them at the module level.
SentimentAnalyzer = None  # type: ignore[assignment]
WebResearcher = None      # type: ignore[assignment]
StrategyAnalyzer = None   # type: ignore[assignment]
ReportGenerator = None    # type: ignore[assignment]


def _ensure_imports() -> None:
    """Lazily import stage modules that depend on Issues #6-9."""
    global SentimentAnalyzer, WebResearcher, StrategyAnalyzer, ReportGenerator
    if SentimentAnalyzer is None:
        from sentiment_analyzer import SentimentAnalyzer as _SA
        SentimentAnalyzer = _SA
    if WebResearcher is None:
        from web_researcher import WebResearcher as _WR
        WebResearcher = _WR
    if StrategyAnalyzer is None:
        from strategy_analyzer import StrategyAnalyzer as _StA
        StrategyAnalyzer = _StA
    if ReportGenerator is None:
        from report_generator import ReportGenerator as _RG
        ReportGenerator = _RG

STAGES = ["interview", "stats", "sentiment", "web", "strategy", "report"]

_ARTIFACT_FILENAMES = {
    "interview": "user_profile.json",
    "stats": "stats_report.json",
    "sentiment": "sentiment_report.json",
    "web": "web_research.json",
    "strategy": "strategy_report.json",
    "report": "final_report.md",
}

# Stages that make LLM API calls and are subject to the token budget check.
_LLM_STAGES = {"sentiment", "web", "strategy", "report"}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the market analysis pipeline on grid search data.",
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to grid search output JSON (required).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: auto-generated from input filename).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip stages that already have artifacts on disk.",
    )
    parser.add_argument(
        "--stage", choices=STAGES, default=None,
        help="Run a single stage instead of the full pipeline.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what each stage would do without making API calls, then exit.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Stop the pipeline if cumulative token usage exceeds this budget.",
    )
    return parser.parse_args(argv)


def build_output_dir(output_dir: Optional[str], input_path: str) -> str:
    if output_dir is not None:
        return output_dir
    p = Path(input_path)
    return str(p.parent / (p.stem + "_analysis"))


def stage_artifact_path(output_dir: str, stage: str) -> str:
    return str(Path(output_dir) / _ARTIFACT_FILENAMES[stage])


def _load_artifact(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_artifact(data: Any, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        p.write_text(data, encoding="utf-8")
    else:
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _get_token_total(usage_summary: Dict[str, Any]) -> int:
    """Return total tokens (input + output) from a usage summary dict."""
    return (
        usage_summary.get("total_input_tokens", 0)
        + usage_summary.get("total_output_tokens", 0)
    )


def run_pipeline(
    businesses: List[dict],
    output_dir: str,
    resume: bool = False,
    stage: Optional[str] = None,
    llm_client=None,
    prompt_engine=None,
    web_searcher=None,
    dry_run: bool = False,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the analysis pipeline and return a result dict.

    The ``web`` stage is treated as non-fatal: if it fails (e.g. because
    ``TAVILY_API_KEY`` is not configured), the error is recorded in
    ``warnings`` and the pipeline continues with an empty ``web_research``
    artifact so that the strategy and report stages can still produce output.

    Returns:
        Dict with keys: success (bool), completed_stages (list),
        failed_stage (str or None), error (str or None), warnings (list),
        usage (dict), stage_usage (dict mapping stage name to
        {input_tokens, output_tokens, cost_usd}).
    """
    # Dry-run: print plan and return without making any API calls
    if dry_run:
        from sentiment_analyzer import REVIEWS_PER_BATCH
        stages_to_show = [stage] if stage else list(STAGES)
        total_reviews = sum(
            len(b.get("reviews") or []) for b in businesses
        )
        batch_count = (
            (total_reviews + REVIEWS_PER_BATCH - 1) // REVIEWS_PER_BATCH
            if total_reviews else 0
        )
        # Best-effort business type / location from existing interview artifact
        _profile = {}
        if resume or stage:
            _ipath = stage_artifact_path(output_dir, "interview")
            if Path(_ipath).exists():
                try:
                    _profile = _load_artifact(_ipath)
                except (json.JSONDecodeError, OSError):
                    pass
        _btype = _profile.get("business_type", "unknown")
        _loc = _profile.get("data_source") or "unknown location"

        for s in stages_to_show:
            if s == "interview":
                print(f"  [dry-run] interview: would interview with {len(businesses)} businesses")
            elif s == "stats":
                print(f"  [dry-run] stats: would analyze {len(businesses)} businesses")
            elif s == "sentiment":
                print(
                    f"  [dry-run] sentiment: would analyze {total_reviews} reviews "
                    f"in {batch_count} batch(es)"
                )
            elif s == "web":
                print(f"  [dry-run] web: would search for {_btype} in {_loc}")
            elif s == "strategy":
                print("  [dry-run] strategy: would analyze with all prior stage data")
            elif s == "report":
                print("  [dry-run] report: would generate final report")
        return {
            "success": True,
            "completed_stages": [],
            "failed_stage": None,
            "error": None,
            "warnings": [],
            "usage": {},
            "stage_usage": {},
            "dry_run": True,
        }

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    stages_to_run = [stage] if stage else list(STAGES)
    completed_stages: List[str] = []
    warnings: List[str] = []
    artifacts: Dict[str, Any] = {}
    stage_usage: Dict[str, Dict[str, Any]] = {}

    # Pre-load existing artifacts when resuming or running a single late stage
    if resume or stage:
        for s in STAGES:
            path = stage_artifact_path(output_dir, s)
            if Path(path).exists():
                if s == "report":
                    # report is a Markdown file, not JSON — just record that it
                    # exists so the resume logic can skip it.
                    artifacts[s] = True
                else:
                    try:
                        artifacts[s] = _load_artifact(path)
                    except (json.JSONDecodeError, OSError):
                        pass

    for current_stage in stages_to_run:
        # Resume: skip if artifact already exists
        if resume and current_stage in artifacts:
            completed_stages.append(current_stage)
            continue

        # Token budget check before LLM stages
        if current_stage in _LLM_STAGES and max_tokens is not None and llm_client is not None:
            current_usage = llm_client.get_usage_summary()
            if _get_token_total(current_usage) >= max_tokens:
                return {
                    "success": False,
                    "completed_stages": completed_stages,
                    "failed_stage": current_stage,
                    "error": (
                        f"Token budget exceeded before stage '{current_stage}': "
                        f"used {_get_token_total(current_usage)} tokens, "
                        f"budget is {max_tokens} tokens."
                    ),
                    "warnings": warnings,
                    "usage": current_usage,
                    "stage_usage": stage_usage,
                }

        # Snapshot token usage before this stage
        before = llm_client.get_usage_summary() if llm_client else {}

        try:
            result = _run_stage(
                current_stage, businesses, artifacts,
                llm_client, prompt_engine, web_searcher,
            )
            artifacts[current_stage] = result
            _save_artifact(result, stage_artifact_path(output_dir, current_stage))
            completed_stages.append(current_stage)
        except Exception as exc:
            if current_stage == "web":
                # Web stage failure is non-fatal: downstream stages can work
                # with empty web_research (e.g. when TAVILY_API_KEY is absent).
                warning_msg = f"web stage skipped: {exc}"
                logger.warning(warning_msg)
                warnings.append(warning_msg)
                artifacts["web"] = {}
                completed_stages.append(current_stage)
            else:
                return {
                    "success": False,
                    "completed_stages": completed_stages,
                    "failed_stage": current_stage,
                    "error": str(exc),
                    "warnings": warnings,
                    "usage": llm_client.get_usage_summary() if llm_client else {},
                    "stage_usage": stage_usage,
                }

        # Record per-stage delta
        after = llm_client.get_usage_summary() if llm_client else {}
        stage_usage[current_stage] = {
            "input_tokens": (
                after.get("total_input_tokens", 0)
                - before.get("total_input_tokens", 0)
            ),
            "output_tokens": (
                after.get("total_output_tokens", 0)
                - before.get("total_output_tokens", 0)
            ),
            "cost_usd": (
                after.get("estimated_cost_usd", 0.0)
                - before.get("estimated_cost_usd", 0.0)
            ),
        }

    return {
        "success": True,
        "completed_stages": completed_stages,
        "failed_stage": None,
        "error": None,
        "warnings": warnings,
        "usage": llm_client.get_usage_summary() if llm_client else {},
        "stage_usage": stage_usage,
    }


def _run_stage(
    stage: str,
    businesses: List[dict],
    artifacts: Dict[str, Any],
    llm_client, prompt_engine, web_searcher,
) -> Any:
    """Execute a single stage and return its result."""
    user_profile = artifacts.get("interview", {})
    stats_report = artifacts.get("stats", {})
    sentiment_report = artifacts.get("sentiment", {})
    web_research = artifacts.get("web", {})
    strategy = artifacts.get("strategy", {})
    business_type = user_profile.get("business_type", "unknown")
    location = user_profile.get("data_source") or "unknown location"

    if stage == "interview":
        interviewer = UserInterviewer(businesses)
        return interviewer.run()

    elif stage == "stats":
        analyzer = StatsAnalyzer(businesses)
        return analyzer.analyze()

    elif stage == "sentiment":
        _ensure_imports()
        analyzer = SentimentAnalyzer(llm_client, prompt_engine)
        return analyzer.analyze(businesses, business_type, location)

    elif stage == "web":
        if web_searcher is None:
            raise EnvironmentError(
                "Web research unavailable: TAVILY_API_KEY is not configured."
            )
        _ensure_imports()
        researcher = WebResearcher(llm_client, web_searcher, prompt_engine)
        return researcher.research(business_type, location, user_profile)

    elif stage == "strategy":
        _ensure_imports()
        analyzer = StrategyAnalyzer(llm_client, prompt_engine)
        return analyzer.analyze(
            business_type=business_type,
            location=location,
            user_profile=user_profile,
            stats_report=stats_report,
            sentiment_report=sentiment_report,
            web_research=web_research,
        )

    elif stage == "report":
        _ensure_imports()
        generator = ReportGenerator(llm_client, prompt_engine)
        return generator.generate(
            business_type=business_type,
            location=location,
            user_profile=user_profile,
            stats_report=stats_report,
            sentiment_report=sentiment_report,
            web_research=web_research,
            strategy=strategy,
        )

    else:
        raise ValueError(f"Unknown stage: {stage}")


def main() -> None:
    args = parse_args()

    # Load input data
    with open(args.input, "r", encoding="utf-8") as f:
        businesses = json.load(f)

    output_dir = build_output_dir(args.output_dir, args.input)

    print(f"Input: {args.input} ({len(businesses)} businesses)")
    print(f"Output: {output_dir}")
    if args.resume:
        print("Resume mode: skipping completed stages")
    if args.stage:
        print(f"Single stage mode: {args.stage}")
    if args.dry_run:
        print("Dry-run mode: no API calls will be made")
    if args.max_tokens is not None:
        print(f"Token budget: {args.max_tokens} tokens")
    print()

    from llm_client import LLMClient
    from prompt_engine import PromptEngine
    from web_searcher import WebSearcher

    llm_client = LLMClient()
    prompt_engine = PromptEngine()

    # WebSearcher may fail if TAVILY_API_KEY is missing — this is non-fatal
    # since the pipeline treats web stage failures gracefully.
    try:
        web_searcher = WebSearcher()
    except EnvironmentError:
        web_searcher = None

    result = run_pipeline(
        businesses=businesses,
        output_dir=output_dir,
        resume=args.resume,
        stage=args.stage,
        llm_client=llm_client,
        prompt_engine=prompt_engine,
        web_searcher=web_searcher,
        dry_run=args.dry_run,
        max_tokens=args.max_tokens,
    )

    if result.get("dry_run"):
        sys.exit(0)

    # Print results
    print()
    if result["success"]:
        print("Pipeline completed successfully!")
        print(f"Stages completed: {', '.join(result['completed_stages'])}")
    else:
        print(f"Pipeline failed at stage: {result['failed_stage']}")
        print(f"Error: {result['error']}")
        print(f"Stages completed before failure: {', '.join(result['completed_stages'])}")

    # Per-stage usage breakdown
    stage_usage = result.get("stage_usage", {})
    if stage_usage:
        print()
        print("Per-Stage Token Usage:")
        print(f"  {'Stage':<12} {'Input':>10} {'Output':>10} {'Cost (USD)':>12}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*12}")
        for s, su in stage_usage.items():
            print(
                f"  {s:<12} {su['input_tokens']:>10} {su['output_tokens']:>10} "
                f"${su['cost_usd']:>11.4f}"
            )

    # Token usage summary
    usage = result.get("usage", {})
    if usage:
        print()
        print("Token Usage Summary:")
        print(f"  Total calls: {usage.get('total_calls', 0)}")
        print(f"  Input tokens: {usage.get('total_input_tokens', 0)}")
        print(f"  Output tokens: {usage.get('total_output_tokens', 0)}")
        print(f"  Estimated cost: ${usage.get('estimated_cost_usd', 0):.4f}")

    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
