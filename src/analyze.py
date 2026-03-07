"""
analyze.py — CLI orchestrator that wires all analysis stages with resume support.

Usage::

    python src/analyze.py --input grid_search_results.json
    python src/analyze.py --input data.json --output-dir ./output --resume
    python src/analyze.py --input data.json --stage sentiment
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def run_pipeline(
    businesses: List[dict],
    output_dir: str,
    resume: bool = False,
    stage: Optional[str] = None,
    llm_client=None,
    prompt_engine=None,
    web_searcher=None,
) -> Dict[str, Any]:
    """Run the analysis pipeline and return a result dict.

    Returns:
        Dict with keys: success (bool), completed_stages (list),
        failed_stage (str or None), error (str or None), usage (dict).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    stages_to_run = [stage] if stage else list(STAGES)
    completed_stages: List[str] = []
    artifacts: Dict[str, Any] = {}

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

        try:
            result = _run_stage(
                current_stage, businesses, output_dir, artifacts,
                llm_client, prompt_engine, web_searcher,
            )
            artifacts[current_stage] = result
            _save_artifact(result, stage_artifact_path(output_dir, current_stage))
            completed_stages.append(current_stage)
        except Exception as exc:
            return {
                "success": False,
                "completed_stages": completed_stages,
                "failed_stage": current_stage,
                "error": str(exc),
                "usage": llm_client.get_usage_summary() if llm_client else {},
            }

    return {
        "success": True,
        "completed_stages": completed_stages,
        "failed_stage": None,
        "error": None,
        "usage": llm_client.get_usage_summary() if llm_client else {},
    }


def _run_stage(
    stage: str,
    businesses: List[dict],
    output_dir: str,
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
    print()

    from llm_client import LLMClient
    from prompt_engine import PromptEngine
    from web_searcher import WebSearcher

    llm_client = LLMClient()
    prompt_engine = PromptEngine()
    web_searcher = WebSearcher()

    result = run_pipeline(
        businesses=businesses,
        output_dir=output_dir,
        resume=args.resume,
        stage=args.stage,
        llm_client=llm_client,
        prompt_engine=prompt_engine,
        web_searcher=web_searcher,
    )

    # Print results
    print()
    if result["success"]:
        print("Pipeline completed successfully!")
        print(f"Stages completed: {', '.join(result['completed_stages'])}")
    else:
        print(f"Pipeline failed at stage: {result['failed_stage']}")
        print(f"Error: {result['error']}")
        print(f"Stages completed before failure: {', '.join(result['completed_stages'])}")

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
