#!/usr/bin/env python3
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
End-to-end example: run the competitive analysis pipeline on sample data.

This script demonstrates the full 6-stage market research pipeline using
included sample data (10 massage/spa businesses in San Jose, CA). It
produces a polished Markdown report with competitive intelligence,
sentiment analysis, market research, and strategic recommendations.

==========================================================================
PREREQUISITES
==========================================================================

1. Install Python dependencies:

       pip install -r requirements.txt

2. Configure API keys (see "API KEY SETUP" below).

3. Run this example:

       cd places-research
       PYTHONPATH=src python3 examples/run_analysis.py

==========================================================================
API KEY SETUP
==========================================================================

The pipeline needs at least ONE LLM key and optionally a web search key.

Option A — Anthropic API key (simplest):

    1. Sign up at https://console.anthropic.com
    2. Create an API key under Settings > API Keys
    3. Set the environment variable:

           export ANTHROPIC_API_KEY="sk-ant-..."

Option B — AWS Bedrock with bearer token (no boto3 needed):

    1. Obtain a Bedrock bearer token from your AWS admin
    2. Set the environment variables:

           export AWS_BEARER_TOKEN_BEDROCK="your-bearer-token"
           export AWS_REGION="us-west-2"    # optional, defaults to us-west-2

Option C — AWS Bedrock with IAM credentials:

    1. Install the Bedrock extras:  pip install anthropic[bedrock]
    2. Configure standard AWS credentials (via env vars, ~/.aws/credentials,
       or IAM role). The pipeline auto-detects them.

Web search key (optional — the pipeline works without it):

    1. Sign up at https://tavily.com for a free API key
    2. Set the environment variable:

           export TAVILY_API_KEY="tvly-..."

    Without this key, the web research stage is skipped gracefully and
    downstream stages (strategy, report) still produce useful output.

Alternatively, copy .env.example to .env and fill in your keys there:

    cp .env.example .env
    # Edit .env with your actual keys

==========================================================================
WHAT THE PIPELINE DOES
==========================================================================

The pipeline runs 6 stages in sequence:

  Stage 1 — Interview (interactive)
      Asks 6 quick questions about your business plan: type of business,
      services you'll offer, target clientele, budget, differentiators,
      and location preference. Takes ~1 minute.

  Stage 2 — Stats (local, no API calls)
      Analyzes competitor data: rating distributions, review volumes,
      geographic density, operating hours, and online presence.

  Stage 3 — Sentiment (LLM)
      Sends customer reviews to Claude in batches. Extracts positive/
      negative themes, service quality patterns, and unmet customer needs.

  Stage 4 — Web Research (LLM + Tavily)
      Searches the web for commercial rent ranges, zoning requirements,
      and industry trends relevant to your business type and location.
      Skipped if TAVILY_API_KEY is not set.

  Stage 5 — Strategy (LLM)
      Combines all prior data to produce competitive positioning advice:
      market saturation assessment, competitor tiers, pricing strategy,
      location recommendations, and a SWOT analysis.

  Stage 6 — Report (LLM)
      Generates a polished Markdown report suitable for sharing with
      partners or investors. Includes executive summary, market overview,
      competitive landscape, and actionable next steps.

==========================================================================
OUTPUT
==========================================================================

All artifacts are saved to an output directory (printed at startup):

    <output_dir>/
        user_profile.json       — Your interview answers
        stats_report.json       — Statistical competitor analysis
        sentiment_report.json   — Review sentiment and themes
        web_research.json       — Market intelligence from web
        strategy_report.json    — Strategic recommendations
        final_report.md         — The polished report (open this!)

Typical cost: ~$0.10-$0.20 for the 10-business sample, ~$0.50-$1.00 for
a full 100-business dataset. The pipeline tracks token usage and prints
a cost summary at the end.

==========================================================================
ADVANCED USAGE
==========================================================================

After running this example, try these variations:

    # Preview stages without making API calls
    PYTHONPATH=src python3 src/analyze.py \\
        --input examples/sample_massage_san_jose.json --dry-run

    # Re-run and skip already-completed stages
    PYTHONPATH=src python3 src/analyze.py \\
        --input examples/sample_massage_san_jose.json --resume

    # Run just one stage
    PYTHONPATH=src python3 src/analyze.py \\
        --input examples/sample_massage_san_jose.json --stage sentiment

    # Set a token budget (pipeline stops if exceeded)
    PYTHONPATH=src python3 src/analyze.py \\
        --input examples/sample_massage_san_jose.json --max-tokens 50000

    # Collect your OWN data and analyze it (requires GOOGLE_MAPS_API_KEY)
    PYTHONPATH=src python3 -m grid_search \\
        --zip 94040 --business "coffee shop" --details
    PYTHONPATH=src python3 src/analyze.py \\
        --input output/output_coffee_shop_mountain_view_94040.json

==========================================================================
"""

import os
import sys

# Add src/ to the path so imports work when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analyze import build_output_dir, parse_args, run_pipeline, STAGES, stage_artifact_path


def main():
    # Path to included sample data (10 massage/spa businesses in San Jose)
    sample_data = os.path.join(os.path.dirname(__file__), "sample_massage_san_jose.json")

    if not os.path.exists(sample_data):
        print(f"ERROR: Sample data not found at {sample_data}")
        print("Make sure you're running from the places-research repository root.")
        sys.exit(1)

    # Parse the same CLI args as analyze.py, with sample data as default input
    argv = sys.argv[1:] if len(sys.argv) > 1 else []
    if "--input" not in argv:
        argv = ["--input", sample_data] + argv

    args = parse_args(argv)

    # Load business data
    import json
    with open(args.input, "r", encoding="utf-8") as f:
        businesses = json.load(f)

    output_dir = build_output_dir(args.output_dir, args.input)

    print("=" * 60)
    print("  Competitive Analysis Pipeline — Example Run")
    print("=" * 60)
    print()
    print(f"  Input:      {args.input} ({len(businesses)} businesses)")
    print(f"  Output:     {output_dir}")
    print()

    # Check API key availability (best-effort: IAM credentials can't be
    # easily detected here, so we only warn — LLMClient handles the real check)
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_bearer = bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK"))
    has_web = bool(os.environ.get("TAVILY_API_KEY"))

    # Try to detect if anthropic[bedrock] is installed (IAM auth)
    try:
        from anthropic import AnthropicBedrock  # noqa: F401
        has_bedrock_sdk = True
    except ImportError:
        has_bedrock_sdk = False

    has_llm = has_api_key or has_bearer or has_bedrock_sdk

    print("  API Keys:")
    if has_api_key:
        print("    LLM auth: Anthropic API key")
    elif has_bearer:
        print("    LLM auth: AWS Bedrock (bearer token)")
    elif has_bedrock_sdk:
        print("    LLM auth: AWS Bedrock (IAM credentials)")
    else:
        print("    LLM auth: MISSING — required")
    print(f"    Web search (Tavily):      {'configured' if has_web else 'not set (web stage will be skipped)'}")
    print()

    if not has_llm and not args.dry_run:
        print("ERROR: No LLM authentication method found.")
        print()
        print("Configure one of the following:")
        print("  Option A: export ANTHROPIC_API_KEY='sk-ant-...'")
        print("  Option B: export AWS_BEARER_TOKEN_BEDROCK='your-token'")
        print("  Option C: pip install anthropic[bedrock]  (+ AWS IAM credentials)")
        print()
        print("Or run with --dry-run to preview without API calls:")
        print(f"  PYTHONPATH=src python3 {__file__} --dry-run")
        sys.exit(1)

    if args.dry_run:
        print("  Mode: DRY RUN (no API calls)")
        print()
        llm_client = None
        prompt_engine = None
        web_searcher = None
    else:
        from llm_client import LLMClient
        from prompt_engine import PromptEngine
        from web_searcher import WebSearcher

        print("  Initializing API clients...")
        llm_client = LLMClient()
        prompt_engine = PromptEngine()
        try:
            web_searcher = WebSearcher()
        except EnvironmentError:
            web_searcher = None
        print(f"  LLM auth method: {llm_client.auth_method}")
        print(f"  LLM model: {llm_client.default_model}")
        print()

    print("-" * 60)
    print("  Starting pipeline... (the interview stage will ask you")
    print("  6 quick questions about your business plan)")
    print("-" * 60)
    print()

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
        print()
        print("Dry run complete. No files were created.")
        sys.exit(0)

    # Print results
    print()
    print("=" * 60)
    if result["success"]:
        print("  Pipeline completed successfully!")
    else:
        print(f"  Pipeline FAILED at stage: {result['failed_stage']}")
        print(f"  Error: {result['error']}")
    print("=" * 60)
    print()

    # Show generated artifacts
    print("  Generated artifacts:")
    for stage_name in STAGES:
        path = stage_artifact_path(output_dir, stage_name)
        if os.path.exists(path):
            size = os.path.getsize(path)
            label = "  <-- open this!" if stage_name == "report" else ""
            print(f"    {path} ({size:,} bytes){label}")
    print()

    # Warnings
    for w in result.get("warnings", []):
        print(f"  Warning: {w}")

    # Per-stage usage
    stage_usage = result.get("stage_usage", {})
    if stage_usage:
        print("  Per-stage token usage:")
        print(f"    {'Stage':<12} {'Input':>10} {'Output':>10} {'Cost':>10}")
        print(f"    {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        for s, su in stage_usage.items():
            print(
                f"    {s:<12} {su['input_tokens']:>10,} "
                f"{su['output_tokens']:>10,} "
                f"${su['cost_usd']:>9.4f}"
            )
        print()

    # Total usage
    usage = result.get("usage", {})
    if usage:
        print(f"  Total: {usage.get('total_calls', 0)} API calls, "
              f"${usage.get('estimated_cost_usd', 0):.4f}")
    print()

    if result["success"]:
        report_path = stage_artifact_path(output_dir, "report")
        print(f"  To view your report:")
        print(f"    cat {report_path}")
        print()
        print("  To re-run (skipping completed stages):")
        print(f"    PYTHONPATH=src python3 {__file__} --resume")

    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
