# Competitive Analysis Agent — Design Document

> **Date:** 2026-03-06
> **Project:** places-research
> **Goal:** Build an interactive CLI agent that analyzes grid search output to produce professional competitive analysis reports for business location decisions.

---

## Problem

You have exhaustive Google Places data for a target market (e.g., 99 massage businesses in zip 95125 with ratings, reviews, hours, contact info). You need actionable intelligence: how competitive is the area, what do customers care about, where are the gaps, and where should you open your business. This requires combining statistical analysis, review sentiment, web research, and strategic reasoning.

## Approach

**Multi-stage pipeline with specialized prompts.** Each stage tackles one dimension of analysis with a focused prompt, producing an intermediate JSON artifact. A final stage synthesizes everything into a polished Markdown report.

Why multi-stage over single-pass: each prompt stays focused (better LLM output quality), stages can be retried independently, intermediate artifacts enable debugging and resume.

Why multi-stage over autonomous agent: the analysis task is well-defined — a structured pipeline produces more reliable, predictable results than a ReAct loop.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  CLI Orchestrator                │
│            (src/analyze.py — main entry)         │
├─────────────────────────────────────────────────┤
│                                                  │
│  Stage 1: User Interview (interactive CLI)       │
│     └→ user_profile.json                        │
│                                                  │
│  Stage 2: Statistical Analysis (pure Python)     │
│     └→ stats_report.json                        │
│                                                  │
│  Stage 3: Review Sentiment Analysis (Claude)     │
│     └→ sentiment_report.json                    │
│                                                  │
│  Stage 4: Web Research (search API + Claude)     │
│     └→ web_research.json                        │
│                                                  │
│  Stage 5: Competitive Strategy (Claude)          │
│     └→ strategy.json                            │
│                                                  │
│  Stage 6: Report Generation (Claude)             │
│     └→ final_report.md                          │
│                                                  │
├─────────────────────────────────────────────────┤
│  Support Modules:                                │
│    src/llm_client.py    — Bedrock Claude wrapper │
│    src/web_searcher.py  — Web search wrapper     │
│    src/prompts/         — Prompt templates (MD)  │
│    src/report_builder.py— Markdown assembly      │
└─────────────────────────────────────────────────┘
```

---

## Stage Details

### Stage 1: User Interview (`src/user_interview.py`)

Interactive CLI prompts collecting:
- Business type confirmation (pre-filled from input data)
- Planned services and specialties
- Target clientele
- Budget range
- Differentiators being considered
- Location preferences (foot traffic vs. low rent, etc.)

Output: `user_profile.json`

### Stage 2: Statistical Analysis (`src/stats_analyzer.py`)

Pure Python, no LLM. Computes from grid search data:
- Rating distribution (histogram buckets, mean, median, std)
- Review volume distribution
- Geographic density clustering (which sub-areas are saturated vs. sparse)
- Operating hours patterns (peak hours, weekend coverage, gaps)
- Online presence analysis (% with website, phone)
- Price tier distribution (if available from Places data)

Output: `stats_report.json`

### Stage 3: Review Sentiment Analysis (`src/sentiment_analyzer.py`)

Sends reviews to Claude in batches:
- Positive theme extraction (what customers love)
- Negative theme extraction (common complaints)
- Service quality patterns
- Unmet needs mentioned in reviews
- Per-business sentiment summary

Model: Claude Sonnet (fast, cost-effective for text classification)

Output: `sentiment_report.json`

### Stage 4: Web Research (`src/web_researcher.py`)

Uses web search API to find and Claude to synthesize:
- Local commercial rent ranges for the zip code
- Zoning regulations relevant to the business type
- Industry trends and growth rates
- Yelp/social media presence of top competitors
- Foot traffic / demographics data for the area

Model: Claude Sonnet for synthesis

Output: `web_research.json`

### Stage 5: Competitive Strategy (`src/strategy_analyzer.py`)

Claude synthesizes all prior artifacts + user profile:
- Market saturation assessment
- Competitor tier ranking (leaders, mid-tier, vulnerable)
- Service gap analysis (what's missing in the market)
- Differentiation recommendations tailored to user's plans
- Location recommendations with reasoning (specific sub-areas)
- Risk factors and mitigation

Model: Claude Opus (strategic reasoning benefits from stronger model)

Output: `strategy.json`

### Stage 6: Report Generation (`src/report_generator.py`)

Claude assembles all artifacts into a polished Markdown report:
- Executive summary
- Market overview with stats
- Competitive landscape
- Customer sentiment insights
- Location analysis and recommendation
- Strategic recommendations
- Appendix (data tables, methodology)

Model: Claude Opus

Output: `final_report.md`

---

## Support Modules

### LLM Client (`src/llm_client.py`)

Thin wrapper around `AnthropicBedrock`:
- Authenticates via Bedrock bearer token
- Default model: Sonnet for stages 3-4, Opus for stages 5-6
- Retry with exponential backoff on rate limits
- Token usage tracking and cost estimation per stage
- Configurable via environment variables (`ANALYSIS_MODEL`, `AWS_REGION`, etc.)

### Web Searcher (`src/web_searcher.py`)

- Abstracts search API behind: `search(query) → list[results]`
- Start with Tavily (best free tier for agents), easy to swap
- Rate limiting built in
- Results cached to avoid redundant searches on re-runs

### Prompt Templates (`src/prompts/`)

```
src/prompts/
  sentiment_analysis.md      — Review batch analysis
  web_research_synthesis.md  — Synthesize search results
  competitive_strategy.md    — Strategic analysis
  report_generation.md       — Final report assembly
  business_types/
    massage.md               — Massage-specific context & terminology
    default.md               — Generic fallback
```

Each template has:
- System prompt section (role, expertise, output format)
- Data injection section (`{{placeholder}}` variables populated at runtime)
- Output format specification (JSON schema for stages 3-5, Markdown for stage 6)

Business type templates provide domain-specific context (e.g., for massage: common service types, licensing requirements, typical pricing tiers).

### Prompt Template Engine (`src/prompt_engine.py`)

- Loads templates from `src/prompts/`
- Selects business-type-specific template, falls back to default
- Populates `{{placeholders}}` with runtime data
- Validates all required placeholders are filled

---

## CLI Interface

```bash
# Full run — interactive interview + all stages
PYTHONPATH=src python -m analyze --input output/massage_san_jose_95125.json

# Resume from a specific stage (skips stages with existing artifacts)
PYTHONPATH=src python -m analyze --input output/massage_san_jose_95125.json --resume

# Run a single stage (for debugging/re-running)
PYTHONPATH=src python -m analyze --input output/massage_san_jose_95125.json --stage sentiment

# Specify output directory
PYTHONPATH=src python -m analyze --input output/massage_san_jose_95125.json --output-dir output/analysis_95125
```

Output directory structure:
```
output/analysis_95125/
  user_profile.json
  stats_report.json
  sentiment_report.json
  web_research.json
  strategy.json
  final_report.md
```

---

## GitHub Issues Breakdown

Each issue is an independently testable unit of work, implemented TDD-style (tests first, then implementation).

| Issue | Title | Dependencies | TDD Focus |
|-------|-------|-------------|-----------|
| #1 | LLM client — Bedrock Claude wrapper with retry and token tracking | None | Mock Bedrock responses, test retry logic, token counting |
| #2 | Web searcher — Search API wrapper with caching | None | Mock search API, test caching, rate limiting |
| #3 | Prompt template engine — Load, select, and populate templates | None | Test template loading, placeholder substitution, fallback |
| #4 | User interview — Interactive CLI questionnaire | None | Mock stdin, test question flow, output schema |
| #5 | Statistical analysis stage | None | Test with fixture data, verify all metrics computed |
| #6 | Review sentiment analysis stage | #1, #3 | Mock LLM, test batch processing, theme extraction |
| #7 | Web research stage | #1, #2, #3 | Mock search + LLM, test query generation, synthesis |
| #8 | Competitive strategy stage | #1, #3 | Mock LLM, test artifact loading, strategy output |
| #9 | Report generation stage | #1, #3 | Mock LLM, test Markdown structure, section assembly |
| #10 | CLI orchestrator — Wire all stages with resume support | #4-#9 | Test stage sequencing, resume logic, error handling |

Issues #1-5 have no dependencies and can be worked in parallel. Issues #6-9 depend on foundation modules. Issue #10 is integration.

---

## Dependencies (additions to requirements.txt)

```
anthropic[bedrock]>=0.40.0    # Claude API via Bedrock
tavily-python>=0.3.0          # Web search
```

---

## Future Extensions

- Multi-zip-code comparison (take multiple grid search JSONs, compare markets)
- PDF export from the Markdown report
- Additional business type templates
