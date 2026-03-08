# places-research

Minimal Google Places market research toolkit with grid-based exhaustive business discovery.

Overcomes the Google Places API's 60-result limit per query by dividing geographic areas into adaptive grids and searching each cell independently, then deduplicating results.

## Quick Start

```bash
# Clone and install
git clone https://github.com/wangmiao1981/places-research.git
cd places-research
pip install -r requirements.txt

# Set up your API key
cp .env.example .env
# Edit .env and add your Google Maps API key
```

## Usage

### Simple Text Search

Search for businesses in a specific city and zip code:

```bash
PYTHONPATH=src python -m data_collector --city "San Francisco" --zip 94107 --keywords "artisan bakery"
```

### Exhaustive Grid Search

Find all businesses of a given type in a zip code area using adaptive grid subdivision:

```bash
PYTHONPATH=src python -m grid_search --zip 95125 --business "massage" --details
```

Grid search options:

```bash
# Basic search
PYTHONPATH=src python -m grid_search --zip 94040 --business "coffee shop"

# With detailed info (reviews, hours, contact) and custom concurrency
PYTHONPATH=src python -m grid_search --zip 95125 --business "massage" --details --max-workers 5

# Disable hybrid search mode
PYTHONPATH=src python -m grid_search --zip 94107 --business "restaurant" --no-hybrid
```

## How Grid Search Works

1. **Geocode** the zip code to get its center coordinates
2. **Create an initial grid** covering the area with overlapping cells
3. **Search each cell** using the Google Places Nearby Search API
4. **Adaptively subdivide** cells that hit the 60-result cap, creating smaller grids for denser areas
5. **Deduplicate** results across all grid cells using place IDs
6. Optionally **fetch detailed info** (reviews, hours, contact) for each unique business

This approach achieves near-complete coverage of businesses in an area, compared to the standard API which caps at 60 results per query.

## Competitive Analysis Pipeline

Once you have business data (from grid search or the included sample), run the
6-stage AI-powered analysis pipeline to get a full market research report.

### Quick Start with Sample Data

No Google Maps key needed — try it immediately with the included sample:

```bash
# Install all dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up API keys (only LLM key is required; web search is optional)
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY (or AWS_BEARER_TOKEN_BEDROCK)
# Optionally set TAVILY_API_KEY for web research

# Run the full pipeline on sample data
PYTHONPATH=src python3 src/analyze.py --input examples/sample_massage_san_jose.json
```

The interactive interview asks 6 quick questions about your business plan, then
the pipeline runs automatically through all stages.

### End-to-End: From Data Collection to Report

```bash
# Step 1: Collect business data (requires GOOGLE_MAPS_API_KEY)
PYTHONPATH=src python3 -m grid_search --zip 95125 --business "massage" --details

# Step 2: Run the analysis pipeline on collected data
PYTHONPATH=src python3 src/analyze.py --input output/output_massage_san_jose_95125.json
```

### Pipeline Stages

| # | Stage | What it does | Output |
|---|-------|-------------|--------|
| 1 | **interview** | Interactive questionnaire about your business plan | `user_profile.json` |
| 2 | **stats** | Statistical analysis of competitors (ratings, reviews, hours, geography) | `stats_report.json` |
| 3 | **sentiment** | LLM-powered theme extraction from customer reviews | `sentiment_report.json` |
| 4 | **web** | Web research on rents, zoning, industry trends | `web_research.json` |
| 5 | **strategy** | Strategic recommendations combining all data | `strategy_report.json` |
| 6 | **report** | Polished Markdown report ready to share | `final_report.md` |

### Pipeline Options

```bash
# Preview what each stage will do (no API calls)
PYTHONPATH=src python3 src/analyze.py --input data.json --dry-run

# Resume after interruption (skips completed stages)
PYTHONPATH=src python3 src/analyze.py --input data.json --resume

# Run a single stage
PYTHONPATH=src python3 src/analyze.py --input data.json --stage sentiment

# Cap spending with a token budget
PYTHONPATH=src python3 src/analyze.py --input data.json --max-tokens 50000

# Custom output directory
PYTHONPATH=src python3 src/analyze.py --input data.json --output-dir ./my_analysis
```

### Cost Estimate

Running the full pipeline on ~100 businesses with Claude Sonnet costs roughly
**$0.50–$1.00** (25 LLM calls). The web research stage adds ~$0.02 if a Tavily
key is configured. Without it, the pipeline continues gracefully — web research
is skipped and downstream stages still produce useful output.

### Research Any Business Type

The pipeline works with any business type — just change the grid search keyword:

```bash
# Coffee shops in Mountain View
PYTHONPATH=src python3 -m grid_search --zip 94040 --business "coffee shop" --details
PYTHONPATH=src python3 src/analyze.py --input output/output_coffee_shop_mountain_view_94040.json

# Yoga studios in Austin
PYTHONPATH=src python3 -m grid_search --zip 78701 --business "yoga studio" --details
PYTHONPATH=src python3 src/analyze.py --input output/output_yoga_studio_austin_78701.json

# Dental clinics in Brooklyn
PYTHONPATH=src python3 -m grid_search --zip 11201 --business "dentist" --details
PYTHONPATH=src python3 src/analyze.py --input output/output_dentist_brooklyn_11201.json
```

## Project Structure

```
places-research/
  src/
    data_collector.py      # Simple text-based search
    grid_search.py         # Exhaustive grid-based search
    analyze.py             # 6-stage analysis pipeline orchestrator
    stats_analyzer.py      # Statistical analysis of competitors
    sentiment_analyzer.py  # LLM-powered review sentiment analysis
    web_researcher.py      # Web search for market intelligence
    strategy_analyzer.py   # Strategic recommendation engine
    report_generator.py    # Final Markdown report generator
    llm_client.py          # Claude API wrapper with retry & tracking
    prompt_engine.py       # Prompt template engine
    web_searcher.py        # Tavily search API with caching
    user_interview.py      # Interactive business plan questionnaire
    rate_limiter.py        # API rate limiting
    security_utils.py      # Safe file I/O utilities
  src/prompts/             # Prompt templates for each analysis stage
  examples/                # Sample data for quick start
  tests/                   # Unit tests
  .github/workflows/       # CI/CD pipelines
```

## Testing

```bash
PYTHONPATH=src python3 -m pytest tests/ -v
```

All tests use mocked API responses and require no API key.

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.
