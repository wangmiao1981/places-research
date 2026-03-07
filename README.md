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

## Project Structure

```
places-research/
  src/
    data_collector.py      # Simple text-based search
    grid_search.py         # Exhaustive grid-based search
    rate_limiter.py        # API rate limiting
    security_utils.py      # Safe file I/O utilities
  tests/                   # 120 unit tests
  .github/workflows/       # CI/CD pipelines
```

## Testing

```bash
PYTHONPATH=src python -m unittest discover tests -v
```

All tests use mocked API responses and require no API key.

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.
