import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


class StatsAnalyzer:
    """Pure Python statistical analysis of grid search business data."""

    def __init__(self, businesses: List[Dict[str, Any]]):
        self.businesses = businesses

    def analyze(self) -> Dict[str, Any]:
        return {
            "total_businesses": len(self.businesses),
            "rating_stats": self.compute_rating_stats(),
            "review_volume": self.compute_review_volume(),
            "geographic_density": self.compute_geographic_density(),
            "operating_hours": self.compute_operating_hours(),
            "online_presence": self.compute_online_presence(),
        }

    def save_report(self, report: Dict[str, Any], path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(report, f, indent=2)

    def compute_rating_stats(self) -> Dict[str, Any]:
        ratings = [b["rating"] for b in self.businesses if b.get("rating") is not None]
        without = len(self.businesses) - len(ratings)

        if not ratings:
            return {
                "mean": 0, "median": 0, "std_dev": 0,
                "with_rating": 0, "without_rating": without,
                "distribution": {"1.0-2.0": 0, "2.0-3.0": 0, "3.0-4.0": 0, "4.0-4.5": 0, "4.5-5.0": 0},
            }

        buckets = {"1.0-2.0": 0, "2.0-3.0": 0, "3.0-4.0": 0, "4.0-4.5": 0, "4.5-5.0": 0}
        for r in ratings:
            if r < 2.0:
                buckets["1.0-2.0"] += 1
            elif r < 3.0:
                buckets["2.0-3.0"] += 1
            elif r < 4.0:
                buckets["3.0-4.0"] += 1
            elif r <= 4.5:
                buckets["4.0-4.5"] += 1
            else:
                buckets["4.5-5.0"] += 1

        return {
            "mean": round(statistics.mean(ratings), 2),
            "median": round(statistics.median(ratings), 2),
            "std_dev": round(statistics.stdev(ratings), 2) if len(ratings) > 1 else 0,
            "with_rating": len(ratings),
            "without_rating": without,
            "distribution": buckets,
        }

    def compute_review_volume(self) -> Dict[str, Any]:
        totals = [b.get("total_ratings", 0) or 0 for b in self.businesses]

        if not totals:
            return {
                "mean": 0, "median": 0, "max": 0, "min": 0,
                "distribution": {"0": 0, "1-10": 0, "11-50": 0, "51-100": 0, "100+": 0},
            }

        buckets = {"0": 0, "1-10": 0, "11-50": 0, "51-100": 0, "100+": 0}
        for t in totals:
            if t == 0:
                buckets["0"] += 1
            elif t <= 10:
                buckets["1-10"] += 1
            elif t <= 50:
                buckets["11-50"] += 1
            elif t <= 100:
                buckets["51-100"] += 1
            else:
                buckets["100+"] += 1

        return {
            "mean": round(statistics.mean(totals), 1),
            "median": round(statistics.median(totals), 1),
            "max": max(totals),
            "min": min(totals),
            "distribution": buckets,
        }

    def compute_geographic_density(self) -> Dict[str, Any]:
        if not self.businesses:
            return {"cells": {}, "densest": [], "sparsest": []}

        groups = defaultdict(list)
        for b in self.businesses:
            grid = b.get("source_grid", "unknown")
            groups[grid].append(b)

        cells = {}
        for grid, businesses in groups.items():
            lats = [b["latitude"] for b in businesses if b.get("latitude") is not None]
            lons = [b["longitude"] for b in businesses if b.get("longitude") is not None]
            cells[grid] = {
                "count": len(businesses),
                "centroid_lat": round(statistics.mean(lats), 4) if lats else 0,
                "centroid_lon": round(statistics.mean(lons), 4) if lons else 0,
            }

        sorted_cells = sorted(cells.items(), key=lambda x: x[1]["count"], reverse=True)
        densest = [{"grid": g, "count": c["count"]} for g, c in sorted_cells[:3]]
        # Sparsest: ascending order, exclude grids already in densest
        densest_grids = {d["grid"] for d in densest}
        sparsest_candidates = [
            {"grid": g, "count": c["count"]}
            for g, c in reversed(sorted_cells)
            if g not in densest_grids
        ][:3]
        return {"cells": cells, "densest": densest, "sparsest": sparsest_candidates}

    def compute_operating_hours(self) -> Dict[str, Any]:
        with_hours = []
        without_hours = 0
        for b in self.businesses:
            hours = b.get("opening_hours")
            if hours:
                with_hours.append(hours)
            else:
                without_hours += 1

        if not with_hours:
            return {
                "with_hours": 0, "without_hours": without_hours,
                "weekend_open_pct": 0, "evening_pct": 0, "early_morning_pct": 0,
                "common_open_hour": None, "common_close_hour": None,
            }

        weekend_open = 0
        evening_open = 0
        early_morning = 0
        open_hours_counter = defaultdict(int)
        close_hours_counter = defaultdict(int)

        for hours_list in with_hours:
            parsed = self._parse_hours_list(hours_list)
            weekday_hours = {d: h for d, h in parsed.items() if d in ("monday", "tuesday", "wednesday", "thursday", "friday")}
            weekend_hours = {d: h for d, h in parsed.items() if d in ("saturday", "sunday")}

            # Weekend: at least one weekend day open
            if any(h is not None for h in weekend_hours.values()):
                weekend_open += 1

            # Evening: any weekday closing past 8pm (20:00)
            for h in weekday_hours.values():
                if h and h[1] > 20:
                    evening_open += 1
                    break

            # Early morning: any weekday opening before 9am
            for h in weekday_hours.values():
                if h and h[0] < 9:
                    early_morning += 1
                    break

            # Track open/close hours for weekdays
            for h in weekday_hours.values():
                if h:
                    open_hours_counter[h[0]] += 1
                    close_hours_counter[h[1]] += 1

        total = len(with_hours)
        common_open = max(open_hours_counter, key=open_hours_counter.get) if open_hours_counter else None
        common_close = max(close_hours_counter, key=close_hours_counter.get) if close_hours_counter else None

        return {
            "with_hours": total,
            "without_hours": without_hours,
            "weekend_open_pct": round(weekend_open / total * 100, 1),
            "evening_pct": round(evening_open / total * 100, 1),
            "early_morning_pct": round(early_morning / total * 100, 1),
            "common_open_hour": common_open,
            "common_close_hour": common_close,
        }

    def _parse_hours_list(self, hours_list: List[str]) -> Dict:
        """Parse opening_hours list into {day: (open_hour, close_hour) or None}."""
        result = {}
        for entry in hours_list:
            match = re.match(r"(\w+):\s*(.+)", entry)
            if not match:
                continue
            day = match.group(1).lower()
            time_str = match.group(2).strip()
            if time_str.lower() == "closed":
                result[day] = None
            else:
                # Split on en-dash/hyphen with optional surrounding unicode whitespace
                times = re.split(r"[\s\u2009\u202f]*[\u2013–-][\s\u2009\u202f]*", time_str)
                if len(times) == 2:
                    open_h = self._parse_time(times[0].strip())
                    close_h = self._parse_time(times[1].strip())
                    if open_h is not None and close_h is not None:
                        result[day] = (open_h, close_h)
                    else:
                        result[day] = None
                else:
                    result[day] = None
        return result

    def _parse_time(self, time_str: str) -> Optional[float]:
        """Parse time string like '9:00 AM' into 24h float (9.0)."""
        # Remove unicode whitespace
        cleaned = re.sub(r"[\u202f\u2009\u00a0]", " ", time_str).strip()
        match = re.match(r"(\d{1,2}):(\d{2})\s*(AM|PM)", cleaned, re.IGNORECASE)
        if not match:
            return None
        hour = int(match.group(1))
        minute = int(match.group(2))
        period = match.group(3).upper()
        if period == "PM" and hour != 12:
            hour += 12
        elif period == "AM" and hour == 12:
            hour = 0
        return hour + minute / 60.0

    def compute_online_presence(self) -> Dict[str, Any]:
        total = len(self.businesses)
        if total == 0:
            return {"website_pct": 0, "phone_pct": 0, "hours_pct": 0}

        with_website = sum(1 for b in self.businesses if b.get("website"))
        with_phone = sum(1 for b in self.businesses if b.get("phone_number"))
        with_hours = sum(1 for b in self.businesses if b.get("opening_hours"))

        return {
            "website_pct": round(with_website / total * 100, 1),
            "phone_pct": round(with_phone / total * 100, 1),
            "hours_pct": round(with_hours / total * 100, 1),
        }
