import unittest
import json
import os
import tempfile
import statistics
from pathlib import Path

# Load fixture data
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_businesses.json"
with open(FIXTURE_PATH) as f:
    SAMPLE_BUSINESSES = json.load(f)


class TestStatsAnalyzerInit(unittest.TestCase):
    def test_init_stores_businesses(self):
        from stats_analyzer import StatsAnalyzer
        analyzer = StatsAnalyzer(SAMPLE_BUSINESSES)
        self.assertEqual(len(analyzer.businesses), 10)

    def test_init_empty_list(self):
        from stats_analyzer import StatsAnalyzer
        analyzer = StatsAnalyzer([])
        self.assertEqual(len(analyzer.businesses), 0)


class TestRatingStats(unittest.TestCase):
    def setUp(self):
        from stats_analyzer import StatsAnalyzer
        self.analyzer = StatsAnalyzer(SAMPLE_BUSINESSES)

    def test_rating_mean(self):
        report = self.analyzer.compute_rating_stats()
        # Ratings: 4.8, 4.5, 5.0, 2.5, 3.8, 4.2, 4.9, 3.2, 4.6 (9 with rating)
        ratings = [4.8, 4.5, 5.0, 2.5, 3.8, 4.2, 4.9, 3.2, 4.6]
        self.assertAlmostEqual(report["mean"], statistics.mean(ratings), places=2)

    def test_rating_median(self):
        report = self.analyzer.compute_rating_stats()
        ratings = [4.8, 4.5, 5.0, 2.5, 3.8, 4.2, 4.9, 3.2, 4.6]
        self.assertAlmostEqual(report["median"], statistics.median(ratings), places=2)

    def test_rating_std_dev(self):
        report = self.analyzer.compute_rating_stats()
        ratings = [4.8, 4.5, 5.0, 2.5, 3.8, 4.2, 4.9, 3.2, 4.6]
        self.assertAlmostEqual(report["std_dev"], statistics.stdev(ratings), places=2)

    def test_rating_count(self):
        report = self.analyzer.compute_rating_stats()
        self.assertEqual(report["with_rating"], 9)
        self.assertEqual(report["without_rating"], 1)

    def test_rating_distribution_buckets(self):
        report = self.analyzer.compute_rating_stats()
        dist = report["distribution"]
        self.assertEqual(dist["1.0-2.0"], 0)
        self.assertEqual(dist["2.0-3.0"], 1)   # 2.5
        self.assertEqual(dist["3.0-4.0"], 2)   # 3.2, 3.8
        self.assertEqual(dist["4.0-4.5"], 2)   # 4.2, 4.5
        self.assertEqual(dist["4.5-5.0"], 4)   # 4.6, 4.8, 4.9, 5.0

    def test_rating_stats_empty(self):
        from stats_analyzer import StatsAnalyzer
        analyzer = StatsAnalyzer([])
        report = analyzer.compute_rating_stats()
        self.assertEqual(report["with_rating"], 0)
        self.assertEqual(report["mean"], 0)

    def test_rating_stats_all_missing(self):
        from stats_analyzer import StatsAnalyzer
        businesses = [{"rating": None}, {"rating": None}]
        analyzer = StatsAnalyzer(businesses)
        report = analyzer.compute_rating_stats()
        self.assertEqual(report["with_rating"], 0)
        self.assertEqual(report["without_rating"], 2)
        self.assertEqual(report["mean"], 0)


class TestReviewVolume(unittest.TestCase):
    def setUp(self):
        from stats_analyzer import StatsAnalyzer
        self.analyzer = StatsAnalyzer(SAMPLE_BUSINESSES)

    def test_review_volume_mean(self):
        report = self.analyzer.compute_review_volume()
        totals = [200, 50, 5, 120, 30, 80, 0, 15, 8, 45]
        self.assertAlmostEqual(report["mean"], statistics.mean(totals), places=1)

    def test_review_volume_median(self):
        report = self.analyzer.compute_review_volume()
        totals = [200, 50, 5, 120, 30, 80, 0, 15, 8, 45]
        self.assertAlmostEqual(report["median"], statistics.median(totals), places=1)

    def test_review_volume_max_min(self):
        report = self.analyzer.compute_review_volume()
        self.assertEqual(report["max"], 200)
        self.assertEqual(report["min"], 0)

    def test_review_volume_distribution(self):
        report = self.analyzer.compute_review_volume()
        dist = report["distribution"]
        self.assertEqual(dist["0"], 1)        # place_7
        self.assertEqual(dist["1-10"], 2)     # 5, 8
        self.assertEqual(dist["11-50"], 4)    # 15, 30, 45, 50
        self.assertEqual(dist["51-100"], 1)   # 80
        self.assertEqual(dist["100+"], 2)     # 120, 200

    def test_review_volume_empty(self):
        from stats_analyzer import StatsAnalyzer
        analyzer = StatsAnalyzer([])
        report = analyzer.compute_review_volume()
        self.assertEqual(report["mean"], 0)
        self.assertEqual(report["max"], 0)


class TestGeographicDensity(unittest.TestCase):
    def setUp(self):
        from stats_analyzer import StatsAnalyzer
        self.analyzer = StatsAnalyzer(SAMPLE_BUSINESSES)

    def test_cell_counts(self):
        report = self.analyzer.compute_geographic_density()
        cells = report["cells"]
        self.assertEqual(cells["grid_0_0"]["count"], 4)  # place 1,2,6,8
        self.assertEqual(cells["grid_1_1"]["count"], 3)   # place 3,4,10
        self.assertEqual(cells["grid_2_2"]["count"], 3)   # place 5,7,9

    def test_densest_cells(self):
        report = self.analyzer.compute_geographic_density()
        densest = report["densest"]
        self.assertEqual(densest[0]["grid"], "grid_0_0")
        self.assertEqual(densest[0]["count"], 4)

    def test_sparsest_cells(self):
        report = self.analyzer.compute_geographic_density()
        sparsest = report["sparsest"]
        # grid_1_1 and grid_2_2 both have 3
        counts = [s["count"] for s in sparsest]
        self.assertIn(3, counts)

    def test_cell_centroids(self):
        report = self.analyzer.compute_geographic_density()
        grid_0_0 = report["cells"]["grid_0_0"]
        # place 1: 37.28,-121.92; place 2: 37.28,-121.91; place 6: 37.29,-121.92; place 8: 37.28,-121.91
        expected_lat = (37.28 + 37.28 + 37.29 + 37.28) / 4
        expected_lon = (-121.92 + -121.91 + -121.92 + -121.91) / 4
        self.assertAlmostEqual(grid_0_0["centroid_lat"], expected_lat, places=3)
        self.assertAlmostEqual(grid_0_0["centroid_lon"], expected_lon, places=3)

    def test_empty_businesses(self):
        from stats_analyzer import StatsAnalyzer
        analyzer = StatsAnalyzer([])
        report = analyzer.compute_geographic_density()
        self.assertEqual(report["cells"], {})
        self.assertEqual(report["densest"], [])
        self.assertEqual(report["sparsest"], [])


class TestOperatingHours(unittest.TestCase):
    def setUp(self):
        from stats_analyzer import StatsAnalyzer
        self.analyzer = StatsAnalyzer(SAMPLE_BUSINESSES)

    def test_weekend_open_pct(self):
        report = self.analyzer.compute_operating_hours()
        # 9 businesses with hours. Weekend open (at least one of Sat/Sun not Closed):
        # place1: Sat open, Sun closed -> yes
        # place2: Sat+Sun open -> yes
        # place3: Sat+Sun closed -> no
        # place4: Sat+Sun open -> yes
        # place5: Sat+Sun closed -> no
        # place6: Sat+Sun open -> yes
        # place8: Sat open, Sun closed -> yes
        # place9: Sat open, Sun closed -> yes
        # place10: Sat open, Sun closed -> yes
        # 7/9 = 77.8%
        self.assertAlmostEqual(report["weekend_open_pct"], 7 / 9 * 100, places=1)

    def test_evening_pct(self):
        report = self.analyzer.compute_operating_hours()
        # Open past 8pm on any weekday:
        # place1: closes 9pm -> yes
        # place2: closes 10pm -> yes
        # place3: closes 7pm -> no
        # place4: closes 11pm -> yes
        # place5: closes 6pm -> no
        # place6: closes 9pm -> yes
        # place8: closes 8pm -> no (not past 8pm)
        # place9: closes 7pm -> no
        # place10: closes 2pm -> no
        # 4/9 = 44.4%
        self.assertAlmostEqual(report["evening_pct"], 4 / 9 * 100, places=1)

    def test_early_morning_pct(self):
        report = self.analyzer.compute_operating_hours()
        # Open before 9am on any weekday:
        # place1: 9am -> no
        # place2: 8am -> yes
        # place3: 10am -> no
        # place4: 7am -> yes
        # place5: 9am -> no
        # place6: 10am -> no
        # place8: 9am -> no
        # place9: 11am -> no
        # place10: 6am -> yes
        # 3/9 = 33.3%
        self.assertAlmostEqual(report["early_morning_pct"], 3 / 9 * 100, places=1)

    def test_no_hours_businesses(self):
        report = self.analyzer.compute_operating_hours()
        self.assertEqual(report["without_hours"], 1)  # place_7

    def test_empty_businesses(self):
        from stats_analyzer import StatsAnalyzer
        analyzer = StatsAnalyzer([])
        report = analyzer.compute_operating_hours()
        self.assertEqual(report["weekend_open_pct"], 0)
        self.assertEqual(report["evening_pct"], 0)

    def test_all_no_hours(self):
        from stats_analyzer import StatsAnalyzer
        businesses = [{"opening_hours": None}, {"opening_hours": None}]
        analyzer = StatsAnalyzer(businesses)
        report = analyzer.compute_operating_hours()
        self.assertEqual(report["with_hours"], 0)
        self.assertEqual(report["without_hours"], 2)


class TestOnlinePresence(unittest.TestCase):
    def setUp(self):
        from stats_analyzer import StatsAnalyzer
        self.analyzer = StatsAnalyzer(SAMPLE_BUSINESSES)

    def test_website_pct(self):
        report = self.analyzer.compute_online_presence()
        # Places with website: 1,2,4,6,8,10 = 6/10 = 60%
        self.assertAlmostEqual(report["website_pct"], 60.0, places=1)

    def test_phone_pct(self):
        report = self.analyzer.compute_online_presence()
        # Places with phone: 1,2,3,4,6,7,8,10 = 8/10 = 80%
        self.assertAlmostEqual(report["phone_pct"], 80.0, places=1)

    def test_hours_pct(self):
        report = self.analyzer.compute_online_presence()
        # Places with hours: all except place_7 = 9/10 = 90%
        self.assertAlmostEqual(report["hours_pct"], 90.0, places=1)

    def test_empty_businesses(self):
        from stats_analyzer import StatsAnalyzer
        analyzer = StatsAnalyzer([])
        report = analyzer.compute_online_presence()
        self.assertEqual(report["website_pct"], 0)
        self.assertEqual(report["phone_pct"], 0)
        self.assertEqual(report["hours_pct"], 0)


class TestAnalyze(unittest.TestCase):
    def test_full_report_keys(self):
        from stats_analyzer import StatsAnalyzer
        analyzer = StatsAnalyzer(SAMPLE_BUSINESSES)
        report = analyzer.analyze()
        self.assertIn("total_businesses", report)
        self.assertIn("rating_stats", report)
        self.assertIn("review_volume", report)
        self.assertIn("geographic_density", report)
        self.assertIn("operating_hours", report)
        self.assertIn("online_presence", report)

    def test_total_businesses(self):
        from stats_analyzer import StatsAnalyzer
        analyzer = StatsAnalyzer(SAMPLE_BUSINESSES)
        report = analyzer.analyze()
        self.assertEqual(report["total_businesses"], 10)

    def test_empty_analyze(self):
        from stats_analyzer import StatsAnalyzer
        analyzer = StatsAnalyzer([])
        report = analyzer.analyze()
        self.assertEqual(report["total_businesses"], 0)


class TestSaveReport(unittest.TestCase):
    def test_save_creates_valid_json(self):
        from stats_analyzer import StatsAnalyzer
        analyzer = StatsAnalyzer(SAMPLE_BUSINESSES)
        report = analyzer.analyze()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            analyzer.save_report(report, path)
            with open(path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["total_businesses"], 10)
            self.assertIn("rating_stats", loaded)
        finally:
            os.unlink(path)

    def test_save_creates_parent_dirs(self):
        from stats_analyzer import StatsAnalyzer
        analyzer = StatsAnalyzer(SAMPLE_BUSINESSES)
        report = analyzer.analyze()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "report.json")
            analyzer.save_report(report, path)
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
