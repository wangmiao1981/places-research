import unittest
import tempfile
import os
from pathlib import Path


class TestPromptEngineInit(unittest.TestCase):
    def test_default_prompts_dir(self):
        from prompt_engine import PromptEngine
        engine = PromptEngine()
        self.assertTrue(engine.prompts_dir.endswith("prompts"))

    def test_custom_prompts_dir(self):
        from prompt_engine import PromptEngine
        engine = PromptEngine(prompts_dir="/tmp/test_prompts")
        self.assertEqual(engine.prompts_dir, "/tmp/test_prompts")


class TestLoadTemplate(unittest.TestCase):
    def setUp(self):
        from prompt_engine import PromptEngine
        prompts_dir = str(Path(__file__).parent.parent / "src" / "prompts")
        self.engine = PromptEngine(prompts_dir=prompts_dir)

    def test_load_existing_template(self):
        content = self.engine.load_template("sentiment_analysis")
        self.assertIn("customer sentiment analysis", content)
        self.assertIn("{{business_type}}", content)

    def test_load_competitive_strategy(self):
        content = self.engine.load_template("competitive_strategy")
        self.assertIn("{{user_profile}}", content)
        self.assertIn("{{stats_report}}", content)

    def test_load_report_generation(self):
        content = self.engine.load_template("report_generation")
        self.assertIn("Executive Summary", content)

    def test_load_web_research(self):
        content = self.engine.load_template("web_research_synthesis")
        self.assertIn("{{search_results}}", content)

    def test_load_nonexistent_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.engine.load_template("nonexistent_template")


class TestGetBusinessContext(unittest.TestCase):
    def setUp(self):
        from prompt_engine import PromptEngine
        prompts_dir = str(Path(__file__).parent.parent / "src" / "prompts")
        self.engine = PromptEngine(prompts_dir=prompts_dir)

    def test_massage_context(self):
        ctx = self.engine.get_business_context("massage")
        self.assertIn("Massage Business Context", ctx)
        self.assertIn("Swedish", ctx)

    def test_unknown_type_falls_back_to_default(self):
        ctx = self.engine.get_business_context("underwater_basket_weaving")
        self.assertIn("general business analysis", ctx)

    def test_default_context(self):
        ctx = self.engine.get_business_context("default")
        self.assertIn("general business analysis", ctx)


class TestRender(unittest.TestCase):
    def setUp(self):
        from prompt_engine import PromptEngine
        self.tmpdir = tempfile.mkdtemp()
        # Create a simple test template
        os.makedirs(os.path.join(self.tmpdir, "business_types"))
        with open(os.path.join(self.tmpdir, "test_template.md"), "w") as f:
            f.write("Hello {{name}}, welcome to {{city}}!")
        with open(os.path.join(self.tmpdir, "no_placeholders.md"), "w") as f:
            f.write("This template has no placeholders.")
        with open(os.path.join(self.tmpdir, "multi_same.md"), "w") as f:
            f.write("{{x}} and {{x}} and {{y}}")
        with open(os.path.join(self.tmpdir, "business_types", "default.md"), "w") as f:
            f.write("Default context for general business.")
        with open(os.path.join(self.tmpdir, "business_types", "massage.md"), "w") as f:
            f.write("Massage context info.")
        self.engine = PromptEngine(prompts_dir=self.tmpdir)

    def test_render_substitutes_all(self):
        result = self.engine.render("test_template", {"name": "Alice", "city": "San Jose"})
        self.assertEqual(result, "Hello Alice, welcome to San Jose!")

    def test_render_raises_on_missing_placeholder(self):
        with self.assertRaises(ValueError) as ctx:
            self.engine.render("test_template", {"name": "Alice"})
        self.assertIn("city", str(ctx.exception))

    def test_render_no_placeholders(self):
        result = self.engine.render("no_placeholders", {})
        self.assertEqual(result, "This template has no placeholders.")

    def test_render_multiple_same_placeholder(self):
        result = self.engine.render("multi_same", {"x": "A", "y": "B"})
        self.assertEqual(result, "A and A and B")

    def test_render_with_business_type_prepends_context(self):
        result = self.engine.render("test_template", {"name": "Bob", "city": "LA"}, business_type="massage")
        self.assertIn("Massage context info.", result)
        self.assertIn("Hello Bob, welcome to LA!", result)

    def test_render_with_unknown_business_type_uses_default(self):
        result = self.engine.render("no_placeholders", {}, business_type="unknown_biz")
        self.assertIn("Default context", result)

    def test_render_empty_variables_with_placeholders_raises(self):
        with self.assertRaises(ValueError):
            self.engine.render("test_template", {})

    def test_render_value_containing_braces_no_false_positive(self):
        """Values with {{...}} patterns should not trigger unfilled placeholder error."""
        result = self.engine.render("test_template", {
            "name": "Alice",
            "city": "code: {{some_var}} here",
        })
        self.assertIn("{{some_var}}", result)


class TestListTemplates(unittest.TestCase):
    def setUp(self):
        from prompt_engine import PromptEngine
        prompts_dir = str(Path(__file__).parent.parent / "src" / "prompts")
        self.engine = PromptEngine(prompts_dir=prompts_dir)

    def test_list_includes_known_templates(self):
        templates = self.engine.list_templates()
        self.assertIn("sentiment_analysis", templates)
        self.assertIn("competitive_strategy", templates)
        self.assertIn("report_generation", templates)
        self.assertIn("web_research_synthesis", templates)

    def test_list_excludes_business_types_dir(self):
        templates = self.engine.list_templates()
        self.assertNotIn("business_types", templates)

    def test_list_empty_dir(self):
        from prompt_engine import PromptEngine
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PromptEngine(prompts_dir=tmpdir)
            self.assertEqual(engine.list_templates(), [])


class TestIntegration(unittest.TestCase):
    """Integration test: render a full template with all variables."""

    def setUp(self):
        from prompt_engine import PromptEngine
        prompts_dir = str(Path(__file__).parent.parent / "src" / "prompts")
        self.engine = PromptEngine(prompts_dir=prompts_dir)

    def test_render_sentiment_full(self):
        result = self.engine.render("sentiment_analysis", {
            "business_type": "massage",
            "location": "San Jose 95125",
            "reviews_data": "[review data here]",
        })
        self.assertIn("massage", result)
        self.assertIn("San Jose 95125", result)
        self.assertIn("[review data here]", result)

    def test_render_competitive_strategy_with_context(self):
        result = self.engine.render("competitive_strategy", {
            "business_type": "massage",
            "location": "San Jose 95125",
            "user_profile": "{}",
            "stats_report": "{}",
            "sentiment_report": "{}",
            "web_research": "{}",
        }, business_type="massage")
        self.assertIn("Massage Business Context", result)
        self.assertIn("senior business strategy consultant", result)

    def test_render_report_generation_full(self):
        result = self.engine.render("report_generation", {
            "business_type": "massage",
            "location": "San Jose 95125",
            "user_profile": "{}",
            "stats_report": "{}",
            "sentiment_report": "{}",
            "web_research": "{}",
            "strategy": "{}",
        })
        self.assertIn("Executive Summary", result)


class TestPromptEngineEdgeCases(unittest.TestCase):

    def setUp(self):
        from prompt_engine import PromptEngine
        self.tmpdir = tempfile.mkdtemp()
        with open(os.path.join(self.tmpdir, "test.md"), "w") as f:
            f.write("Hello {{name}}!")
        self.engine = PromptEngine(prompts_dir=self.tmpdir)

    def test_path_traversal_in_template_name_raises(self):
        """Template names with .. should be rejected."""
        with self.assertRaises(ValueError):
            self.engine.load_template("../../etc/passwd")

    def test_path_traversal_in_business_type_falls_back(self):
        """Business type with .. should fall back to default, not traverse."""
        os.makedirs(os.path.join(self.tmpdir, "business_types"))
        with open(os.path.join(self.tmpdir, "business_types", "default.md"), "w") as f:
            f.write("safe default")
        result = self.engine.get_business_context("../../etc/passwd")
        self.assertIn("safe default", result)

    def test_extra_variables_silently_ignored(self):
        """Extra variables not in template are silently ignored."""
        result = self.engine.render("test", {"name": "Alice", "extra": "ignored"})
        self.assertEqual(result, "Hello Alice!")


if __name__ == "__main__":
    unittest.main()
