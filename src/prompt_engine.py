import os
import re
from pathlib import Path
from typing import Dict, List, Optional


class PromptEngine:
    """Load, select, and populate Markdown prompt templates."""

    def __init__(self, prompts_dir: Optional[str] = None):
        if prompts_dir is None:
            prompts_dir = str(Path(__file__).parent / "prompts")
        self.prompts_dir = prompts_dir

    def load_template(self, name: str) -> str:
        path = Path(self.prompts_dir) / f"{name}.md"
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        return path.read_text()

    def get_business_context(self, business_type: str) -> str:
        path = Path(self.prompts_dir) / "business_types" / f"{business_type}.md"
        if not path.exists():
            path = Path(self.prompts_dir) / "business_types" / "default.md"
        if not path.exists():
            return ""
        return path.read_text()

    def render(self, template_name: str, variables: Dict[str, str],
               business_type: Optional[str] = None) -> str:
        template = self.load_template(template_name)

        if business_type:
            context = self.get_business_context(business_type)
            template = context + "\n\n" + template

        # Substitute all {{placeholder}} with values
        for key, value in variables.items():
            template = template.replace("{{" + key + "}}", str(value))

        # Check for unfilled placeholders
        remaining = re.findall(r"\{\{(\w+)\}\}", template)
        if remaining:
            raise ValueError(f"Unfilled placeholders: {', '.join(remaining)}")

        return template

    def list_templates(self) -> List[str]:
        prompts_path = Path(self.prompts_dir)
        if not prompts_path.exists():
            return []
        templates = []
        for f in prompts_path.iterdir():
            if f.is_file() and f.suffix == ".md":
                templates.append(f.stem)
        return sorted(templates)
