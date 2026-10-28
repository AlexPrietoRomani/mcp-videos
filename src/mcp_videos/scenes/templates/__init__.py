"""Motion-graphics HTML templates bundled with the scene generator."""

from __future__ import annotations

import re
from pathlib import Path

TEMPLATES_DIR = Path(__file__).resolve().parent
BASE_SCENE_TEMPLATE = "base_scene.html"
TECHNICAL_EXPLAINER_TEMPLATE = "technical_explainer.html"
AVAILABLE_TEMPLATES = (BASE_SCENE_TEMPLATE, TECHNICAL_EXPLAINER_TEMPLATE)
_PLACEHOLDER_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


class TemplateNotFoundError(FileNotFoundError):
    """Raised when a template name is not part of the catalog."""


def list_templates() -> list[str]:
    """Return the catalog of bundled HTML templates."""

    return list(AVAILABLE_TEMPLATES)


def load_template(name: str) -> str:
    """Read one bundled template, rejecting path traversal attempts."""

    if name not in AVAILABLE_TEMPLATES:
        raise TemplateNotFoundError(f"Plantilla no disponible: {name!r}")
    return (TEMPLATES_DIR / name).read_text(encoding="utf-8")


def unresolved_placeholders(template: str) -> list[str]:
    """List placeholders still present after rendering."""

    return sorted({match.group(1) for match in _PLACEHOLDER_RE.finditer(template)})


def render_template(name: str, context: dict[str, object]) -> str:
    """Render a template through literal ``{{PLACEHOLDER}}`` substitutions."""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in context:
            return str(context[key])
        return match.group(0)

    rendered = _PLACEHOLDER_RE.sub(replace, load_template(name))
    missing = unresolved_placeholders(rendered)
    if missing:
        raise ValueError(
            f"Placeholders sin resolver en {name}: {', '.join(missing)}"
        )
    return rendered


__all__ = [
    "AVAILABLE_TEMPLATES",
    "BASE_SCENE_TEMPLATE",
    "TECHNICAL_EXPLAINER_TEMPLATE",
    "TEMPLATES_DIR",
    "TemplateNotFoundError",
    "list_templates",
    "load_template",
    "render_template",
    "unresolved_placeholders",
]
