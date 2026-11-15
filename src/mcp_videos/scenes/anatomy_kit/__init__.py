"""Empaquetado y utilidades de inlining para Anatomy IsoKit (wheresryan22/anatomy)."""

from __future__ import annotations

import re
from pathlib import Path

ANATOMY_KIT_DIR: Path = Path(__file__).resolve().parent

ISO_KIT_EXPORTS: tuple[str, ...] = (
    "AZIMUTH",
    "ELEVATION",
    "SHADES",
    "cameraOf",
    "iso",
    "towardViewer",
    "depthOf",
    "pathOf",
    "joinPaths",
    "roundedPlan",
    "extrude",
    "ringOf",
    "slabOf",
    "circleRing",
    "cylinder",
    "insetPlan",
    "onTop",
    "planOutline",
    "lineOnTop",
    "segment",
    "rise",
    "ring",
    "pointsAt",
    "dotGrid",
    "corners",
    "boxCorners",
    "knurl",
    "sideArc",
    "ringSeam",
    "sideSeam",
    "sideRing",
    "coil",
    "topTicks",
    "sideTicks",
    "radialTicks",
    "haloOf",
    "axisVector",
    "translateAlong",
    "holdStill",
    "topMatrix",
    "sideMatrix",
    "screenRise",
    "fitProjection",
    "solidSvg",
    "lineSvg",
    "dotsSvg",
    "fadedSvg",
    "fadeLineSvg",
    "wireSvg",
    "faceTextSvg",
    "groupSvg",
    "figureSvg",
    "plateHtml",
    "pageHtml",
    "ISO_CSS",
)

_ISO_CSS_CACHE: str | None = None
_ISO_KIT_SCRIPT_CACHE: str | None = None
_LINE_ENDS_SCRIPT_CACHE: str | None = None


def load_iso_css() -> str:
    """Devuelve la hoja de estilos CSS oficial de Anatomy (`iso.css`)."""
    global _ISO_CSS_CACHE
    if _ISO_CSS_CACHE is None:
        css_path = ANATOMY_KIT_DIR / "iso.css"
        _ISO_CSS_CACHE = css_path.read_text(encoding="utf-8")
    return _ISO_CSS_CACHE


def load_iso_kit_script() -> str:
    """Devuelve el script JavaScript de IsoKit empaquetado como IIFE auto-ejecutable."""
    global _ISO_KIT_SCRIPT_CACHE
    if _ISO_KIT_SCRIPT_CACHE is None:
        kit_path = ANATOMY_KIT_DIR / "iso-kit.mjs"
        raw_code = kit_path.read_text(encoding="utf-8")
        cleaned = re.sub(r"\bexport\s+(const|function|let|var)\b", r"\1", raw_code)

        mapping = ",\n    ".join(f"{name}: {name}" for name in ISO_KIT_EXPORTS)
        _ISO_KIT_SCRIPT_CACHE = f"""(function (global) {{
  "use strict";

  if (global.IsoKit && typeof global.IsoKit.fitProjection === "function") {{
    return;
  }}

{cleaned}

  var IsoKit = {{
    {mapping},
    ensureStyles: function (doc) {{
      var d = doc || (typeof document !== "undefined" ? document : null);
      if (!d || !d.head) return;
      if (d.querySelector("style[data-mcp-iso-css]")) return;
      var style = d.createElement("style");
      style.setAttribute("data-mcp-iso-css", "1");
      style.textContent = ISO_CSS;
      d.head.appendChild(style);
    }}
  }};

  global.IsoKit = IsoKit;
  if (typeof window !== "undefined") {{
    window.IsoKit = IsoKit;
  }}
}})(typeof window !== "undefined" ? window : globalThis);
"""
    return _ISO_KIT_SCRIPT_CACHE


def load_line_ends_script() -> str:
    """Devuelve el script del auditor geométrico `lineEnds` de Anatomy como IIFE global."""
    global _LINE_ENDS_SCRIPT_CACHE
    if _LINE_ENDS_SCRIPT_CACHE is None:
        script_path = ANATOMY_KIT_DIR / "line-ends.mjs"
        raw_code = script_path.read_text(encoding="utf-8")
        cleaned = re.sub(
            r"\bexport\s+const\s+LINE_ENDS\s*=\s*lineEnds\.toString\(\);?",
            "",
            raw_code,
        )
        _LINE_ENDS_SCRIPT_CACHE = f"""(function (global) {{
  "use strict";

{cleaned}

  global.lineEnds = lineEnds;
  if (typeof window !== "undefined") {{
    window.lineEnds = lineEnds;
  }}
}})(typeof window !== "undefined" ? window : globalThis);
"""
    return _LINE_ENDS_SCRIPT_CACHE


__all__ = [
    "ANATOMY_KIT_DIR",
    "ISO_KIT_EXPORTS",
    "load_iso_css",
    "load_iso_kit_script",
    "load_line_ends_script",
]
