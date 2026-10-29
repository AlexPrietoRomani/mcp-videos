#!/usr/bin/env python3
# ruff: noqa: E501
"""Generate the forecasting talk short end-to-end through the MCP tools.

The script is intentionally self-contained: it registers two Edge-TTS preset
voices, synthesizes the dialogue, builds four deterministic motion-graphics
scenes, renders at 30 FPS, copies the requested deliverable and verifies the
result with ffprobe/ffmpeg.
"""

from __future__ import annotations

import asyncio
import html
import itertools
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = "charla-forecasting"
PROJECT_TITLE = "¿Puede la IA predecir el futuro? Forecasting Real"
PROJECT_TOPIC = "Cono de incertidumbre, Conformal Prediction y Café de Jaén"
RENDER_DIR = ROOT / "projects" / PROJECT_ID / "renders"
FINAL_MP4 = RENDER_DIR / f"{PROJECT_ID}_final.mp4"
CAPTURE_DIR = RENDER_DIR / "captures"
LOG_PATH = ROOT / "docs" / "logs" / "log.md"
FFMPEG = "/usr/bin/ffmpeg"
FFPROBE = "/usr/bin/ffprobe"

VOICE_HOST = "host-es"
VOICE_EXPERT = "experto-es"
TURN_PAUSES_MS = (140, 160, 140, 180)

DIALOGUE: tuple[dict[str, Any], ...] = (
    {
        "voice_id": VOICE_HOST,
        "speaker_label": "Host",
        "text_content": (
            "¿Tu app promete que lloverá exactamente a las tres con treinta y un grados? "
            "¿Por qué la inteligencia artificial siempre falla esa predicción puntual?"
        ),
        "pause_after_ms": TURN_PAUSES_MS[0],
    },
    {
        "voice_id": VOICE_EXPERT,
        "speaker_label": "Experto",
        "text_content": (
            "Porque prometer un número exacto es una mentira matemática. El futuro es un "
            "cono de incertidumbre caótico que se ensancha conforme miras más adelante."
        ),
        "pause_after_ms": TURN_PAUSES_MS[1],
    },
    {
        "voice_id": VOICE_HOST,
        "speaker_label": "Host",
        "text_content": (
            "¡Totalmente! En Cajamarca los satélites miraron el verdor del café y fallaron "
            "por casi seis quintales por hectárea."
        ),
        "pause_after_ms": TURN_PAUSES_MS[2],
    },
    {
        "voice_id": VOICE_EXPERT,
        "speaker_label": "Experto",
        "text_content": (
            "Exacto, el satélite se saturó de verde y quedó ciego. Al mirar el reloj "
            "biológico de floración, el error cayó a apenas cero punto veintiséis "
            "quintales. La física venció al algoritmo."
        ),
        "pause_after_ms": TURN_PAUSES_MS[3],
    },
)

VOICES: tuple[dict[str, Any], ...] = (
    {
        "voice_id": VOICE_HOST,
        "name": "Host Español",
        "engine": "edge_tts",
        "mode": "preset",
        "language": "es",
        "config": {
            "voice_code": "es-ES-ElviraNeural",
            "rate": "+18%",
            "pitch": "+2Hz",
        },
    },
    {
        "voice_id": VOICE_EXPERT,
        "name": "Experto Español",
        "engine": "edge_tts",
        "mode": "preset",
        "language": "es",
        "config": {
            "voice_code": "es-ES-AlvaroNeural",
            "rate": "+15%",
            "pitch": "-2Hz",
        },
    },
)


class MCPPipelineError(RuntimeError):
    """Raised when one MCP tool returns an error payload."""


class MCPClient:
    """Small synchronous wrapper around ``FastMCP.call_tool``."""

    def __init__(self) -> None:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        self.server = import_module("mcp_videos.mcp_server")

    def call(self, tool: str, arguments: Mapping[str, Any] | None = None) -> Any:
        result = asyncio.run(self.server.mcp.call_tool(tool, dict(arguments or {})))
        if result.is_error:
            raise MCPPipelineError(f"{tool} devolvió un error de protocolo: {result.content}")
        payload = result.structured_content
        if payload is None:
            payload = json.loads(result.content[0].text)
        if isinstance(payload, dict) and set(payload) == {"result"}:
            payload = payload["result"]
        if isinstance(payload, Mapping) and payload.get("status") == "error":
            raise MCPPipelineError(
                f"{tool}: {payload.get('error_type', 'Error')}: {payload.get('error')}"
            )
        return payload


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _metric_cards(items: Iterable[tuple[str, str, str]]) -> str:
    cards: list[str] = []
    for value, label, tone in items:
        cards.append(
            '<div class="metric is-'
            + html.escape(tone)
            + '"><div class="value">'
            + html.escape(value)
            + '</div><div class="label">'
            + html.escape(label)
            + "</div></div>"
        )
    return "".join(cards)


def _scene_boundaries(
    lines: Sequence[Mapping[str, Any]], total_duration_ms: int
) -> list[tuple[int, int]]:
    if len(lines) != 4:
        raise MCPPipelineError(f"se esperaban 4 líneas de diálogo, llegaron {len(lines)}")
    starts = [0]
    for line in lines[1:]:
        starts.append(int(line["start_ms"]))
    if any(right <= left for left, right in itertools.pairwise(starts)):
        raise MCPPipelineError(f"los inicios de escena no son crecientes: {starts}")
    ends = [*starts[1:], total_duration_ms]
    if ends[-1] != total_duration_ms or any(
        end <= start for start, end in zip(starts, ends, strict=True)
    ):
        raise MCPPipelineError(
            f"fronteras de escena inválidas: {list(zip(starts, ends, strict=True))}"
        )
    return list(zip(starts, ends, strict=True))


def _cues_for_scene(
    words: Sequence[Mapping[str, Any]],
    start_ms: int,
    end_ms: int,
    fallback_text: str,
) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    for word in words:
        word_start = int(word["start_ms"])
        word_end = int(word["end_ms"])
        if word_end <= start_ms or word_start >= end_ms:
            continue
        local_start = max(0, word_start - start_ms)
        local_end = min(end_ms - start_ms, word_end - start_ms)
        if local_end <= local_start:
            local_end = min(end_ms - start_ms, local_start + 180)
        cues.append(
            {
                "word": str(word["word"]),
                "start_ms": local_start,
                "end_ms": local_end,
            }
        )
    if cues:
        return cues

    tokens = [token for token in fallback_text.split() if token]
    step = max(1, (end_ms - start_ms) // max(1, len(tokens)))
    return [
        {
            "word": token,
            "start_ms": index * step,
            "end_ms": min(end_ms - start_ms, (index + 1) * step),
        }
        for index, token in enumerate(tokens)
    ]


HTML_TEMPLATE = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=1080, height=1920">
<title>%%PAGE_TITLE%%</title>
<style>
  :root {
    --bg: #05070d;
    --panel: #0b1220;
    --ink: #f7fbff;
    --muted: #8ea0b8;
    --accent: %%ACCENT%%;
    --accent-glow: %%ACCENT_GLOW%%;
    --accent2: #f5c842;
    --danger: #fb7185;
    --good: #34d399;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: var(--bg);
    color: var(--ink);
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
  }
  body::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(148,163,184,.055) 1px, transparent 1px),
      linear-gradient(90deg, rgba(148,163,184,.055) 1px, transparent 1px);
    background-size: 60px 60px;
    mask-image: radial-gradient(circle at 50% 35%, black, transparent 78%);
    pointer-events: none;
  }
  body::after {
    content: "";
    position: fixed;
    inset: -20%;
    background: radial-gradient(circle at 50% 18%, var(--accent-glow), transparent 55%);
    pointer-events: none;
  }
  .shell {
    position: relative;
    z-index: 1;
    width: 100%;
    height: 100%;
    padding: 62px 62px 54px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  .topline {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 21px;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 800;
  }
  .brand { display: flex; align-items: center; gap: 12px; color: var(--ink); }
  .brand::before {
    content: "";
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 26px var(--accent);
  }
  .signal { color: var(--accent); }
  .eyebrow {
    color: var(--accent);
    font-size: 21px;
    font-weight: 900;
    letter-spacing: .19em;
    text-transform: uppercase;
    margin-bottom: 10px;
  }
  h1 {
    margin: 0;
    font-size: 76px;
    line-height: .98;
    letter-spacing: -.045em;
    font-weight: 900;
    max-width: 950px;
    text-wrap: balance;
  }
  h1 .accent { color: var(--accent); }
  .subtitle {
    margin: 16px 0 0;
    font-size: 29px;
    line-height: 1.28;
    color: #c7d3e3;
    max-width: 920px;
  }
  .stage-wrap { flex: 1; min-height: 0; display: flex; }
  .stage {
    position: relative;
    flex: 1;
    min-height: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    border: 1px solid rgba(148,163,184,.2);
    border-radius: 34px;
    background: linear-gradient(180deg, rgba(15,23,42,.8), rgba(5,7,13,.92));
    box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 28px 70px rgba(0,0,0,.42);
  }
  .stage::before {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 50% 0%, var(--accent-glow), transparent 58%);
    pointer-events: none;
  }
  .stage svg { width: 100%; height: 100%; display: block; }
  .corner { position: absolute; width: 26px; height: 26px; opacity: .55; }
  .corner.tl { top: 18px; left: 18px; border-top: 2px solid var(--accent); border-left: 2px solid var(--accent); }
  .corner.tr { top: 18px; right: 18px; border-top: 2px solid var(--accent); border-right: 2px solid var(--accent); }
  .corner.bl { bottom: 18px; left: 18px; border-bottom: 2px solid var(--accent); border-left: 2px solid var(--accent); }
  .corner.br { bottom: 18px; right: 18px; border-bottom: 2px solid var(--accent); border-right: 2px solid var(--accent); }
  .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
  .metric {
    border: 1px solid rgba(148,163,184,.18);
    border-radius: 22px;
    padding: 16px 19px;
    background: rgba(15,23,42,.66);
  }
  .metric .value { font-size: 35px; line-height: 1.05; font-weight: 900; letter-spacing: -.03em; }
  .metric .label {
    margin-top: 5px;
    font-size: 16px;
    line-height: 1.2;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .metric.is-accent .value { color: var(--accent); }
  .metric.is-danger .value { color: var(--danger); }
  .metric.is-good .value { color: var(--good); }
  .metric.is-gold .value { color: var(--accent2); }
  .cues {
    border: 1px solid rgba(148,163,184,.18);
    border-radius: 24px;
    padding: 15px 20px 17px;
    background: rgba(15,23,42,.66);
  }
  .cues-head {
    display: flex;
    justify-content: space-between;
    font-size: 15px;
    letter-spacing: .13em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 800;
  }
  .cues-wrap { overflow: hidden; margin-top: 11px; }
  .cues-inner { position: relative; display: flex; gap: 10px; width: max-content; }
  .cue {
    white-space: nowrap;
    padding: 7px 12px;
    border-radius: 12px;
    background: rgba(148,163,184,.085);
    color: var(--muted);
    font-size: 19px;
    font-weight: 750;
    border: 1px solid transparent;
  }
  .cue.active {
    color: #031018;
    background: var(--accent);
    border-color: var(--accent);
    box-shadow: 0 0 30px var(--accent-glow);
  }
  .progress {
    height: 6px;
    border-radius: 999px;
    background: rgba(148,163,184,.18);
    overflow: hidden;
    margin-top: 13px;
  }
  .progress > i {
    display: block;
    height: 100%;
    width: 0;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
  }
  .svg-clock { font-size: 102px; font-weight: 900; fill: var(--ink); letter-spacing: -.06em; }
  .svg-big { font-size: 66px; font-weight: 900; letter-spacing: -.03em; }
  .svg-small { font-size: 20px; font-weight: 850; letter-spacing: .14em; fill: var(--muted); }
  .svg-tiny { font-size: 18px; font-weight: 700; fill: var(--muted); }
</style>
</head>
<body>
<div class="shell">
  <div class="topline">
    <div class="brand">MCP VIDEOS / FORECASTING</div>
    <div class="signal">ESCENA %%SCENE_NUMBER%% / 04</div>
  </div>
  <div>
    <div class="eyebrow">%%EYEBROW%%</div>
    <h1>%%TITLE%%</h1>
    <p class="subtitle">%%SUBTITLE%%</p>
  </div>
  <div class="stage-wrap">
    <div class="stage">
      <span class="corner tl"></span>
      <span class="corner tr"></span>
      <span class="corner bl"></span>
      <span class="corner br"></span>
      %%SCENE_BODY%%
    </div>
  </div>
  <div class="metrics">%%METRICS%%</div>
  <div class="cues">
    <div class="cues-head"><span>Transcripción sincronizada</span><span id="timecode">0.00 s</span></div>
    <div class="cues-wrap" id="cues-wrap">
      <div class="cues-inner" id="cues"></div>
    </div>
    <div class="progress"><i id="progress-fill"></i></div>
  </div>
</div>
<script>
%%SCENE_SCRIPT%%
(function () {
  "use strict";
  const DURATION_MS = %%DURATION_MS%%;
  const CUES = %%CUES_JSON%%;
  const root = document.documentElement;
  const cueWrap = document.getElementById("cues-wrap");
  const cueInner = document.getElementById("cues");
  const progressFill = document.getElementById("progress-fill");
  const timecode = document.getElementById("timecode");

  const cueEls = CUES.map(function (cue, index) {
    const el = document.createElement("span");
    el.className = "cue";
    el.id = "cue-" + index;
    el.textContent = cue.word;
    cueInner.appendChild(el);
    return el;
  });

  function clamp01(value) {
    return Math.max(0, Math.min(1, value));
  }

  function easeOutCubic(value) {
    return 1 - Math.pow(1 - value, 3);
  }

  function updateCues(timeMs) {
    let active = -1;
    for (let index = 0; index < CUES.length; index += 1) {
      if (timeMs >= CUES[index].start_ms && timeMs < CUES[index].end_ms) {
        active = index;
        break;
      }
    }
    if (active < 0 && CUES.length > 0 && timeMs >= CUES[CUES.length - 1].end_ms) {
      active = CUES.length - 1;
    }
    cueEls.forEach(function (el, index) {
      el.classList.toggle("active", index === active);
    });
    if (active >= 0) {
      const el = cueEls[active];
      const target = el.offsetLeft - (cueWrap.clientWidth - el.offsetWidth) / 2;
      const maxScroll = Math.max(0, cueWrap.scrollWidth - cueWrap.clientWidth);
      cueWrap.scrollLeft = Math.max(0, Math.min(maxScroll, target));
    }
  }

  window.__setFrameTime = function (timeMs) {
    const duration = Math.max(1, DURATION_MS);
    const t = Math.max(0, Math.min(duration, Number(timeMs) || 0));
    const progress = clamp01(t / duration);
    const eased = easeOutCubic(progress);
    window.__currentTimeMs = t;
    root.style.setProperty("--time-ms", String(t));
    progressFill.style.width = (progress * 100).toFixed(3) + "%";
    timecode.textContent = (t / 1000).toFixed(2) + " s";
    updateCues(t);
    renderScene(t, progress, eased, duration);
  };

  window.__setFrameTime(0);
})();
</script>
</body>
</html>
"""


SCENE_0_SCRIPT = """
function renderScene(t, p, e, duration) {
  const hand = document.getElementById("clock-hand");
  if (hand) {
    hand.setAttribute("transform", "rotate(" + ((p * 1080) % 360).toFixed(2) + " 450 430)");
  }
  const ring = document.getElementById("clock-ring");
  if (ring) ring.style.strokeDashoffset = String((1540 * (1 - e)).toFixed(2));
  const card = document.getElementById("forecast-card");
  if (card) {
    card.setAttribute("transform", "translate(" + (Math.sin(p * 38) * 6).toFixed(2) + " 0)");
  }
  const crack = document.getElementById("crack");
  if (crack) {
    const reveal = Math.max(0, Math.min(1, (p - 0.36) / 0.64));
    crack.style.strokeDashoffset = String((620 * (1 - reveal)).toFixed(2));
  }
  const shards = document.querySelectorAll("#shards circle");
  shards.forEach(function (el, index) {
    const angle = (index / Math.max(1, shards.length)) * Math.PI * 2 + 0.35;
    const distance = Math.max(0, p - 0.52) * 260;
    el.setAttribute(
      "transform",
      "translate(" + (Math.cos(angle) * distance).toFixed(2) + " " +
        (Math.sin(angle) * distance).toFixed(2) + ")"
    );
    el.setAttribute("opacity", String(Math.max(0, Math.min(1, (p - 0.52) * 6))));
  });
  const verdict = document.getElementById("verdict-0");
  if (verdict) verdict.setAttribute("opacity", String(Math.max(0, Math.min(1, (p - 0.62) * 4))));
}
"""

SCENE_1_SCRIPT = """
function renderScene(t, p, e, duration) {
  const outer = document.getElementById("cone-outer");
  const inner = document.getElementById("band-inner");
  if (outer) {
    const half = 18 + 312 * e;
    outer.setAttribute(
      "points",
      "450,300 " + (450 - half).toFixed(2) + ",880 " + (450 + half).toFixed(2) + ",880"
    );
  }
  if (inner) {
    const half = 6 + 156 * e;
    inner.setAttribute(
      "points",
      "450,300 " + (450 - half).toFixed(2) + ",880 " + (450 + half).toFixed(2) + ",880"
    );
  }
  const dot = document.getElementById("truth-dot");
  if (dot) {
    dot.setAttribute("cx", String((450 + Math.sin(p * 7.5) * 70 * p).toFixed(2)));
    dot.setAttribute("cy", String((300 + p * 580).toFixed(2)));
  }
  const outerLabel = document.getElementById("outer-label");
  if (outerLabel) outerLabel.setAttribute("opacity", String(0.2 + 0.8 * e));
  const innerLabel = document.getElementById("inner-label");
  if (innerLabel) innerLabel.setAttribute("opacity", String(Math.max(0, Math.min(1, (p - 0.18) * 4))));
  const badge = document.getElementById("coverage-badge");
  if (badge) badge.setAttribute("opacity", String(Math.max(0, Math.min(1, (p - 0.22) * 4))));
}
"""

SCENE_2_SCRIPT = """
function renderScene(t, p, e, duration) {
  const curve = document.getElementById("ndvi-curve");
  if (curve) curve.style.strokeDashoffset = String((1100 * (1 - e)).toFixed(2));
  const zone = document.getElementById("saturation-zone");
  if (zone) zone.setAttribute("opacity", String(0.12 + 0.28 * e));
  const points = document.querySelectorAll(".ndvi-point");
  points.forEach(function (el, index) {
    const threshold = (index + 1) / Math.max(1, points.length);
    el.setAttribute("opacity", String(Math.max(0, Math.min(1, (e - threshold * 0.8) * 5))));
  });
  const trap = document.getElementById("trap-label");
  if (trap) trap.setAttribute("opacity", String(Math.max(0, Math.min(1, (p - 0.34) * 3))));
  const satDot = document.getElementById("sat-dot");
  if (satDot) satDot.setAttribute("opacity", String(Math.max(0, Math.min(1, (p - 0.42) * 4))));
  const slash = document.getElementById("blind-slash");
  if (slash) slash.setAttribute("opacity", String(Math.max(0, Math.min(1, (p - 0.28) * 4))));
}
"""

SCENE_3_SCRIPT = """
function renderScene(t, p, e, duration) {
  const satBar = document.getElementById("sat-bar");
  if (satBar) satBar.setAttribute("width", String((520 * e).toFixed(2)));
  const phenoBar = document.getElementById("pheno-bar");
  if (phenoBar) phenoBar.setAttribute("width", String(Math.max(7, 520 * (0.26 / 5.80) * e).toFixed(2)));
  const satValue = document.getElementById("sat-value");
  if (satValue) satValue.textContent = (5.80 * e).toFixed(2);
  const phenoValue = document.getElementById("pheno-value");
  if (phenoValue) phenoValue.textContent = (0.26 * e).toFixed(2);
  const verdict = document.getElementById("verdict-3");
  if (verdict) verdict.setAttribute("opacity", String(Math.max(0, Math.min(1, (p - 0.54) * 3))));
  const regLine = document.getElementById("reg-line");
  if (regLine) regLine.style.strokeDashoffset = String((620 * (1 - e)).toFixed(2));
  const regPoints = document.querySelectorAll(".reg-point");
  regPoints.forEach(function (el, index) {
    const threshold = (index + 1) / Math.max(1, regPoints.length);
    el.setAttribute("opacity", String(Math.max(0, Math.min(1, (e - threshold * 0.7) * 5))));
  });
}
"""


def _scene_0_body() -> str:
    ticks = "\n".join(
        f'    <line x1="450" y1="188" x2="450" y2="216" transform="rotate({angle} 450 430)"/>'
        for angle in range(0, 360, 30)
    )
    shards = "\n".join(
        f'    <circle cx="{cx}" cy="{cy}" r="{radius}" opacity="0"/>'
        for cx, cy, radius in (
            (286, 742, 8),
            (352, 702, 5),
            (520, 716, 7),
            (610, 768, 5),
            (308, 838, 6),
            (566, 862, 9),
            (646, 828, 4),
            (430, 936, 5),
        )
    )
    return f"""
<svg viewBox="0 0 900 1080" role="img" aria-label="Reloj y pronóstico puntual rompiéndose">
  <defs>
    <radialGradient id="clock-glow" cx="50%" cy="40%" r="55%">
      <stop offset="0%" stop-color="#67e8f9" stop-opacity="0.2"/>
      <stop offset="100%" stop-color="#67e8f9" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <circle cx="450" cy="430" r="350" fill="url(#clock-glow)"/>
  <circle cx="450" cy="430" r="300" fill="none" stroke="rgba(148,163,184,.25)" stroke-width="2" stroke-dasharray="3 18"/>
  <circle id="clock-ring" cx="450" cy="430" r="250" fill="rgba(8,15,30,.72)" stroke="#67e8f9" stroke-width="10" stroke-dasharray="1540" stroke-dashoffset="1540" stroke-linecap="round" transform="rotate(-90 450 430)"/>
  <circle cx="450" cy="430" r="218" fill="none" stroke="rgba(103,232,249,.2)" stroke-width="2"/>
  <g stroke="rgba(226,232,240,.55)" stroke-width="3" stroke-linecap="round">
{ticks}
  </g>
  <line id="clock-hand" x1="450" y1="430" x2="450" y2="238" stroke="#67e8f9" stroke-width="8" stroke-linecap="round"/>
  <circle cx="450" cy="430" r="16" fill="#67e8f9"/>
  <text x="450" y="520" text-anchor="middle" class="svg-small">HORA PROMETIDA</text>
  <text x="450" y="612" text-anchor="middle" class="svg-clock">15:30</text>
  <g id="forecast-card">
    <rect x="235" y="700" width="430" height="190" rx="28" fill="rgba(8,15,30,.94)" stroke="rgba(251,113,133,.72)" stroke-width="3"/>
    <text x="450" y="750" text-anchor="middle" class="svg-small">PRONÓSTICO PUNTUAL</text>
    <text x="450" y="832" text-anchor="middle" class="svg-big" fill="#fb7185">31.4 °C</text>
    <text x="450" y="870" text-anchor="middle" class="svg-tiny">“Lluvia exacta a las 15:30”</text>
  </g>
  <path id="crack" d="M235 700 L318 760 L286 790 L372 830 L338 858 L446 900 L420 940" fill="none" stroke="#fecdd3" stroke-width="6" stroke-linejoin="round" stroke-dasharray="620" stroke-dashoffset="620"/>
  <g id="shards" fill="#fda4af">
{shards}
  </g>
  <text id="verdict-0" x="450" y="995" text-anchor="middle" class="svg-small" fill="#fb7185" opacity="0">LA CERTEZA SE ROMPE</text>
</svg>
"""


def _scene_1_body() -> str:
    return """
<svg viewBox="0 0 900 1080" role="img" aria-label="Cono de incertidumbre y banda conformal">
  <defs>
    <linearGradient id="cone-fill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#f5c842" stop-opacity="0.03"/>
      <stop offset="100%" stop-color="#f5c842" stop-opacity="0.3"/>
    </linearGradient>
  </defs>
  <text x="450" y="172" text-anchor="middle" class="svg-small">EL FUTURO ES UN CONO</text>
  <text x="450" y="240" text-anchor="middle" class="svg-tiny">λ &gt; 0 · la incertidumbre se ensancha</text>
  <line x1="80" y1="880" x2="820" y2="880" stroke="rgba(148,163,184,.35)" stroke-width="2"/>
  <line x1="450" y1="300" x2="450" y2="900" stroke="rgba(148,163,184,.25)" stroke-width="2" stroke-dasharray="6 12"/>
  <polygon id="cone-outer" points="450,300 432,880 468,880" fill="url(#cone-fill)" stroke="#f5c842" stroke-width="4" stroke-dasharray="12 10"/>
  <polygon id="band-inner" points="450,300 444,880 456,880" fill="rgba(103,232,249,.08)" stroke="#67e8f9" stroke-width="5"/>
  <path d="M450 300 C 520 430, 405 540, 505 660 S 430 800 455 880" fill="none" stroke="#34d399" stroke-width="4" stroke-dasharray="7 11" opacity=".8"/>
  <circle id="truth-dot" cx="450" cy="300" r="10" fill="#34d399"/>
  <text x="450" y="946" text-anchor="middle" class="svg-small">HOY</text>
  <text x="780" y="946" text-anchor="middle" class="svg-small">t + H</text>
  <text id="outer-label" x="112" y="826" class="svg-small" fill="#f5c842" opacity="0.2">CONO DE LYAPUNOV</text>
  <text id="inner-label" x="592" y="782" class="svg-small" fill="#67e8f9" opacity="0">SPLIT CONFORMAL 90%</text>
  <g id="coverage-badge" opacity="0">
    <rect x="620" y="612" width="150" height="58" rx="29" fill="rgba(245,200,66,.16)" stroke="#f5c842" stroke-width="2"/>
    <text x="695" y="650" text-anchor="middle" font-size="27" font-weight="900" fill="#f5c842">90%</text>
  </g>
</svg>
"""


def _ndvi_curve_path() -> str:
    points: list[str] = []
    for index in range(31):
        lai = index * 6 / 30
        ndvi = 0.18 + 0.72 * (1 - math.exp(-lai / 1.1))
        x = 150 + (lai / 6) * 600
        y = 780 - ndvi * 480
        points.append(f"{x:.1f} {y:.1f}")
    return "M" + " L".join(points)


def _ndvi_scatter() -> str:
    circles: list[str] = []
    for index, lai in enumerate(
        (0.3, 0.55, 0.8, 1.05, 1.3, 1.6, 1.95, 2.3, 2.7, 3.1, 3.5, 4.0, 4.5, 5.0, 5.5)
    ):
        base = 0.18 + 0.72 * (1 - math.exp(-lai / 1.1))
        ndvi = min(0.86, base + 0.018 * math.sin(index * 2.1))
        x = 150 + (lai / 6) * 600
        y = 780 - ndvi * 480
        radius = 8 if lai <= 3 else 10
        circles.append(
            f'    <circle class="ndvi-point" cx="{x:.1f}" cy="{y:.1f}" r="{radius}" '
            'fill="#fb7185" stroke="#fecdd3" stroke-width="1.5" opacity="0"/>'
        )
    return "\n".join(circles)


def _scene_2_body() -> str:
    x_ticks: list[str] = []
    for value in range(7):
        x = 150 + (value / 6) * 600
        x_ticks.append(
            f'    <line x1="{x:.1f}" y1="780" x2="{x:.1f}" y2="792" stroke="rgba(148,163,184,.4)" stroke-width="2"/>'
            f'<text x="{x:.1f}" y="822" text-anchor="middle" class="svg-tiny">{value}</text>'
        )
    y_ticks: list[str] = []
    for value in (0.0, 0.3, 0.6, 0.9):
        y = 780 - value * 480
        y_ticks.append(
            f'    <line x1="138" y1="{y:.1f}" x2="150" y2="{y:.1f}" stroke="rgba(148,163,184,.4)" stroke-width="2"/>'
            f'<text x="130" y="{y + 6:.1f}" text-anchor="end" class="svg-tiny">{value:.1f}</text>'
        )
    return f"""
<svg viewBox="0 0 900 1080" role="img" aria-label="NDVI saturado y error del satélite">
  <defs>
    <linearGradient id="ndvi-area" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#67e8f9" stop-opacity="0.16"/>
      <stop offset="100%" stop-color="#67e8f9" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <text x="450" y="166" text-anchor="middle" class="svg-small">LA TRAMPA DEL VERDE</text>
  <text x="150" y="268" class="svg-tiny" fill="#fb7185">SENSOR NDVI · CAJAMARCA</text>
  <g id="satellite" transform="translate(760 212)">
    <rect x="-56" y="-16" width="34" height="32" rx="5" fill="#1e293b" stroke="#94a3b8" stroke-width="2"/>
    <rect x="22" y="-16" width="34" height="32" rx="5" fill="#1e293b" stroke="#94a3b8" stroke-width="2"/>
    <rect x="-18" y="-23" width="36" height="46" rx="9" fill="#0f172a" stroke="#67e8f9" stroke-width="2"/>
    <circle id="sat-lens" cx="0" cy="0" r="10" fill="#67e8f9" opacity=".8"/>
    <line id="blind-slash" x1="-30" y1="-34" x2="30" y2="34" stroke="#fb7185" stroke-width="6" stroke-linecap="round" opacity="0"/>
    <text x="0" y="62" text-anchor="middle" class="svg-tiny" fill="#fb7185">CEGADO</text>
  </g>
  <line x1="150" y1="780" x2="750" y2="780" stroke="rgba(148,163,184,.42)" stroke-width="2"/>
  <line x1="150" y1="300" x2="150" y2="780" stroke="rgba(148,163,184,.42)" stroke-width="2"/>
{x_ticks}
{y_ticks}
  <rect id="saturation-zone" x="450" y="300" width="300" height="480" fill="#fb7185" opacity=".12"/>
  <line x1="450" y1="300" x2="450" y2="780" stroke="#fb7185" stroke-width="3" stroke-dasharray="7 9"/>
  <text x="462" y="334" class="svg-tiny" fill="#fb7185">LAI &gt; 3</text>
  <line x1="150" y1="367" x2="750" y2="367" stroke="#f5c842" stroke-width="2" stroke-dasharray="4 8" opacity=".65"/>
  <text x="160" y="352" class="svg-tiny" fill="#f5c842">NDVI SATURADO</text>
  <path id="ndvi-curve" d="{_ndvi_curve_path()}" fill="none" stroke="#67e8f9" stroke-width="7" stroke-linecap="round" stroke-dasharray="1100" stroke-dashoffset="1100"/>
  <g id="ndvi-points">
{_ndvi_scatter()}
  </g>
  <circle id="sat-dot" cx="626" cy="350" r="9" fill="#fb7185" opacity="0"/>
  <g id="trap-label" opacity="0">
    <rect x="536" y="604" width="198" height="76" rx="18" fill="rgba(251,113,133,.14)" stroke="#fb7185" stroke-width="2"/>
    <text x="635" y="636" text-anchor="middle" class="svg-tiny" fill="#fb7185">VERDE ENGAÑOSO</text>
    <text x="635" y="665" text-anchor="middle" font-size="25" font-weight="900" fill="#fb7185">r = 0.154</text>
  </g>
  <text x="150" y="852" class="svg-small">LAI</text>
  <text x="78" y="540" transform="rotate(-90 78 540)" text-anchor="middle" class="svg-small">NDVI</text>
</svg>
"""


def _regression_points() -> str:
    circles: list[str] = []
    for index, x in enumerate((220, 278, 336, 394, 452, 510, 568, 626, 684)):
        y = 930 - ((x - 190) / 520) * 125 + math.sin(index * 1.7) * 13
        circles.append(
            f'    <circle class="reg-point" cx="{x}" cy="{y:.1f}" r="6" fill="#34d399" opacity="0"/>'
        )
    return "\n".join(circles)


def _scene_3_body() -> str:
    x_ticks: list[str] = []
    for value in (0, 2, 4, 6):
        x = 250 + (value / 6) * 520
        x_ticks.append(
            f'    <line x1="{x:.1f}" y1="612" x2="{x:.1f}" y2="624" stroke="rgba(148,163,184,.4)" stroke-width="2"/>'
            f'<text x="{x:.1f}" y="654" text-anchor="middle" class="svg-tiny">{value}</text>'
        )
    return f"""
<svg viewBox="0 0 900 1080" role="img" aria-label="Comparación satélite frente a reloj fenológico">
  <text x="450" y="168" text-anchor="middle" class="svg-small">ERROR ABSOLUTO · qq/ha</text>
  <text x="110" y="300" class="svg-tiny" fill="#fb7185">SATÉLITE NDVI</text>
  <rect x="250" y="312" width="520" height="66" rx="33" fill="rgba(148,163,184,.12)" stroke="rgba(148,163,184,.22)" stroke-width="2"/>
  <rect id="sat-bar" x="250" y="312" width="0" height="66" rx="33" fill="#fb7185"/>
  <text id="sat-value" x="880" y="360" text-anchor="end" font-size="42" font-weight="900" fill="#fb7185">0.00</text>
  <text x="110" y="504" class="svg-tiny" fill="#34d399">RELOJ FENOLÓGICO</text>
  <rect x="250" y="516" width="520" height="66" rx="33" fill="rgba(148,163,184,.12)" stroke="rgba(148,163,184,.22)" stroke-width="2"/>
  <rect id="pheno-bar" x="250" y="516" width="0" height="66" rx="33" fill="#34d399"/>
  <text id="pheno-value" x="880" y="564" text-anchor="end" font-size="42" font-weight="900" fill="#34d399">0.00</text>
  <line x1="250" y1="612" x2="770" y2="612" stroke="rgba(148,163,184,.42)" stroke-width="2"/>
{x_ticks}
  <text id="verdict-3" x="450" y="716" text-anchor="middle" font-size="28" font-weight="900" fill="#34d399" letter-spacing="2" opacity="0">LA FÍSICA VENCIÓ AL ALGORITMO</text>
  <rect x="120" y="758" width="660" height="242" rx="22" fill="rgba(8,15,30,.8)" stroke="rgba(148,163,184,.2)" stroke-width="2"/>
  <text x="150" y="790" class="svg-tiny">AJUSTE FENOLÓGICO</text>
  <line x1="190" y1="958" x2="710" y2="958" stroke="rgba(148,163,184,.35)" stroke-width="2"/>
  <line x1="190" y1="790" x2="190" y2="958" stroke="rgba(148,163,184,.35)" stroke-width="2"/>
  <path id="reg-line" d="M190 935 L710 805" fill="none" stroke="#34d399" stroke-width="5" stroke-linecap="round" stroke-dasharray="620" stroke-dashoffset="620"/>
  <g id="reg-points">
{_regression_points()}
  </g>
  <text x="730" y="916" text-anchor="end" font-size="25" font-weight="900" fill="#34d399">r = 0.751</text>
  <text x="730" y="946" text-anchor="end" class="svg-tiny">R² = 56.4%</text>
</svg>
"""


SCENE_BODIES = (_scene_0_body, _scene_1_body, _scene_2_body, _scene_3_body)
SCENE_SCRIPTS = (SCENE_0_SCRIPT, SCENE_1_SCRIPT, SCENE_2_SCRIPT, SCENE_3_SCRIPT)

SCENE_META = (
    {
        "title": '¿Lluvia exacta<br><span class="accent">a las 15:30?</span>',
        "eyebrow": "Escena 01 · La promesa",
        "subtitle": "Un pronóstico puntual suena preciso. También es la forma más rápida de fallar.",
        "accent": "#67e8f9",
        "accent_glow": "rgba(103,232,249,.17)",
        "metrics": (
            ("31.4 °C", "Pronóstico puntual", "danger"),
            ("±0.0", "Margen prometido", "accent"),
            ("CAOS", "Realidad", "gold"),
        ),
        "prompt": "Reloj, pronóstico 31.4 °C y tarjeta agrietándose.",
    },
    {
        "title": 'El futuro no es un punto:<br><span class="accent">es un cono.</span>',
        "eyebrow": "Escena 02 · Incertidumbre",
        "subtitle": "Lyapunov y Split Conformal: la banda se ensancha y la cobertura se vuelve honesta.",
        "accent": "#f5c842",
        "accent_glow": "rgba(245,200,66,.18)",
        "metrics": (
            ("λ > 0", "Expansión caótica", "gold"),
            ("90%", "Cobertura conformal", "accent"),
            ("±q̂₉₀", "Banda empírica", "good"),
        ),
        "prompt": "Cono de Lyapunov y banda Split Conformal 90%.",
    },
    {
        "title": 'El satélite vio verde.<br><span class="accent">Y se quedó ciego.</span>',
        "eyebrow": "Escena 03 · Cajamarca",
        "subtitle": "El NDVI se saturó con LAI mayor que 3: la señal dejó de distinguir el café sano.",
        "accent": "#fb7185",
        "accent_glow": "rgba(251,113,133,.2)",
        "metrics": (
            ("r = 0.154", "Correlación NDVI", "danger"),
            ("+5.80", "Error qq/ha", "danger"),
            ("LAI > 3", "Zona saturada", "gold"),
        ),
        "prompt": "NDVI saturado y error de +5.80 qq/ha.",
    },
    {
        "title": 'La física venció<br><span class="accent">al algoritmo.</span>',
        "eyebrow": "Escena 04 · Reloj fenológico",
        "subtitle": "El satélite falló por 5.80 qq/ha; el reloj biológico de floración cayó a 0.26 qq/ha.",
        "accent": "#34d399",
        "accent_glow": "rgba(52,211,153,.18)",
        "metrics": (
            ("+5.80", "Satélite qq/ha", "danger"),
            ("0.26", "Reloj qq/ha", "good"),
            ("r = 0.751", "R² 56.4%", "accent"),
        ),
        "prompt": "Comparativa +5.80 vs 0.26 qq/ha y ajuste r=0.751.",
    },
)


def build_scene_html(
    *,
    scene_index: int,
    meta: Mapping[str, Any],
    duration_ms: int,
    cues: Sequence[Mapping[str, Any]],
) -> str:
    cues_json = (
        json.dumps(list(cues), ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    replacements = {
        "%%PAGE_TITLE%%": html.escape(f"{PROJECT_TITLE} · Escena {scene_index + 1}"),
        "%%ACCENT%%": str(meta["accent"]),
        "%%ACCENT_GLOW%%": str(meta["accent_glow"]),
        "%%SCENE_NUMBER%%": f"{scene_index + 1:02d}",
        "%%EYEBROW%%": html.escape(str(meta["eyebrow"])),
        "%%TITLE%%": str(meta["title"]),
        "%%SUBTITLE%%": html.escape(str(meta["subtitle"])),
        "%%SCENE_BODY%%": SCENE_BODIES[scene_index](),
        "%%METRICS%%": _metric_cards(meta["metrics"]),
        "%%SCENE_SCRIPT%%": SCENE_SCRIPTS[scene_index],
        "%%DURATION_MS%%": str(duration_ms),
        "%%CUES_JSON%%": cues_json,
    }
    document = HTML_TEMPLATE
    for token, value in replacements.items():
        document = document.replace(token, value)
    return document


def _ffprobe_json(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise MCPPipelineError(f"ffprobe falló: {completed.stderr.strip()}")
    return json.loads(completed.stdout)


def _verify_video(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MCPPipelineError(f"no se generó el MP4: {path}")
    if path.stat().st_size < 500_000:
        raise MCPPipelineError(f"el MP4 es demasiado pequeño ({path.stat().st_size} bytes)")
    probe = _ffprobe_json(path)
    streams = probe.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if video is None or audio is None:
        raise MCPPipelineError("el MP4 debe contener video y audio")
    if video.get("codec_name") != "h264":
        raise MCPPipelineError(f"códec de video inesperado: {video.get('codec_name')}")
    if audio.get("codec_name") != "aac":
        raise MCPPipelineError(f"códec de audio inesperado: {audio.get('codec_name')}")
    if (video.get("width"), video.get("height")) != (1080, 1920):
        raise MCPPipelineError(f"resolución inesperada: {video.get('width')}x{video.get('height')}")
    fps_text = str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1")
    numerator, _, denominator = fps_text.partition("/")
    fps = float(numerator) / float(denominator or "1")
    if abs(fps - 30.0) > 0.01:
        raise MCPPipelineError(f"FPS inesperado: {fps_text}")
    duration = float((probe.get("format") or {}).get("duration") or 0.0)
    if duration <= 15.0:
        raise MCPPipelineError(f"duración insuficiente: {duration:.2f}s")
    return {
        "duration_s": round(duration, 3),
        "fps": fps,
        "width": video.get("width"),
        "height": video.get("height"),
        "video_codec": video.get("codec_name"),
        "audio_codec": audio.get("codec_name"),
        "size_bytes": path.stat().st_size,
    }


def _mean_luma(path: Path) -> float:
    completed = subprocess.run(
        [
            FFMPEG,
            "-v",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-vf",
            "scale=1:1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout:
        raise MCPPipelineError(f"no se pudo medir el brillo de {path.name}")
    return float(completed.stdout[0])


def _extract_captures(video: Path, boundaries: Sequence[tuple[int, int]]) -> list[Path]:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    captures: list[Path] = []
    for index, (start_ms, end_ms) in enumerate(boundaries):
        capture_ms = max(start_ms + 300, end_ms - 250)
        midpoint_s = capture_ms / 1000
        target = CAPTURE_DIR / f"scene-{index:02d}.png"
        completed = subprocess.run(
            [
                FFMPEG,
                "-y",
                "-v",
                "error",
                "-i",
                str(video),
                "-ss",
                f"{midpoint_s:.3f}",
                "-frames:v",
                "1",
                str(target),
            ],
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise MCPPipelineError(
                f"ffmpeg no pudo extraer la captura {index}: {completed.stderr.decode(errors='replace')}"
            )
        luma = _mean_luma(target)
        if luma < 10.0:
            raise MCPPipelineError(f"la captura {target.name} parece negra (luma={luma:.1f})")
        captures.append(target)
    return captures


def _log_incidents() -> None:
    if not LOG_PATH.is_file():
        return
    text = LOG_PATH.read_text(encoding="utf-8")
    marker = "## [Plantilla para Nuevas Entradas]"
    if marker not in text:
        return
    entries = [
        (
            "Voces `edge_tts` rechazadas por el CHECK de SQLite",
            "El registro de una voz con `engine='edge_tts'` fallaba porque la tabla "
            "`voices` de `0001_initial_schema.sql` solo permitía kokoro, piper, xtts, "
            "f5, elevenlabs y openai.",
            "Se añadió `VoiceEngine.EDGE_TTS`, el provider Edge-TTS con fallback local y "
            "la migración `0002_add_edge_tts_engine.sql`, que reconstruye `voices` con el "
            "nuevo CHECK. El runner desactiva `PRAGMA foreign_keys` antes de `BEGIN` y lo "
            "reactiva al terminar.",
            "`uv run pytest tests/unit/test_audio_providers.py tests/unit/test_db.py "
            "tests/unit/test_audio_pipeline.py -q` y la generación real del short.",
        ),
        (
            "Las tools MCP no aceptaban `topic` ni `fps`",
            "El script necesitaba crear el proyecto con un topic explícito y forzar 30 FPS, "
            "pero `create_project` solo exponía `description` y `render_project_video` no "
            "tenía `fps`.",
            "`create_project` acepta `topic` como alias de `description`; "
            "`render_project_video` acepta `fps` opcional y el orquestador lo persiste con "
            "validación Pydantic antes de renderizar.",
            "Tests de `tests/unit/test_mcp_cli.py` y `tests/unit/test_orchestrator.py`, más "
            "el render E2E del short.",
        ),
        (
            "El MP4 del motor no cae en la ruta entregable solicitada",
            "`render_project_video` escribe en `workspaces/charla-forecasting/output/final.mp4`, "
            "mientras el entregable pedido vive en "
            "`projects/charla-forecasting/renders/charla-forecasting_final.mp4`.",
            "El script copia el MP4 validado a la ruta de entregables y extrae allí las cuatro "
            "capturas PNG, sin duplicar el render.",
            "Verificación con `ffprobe` (H.264/AAC, 1080x1920, 30 FPS, >15 s) y brillo medio "
            "de las capturas.",
        ),
    ]
    new_entries: list[str] = []
    for title, symptom, resolution, verification in entries:
        if title in text:
            continue
        new_entries.append(
            f"## {datetime.now(UTC).date().isoformat()} — {title} — Pipeline MCP\n\n"
            "### 1. Síntoma Observado / Reporte\n"
            f"*   **Qué falló:** {symptom}\n"
            "*   **Entorno:** Local CPU / Chromium headless / SQLite\n\n"
            "### 2. Causa Raíz\n"
            f"*   {symptom}\n\n"
            "### 3. Resolución e Implementación\n"
            f"*   {resolution}\n\n"
            "### 4. Verificación y Mediciones\n"
            f"*   {verification}\n\n"
            "### 5. Lecciones Aprendidas\n"
            "*   Mantener el contrato MCP y el esquema SQLite sincronizados con los motores "
            "soportados; validar el artefacto final en su ruta de entrega, no solo en el "
            "workspace interno.\n"
        )
    if not new_entries:
        return
    insertion = "\n".join(new_entries) + "\n"
    text = text.replace(marker, insertion + marker, 1)
    LOG_PATH.write_text(text, encoding="utf-8")


def _copy_deliverable(source: Path) -> None:
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    if source.resolve() == FINAL_MP4.resolve():
        return
    shutil.copy2(source, FINAL_MP4)


def main() -> int:
    client = MCPClient()

    existing_voices = client.call("list_voices", {"language": "es"})
    print(f"Voces en catálogo: {len(existing_voices)}")

    for voice in VOICES:
        payload = client.call("register_voice", voice)
        print(f"Voz lista: {payload.get('voice_id')} ({payload.get('engine')})")

    project = client.call(
        "create_project",
        {
            "project_id": PROJECT_ID,
            "title": PROJECT_TITLE,
            "topic": PROJECT_TOPIC,
            "width": 1080,
            "height": 1920,
            "fps": 30,
        },
    )
    print(f"Proyecto listo: {project.get('project_id')} -> {project.get('workspace_dir')}")

    synthesis = client.call(
        "synthesize_dialogue",
        {"project_id": PROJECT_ID, "lines": list(DIALOGUE)},
    )
    total_duration_ms = int(synthesis["total_duration_ms"])
    lines = list(synthesis["lines"])
    words = list(synthesis["words"])
    print(
        f"Audio listo: {total_duration_ms} ms, "
        f"{synthesis.get('lines_count')} líneas, {synthesis.get('words_count')} palabras"
    )

    boundaries = _scene_boundaries(lines, total_duration_ms)
    for scene_index, ((start_ms, end_ms), meta) in enumerate(
        zip(boundaries, SCENE_META, strict=True)
    ):
        cues = _cues_for_scene(
            words,
            start_ms,
            end_ms,
            str(lines[scene_index]["text_content"]),
        )
        document = build_scene_html(
            scene_index=scene_index,
            meta=meta,
            duration_ms=end_ms - start_ms,
            cues=cues,
        )
        payload = client.call(
            "save_and_validate_scene",
            {
                "project_id": PROJECT_ID,
                "scene_index": scene_index,
                "title": f"Escena {scene_index + 1}",
                "start_ms": start_ms,
                "end_ms": end_ms,
                "html_content": document,
                "prompt": meta["prompt"],
            },
        )
        status = payload.get("validation_status")
        if status != "valid":
            raise MCPPipelineError(
                f"la escena {scene_index} no validó: {payload.get('validation_errors')}"
            )
        print(f"Escena {scene_index + 1}/4 validada: {start_ms}-{end_ms} ms, {len(cues)} cues")

    print("Renderizando 30 FPS...")
    render = client.call(
        "render_project_video",
        {"project_id": PROJECT_ID, "burn_subtitles": True, "fps": 30},
    )
    if render.get("status") != "completed":
        raise MCPPipelineError(f"el render no terminó: {render}")
    source = Path(str(render["output_mp4_path"])).expanduser().resolve()
    print(
        f"Render terminado: {render.get('rendered_frames')} frames, {render.get('duration_ms')} ms"
    )

    _copy_deliverable(source)
    metrics = _verify_video(FINAL_MP4)
    captures = _extract_captures(FINAL_MP4, boundaries)
    _log_incidents()

    print(f"MP4 final: {FINAL_MP4}")
    print(
        "ffprobe: "
        f"{metrics['video_codec']}+{metrics['audio_codec']} "
        f"{metrics['width']}x{metrics['height']} @ {metrics['fps']:.0f} FPS, "
        f"{metrics['duration_s']:.2f}s, {metrics['size_bytes']} bytes"
    )
    for capture in captures:
        print(f"Captura: {capture} (luma={_mean_luma(capture):.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
