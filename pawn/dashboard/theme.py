"""Design tokens for the PAWN dashboard.

Centralizes colors, typography, spacing and Plotly layout defaults so the
dashboard has a single source of truth for its visual identity.
"""

from __future__ import annotations

import colorsys

FONT_FAMILY = (
    "'Inter', 'SF Pro Text', -apple-system, BlinkMacSystemFont, "
    "'Segoe UI', Roboto, sans-serif"
)
FONT_MONO = (
    "'JetBrains Mono', 'SF Mono', 'Menlo', 'Consolas', 'Liberation Mono', monospace"
)

# ---------------------------------------------------------------------------
# Surface palette — three-step elevation: near-black page > cards > inputs.
# ---------------------------------------------------------------------------
BG = "#05070d"
BG_GRADIENT = BG  # kept for backward compat; solid bg now
SURFACE = "#121826"
SURFACE_ELEVATED = "#1a2232"
HEADER_BG = "#141b2b"  # solid grey-blue, no gradient
BORDER = "#222b3d"
BORDER_STRONG = "#2f3a52"

TEXT = "#e8ecf4"
TEXT_MUTED = "#9aa3b8"
TEXT_FAINT = "#5a6379"

# ---------------------------------------------------------------------------
# Data palette — 8 harmonious hues, cool-leaning. Semantic aliases below.
# ---------------------------------------------------------------------------
SKY = "#60a5fa"       # train, primary
ROSE = "#f472b6"      # val, accent
EMERALD = "#34d399"   # top-5, success, legal
AMBER = "#fbbf24"     # learning rate, warning
VIOLET = "#a78bfa"    # gradient norm
CYAN = "#22d3ee"      # gpu, info
SLATE = "#94a3b8"     # time, neutral trace
CORAL = "#fb7185"     # forfeit, error

# Backward-compatible semantic names used by charts.py
COLORS: dict[str, str] = {
    # Semantic
    "primary": SKY,
    "accent": ROSE,
    "success": EMERALD,
    "warning": AMBER,
    "info": CYAN,
    "violet": VIOLET,
    "slate": SLATE,
    "error": CORAL,
    # Legacy aliases (keep existing chart code happy)
    "blue": SKY,
    "red": ROSE,
    "green": EMERALD,
    "orange": "#fb923c",
    "purple": VIOLET,
    "gold": AMBER,
    "brown": SLATE,
}

# KPI severity tints
GOOD = EMERALD
NEUTRAL = SKY
WARN = AMBER
BAD = CORAL


def layer_color(
    layer_idx: int,
    n_layers: int = 8,
    saturation: float = 0.60,
    lightness: float = 0.64,
) -> str:
    """Cyan->violet gradient for layer indices. Cohesive with the data palette."""
    t = layer_idx / max(n_layers - 1, 1)
    hue = 0.52 + t * 0.22  # cyan (0.52) -> violet (0.74)
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


LAYER_COLORS = [layer_color(i) for i in range(8)]
OUTPUT_LAYER_COLOR = AMBER  # warm contrast vs. cool layer gradient


def desaturate(hex_color: str, saturation: float = 0.35, lightness_delta: float = 0.08) -> str:
    """Return a softened version of a hex color, used for fit overlays."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    hue, lig, sat = colorsys.rgb_to_hls(r, g, b)
    new_lig = min(1.0, lig + lightness_delta)
    new_sat = sat * saturation
    nr, ng, nb = colorsys.hls_to_rgb(hue, new_lig, new_sat)
    return f"#{int(nr * 255):02x}{int(ng * 255):02x}{int(nb * 255):02x}"


# ---------------------------------------------------------------------------
# Plotly layout defaults — transparent background lets the Solara card show.
# ---------------------------------------------------------------------------
GRID = "rgba(148,163,184,0.08)"
ZERO = "rgba(148,163,184,0.18)"
AXIS_LINE = "rgba(148,163,184,0.22)"

PLOTLY_LAYOUT: dict = dict(
    template="plotly_dark",
    height=360,
    margin=dict(l=58, r=18, t=56, b=96),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT_FAMILY, color=TEXT_MUTED, size=12),
    title=dict(
        font=dict(family=FONT_FAMILY, size=15, color=TEXT),
        x=0.01,
        xanchor="left",
        y=0.97,
        yanchor="top",
        pad=dict(l=4, t=2),
    ),
    xaxis=dict(
        gridcolor=GRID,
        zerolinecolor=ZERO,
        linecolor=AXIS_LINE,
        tickcolor=AXIS_LINE,
        tickfont=dict(color=TEXT_MUTED, size=11),
        title=dict(font=dict(color=TEXT_MUTED, size=12)),
        showspikes=True,
        spikecolor=AXIS_LINE,
        spikethickness=1,
        spikedash="dot",
        spikemode="across",
    ),
    yaxis=dict(
        gridcolor=GRID,
        zerolinecolor=ZERO,
        linecolor=AXIS_LINE,
        tickcolor=AXIS_LINE,
        tickfont=dict(color=TEXT_MUTED, size=11),
        title=dict(font=dict(color=TEXT_MUTED, size=12)),
    ),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.22,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, size=11),
        itemsizing="constant",
    ),
    hoverlabel=dict(
        bgcolor=SURFACE_ELEVATED,
        bordercolor=BORDER_STRONG,
        font=dict(family=FONT_FAMILY, size=12, color=TEXT),
    ),
    hovermode="x unified",
    colorway=[SKY, ROSE, EMERALD, AMBER, VIOLET, CYAN, SLATE, CORAL],
)

TRACE_WIDTH = 2.3
