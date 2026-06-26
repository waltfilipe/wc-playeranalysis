from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mplsoccer import Pitch
from PIL import Image

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="WC Player Analysis — Top ΔxT")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .player-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #eef1f7;
        margin-bottom: 0.15rem;
    }
    .player-sub {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 0.75rem;
    }
    .map-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #c7cdda;
        margin: 0.25rem 0 0.35rem 0;
        text-align: center;
    }
  </style>
    """,
    unsafe_allow_html=True,
)

# ── CONSTANTS ────────────────────────────────────────────────
FIELD_X, FIELD_Y = 120.0, 80.0
HALF_LINE_X = FIELD_X / 2
FINAL_THIRD_LINE_X = 80.0
GOAL_X, GOAL_Y = 120.0, 40.0
FIG_W, FIG_H = 5.4, 3.5
FIG_DPI = 160
PASS_START_MARKER_SIZE = 6
CARRY_START_MARKER_SIZE = 6
MAP_REF_WIDTH = 5.4
ARROW_WIDTH = 0.75
ARROW_HEADWIDTH = 1.15
ARROW_HEADLENGTH = 1.15
ARROW_ALPHA = 0.68
ARROW_ALPHA_EMPH = 0.82
ALL_MATCHES_LABEL = "All Matches"
DATA_CACHE_VERSION = 2
XT_ZONE_COLS = 3
XT_ZONE_ROWS = 2
NX_XT = 16
NY_XT = 12
XT_GRID_CMAP = LinearSegmentedColormap.from_list(
    "xt_grid", ["#1a1a2e", "#3b82f6", "#fbbf24", "#ef4444"]
)
WYSCOUT_PITCH_SIZE = 100.0
OPT_ATTACKING_TWO_THIRDS_X = 40.0
WYSCOUT_PROG_OWN_HALF = 30.0
WYSCOUT_PROG_CROSS_HALF = 15.0
WYSCOUT_PROG_OPP_HALF = 10.0
XT_MODEL_HEURISTIC_V3 = "heuristic_v3"
XT_MIN_PASS_DISTANCE = 9.5
XT_V3_FINE_NX = 96
XT_V3_FINE_NY = 64
XT_V3_DEF_MAX = 0.25
XT_V3_MID_MAX = 0.60
XT_V3_ATT_BYLINE = 0.94
XT_V3_SURFACE_MAX = 1.02
XT_V3_ZONE_BLEND_WIDTH = 22.0
XT_V3_LAT_DISC_MAX = 0.16
XT_V3_LAT_CURVE_POWER = 1.0
XT_V3_PROG_SCALE = 0.15
XT_V3_HIGH_SCALE = 0.35
XT_V3_PROG_FLOOR = 0.08
XT_V3_HIGH_FLOOR = 0.18
XT_V3_PROG_SCALE_CLASS = 0.17
XT_V3_HIGH_SCALE_CLASS = 0.40
XT_V3_PROG_FLOOR_CLASS = 0.10
XT_V3_HIGH_FLOOR_CLASS = 0.22
XT_V3_MIN_PASS_DISTANCE = 10.5
XT_V3_NEG_PENALTY_FACTOR = 0.55
XT_V3_PRESSURE_ESCAPE_BONUS = 0.02
XT_V3_PRESSURE_X_MAX = 50.0
XT_V3_WIDE_FRAC = 0.60
XT_V3_NEG_RECYCLE_X_MAX = 60.0
XT_V4_BOX_X_START = 90.0
XT_V4_BOX_X_FULL = 112.0
XT_V4_CORNER_LAT_ON = 0.58
XT_V4_CORNER_PENALTY = 0.10
XT_V4_CENTRAL_PREMIUM = 0.06
XT_V4_SHORT_PASS_DIST = 8.0
XT_V4_SHORT_PASS_FACTOR = 0.55
XT_V4_V1_WING_BASE = 0.80
XT_V4_V1_CENT_MULT = 1.00
XT_V5_MAX_DELTA_DEF = 0.28
XT_V5_MAX_DELTA_MID = 0.36
XT_V5_MAX_DELTA_ATT = 0.42
XT_V5_MAX_DELTA_BOX = 0.52

# xT Heurístico v3c — conservador (0–0.5), transições mais suaves
XT_MODEL_HEURISTIC_V3C = "heuristic_v3c"
XT_V3C_SURFACE_MAX = 0.50
XT_V3C_DEF_MAX = 0.125
XT_V3C_MID_MAX = 0.30
XT_V3C_ATT_BYLINE = 0.47
XT_V3C_ZONE_BLEND_WIDTH = 42.0
XT_V3C_LAT_DISC_MAX = 0.08
XT_V3C_CENTRAL_PREMIUM = 0.03
XT_V3C_CORNER_PENALTY = 0.05
XT_V3C_MAX_DELTA_DEF = 0.14
XT_V3C_MAX_DELTA_MID = 0.18
XT_V3C_MAX_DELTA_ATT = 0.21
XT_V3C_MAX_DELTA_BOX = 0.26

CARD_TITLE_TEXT = "14px"
CARD_LABEL_TEXT = "16px"
CARD_INNER_BORDER = "rgba(107,114,128,0.45)"
TOP_DELTAXT_N = 10
EXCLUDED_CSV = {"enzo.csv"}

PLAYERS = [
    {"code": "BG", "name": "Bruno Guimarães", "tone": "#5b9bd5"},
    {"code": "CS", "name": "Casemiro", "tone": "#e67e22"},
    {"code": "LP", "name": "Lucas Paquetá", "tone": "#22c55e"},
]

CMAP_PASS = LinearSegmentedColormap.from_list(
    "pass_dxt", ["#bfdbfe", "#60a5fa", "#2563eb", "#1e3a8a"]
)
CMAP_CARRY = LinearSegmentedColormap.from_list(
    "carry_dxt", ["#fde68a", "#fbbf24", "#f59e0b", "#b45309"]
)

COLOR_SUCCESS = "#6ee7b7"
COLOR_PROGRESSIVE = "#7dd3fc"
COLOR_HIGHLY_PROGRESSIVE = "#fcd34d"
COLOR_FAIL = "#fca5a5"
COLOR_CARRY = "#c4b5fd"
ALPHA_SUCCESS = 0.50
COLOR_CARRY_BASE_ALPHA = 0.50


# ── COORDINATE HELPERS ───────────────────────────────────────
def wyscout_to_statsbomb(x: float, y: float, *, flip_x: bool = False) -> tuple[float, float]:
    """Wyscout 0–100 → StatsBomb 120×80; espelha Y; flip X em jogos fora (isHome=false)."""
    x_sb = x * FIELD_X / WYSCOUT_PITCH_SIZE
    y_sb = FIELD_Y - (y * FIELD_Y / WYSCOUT_PITCH_SIZE)
    if flip_x:
        x_sb = FIELD_X - x_sb
    return x_sb, y_sb


def distance_to_goal(x: float, y: float) -> float:
    return float(np.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2))


def is_progressive_wyscout(x_start: float, y_start: float, x_end: float, y_end: float) -> bool:
    start_dist = distance_to_goal(x_start, y_start)
    end_dist = distance_to_goal(x_end, y_end)
    progress = start_dist - end_dist
    if progress <= 0:
        return False
    start_own = x_start < HALF_LINE_X
    end_own = x_end < HALF_LINE_X
    start_opp = x_start >= HALF_LINE_X
    end_opp = x_end >= HALF_LINE_X
    if start_own and end_own:
        return progress >= WYSCOUT_PROG_OWN_HALF
    if start_opp and end_opp:
        return progress >= WYSCOUT_PROG_OPP_HALF
    return progress >= WYSCOUT_PROG_CROSS_HALF


def _parse_bool(value) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "successful"}


def _has_coords(row, prefix: str) -> bool:
    x_col, y_col = f"{prefix}_x", f"{prefix}_y"
    return pd.notna(row.get(x_col)) and pd.notna(row.get(y_col))


# ── xT HEURÍSTICO v3 ─────────────────────────────────────────
def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smootherstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _centrality(y: np.ndarray) -> np.ndarray:
    return 1.0 - np.abs((y / FIELD_Y) - 0.5) * 2.0


def _lateral_frac(y: float) -> float:
    return float(abs(y - GOAL_Y) / (FIELD_Y / 2.0))


def _lateral_relative_position(y: np.ndarray) -> np.ndarray:
    return np.abs(y - GOAL_Y) / (FIELD_Y / 2.0)


def _location_factor_v3(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    lat = _lateral_relative_position(y)
    depth = np.clip(
        (x - OPT_ATTACKING_TWO_THIRDS_X) / (FIELD_X - OPT_ATTACKING_TWO_THIRDS_X),
        0.0,
        1.0,
    )
    zone_gate = _smoothstep(depth)
    max_discount = XT_V3_LAT_DISC_MAX * zone_gate
    lateral_curve = _smoothstep(lat ** XT_V3_LAT_CURVE_POWER)
    return 1.0 - max_discount * lateral_curve


def _v4_box_gate(x: np.ndarray) -> np.ndarray:
    span = max(XT_V4_BOX_X_FULL - XT_V4_BOX_X_START, 1.0)
    t = np.clip((x - XT_V4_BOX_X_START) / span, 0.0, 1.0)
    return _smoothstep(t)


def _v4_xg_finishing_factor(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    box_gate = _v4_box_gate(x)
    cent = _centrality(y)
    lat = _lateral_relative_position(y)
    central_bonus = XT_V4_CENTRAL_PREMIUM * box_gate * _smoothstep(cent)
    wide_in_box = box_gate * _smoothstep(np.clip((lat - XT_V4_CORNER_LAT_ON) / 0.42, 0.0, 1.0))
    wide_discount = XT_V4_CORNER_PENALTY * wide_in_box
    return np.clip(1.0 + central_bonus - wide_discount, 0.94, 1.06)


def _enforce_row_monotonic_x(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for iy in range(out.shape[0]):
        for ix in range(1, out.shape[1]):
            if out[iy, ix] < out[iy, ix - 1]:
                out[iy, ix] = out[iy, ix - 1]
    return out


def _map_zonal_threat_v3_smooth(x: np.ndarray) -> np.ndarray:
    blend = XT_V3_ZONE_BLEND_WIDTH
    x = np.clip(x, 0.0, FIELD_X)
    threat_def = XT_V3_DEF_MAX * np.clip(x / OPT_ATTACKING_TWO_THIRDS_X, 0.0, 1.0)
    mid_span = max(FINAL_THIRD_LINE_X - OPT_ATTACKING_TWO_THIRDS_X, 1.0)
    mid_t = np.clip((x - OPT_ATTACKING_TWO_THIRDS_X) / mid_span, 0.0, 1.0)
    threat_mid = XT_V3_DEF_MAX + (XT_V3_MID_MAX - XT_V3_DEF_MAX) * _smootherstep(mid_t)
    att_span = max(FIELD_X - FINAL_THIRD_LINE_X, 1.0)
    att_t = np.clip((x - FINAL_THIRD_LINE_X) / att_span, 0.0, 1.0)
    threat_att = XT_V3_MID_MAX + (XT_V3_ATT_BYLINE - XT_V3_MID_MAX) * _smootherstep(att_t)
    w_def = 1.0 - _smootherstep(np.clip((x - (OPT_ATTACKING_TWO_THIRDS_X - blend)) / blend, 0.0, 1.0))
    w_att = _smootherstep(np.clip((x - (FINAL_THIRD_LINE_X - blend)) / blend, 0.0, 1.0))
    w_mid = np.clip(1.0 - w_def - w_att, 0.0, 1.0)
    w_sum = w_def + w_mid + w_att + 1e-12
    return (w_def * threat_def + w_mid * threat_mid + w_att * threat_att) / w_sum


def _build_heuristic_v3_threat_surface(Xc: np.ndarray, Yc: np.ndarray) -> np.ndarray:
    zonal = _map_zonal_threat_v3_smooth(Xc)
    surface = zonal * _location_factor_v3(Xc, Yc) * _v4_xg_finishing_factor(Xc, Yc)
    surface = np.clip(surface, 0.0, XT_V3_SURFACE_MAX)
    return _enforce_row_monotonic_x(surface)


@st.cache_data(show_spinner=False)
def compute_heuristic_v3_xt_grid(
    n_x: int = NX_XT, n_y: int = NY_XT,
) -> np.ndarray:
    """16×12 display grid sampled from the v3 fine threat surface."""
    fine = compute_heuristic_v3_fine_grid()
    grid = zone_xt_means(fine, n_x=n_x, n_y=n_y)
    return _enforce_row_monotonic_x(grid)


@st.cache_data(show_spinner=False)
def compute_heuristic_v3c_xt_grid(
    n_x: int = NX_XT, n_y: int = NY_XT,
) -> np.ndarray:
    """16×12 display grid sampled from the v3c fine threat surface."""
    fine = compute_heuristic_v3c_fine_grid()
    grid = zone_xt_means(fine, n_x=n_x, n_y=n_y)
    return _enforce_row_monotonic_x(grid)


@st.cache_data(show_spinner=False)
def compute_heuristic_v3_fine_grid(nx: int = XT_V3_FINE_NX, ny: int = XT_V3_FINE_NY) -> np.ndarray:
    xe = np.linspace(0.0, FIELD_X, nx)
    ye = np.linspace(0.0, FIELD_Y, ny)
    Xc, Yc = np.meshgrid(xe, ye)
    return _build_heuristic_v3_threat_surface(Xc, Yc)


def xt_value_bilinear(x: float, y: float, fine_grid: np.ndarray) -> float:
    nx, ny = fine_grid.shape[1], fine_grid.shape[0]
    fx = float(np.clip(x / FIELD_X * (nx - 1), 0.0, nx - 1))
    fy = float(np.clip(y / FIELD_Y * (ny - 1), 0.0, ny - 1))
    x0, y0 = int(fx), int(fy)
    x1, y1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1)
    tx, ty = fx - x0, fy - y0
    v00, v10 = fine_grid[y0, x0], fine_grid[y0, x1]
    v01, v11 = fine_grid[y1, x0], fine_grid[y1, x1]
    return float(
        (1 - tx) * (1 - ty) * v00
        + tx * (1 - ty) * v10
        + (1 - tx) * ty * v01
        + tx * ty * v11
    )


def _v3_short_pass_multiplier(pass_distance: float) -> float:
    short_dist = XT_V4_SHORT_PASS_DIST
    short_factor = XT_V4_SHORT_PASS_FACTOR
    blend_span = 4.0
    if pass_distance < short_dist:
        return short_factor
    if pass_distance < short_dist + blend_span:
        blend = (pass_distance - short_dist) / blend_span
        return short_factor + (1.0 - short_factor) * blend
    return 1.0


def _v3_zone_max_pass_delta(x_start: float) -> float:
    x = float(np.clip(x_start, 0.0, FIELD_X))
    control_points = [
        (0.0, XT_V5_MAX_DELTA_DEF),
        (OPT_ATTACKING_TWO_THIRDS_X, XT_V5_MAX_DELTA_MID),
        (FINAL_THIRD_LINE_X, XT_V5_MAX_DELTA_ATT),
        (XT_V4_BOX_X_START, XT_V5_MAX_DELTA_BOX),
        (FIELD_X, XT_V5_MAX_DELTA_BOX),
    ]
    for idx in range(len(control_points) - 1):
        x0, cap0 = control_points[idx]
        x1, cap1 = control_points[idx + 1]
        if x <= x1:
            if x1 <= x0:
                return cap1
            t = float(_smoothstep(np.array([(x - x0) / (x1 - x0)]))[0])
            return cap0 + (cap1 - cap0) * t
    return control_points[-1][1]


def _adjust_heuristic_v3_pass_delta(row) -> float:
    if not row.is_won:
        return 0.0
    raw = float(row.xt_end - row.xt_start)
    if raw >= 0:
        adjusted = raw * _v3_short_pass_multiplier(row.pass_distance)
        return min(adjusted, _v3_zone_max_pass_delta(row.x_start))
    lat_start = _lateral_frac(row.y_start)
    lat_end = _lateral_frac(row.y_end)
    if row.x_start < XT_V3_NEG_RECYCLE_X_MAX:
        adjusted = raw * (XT_V3_NEG_PENALTY_FACTOR if lat_end < lat_start else 1.0)
    else:
        adjusted = raw
    if (
        row.x_start < XT_V3_PRESSURE_X_MAX
        and lat_start > XT_V3_WIDE_FRAC
        and lat_end < lat_start - 0.12
    ):
        adjusted += XT_V3_PRESSURE_ESCAPE_BONUS
    return adjusted


def apply_heuristic_v3_xt(df: pd.DataFrame) -> pd.DataFrame:
    fine = compute_heuristic_v3_fine_grid()
    out = df.copy()
    out["xt_start"] = out.apply(lambda r: xt_value_bilinear(r["x_start"], r["y_start"], fine), axis=1)
    out["xt_end"] = out.apply(lambda r: xt_value_bilinear(r["x_end"], r["y_end"], fine), axis=1)
    out["delta_xt"] = out.apply(_adjust_heuristic_v3_pass_delta, axis=1)
    return out


# ── xT HEURÍSTICO v3c (conservador) ──────────────────────────
def _location_factor_v3c(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    lat = _lateral_relative_position(y)
    depth = np.clip(
        (x - OPT_ATTACKING_TWO_THIRDS_X) / (FIELD_X - OPT_ATTACKING_TWO_THIRDS_X),
        0.0,
        1.0,
    )
    zone_gate = _smootherstep(depth)
    max_discount = XT_V3C_LAT_DISC_MAX * zone_gate
    lateral_curve = _smootherstep(lat ** XT_V3_LAT_CURVE_POWER)
    return 1.0 - max_discount * lateral_curve


def _v4_xg_finishing_factor_v3c(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    box_gate = _v4_box_gate(x)
    cent = _centrality(y)
    lat = _lateral_relative_position(y)
    central_bonus = XT_V3C_CENTRAL_PREMIUM * box_gate * _smootherstep(cent)
    wide_in_box = box_gate * _smootherstep(np.clip((lat - XT_V4_CORNER_LAT_ON) / 0.42, 0.0, 1.0))
    wide_discount = XT_V3C_CORNER_PENALTY * wide_in_box
    return np.clip(1.0 + central_bonus - wide_discount, 0.97, 1.03)


def _map_zonal_threat_v3c_smooth(x: np.ndarray) -> np.ndarray:
    """Mesma lógica de zonas do v3, com blend mais largo e curvas mais suaves."""
    blend = XT_V3C_ZONE_BLEND_WIDTH
    x = np.clip(x, 0.0, FIELD_X)
    threat_def = XT_V3C_DEF_MAX * _smootherstep(np.clip(x / OPT_ATTACKING_TWO_THIRDS_X, 0.0, 1.0))
    mid_span = max(FINAL_THIRD_LINE_X - OPT_ATTACKING_TWO_THIRDS_X, 1.0)
    mid_t = np.clip((x - OPT_ATTACKING_TWO_THIRDS_X) / mid_span, 0.0, 1.0)
    threat_mid = XT_V3C_DEF_MAX + (XT_V3C_MID_MAX - XT_V3C_DEF_MAX) * _smootherstep(mid_t)
    att_span = max(FIELD_X - FINAL_THIRD_LINE_X, 1.0)
    att_t = np.clip((x - FINAL_THIRD_LINE_X) / att_span, 0.0, 1.0)
    threat_att = XT_V3C_MID_MAX + (XT_V3C_ATT_BYLINE - XT_V3C_MID_MAX) * _smootherstep(att_t)
    w_def = 1.0 - _smootherstep(np.clip((x - (OPT_ATTACKING_TWO_THIRDS_X - blend)) / blend, 0.0, 1.0))
    w_att = _smootherstep(np.clip((x - (FINAL_THIRD_LINE_X - blend)) / blend, 0.0, 1.0))
    w_mid = np.clip(1.0 - w_def - w_att, 0.0, 1.0)
    w_sum = w_def + w_mid + w_att + 1e-12
    return (w_def * threat_def + w_mid * threat_mid + w_att * threat_att) / w_sum


def _build_heuristic_v3c_threat_surface(Xc: np.ndarray, Yc: np.ndarray) -> np.ndarray:
    zonal = _map_zonal_threat_v3c_smooth(Xc)
    surface = zonal * _location_factor_v3c(Xc, Yc) * _v4_xg_finishing_factor_v3c(Xc, Yc)
    surface = np.clip(surface, 0.0, XT_V3C_SURFACE_MAX)
    return _enforce_row_monotonic_x(surface)


@st.cache_data(show_spinner=False)
def compute_heuristic_v3c_fine_grid(nx: int = XT_V3_FINE_NX, ny: int = XT_V3_FINE_NY) -> np.ndarray:
    xe = np.linspace(0.0, FIELD_X, nx)
    ye = np.linspace(0.0, FIELD_Y, ny)
    Xc, Yc = np.meshgrid(xe, ye)
    return _build_heuristic_v3c_threat_surface(Xc, Yc)


def _v3c_zone_max_pass_delta(x_start: float) -> float:
    x = float(np.clip(x_start, 0.0, FIELD_X))
    control_points = [
        (0.0, XT_V3C_MAX_DELTA_DEF),
        (OPT_ATTACKING_TWO_THIRDS_X, XT_V3C_MAX_DELTA_MID),
        (FINAL_THIRD_LINE_X, XT_V3C_MAX_DELTA_ATT),
        (XT_V4_BOX_X_START, XT_V3C_MAX_DELTA_BOX),
        (FIELD_X, XT_V3C_MAX_DELTA_BOX),
    ]
    for idx in range(len(control_points) - 1):
        x0, cap0 = control_points[idx]
        x1, cap1 = control_points[idx + 1]
        if x <= x1:
            if x1 <= x0:
                return cap1
            t = float(_smootherstep(np.array([(x - x0) / (x1 - x0)]))[0])
            return cap0 + (cap1 - cap0) * t
    return control_points[-1][1]


def _adjust_heuristic_v3c_pass_delta(row) -> float:
    if not row.is_won:
        return 0.0
    raw = float(row.xt_end_v3c - row.xt_start_v3c)
    if raw >= 0:
        adjusted = raw * _v3_short_pass_multiplier(row.pass_distance)
        return min(adjusted, _v3c_zone_max_pass_delta(row.x_start))
    lat_start = _lateral_frac(row.y_start)
    lat_end = _lateral_frac(row.y_end)
    if row.x_start < XT_V3_NEG_RECYCLE_X_MAX:
        adjusted = raw * (XT_V3_NEG_PENALTY_FACTOR if lat_end < lat_start else 1.0)
    else:
        adjusted = raw
    if (
        row.x_start < XT_V3_PRESSURE_X_MAX
        and lat_start > XT_V3_WIDE_FRAC
        and lat_end < lat_start - 0.12
    ):
        adjusted += XT_V3_PRESSURE_ESCAPE_BONUS * 0.5
    return adjusted


def apply_heuristic_v3c_xt(df: pd.DataFrame) -> pd.DataFrame:
    fine = compute_heuristic_v3c_fine_grid()
    out = df.copy()
    out["xt_start_v3c"] = out.apply(
        lambda r: xt_value_bilinear(r["x_start"], r["y_start"], fine), axis=1
    )
    out["xt_end_v3c"] = out.apply(
        lambda r: xt_value_bilinear(r["x_end"], r["y_end"], fine), axis=1
    )
    out["delta_xt_v3c"] = out.apply(_adjust_heuristic_v3c_pass_delta, axis=1)
    return out


def classify_xt_progressive_v3_adjusted(
    xt_start: float,
    delta_xt: float,
    x_end: float,
    pass_distance: float,
) -> str:
    if pass_distance <= XT_V3_MIN_PASS_DISTANCE:
        return "none"
    if delta_xt <= 0:
        return "none"
    prog_thresh = max(XT_V3_PROG_FLOOR_CLASS, XT_V3_PROG_SCALE_CLASS * (1.0 - xt_start))
    high_thresh = max(XT_V3_HIGH_FLOOR_CLASS, XT_V3_HIGH_SCALE_CLASS * (1.0 - xt_start))
    if delta_xt <= prog_thresh:
        return "none"
    if delta_xt > high_thresh:
        return "highly"
    return "progressive"


def _row_if_won(row):
    if hasattr(row, "_replace"):
        return row._replace(is_won=True)
    if isinstance(row, pd.Series):
        won = row.copy()
        won["is_won"] = True
        return won
    return row


def hypothetical_delta_xt(row) -> float:
    return _adjust_heuristic_v3_pass_delta(_row_if_won(row))


def progressive_delta_for_attempt(row) -> float:
    if row.is_won:
        return float(row.delta_xt)
    return hypothetical_delta_xt(row)


def is_impact_attempt(row) -> bool:
    delta = progressive_delta_for_attempt(row)
    return classify_xt_progressive_v3_adjusted(
        row.xt_start, delta, row.x_end, row.pass_distance
    ) in ("progressive", "highly")


def is_high_impact_attempt(row) -> bool:
    delta = progressive_delta_for_attempt(row)
    return (
        classify_xt_progressive_v3_adjusted(
            row.xt_start, delta, row.x_end, row.pass_distance
        )
        == "highly"
    )


def classification_accuracy(df: pd.DataFrame, success_col: str, attempt_fn) -> dict:
    attempts = df.apply(attempt_fn, axis=1)
    successful = int(df[success_col].astype(bool).sum())
    attempted = int(attempts.sum())
    accuracy_pct = (successful / attempted * 100.0) if attempted else 0.0
    return {
        "successful": successful,
        "attempted": attempted,
        "accuracy_pct": round(accuracy_pct, 1),
    }


def enrich_with_xt_v3(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pass_distance"] = np.where(
        out["has_end"],
        np.sqrt((out["x_end"] - out["x_start"]) ** 2 + (out["y_end"] - out["y_start"]) ** 2),
        0.0,
    )
    out["is_won"] = out["is_success"].astype(bool)
    carry_mask = out["category"] == "ball-carries"
    out.loc[carry_mask, "is_won"] = out.loc[carry_mask, "has_end"]

    for col in ("xt_start", "xt_end", "delta_xt", "xt_start_v3c", "xt_end_v3c", "delta_xt_v3c"):
        out[col] = 0.0
    out["progressive"] = False
    out["impact_pass"] = False
    out["high_impact_pass"] = False
    out["impact_carry"] = False
    out["high_impact_carry"] = False

    xt_mask = out["category"].isin(["passes", "ball-carries"]) & out["has_end"]
    if not xt_mask.any():
        return out

    xt_df = apply_heuristic_v3_xt(out.loc[xt_mask].copy())
    out.loc[xt_mask, ["xt_start", "xt_end", "delta_xt"]] = xt_df[
        ["xt_start", "xt_end", "delta_xt"]
    ].values

    xt_df_v3c = apply_heuristic_v3c_xt(out.loc[xt_mask].copy())
    out.loc[xt_mask, ["xt_start_v3c", "xt_end_v3c", "delta_xt_v3c"]] = xt_df_v3c[
        ["xt_start_v3c", "xt_end_v3c", "delta_xt_v3c"]
    ].values

    pass_mask = out["category"] == "passes"
    for idx, row in out.loc[pass_mask].iterrows():
        out.at[idx, "progressive"] = bool(
            row.is_won
            and is_progressive_wyscout(row.x_start, row.y_start, row.x_end, row.y_end)
        )
        out.at[idx, "impact_pass"] = bool(row.is_won and is_impact_attempt(row))
        out.at[idx, "high_impact_pass"] = bool(row.is_won and is_high_impact_attempt(row))

    carry_mask = out["category"] == "ball-carries"
    for idx, row in out.loc[carry_mask].iterrows():
        out.at[idx, "impact_carry"] = bool(row.is_won and is_impact_attempt(row))
        out.at[idx, "high_impact_carry"] = bool(row.is_won and is_high_impact_attempt(row))

    return out


def ensure_xt_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill v3c columns when serving cached frames from older app versions."""
    if df.empty or "delta_xt_v3c" in df.columns:
        return df

    out = df.copy()
    for col in ("xt_start_v3c", "xt_end_v3c", "delta_xt_v3c"):
        out[col] = 0.0

    xt_mask = out["category"].isin(["passes", "ball-carries"]) & out["has_end"]
    if xt_mask.any():
        xt_df_v3c = apply_heuristic_v3c_xt(out.loc[xt_mask].copy())
        out.loc[xt_mask, ["xt_start_v3c", "xt_end_v3c", "delta_xt_v3c"]] = xt_df_v3c[
            ["xt_start_v3c", "xt_end_v3c", "delta_xt_v3c"]
        ].values
    return out


def _safe_col_sum(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())


# ── DATA LOADING ─────────────────────────────────────────────
def discover_csv_files(base_dir: Path | None = None) -> list[Path]:
    root = base_dir or Path(__file__).resolve().parent
    return sorted(
        p for p in root.glob("*.csv")
        if p.name not in EXCLUDED_CSV
    )


def load_player_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"category", "eventActionType", "start_x", "start_y"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {path.name}: {', '.join(sorted(missing))}")

    rows = []
    is_home = _parse_bool(frame["isHome"].iloc[0]) if "isHome" in frame.columns and len(frame) else True
    flip_x = not is_home

    for idx, row in frame.iterrows():
        sx, sy = wyscout_to_statsbomb(float(row["start_x"]), float(row["start_y"]), flip_x=flip_x)
        has_end = _has_coords(row, "end")
        ex = ey = np.nan
        if has_end:
            ex, ey = wyscout_to_statsbomb(float(row["end_x"]), float(row["end_y"]), flip_x=flip_x)

        rows.append(
            {
                "category": str(row["category"]).strip().lower(),
                "action_type": str(row["eventActionType"]).strip().lower(),
                "is_home": _parse_bool(row.get("isHome")),
                "is_success": _parse_bool(row.get("outcome")),
                "is_key_pass": _parse_bool(row.get("keypass")),
                "is_long_ball": _parse_bool(row.get("isLongBall")),
                "x_start": sx,
                "y_start": sy,
                "x_end": ex,
                "y_end": ey,
                "has_end": has_end,
                "player": path.stem.replace("_", " ").title(),
                "source_file": path.name,
                "row_id": idx + 1,
            }
        )

    return enrich_with_xt_v3(pd.DataFrame(rows))


def load_player_all_matches(code: str, name: str, base_dir: Path | None = None) -> pd.DataFrame:
    """Aggregate all match CSVs for a player code (BG, CS, LP)."""
    root = base_dir or Path(__file__).resolve().parent
    files = sorted(root.glob(f"{code}-vs *.csv"))
    if not files:
        return pd.DataFrame()

    frames = []
    for path in files:
        match_df = load_player_csv(path)
        match_df["match"] = path.stem.replace(f"{code}-", "")
        frames.append(match_df)

    combined = pd.concat(frames, ignore_index=True)
    combined["player"] = name
    return combined


def top_deltaxt_actions(
    df: pd.DataFrame, n: int = TOP_DELTAXT_N, delta_col: str = "delta_xt"
) -> pd.DataFrame:
    """Top N passes and carries by positive delta xT."""
    actions = df[
        df["category"].isin(["passes", "ball-carries"]) & df["has_end"]
    ].copy()
    if delta_col not in actions.columns:
        return pd.DataFrame()
    actions = actions[actions[delta_col] > 0]
    if actions.empty:
        return actions
    return actions.nlargest(n, delta_col)


def get_available_matches(player_data: dict[str, pd.DataFrame]) -> list[str]:
    matches: set[str] = set()
    for df in player_data.values():
        if not df.empty and "match" in df.columns:
            matches.update(df["match"].dropna().unique())
    return sorted(matches)


def filter_by_match(df: pd.DataFrame, match_selection: str) -> pd.DataFrame:
    if df.empty or match_selection == ALL_MATCHES_LABEL:
        return df
    if "match" not in df.columns:
        return df
    return df[df["match"] == match_selection].copy()


def _match_scope_label(match_selection: str) -> str:
    return "todos os jogos" if match_selection == ALL_MATCHES_LABEL else match_selection


# ── STATS ────────────────────────────────────────────────────
def compute_player_stats(df: pd.DataFrame) -> dict:
    passes = df[df["category"] == "passes"]
    carries = df[df["category"] == "ball-carries"]
    empty_cls = {"successful": 0, "attempted": 0, "accuracy_pct": 0.0}

    total_passes = len(passes)
    if total_passes == 0:
        return {
            "total_passes": 0,
            "accuracy_pct": 0.0,
            "progressive_wyscout": empty_cls.copy(),
            "impact_pass": empty_cls.copy(),
            "high_impact_pass": empty_cls.copy(),
            "impact_carry": empty_cls.copy(),
            "high_impact_carry": empty_cls.copy(),
            "sum_dxt_passes": 0.0,
            "sum_dxt_carries": 0.0,
            "sum_xt_end_passes": 0.0,
            "pos_pct": 0.0,
            "carries_total": len(carries),
            "defensive_total": int((df["category"] == "defensive").sum()),
            "total_actions": len(df),
            "by_action_type": df.groupby("action_type").size().to_dict(),
        }

    successful = int(passes["is_success"].sum())
    accuracy = successful / total_passes * 100.0
    progressive_wyscout = classification_accuracy(
        passes,
        "progressive",
        lambda r: is_progressive_wyscout(r["x_start"], r["y_start"], r["x_end"], r["y_end"]),
    )
    impact_pass = classification_accuracy(passes, "impact_pass", is_impact_attempt)
    high_impact_pass = classification_accuracy(passes, "high_impact_pass", is_high_impact_attempt)
    impact_carry = classification_accuracy(carries, "impact_carry", is_impact_attempt)
    high_impact_carry = classification_accuracy(carries, "high_impact_carry", is_high_impact_attempt)

    xt_actions = df[df["category"].isin(["passes", "ball-carries"]) & df["has_end"]]
    pos_count = int((xt_actions["delta_xt"] > 0).sum())
    pos_pct = (pos_count / len(xt_actions) * 100.0) if len(xt_actions) else 0.0

    completed_passes = passes[passes["is_success"]]

    return {
        "total_passes": total_passes,
        "accuracy_pct": accuracy,
        "progressive_wyscout": progressive_wyscout,
        "impact_pass": impact_pass,
        "high_impact_pass": high_impact_pass,
        "impact_carry": impact_carry,
        "high_impact_carry": high_impact_carry,
        "sum_dxt_passes": float(passes["delta_xt"].sum()),
        "sum_dxt_carries": float(carries["delta_xt"].sum()),
        "sum_xt_end_passes": float(completed_passes["xt_end"].sum()) if not completed_passes.empty else 0.0,
        "pos_pct": pos_pct,
        "carries_total": len(carries),
        "defensive_total": int((df["category"] == "defensive").sum()),
        "total_actions": len(df),
        "by_action_type": df.groupby("action_type").size().to_dict(),
    }


def _item_sep(idx: int, total: int) -> str:
    return "" if idx == total - 1 else f"margin-bottom:14px;padding-bottom:14px;border-bottom:1px solid {CARD_INNER_BORDER};"


def _accent_rgb(border_color: str) -> tuple[int, int, int]:
    h = border_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _stats_card_shell_html(title: str, border_color: str, body_html: str) -> str:
    r, g, b = _accent_rgb(border_color)
    grad = (
        f"linear-gradient(150deg, rgba({r},{g},{b},0.18) 0%, "
        f"rgba(24,24,38,0.55) 55%, rgba(16,16,26,0.82) 100%)"
    )
    html = (
        f'<div style="background:{grad};border:1px solid rgba({r},{g},{b},0.55);'
        f'border-radius:14px;padding:18px 20px 14px 20px;margin-bottom:12px;">'
    )
    html += (
        f'<div style="border-bottom:2.5px solid rgb({r},{g},{b});padding-bottom:8px;margin-bottom:12px;">'
        f'<span style="font-size:{CARD_TITLE_TEXT};color:#eef1f7;font-weight:700;letter-spacing:0.04em;">'
        f"{title.upper()}</span></div>"
    )
    html += body_html
    html += "</div>"
    return html


def _simple_body_scoreboard(items: list[tuple[str, str]]) -> str:
    body = ""
    for idx, (label, disp_val) in enumerate(items):
        body += f'<div style="{_item_sep(idx, len(items))}">'
        body += (
            '<div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px;">'
            f'<span style="font-size:{CARD_LABEL_TEXT};color:#c7cdda;font-weight:600;">{label}</span>'
            f'<span style="font-size:28px;color:#ffffff;font-weight:700;line-height:1;">{disp_val}</span>'
            "</div>"
        )
        body += "</div>"
    return body


def stats_section_card(title: str, border_color: str, items: list[tuple[str, str]]) -> None:
    inner = _simple_body_scoreboard(items)
    st.markdown(_stats_card_shell_html(title, border_color, inner), unsafe_allow_html=True)


def render_impact_card(stats: dict, tone: str) -> None:
    impact = stats["impact_pass"]
    high_impact = stats["high_impact_pass"]
    carry_impact = stats["impact_carry"]
    carry_high = stats["high_impact_carry"]
    stats_section_card(
        "Impact",
        tone,
        [
            ("Pass Impact (xT v3)", f"{stats['sum_dxt_passes']:.2f}"),
            ("Σ xT final passes", f"{stats['sum_xt_end_passes']:.2f}"),
            ("Carry Impact (xT v3)", f"{stats['sum_dxt_carries']:.2f}"),
            ("Total Impact (xT v3)", f"{stats['sum_dxt_passes'] + stats['sum_dxt_carries']:.2f}"),
            ("Impact Passes", f"{impact['successful']:.0f}"),
            ("High Impact Passes", f"{high_impact['successful']:.0f}"),
            ("Impact Carries", f"{carry_impact['successful']:.0f}"),
            ("High Impact Carries", f"{carry_high['successful']:.0f}"),
            ("% Positive Impact", f"{stats['pos_pct']:.1f}%"),
        ],
    )


# ── PITCH DRAWING ────────────────────────────────────────────
def _base_pitch(bg="#1a1a2e"):
    pitch = Pitch(pitch_type="statsbomb", pitch_color=bg, line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor(bg)
    fig.set_dpi(FIG_DPI)
    return fig, ax, pitch


def _map_scale() -> float:
    return FIG_W / MAP_REF_WIDTH


def _add_map_legend(ax, handles: list) -> None:
    scale = _map_scale()
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="#1a1a2e",
        edgecolor="#444466",
        fontsize=6.2 * scale,
        labelspacing=0.35 * scale,
        borderpad=0.45 * scale,
        handlelength=1.9 * scale,
    )
    for text in leg.get_texts():
        text.set_color("white")
    leg.get_frame().set_alpha(0.90)


def _attack_arrow(fig, has_cbar: bool = False):
    scale = _map_scale()
    ox = -0.04 if has_cbar else 0.0
    fig.patches.append(
        FancyArrowPatch(
            (0.44 + ox, 0.045),
            (0.56 + ox, 0.045),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=10 * scale,
            linewidth=1.4 * scale,
            color="#aaaaaa",
        )
    )
    fig.text(
        0.50 + ox,
        0.012,
        "Attacking Direction",
        ha="center",
        va="bottom",
        transform=fig.transFigure,
        fontsize=7.0 * scale,
        color="#aaaaaa",
    )


def _save_fig(fig):
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)


def _delicate_arrows(
    pitch, ax, x1, y1, x2, y2, color, scale: float, *, alpha: float | None = None, width_mult: float = 1.0
) -> None:
    pitch.arrows(
        x1, y1, x2, y2,
        color=color,
        width=ARROW_WIDTH * scale * width_mult,
        headwidth=ARROW_HEADWIDTH * scale * width_mult,
        headlength=ARROW_HEADLENGTH * scale * width_mult,
        ax=ax,
        zorder=3,
        alpha=alpha if alpha is not None else ARROW_ALPHA,
    )


def draw_pass_map(df: pd.DataFrame, player_name: str, match_label: str):
    passes = df[df["category"] == "passes"].copy()
    fig, ax, pitch = _base_pitch()
    scale = _map_scale()

    for _, row in passes.iterrows():
        if not row["has_end"]:
            continue
        is_lost = not row["is_success"]
        is_high_impact = bool(row.get("high_impact_pass", False))
        is_prog = bool(row.get("progressive", False))
        if is_lost:
            color, alpha = COLOR_FAIL, ARROW_ALPHA_EMPH
        elif is_high_impact:
            color, alpha = COLOR_HIGHLY_PROGRESSIVE, ARROW_ALPHA_EMPH
        elif is_prog:
            color, alpha = COLOR_PROGRESSIVE, ARROW_ALPHA_EMPH
        else:
            color, alpha = COLOR_SUCCESS, ARROW_ALPHA

        _delicate_arrows(
            pitch, ax,
            row["x_start"], row["y_start"], row["x_end"], row["y_end"],
            color, scale, alpha=alpha,
        )
        pitch.scatter(
            row["x_start"], row["y_start"],
            s=PASS_START_MARKER_SIZE, marker="o", color=color,
            edgecolors="white", linewidths=0.3, ax=ax, zorder=6, alpha=alpha,
        )

    legend_handles = [
        Line2D([0], [0], color=COLOR_SUCCESS, lw=1.4 * scale, label="Completado", alpha=0.65),
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=1.4 * scale, label="Progressivo", alpha=0.80),
        Line2D([0], [0], color=COLOR_HIGHLY_PROGRESSIVE, lw=1.4 * scale, label="High Impact", alpha=0.85),
        Line2D([0], [0], color=COLOR_FAIL, lw=1.4 * scale, label="Incompleto", alpha=0.80),
    ]
    _add_map_legend(ax, legend_handles)
    ax.set_title(
        f"{player_name}\nPasses · {match_label}",
        color="white", fontsize=8.8 * scale, pad=5,
    )
    _attack_arrow(fig)
    return _save_fig(fig), fig


def draw_carry_map(df: pd.DataFrame, player_name: str, match_label: str):
    carries = df[df["category"] == "ball-carries"].copy()
    fig, ax, pitch = _base_pitch()
    scale = _map_scale()

    for _, row in carries.iterrows():
        if not row["has_end"]:
            continue
        is_high_impact = bool(row.get("high_impact_carry", False))
        is_impact = bool(row.get("impact_carry", False))
        if is_high_impact:
            color, alpha = COLOR_HIGHLY_PROGRESSIVE, ARROW_ALPHA_EMPH
        elif is_impact:
            color, alpha = COLOR_PROGRESSIVE, ARROW_ALPHA_EMPH
        else:
            color, alpha = COLOR_CARRY, COLOR_CARRY_BASE_ALPHA

        _delicate_arrows(
            pitch, ax,
            row["x_start"], row["y_start"], row["x_end"], row["y_end"],
            color, scale, alpha=alpha,
        )
        pitch.scatter(
            row["x_start"], row["y_start"],
            s=CARRY_START_MARKER_SIZE, marker="o", color=color,
            edgecolors="white", linewidths=0.3, ax=ax, zorder=6, alpha=alpha,
        )

    legend_handles = [
        Line2D([0], [0], color=COLOR_CARRY, lw=1.4 * scale, label="Condução", alpha=0.60),
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=1.4 * scale, label="Impact", alpha=0.80),
        Line2D([0], [0], color=COLOR_HIGHLY_PROGRESSIVE, lw=1.4 * scale, label="High Impact", alpha=0.85),
    ]
    _add_map_legend(ax, legend_handles)
    ax.set_title(
        f"{player_name}\nConduções · {match_label}",
        color="white", fontsize=8.8 * scale, pad=5,
    )
    _attack_arrow(fig)
    return _save_fig(fig), fig


def draw_top_deltaxt_map(
    df: pd.DataFrame,
    player_name: str,
    match_label: str,
    *,
    delta_col: str = "delta_xt",
    model_label: str = "v3",
):
    """Top delta-xT actions with distinct colormaps for passes vs carries."""
    top = top_deltaxt_actions(df, TOP_DELTAXT_N, delta_col=delta_col)
    fig, ax, pitch = _base_pitch()
    scale = _map_scale()

    if top.empty:
        ax.text(
            60, 40, "Sem ações com ΔxT positivo",
            ha="center", va="center", color="white", fontsize=9,
        )
    else:
        passes = top[top["category"] == "passes"]
        carries = top[top["category"] == "ball-carries"]
        pass_vmax = max(float(passes[delta_col].max()), 0.01) if not passes.empty else 0.01
        carry_vmax = max(float(carries[delta_col].max()), 0.01) if not carries.empty else 0.01

        if not passes.empty:
            norm_pass = Normalize(vmin=0, vmax=pass_vmax)
            for _, row in passes.iterrows():
                color = CMAP_PASS(norm_pass(row[delta_col]))
                _delicate_arrows(
                    pitch, ax,
                    row["x_start"], row["y_start"], row["x_end"], row["y_end"],
                    color, scale, alpha=ARROW_ALPHA_EMPH,
                )
                pitch.scatter(
                    row["x_start"], row["y_start"],
                    s=PASS_START_MARKER_SIZE + 4, marker="o", color=color,
                    edgecolors="white", linewidths=0.3, ax=ax, zorder=6, alpha=0.88,
                )

        if not carries.empty:
            norm_carry = Normalize(vmin=0, vmax=carry_vmax)
            for _, row in carries.iterrows():
                color = CMAP_CARRY(norm_carry(row[delta_col]))
                _delicate_arrows(
                    pitch, ax,
                    row["x_start"], row["y_start"], row["x_end"], row["y_end"],
                    color, scale, alpha=ARROW_ALPHA_EMPH, width_mult=1.05,
                )
                pitch.scatter(
                    row["x_end"], row["y_end"],
                    s=CARRY_START_MARKER_SIZE + 6, marker="s", color=color,
                    edgecolors="white", linewidths=0.3, ax=ax, zorder=6, alpha=0.88,
                )

        legend_handles = [
            Line2D([0], [0], color=CMAP_PASS(0.85), lw=1.4 * scale, label="Passe (ΔxT)"),
            Line2D([0], [0], color=CMAP_CARRY(0.85), lw=1.4 * scale, label="Condução (ΔxT)"),
        ]
        _add_map_legend(ax, legend_handles)

    ax.set_title(
        f"{player_name}\nTop {TOP_DELTAXT_N} ΔxT · {model_label} · {match_label}",
        color="white", fontsize=8.8 * scale, pad=5,
    )
    _attack_arrow(fig)
    return _save_fig(fig), fig


def draw_xt_threat_surface(grid: np.ndarray, title: str, vmax: float):
    """Heatmap of the heuristic xT threat surface on the pitch."""
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e", line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor("#1a1a2e")
    fig.set_dpi(FIG_DPI)
    scale = _map_scale()

    ny, nx = grid.shape
    x_edges = np.linspace(0, FIELD_X, nx + 1)
    y_edges = np.linspace(0, FIELD_Y, ny + 1)
    ax.pcolormesh(
        x_edges, y_edges, grid,
        cmap="magma", vmin=0, vmax=vmax, shading="auto", alpha=0.88, zorder=1,
    )
    pitch.draw(ax=ax)

    ax.set_title(title, color="white", fontsize=8.8 * scale, pad=5)
    _attack_arrow(fig)
    return _save_fig(fig), fig


def draw_xt_grid_map(
    grid: np.ndarray,
    title: str,
    *,
    as_percent: bool = True,
    color_percentile: tuple[float, float] | None = (5, 95),
    value_fmt: str = ".2f",
):
    """16×12 pitch grid with xT value labeled in each cell (Hudson-style)."""
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e", line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(7.8, 5.2))
    fig.set_facecolor("#1a1a2e")
    fig.set_dpi(FIG_DPI)
    scale = 7.8 / MAP_REF_WIDTH

    x_bins = np.linspace(0, FIELD_X, NX_XT + 1)
    y_bins = np.linspace(0, FIELD_Y, NY_XT + 1)
    if color_percentile is not None:
        vmin = float(np.percentile(grid, color_percentile[0]))
        vmax = float(np.percentile(grid, color_percentile[1]))
    else:
        vmin = 0.0
        vmax = max(float(grid.max()), 1e-6)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    norm = Normalize(vmin=vmin, vmax=vmax)
    threshold = vmin + (vmax - vmin) * 0.45

    for iy in range(NY_XT):
        for ix in range(NX_XT):
            value = float(grid[iy, ix])
            x0, x1 = x_bins[ix], x_bins[ix + 1]
            y0, y1 = y_bins[iy], y_bins[iy + 1]
            ax.add_patch(
                Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    facecolor=XT_GRID_CMAP(norm(value)),
                    edgecolor=(1, 1, 1, 0.15),
                    linewidth=0.4,
                    alpha=0.95,
                    zorder=2,
                )
            )
            label = f"{value * 100:.1f}%" if as_percent else f"{value:{value_fmt}}"
            ax.text(
                (x0 + x1) / 2, (y0 + y1) / 2, label,
                ha="center", va="center",
                color="#000000" if value <= threshold else "#ffffff",
                fontsize=5.2 * scale, fontweight="600", zorder=4,
            )

    pitch.draw(ax=ax)
    ax.set_title(title, color="#eef1f7", fontsize=10 * scale, pad=8)
    sm = plt.cm.ScalarMappable(cmap=XT_GRID_CMAP, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.02, shrink=0.55)
    cbar.ax.yaxis.set_tick_params(color="#ffffff", labelsize=6)
    plt.setp(cbar.ax.axes.get_yticklabels(), color="#ffffff")
    _attack_arrow(fig, has_cbar=True)
    return _save_fig(fig), fig


def zone_xt_means(grid: np.ndarray, n_x: int = XT_ZONE_COLS, n_y: int = XT_ZONE_ROWS) -> np.ndarray:
    """Mean xT per pitch zone from the fine threat grid."""
    ny, nx = grid.shape
    zones = np.zeros((n_y, n_x), dtype=float)
    for iy in range(n_y):
        y_start = int(iy * ny / n_y)
        y_end = int((iy + 1) * ny / n_y)
        for ix in range(n_x):
            x_start = int(ix * nx / n_x)
            x_end = int((ix + 1) * nx / n_x)
            zones[iy, ix] = float(grid[y_start:y_end, x_start:x_end].mean())
    return zones


@st.cache_data(show_spinner=False)
def load_all_three_players(_cache_version: int = DATA_CACHE_VERSION) -> dict[str, pd.DataFrame]:
    return {
        player["code"]: load_player_all_matches(player["code"], player["name"])
        for player in PLAYERS
    }


def _show_map(draw_fn, df: pd.DataFrame, player_name: str, match_label: str, empty_msg: str) -> None:
    if df.empty:
        st.info(empty_msg)
        return
    img, fig = draw_fn(df, player_name, match_label)
    plt.close(fig)
    st.image(img, use_container_width=True)


def render_comparison(player_data: dict[str, pd.DataFrame], match_selection: str) -> None:
    match_label = _match_scope_label(match_selection)
    map_cols = st.columns(3)

    for col, player in zip(map_cols, PLAYERS):
        with col:
            st.markdown(f'<div class="player-header">{player["name"]}</div>', unsafe_allow_html=True)
            df = filter_by_match(player_data[player["code"]], match_selection)

            if df.empty:
                st.warning(f"Sem dados para {player['name']}.")
                continue

            st.markdown('<div class="map-label">Passes</div>', unsafe_allow_html=True)
            _show_map(draw_pass_map, df, player["name"], match_label, "Sem passes no recorte.")

            st.markdown('<div class="map-label">Conduções</div>', unsafe_allow_html=True)
            _show_map(draw_carry_map, df, player["name"], match_label, "Sem conduções no recorte.")

            st.markdown(
                f'<div class="map-label">Top {TOP_DELTAXT_N} ΔxT</div>',
                unsafe_allow_html=True,
            )
            _show_map(
                draw_top_deltaxt_map, df, player["name"], match_label,
                "Sem ações com ΔxT positivo.",
            )

    st.markdown("---")
    st.markdown("### Impact")
    stat_cols = st.columns(3)
    for col, player in zip(stat_cols, PLAYERS):
        with col:
            df = filter_by_match(player_data[player["code"]], match_selection)
            if not df.empty:
                render_impact_card(compute_player_stats(df), player["tone"])


def render_xt_model_comparison(
    player_data: dict[str, pd.DataFrame], match_selection: str
) -> None:
    """Compare xT v3 vs conservative v3c threat surfaces and top ΔxT maps."""
    match_label = _match_scope_label(match_selection)

    st.markdown("### Mapa xT por quadrante (16×12)")
    st.caption(
        "Cada célula mostra o xT médio da superfície naquele quadrante, "
        "em percentual (valor × 100). Cores: azul (baixo) → vermelho (alto)."
    )
    grid_v3 = compute_heuristic_v3_xt_grid()
    grid_v3c = compute_heuristic_v3c_xt_grid()
    grid_cols = st.columns(2)
    with grid_cols[0]:
        st.markdown('<div class="map-label">Heurístico v3</div>', unsafe_allow_html=True)
        img_gv3, fig_gv3 = draw_xt_grid_map(grid_v3, "Heurístico v3", as_percent=True)
        plt.close(fig_gv3)
        st.image(img_gv3, use_container_width=True)
        st.caption(f"Máx: {grid_v3.max():.3f} · Média: {grid_v3.mean():.3f}")
    with grid_cols[1]:
        st.markdown('<div class="map-label">Heurístico v3c (conservador)</div>', unsafe_allow_html=True)
        img_gv3c, fig_gv3c = draw_xt_grid_map(grid_v3c, "Heurístico v3c", as_percent=True)
        plt.close(fig_gv3c)
        st.image(img_gv3c, use_container_width=True)
        st.caption(f"Máx: {grid_v3c.max():.3f} · Média: {grid_v3c.mean():.3f}")

    with st.expander("Superfície contínua xT"):
        surf_cols = st.columns(2)
        fine_v3 = compute_heuristic_v3_fine_grid()
        fine_v3c = compute_heuristic_v3c_fine_grid()
        with surf_cols[0]:
            img_v3, fig_v3 = draw_xt_threat_surface(fine_v3, "Superfície v3", XT_V3_SURFACE_MAX)
            plt.close(fig_v3)
            st.image(img_v3, use_container_width=True)
        with surf_cols[1]:
            img_v3c, fig_v3c = draw_xt_threat_surface(fine_v3c, "Superfície v3c", XT_V3C_SURFACE_MAX)
            plt.close(fig_v3c)
            st.image(img_v3c, use_container_width=True)

    st.markdown("---")
    st.markdown("### Top 10 ΔxT — v3 vs v3c")

    summary_rows = []
    for player in PLAYERS:
        df = filter_by_match(player_data[player["code"]], match_selection)
        st.markdown(f'<div class="player-header">{player["name"]}</div>', unsafe_allow_html=True)

        if df.empty:
            st.warning(f"Sem dados para {player['name']}.")
            continue

        xt_actions = df[df["category"].isin(["passes", "ball-carries"]) & df["has_end"]]
        passes = df[df["category"] == "passes"]
        summary_rows.append({
            "Jogador": player["name"],
            "Σ ΔxT v3": round(_safe_col_sum(xt_actions, "delta_xt"), 3),
            "Σ ΔxT v3c": round(_safe_col_sum(xt_actions, "delta_xt_v3c"), 3),
            "Σ xT final v3": round(_safe_col_sum(passes, "xt_end"), 3),
            "Σ xT final v3c": round(_safe_col_sum(passes, "xt_end_v3c"), 3),
        })

        cmp_cols = st.columns(2)
        with cmp_cols[0]:
            st.markdown('<div class="map-label">Top ΔxT · v3</div>', unsafe_allow_html=True)
            _show_map(
                lambda d, n, m: draw_top_deltaxt_map(
                    d, n, m, delta_col="delta_xt", model_label="v3"
                ),
                df, player["name"], match_label, "Sem ações com ΔxT positivo (v3).",
            )
        with cmp_cols[1]:
            st.markdown('<div class="map-label">Top ΔxT · v3c</div>', unsafe_allow_html=True)
            _show_map(
                lambda d, n, m: draw_top_deltaxt_map(
                    d, n, m, delta_col="delta_xt_v3c", model_label="v3c"
                ),
                df, player["name"], match_label, "Sem ações com ΔxT positivo (v3c).",
            )

    if summary_rows:
        st.markdown("---")
        st.markdown("### Resumo comparativo")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ── MAIN ─────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center;margin-bottom:1rem;">
      <h1 style="margin:0;color:#eef1f7;">WC Player Analysis — Top ΔxT</h1>
      <p style="color:#94a3b8;font-size:0.95rem;margin-top:0.35rem;">
        Bruno Guimarães · Casemiro · Lucas Paquetá — xT Heurístico v3
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

player_data = {
    code: ensure_xt_model_columns(df)
    for code, df in load_all_three_players().items()
}
if not any(not df.empty for df in player_data.values()):
    st.error(
        "Nenhum CSV de jogador encontrado. "
        "Esperado: `BG-vs *.csv`, `CS-vs *.csv`, `LP-vs *.csv`."
    )
    st.stop()

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;">
          <h3 style="margin:0;color:#eef1f7;">Partidas</h3>
          <p style="color:#94a3b8;font-size:0.85rem;">Filtrar mapas e stats</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    match_options = [ALL_MATCHES_LABEL, *get_available_matches(player_data)]
    selected_match = st.selectbox("Selecionar partida", match_options, label_visibility="collapsed")
    st.caption("xT v3 · v3c conservador · Progressivos Wyscout")

tab_analysis, tab_compare = st.tabs(["Análise", "Comparar xT v3 vs v3c"])

with tab_analysis:
    render_comparison(player_data, selected_match)

with tab_compare:
    render_xt_model_comparison(player_data, selected_match)
