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
DATA_CACHE_VERSION = 17
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

# xT Heurístico v3.1 — transições suaves e salto reduzido na linha de meio
XT_MODEL_HEURISTIC_V31 = "heuristic_v31"
XT_V31_ZONE_BLEND_WIDTH = 48.0
XT_V31_LAT_DISC_MAX = 0.06
XT_V31_LAT_GATE_X = HALF_LINE_X
XT_V31_GAUSS_SIGMA_X = 3.5
XT_V31_GAUSS_SIGMA_Y = 0.0
XT_V31_COL_SMOOTH_KERNEL = (0.22, 0.56, 0.22)
XT_V31_MAX_COL_STEP_DEF = 0.050
XT_V31_MAX_COL_STEP_ATT = 0.078
XT_V31_ATT_COL_START = 10

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
    """Wyscout 0–100 → StatsBomb 120×80; espelha Y para corrigir corredores laterais."""
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


# ── xT HEURÍSTICO v3.1 (transições graduais) ─────────────────
def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    return kernel / kernel.sum()


def _gaussian_smooth_2d(grid: np.ndarray, sigma_x: float, sigma_y: float) -> np.ndarray:
    out = grid
    if sigma_x > 0:
        kx = _gaussian_kernel_1d(sigma_x)
        out = np.apply_along_axis(lambda row: np.convolve(row, kx, mode="same"), axis=1, arr=out)
    if sigma_y > 0:
        ky = _gaussian_kernel_1d(sigma_y)
        out = np.apply_along_axis(lambda row: np.convolve(row, ky, mode="same"), axis=0, arr=out)
    return out


def _map_zonal_threat_v31(x: np.ndarray) -> np.ndarray:
    """Zonas com blend amplo e curvas smootherstep para transições homogêneas."""
    blend = XT_V31_ZONE_BLEND_WIDTH
    x = np.clip(x, 0.0, FIELD_X)
    threat_def = XT_V3_DEF_MAX * _smootherstep(np.clip(x / OPT_ATTACKING_TWO_THIRDS_X, 0.0, 1.0))
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


def _location_factor_v31(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    lat = _lateral_relative_position(y)
    depth = np.clip(
        (x - XT_V31_LAT_GATE_X) / (FIELD_X - XT_V31_LAT_GATE_X),
        0.0,
        1.0,
    )
    zone_gate = _smootherstep(depth)
    max_discount = XT_V31_LAT_DISC_MAX * zone_gate
    lateral_curve = _smootherstep(lat ** XT_V3_LAT_CURVE_POWER)
    return 1.0 - max_discount * lateral_curve


def _build_heuristic_v31_threat_surface(Xc: np.ndarray, Yc: np.ndarray) -> np.ndarray:
    zonal = _map_zonal_threat_v31(Xc)
    surface = zonal * _location_factor_v31(Xc, Yc)
    surface = np.clip(surface, 0.0, XT_V3_SURFACE_MAX)
    smoothed = _gaussian_smooth_2d(surface, XT_V31_GAUSS_SIGMA_X, XT_V31_GAUSS_SIGMA_Y)
    return np.clip(smoothed, 0.0, XT_V3_SURFACE_MAX)


def _smooth_columns_1d(row: np.ndarray, kernel: tuple[float, ...]) -> np.ndarray:
    k = np.asarray(kernel, dtype=float)
    k = k / k.sum()
    pad = len(k) // 2
    padded = np.pad(row, (pad, pad), mode="edge")
    return np.convolve(padded, k, mode="valid")


def _limit_adjacent_column_step(
    grid: np.ndarray,
    max_step: float,
    *,
    att_col_start: int | None = None,
    max_step_att: float | None = None,
) -> np.ndarray:
    """Cap column-to-column jumps while preserving attack-direction growth."""
    out = grid.copy()
    att_start = att_col_start if att_col_start is not None else grid.shape[1]
    att_step = max_step_att if max_step_att is not None else max_step
    for iy in range(out.shape[0]):
        row = out[iy].copy()
        for ix in range(1, row.shape[0]):
            step = att_step if ix >= att_start else max_step
            lo = row[ix - 1]
            hi = lo + step
            if row[ix] > hi:
                row[ix] = hi
            elif row[ix] < lo:
                row[ix] = lo
        out[iy] = row
    return out


def _sample_display_grid(
    fine: np.ndarray,
    n_x: int = NX_XT,
    n_y: int = NY_XT,
    *,
    post_process=None,
) -> np.ndarray:
    grid = zone_xt_means(fine, n_x=n_x, n_y=n_y)
    if post_process is not None:
        grid = post_process(grid)
    return grid


@st.cache_data(show_spinner=False)
def compute_heuristic_v31_fine_grid(
    nx: int = XT_V3_FINE_NX, ny: int = XT_V3_FINE_NY,
) -> np.ndarray:
    xe = np.linspace(0.0, FIELD_X, nx)
    ye = np.linspace(0.0, FIELD_Y, ny)
    Xc, Yc = np.meshgrid(xe, ye)
    return _build_heuristic_v31_threat_surface(Xc, Yc)


@st.cache_data(show_spinner=False)
def compute_heuristic_v31_xt_grid(n_x: int = NX_XT, n_y: int = NY_XT) -> np.ndarray:
    fine = compute_heuristic_v31_fine_grid()

    def _post(grid: np.ndarray) -> np.ndarray:
        smoothed = np.array([
            _smooth_columns_1d(grid[iy], XT_V31_COL_SMOOTH_KERNEL)
            for iy in range(grid.shape[0])
        ])
        return _limit_adjacent_column_step(
            smoothed,
            XT_V31_MAX_COL_STEP_DEF,
            att_col_start=XT_V31_ATT_COL_START,
            max_step_att=XT_V31_MAX_COL_STEP_ATT,
        )

    return _sample_display_grid(fine, n_x, n_y, post_process=_post)


def _adjust_heuristic_v3_variant_pass_delta(row, start_col: str, end_col: str) -> float:
    if not row.is_won:
        return 0.0
    raw = float(getattr(row, end_col) - getattr(row, start_col))
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


def _apply_heuristic_v3_variant_xt(
    df: pd.DataFrame, fine_fn, prefix: str,
) -> pd.DataFrame:
    fine = fine_fn()
    out = df.copy()
    start_col, end_col, delta_col = f"xt_start_{prefix}", f"xt_end_{prefix}", f"delta_xt_{prefix}"
    out[start_col] = out.apply(
        lambda r: xt_value_bilinear(r["x_start"], r["y_start"], fine), axis=1
    )
    out[end_col] = out.apply(
        lambda r: xt_value_bilinear(r["x_end"], r["y_end"], fine), axis=1
    )
    out[delta_col] = out.apply(
        lambda r: _adjust_heuristic_v3_variant_pass_delta(r, start_col, end_col), axis=1
    )
    return out


def apply_heuristic_v31_xt(df: pd.DataFrame) -> pd.DataFrame:
    return _apply_heuristic_v3_variant_xt(df, compute_heuristic_v31_fine_grid, "v31")


def _max_adjacent_col_jump_pct(grid: np.ndarray) -> float:
    if grid.shape[1] < 2:
        return 0.0
    jumps = np.abs(np.diff(grid, axis=1))
    return float(jumps.max() * 100.0)


def classify_xt_progressive_v3_adjusted(
    xt_start: float,
    delta_xt: float,
    x_end: float,
    pass_distance: float,
) -> str:
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


def _xt_column_set(variant: str = "v3") -> dict[str, str]:
    if variant == "v31":
        return {"start": "xt_start_v31", "end": "xt_end_v31", "delta": "delta_xt_v31"}
    return {"start": "xt_start", "end": "xt_end", "delta": "delta_xt"}


def hypothetical_delta_xt(row, cols: dict[str, str] | None = None) -> float:
    cols = cols or _xt_column_set("v3")
    return _adjust_heuristic_v3_variant_pass_delta(_row_if_won(row), cols["start"], cols["end"])


def progressive_delta_for_attempt(row, cols: dict[str, str] | None = None) -> float:
    cols = cols or _xt_column_set("v3")
    if row.is_won:
        return float(getattr(row, cols["delta"]))
    return hypothetical_delta_xt(row, cols)


def is_impact_attempt(row, cols: dict[str, str] | None = None) -> bool:
    cols = cols or _xt_column_set("v3")
    delta = progressive_delta_for_attempt(row, cols)
    return classify_xt_progressive_v3_adjusted(
        getattr(row, cols["start"]), delta, row.x_end, row.pass_distance
    ) in ("progressive", "highly")


def is_high_impact_attempt(row, cols: dict[str, str] | None = None) -> bool:
    cols = cols or _xt_column_set("v3")
    delta = progressive_delta_for_attempt(row, cols)
    return (
        classify_xt_progressive_v3_adjusted(
            getattr(row, cols["start"]), delta, row.x_end, row.pass_distance
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


def classification_accuracy_fn(
    df: pd.DataFrame, attempt_fn, success_fn,
) -> dict:
    if df.empty:
        return {"successful": 0, "attempted": 0, "accuracy_pct": 0.0}
    attempts = df.apply(attempt_fn, axis=1)
    successful = int(df.apply(success_fn, axis=1).sum())
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

    for col in (
        "xt_start", "xt_end", "delta_xt",
        "xt_start_v31", "xt_end_v31", "delta_xt_v31",
    ):
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

    xt_df_v31 = apply_heuristic_v31_xt(out.loc[xt_mask].copy())
    out.loc[xt_mask, ["xt_start_v31", "xt_end_v31", "delta_xt_v31"]] = xt_df_v31[
        ["xt_start_v31", "xt_end_v31", "delta_xt_v31"]
    ].values

    pass_mask = out["category"] == "passes"
    cols_v31 = _xt_column_set("v31")
    for idx, row in out.loc[pass_mask].iterrows():
        out.at[idx, "progressive"] = bool(
            row.is_won
            and is_progressive_wyscout(row.x_start, row.y_start, row.x_end, row.y_end)
        )
        out.at[idx, "impact_pass"] = bool(row.is_won and is_impact_attempt(row, cols_v31))
        out.at[idx, "high_impact_pass"] = bool(row.is_won and is_high_impact_attempt(row, cols_v31))

    carry_mask = out["category"] == "ball-carries"
    for idx, row in out.loc[carry_mask].iterrows():
        out.at[idx, "impact_carry"] = bool(row.is_won and is_impact_attempt(row, cols_v31))
        out.at[idx, "high_impact_carry"] = bool(row.is_won and is_high_impact_attempt(row, cols_v31))

    return out


def ensure_xt_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill variant xT columns when serving cached frames from older app versions."""
    if df.empty:
        return df

    out = df.copy()
    xt_mask = out["category"].isin(["passes", "ball-carries"]) & out["has_end"]

    variant_specs = [
        ("v31", apply_heuristic_v31_xt),
    ]
    for prefix, apply_fn in variant_specs:
        delta_col = f"delta_xt_{prefix}"
        if delta_col not in out.columns:
            for col in (f"xt_start_{prefix}", f"xt_end_{prefix}", delta_col):
                out[col] = 0.0
            if xt_mask.any():
                xt_df = apply_fn(out.loc[xt_mask].copy())
                out.loc[xt_mask, [f"xt_start_{prefix}", f"xt_end_{prefix}", delta_col]] = xt_df[
                    [f"xt_start_{prefix}", f"xt_end_{prefix}", delta_col]
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

    for idx, row in frame.iterrows():
        sx, sy = wyscout_to_statsbomb(float(row["start_x"]), float(row["start_y"]))
        has_end = _has_coords(row, "end")
        ex = ey = np.nan
        if has_end:
            ex, ey = wyscout_to_statsbomb(float(row["end_x"]), float(row["end_y"]))

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


def _safe_ratio(numerator: float, denominator: int, *, decimals: int = 3) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / denominator, decimals)


def _fmt_count(value: int | float | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.0f}" if value.is_integer() else f"{value:.1f}"
    return f"{int(value)}"


def _fmt_decimal(value: float | None, *, decimals: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}"


def _fmt_pct(value: float | None, *, decimals: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}%"


def _count_action(df: pd.DataFrame, action_type: str) -> int:
    if df.empty:
        return 0
    return int((df["action_type"] == action_type).sum())


def _is_prog_wyscout_row(row) -> bool:
    if not row.has_end:
        return False
    return is_progressive_wyscout(row.x_start, row.y_start, row.x_end, row.y_end)


# ── STATS ────────────────────────────────────────────────────
def compute_player_stats(df: pd.DataFrame) -> dict:
    """Player stats using heuristic xT v3.1 only."""
    passes = df[df["category"] == "passes"]
    carries = df[df["category"] == "ball-carries"]
    empty_cls = {"successful": 0, "attempted": 0, "accuracy_pct": 0.0}
    cols = _xt_column_set("v31")
    delta_col, end_col = cols["delta"], cols["end"]

    total_passes = len(passes)
    completed_passes = passes[passes["is_success"]] if total_passes else passes.iloc[0:0]
    carries_total = len(carries)

    general = {
        "passes_total": total_passes,
        "passes_completed": int(len(completed_passes)),
        "passes_accuracy_pct": round(len(completed_passes) / total_passes * 100.0, 1) if total_passes else 0.0,
        "key_passes": int(passes["is_key_pass"].sum()) if total_passes else 0,
        "crosses": _count_action(passes, "cross"),
        "long_balls": int(passes["is_long_ball"].sum()) if total_passes else 0,
        "carries_total": carries_total,
        "dribbles": int((df["category"] == "dribbles").sum()),
        "tackles": _count_action(df, "tackle"),
        "interceptions": _count_action(df, "interception"),
        "clearances": _count_action(df, "clearance"),
        "ball_recoveries": _count_action(df, "ball-recovery"),
        "blocks": _count_action(df, "block"),
        "defensive_total": int((df["category"] == "defensive").sum()),
        "shots": None,
        "xg": None,
        "assists": None,
        "xa": None,
        "total_actions": len(df),
    }

    if total_passes == 0:
        return {
            **general,
            "accuracy_pct": 0.0,
            "progressive_wyscout": empty_cls.copy(),
            "impact_pass": empty_cls.copy(),
            "high_impact_pass": empty_cls.copy(),
            "impact_carry": empty_cls.copy(),
            "high_impact_carry": empty_cls.copy(),
            "sum_dxt_passes": 0.0,
            "sum_dxt_passes_offensive": 0.0,
            "sum_dxt_carries": 0.0,
            "sum_xt_end_passes": 0.0,
            "sum_xt_end_final_third": 0.0,
            "sum_xt_end_long_balls": 0.0,
            "pos_pct": 0.0,
            "xt_per_pass": 0.0,
            "xt_per_prog_pass": 0.0,
            "xt_per_impact_pass": 0.0,
            "xt_per_long_ball": 0.0,
            "by_action_type": df.groupby("action_type").size().to_dict() if not df.empty else {},
        }

    successful = int(passes["is_success"].sum())
    accuracy = successful / total_passes * 100.0
    progressive_wyscout = classification_accuracy(
        passes,
        "progressive",
        _is_prog_wyscout_row,
    )
    impact_pass = classification_accuracy_fn(
        passes,
        lambda r: is_impact_attempt(r, cols),
        lambda r: bool(r.is_won and is_impact_attempt(r, cols)),
    )
    high_impact_pass = classification_accuracy_fn(
        passes,
        lambda r: is_high_impact_attempt(r, cols),
        lambda r: bool(r.is_won and is_high_impact_attempt(r, cols)),
    )
    impact_carry = classification_accuracy_fn(
        carries,
        lambda r: is_impact_attempt(r, cols),
        lambda r: bool(r.is_won and is_impact_attempt(r, cols)),
    )
    high_impact_carry = classification_accuracy_fn(
        carries,
        lambda r: is_high_impact_attempt(r, cols),
        lambda r: bool(r.is_won and is_high_impact_attempt(r, cols)),
    )

    xt_actions = df[df["category"].isin(["passes", "ball-carries"]) & df["has_end"]]
    pos_count = int((xt_actions[delta_col] > 0).sum())
    pos_pct = (pos_count / len(xt_actions) * 100.0) if len(xt_actions) else 0.0

    sum_dxt_passes = float(passes[delta_col].sum())
    offensive_passes = passes[passes["x_start"] >= HALF_LINE_X]
    sum_dxt_passes_offensive = (
        float(offensive_passes[delta_col].sum()) if not offensive_passes.empty else 0.0
    )
    sum_dxt_carries = float(carries[delta_col].sum())
    sum_xt_end_passes = float(completed_passes[end_col].sum()) if not completed_passes.empty else 0.0
    completed_long_balls = completed_passes[completed_passes["is_long_ball"]]
    sum_xt_end_long_balls = (
        float(completed_long_balls[end_col].sum()) if not completed_long_balls.empty else 0.0
    )

    final_third_won = df[
        df["category"].isin(["passes", "ball-carries"])
        & df["has_end"]
        & df["is_won"]
        & (df["x_end"] >= FINAL_THIRD_LINE_X)
    ]
    sum_xt_end_final_third = float(final_third_won[end_col].sum()) if not final_third_won.empty else 0.0

    prog_success_mask = passes.apply(
        lambda r: bool(r.is_success and _is_prog_wyscout_row(r)), axis=1
    )
    prog_success = passes[prog_success_mask]
    impact_success_mask = passes.apply(
        lambda r: bool(r.is_won and is_impact_attempt(r, cols)), axis=1
    )
    impact_success = passes[impact_success_mask]

    return {
        **general,
        "accuracy_pct": accuracy,
        "progressive_wyscout": progressive_wyscout,
        "impact_pass": impact_pass,
        "high_impact_pass": high_impact_pass,
        "impact_carry": impact_carry,
        "high_impact_carry": high_impact_carry,
        "sum_dxt_passes": sum_dxt_passes,
        "sum_dxt_passes_offensive": sum_dxt_passes_offensive,
        "sum_dxt_carries": sum_dxt_carries,
        "sum_xt_end_passes": sum_xt_end_passes,
        "sum_xt_end_final_third": sum_xt_end_final_third,
        "sum_xt_end_long_balls": sum_xt_end_long_balls,
        "pos_pct": pos_pct,
        "xt_per_pass": _safe_ratio(sum_xt_end_passes, len(completed_passes)),
        "xt_per_prog_pass": _safe_ratio(float(prog_success[delta_col].sum()), len(prog_success)),
        "xt_per_impact_pass": _safe_ratio(float(impact_success[delta_col].sum()), len(impact_success)),
        "xt_per_long_ball": _safe_ratio(sum_xt_end_long_balls, len(completed_long_balls)),
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


def render_general_stats_card(stats: dict, tone: str) -> None:
    stats_section_card(
        "Geral",
        tone,
        [
            ("Passes", _fmt_count(stats["passes_total"])),
            ("Passes completados", _fmt_count(stats["passes_completed"])),
            ("% acerto passes", _fmt_pct(stats["passes_accuracy_pct"])),
            ("Key passes", _fmt_count(stats["key_passes"])),
            ("Crosses", _fmt_count(stats["crosses"])),
            ("Bolas longas", _fmt_count(stats["long_balls"])),
            ("Conduções", _fmt_count(stats["carries_total"])),
            ("Dribles", _fmt_count(stats["dribbles"])),
            ("Finalizações", _fmt_count(stats["shots"])),
            ("xG", _fmt_decimal(stats["xg"], decimals=2)),
            ("Assistências", _fmt_count(stats["assists"])),
            ("xA", _fmt_decimal(stats["xa"], decimals=2)),
            ("Desarmes", _fmt_count(stats["tackles"])),
            ("Interceptações", _fmt_count(stats["interceptions"])),
            ("Cortes", _fmt_count(stats["clearances"])),
            ("Recuperações", _fmt_count(stats["ball_recoveries"])),
            ("Bloqueios", _fmt_count(stats["blocks"])),
            ("Ações defensivas", _fmt_count(stats["defensive_total"])),
        ],
    )


def render_impact_card(stats: dict, tone: str) -> None:
    impact = stats["impact_pass"]
    high_impact = stats["high_impact_pass"]
    carry_impact = stats["impact_carry"]
    carry_high = stats["high_impact_carry"]
    prog = stats["progressive_wyscout"]
    stats_section_card(
        "Impact (xT v3.1)",
        tone,
        [
            ("Pass Impact (xT v3.1)", _fmt_decimal(stats["sum_dxt_passes"])),
            ("ΔxT passes campo ofensivo", _fmt_decimal(stats["sum_dxt_passes_offensive"])),
            ("Σ xT final passes", _fmt_decimal(stats["sum_xt_end_passes"])),
            ("Σ xT final bolas longas", _fmt_decimal(stats["sum_xt_end_long_balls"])),
            ("Carry Impact (xT v3.1)", _fmt_decimal(stats["sum_dxt_carries"])),
            (
                "Total Impact (xT v3.1)",
                _fmt_decimal(stats["sum_dxt_passes"] + stats["sum_dxt_carries"]),
            ),
            ("Σ xT terço final", _fmt_decimal(stats["sum_xt_end_final_third"])),
            ("Impact Passes", _fmt_count(impact["successful"])),
            ("High Impact Passes", _fmt_count(high_impact["successful"])),
            ("Impact Carries", _fmt_count(carry_impact["successful"])),
            ("High Impact Carries", _fmt_count(carry_high["successful"])),
            ("Passes prog. Wyscout", _fmt_count(prog["successful"])),
            ("% Positive Impact", _fmt_pct(stats["pos_pct"])),
        ],
    )


def render_xt_efficiency_card(stats: dict, tone: str) -> None:
    stats_section_card(
        "Eficiência xT (v3.1)",
        tone,
        [
            ("xT / passe", _fmt_decimal(stats["xt_per_pass"], decimals=3)),
            ("xT / passe prog.", _fmt_decimal(stats["xt_per_prog_pass"], decimals=3)),
            ("xT / impact passe", _fmt_decimal(stats["xt_per_impact_pass"], decimals=3)),
            ("xT / bola longa", _fmt_decimal(stats["xt_per_long_ball"], decimals=3)),
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


def filter_impact_plays(df: pd.DataFrame) -> pd.DataFrame:
    """Keep passes/carries classified as impact attempts (xT v3.1)."""
    if df.empty:
        return df
    cols = _xt_column_set("v31")
    mask = df.apply(
        lambda r: (
            r["category"] in ("passes", "ball-carries")
            and r["has_end"]
            and is_impact_attempt(r, cols)
        ),
        axis=1,
    )
    return df[mask].copy()


def draw_impact_plays_map(df: pd.DataFrame, player_name: str, match_label: str):
    """Passes and carries that qualify as impact attempts (xT v3.1)."""
    cols = _xt_column_set("v31")
    actions = df[
        df["category"].isin(["passes", "ball-carries"]) & df["has_end"]
    ].copy()
    actions = actions[actions.apply(lambda r: is_impact_attempt(r, cols), axis=1)]

    fig, ax, pitch = _base_pitch()
    scale = _map_scale()

    if actions.empty:
        ax.text(
            60, 40, "Sem impact plays no recorte",
            ha="center", va="center", color="white", fontsize=9,
        )
    else:
        for _, row in actions.iterrows():
            is_pass = row["category"] == "passes"
            is_lost = is_pass and not row["is_success"]
            is_high = is_high_impact_attempt(row, cols)
            if is_lost:
                color, alpha = COLOR_FAIL, ARROW_ALPHA_EMPH
            elif is_high:
                color, alpha = COLOR_HIGHLY_PROGRESSIVE, ARROW_ALPHA_EMPH
            else:
                color, alpha = COLOR_PROGRESSIVE, ARROW_ALPHA_EMPH

            _delicate_arrows(
                pitch, ax,
                row["x_start"], row["y_start"], row["x_end"], row["y_end"],
                color, scale, alpha=alpha,
            )
            if is_pass:
                pitch.scatter(
                    row["x_start"], row["y_start"],
                    s=PASS_START_MARKER_SIZE + 2, marker="o", color=color,
                    edgecolors="white", linewidths=0.3, ax=ax, zorder=6, alpha=alpha,
                )
            else:
                pitch.scatter(
                    row["x_end"], row["y_end"],
                    s=CARRY_START_MARKER_SIZE + 4, marker="s", color=color,
                    edgecolors="white", linewidths=0.3, ax=ax, zorder=6, alpha=alpha,
                )

    legend_handles = [
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=1.4 * scale, label="Impact", alpha=0.80),
        Line2D([0], [0], color=COLOR_HIGHLY_PROGRESSIVE, lw=1.4 * scale, label="High Impact", alpha=0.85),
        Line2D([0], [0], color=COLOR_FAIL, lw=1.4 * scale, label="Impact incompleto (passe)", alpha=0.80),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_PROGRESSIVE, markersize=4, linestyle="None", label="Passe"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_PROGRESSIVE, markersize=4, linestyle="None", label="Condução"),
    ]
    _add_map_legend(ax, legend_handles)
    ax.set_title(
        f"{player_name}\nImpact Plays · xT v3.1 · {match_label}",
        color="white", fontsize=8.8 * scale, pad=5,
    )
    _attack_arrow(fig)
    return _save_fig(fig), fig


def draw_pass_map(df: pd.DataFrame, player_name: str, match_label: str, *, impact_only: bool = False):
    passes = df[df["category"] == "passes"].copy()
    if impact_only:
        cols = _xt_column_set("v31")
        passes = passes[passes.apply(lambda r: is_impact_attempt(r, cols), axis=1)]
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
    title_suffix = " · Impact" if impact_only else ""
    ax.set_title(
        f"{player_name}\nPasses{title_suffix} · {match_label}",
        color="white", fontsize=8.8 * scale, pad=5,
    )
    _attack_arrow(fig)
    return _save_fig(fig), fig


def draw_carry_map(df: pd.DataFrame, player_name: str, match_label: str, *, impact_only: bool = False):
    carries = df[df["category"] == "ball-carries"].copy()
    if impact_only:
        cols = _xt_column_set("v31")
        carries = carries[carries.apply(lambda r: is_impact_attempt(r, cols), axis=1)]
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
    title_suffix = " · Impact" if impact_only else ""
    ax.set_title(
        f"{player_name}\nConduções{title_suffix} · {match_label}",
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
    n_x: int | None = None,
    n_y: int | None = None,
):
    """Pitch grid with xT value labeled in each cell (Hudson-style)."""
    grid_rows, grid_cols = grid.shape
    cols = n_x if n_x is not None else grid_cols
    rows = n_y if n_y is not None else grid_rows
    if cols != grid_cols or rows != grid_rows:
        cols, rows = grid_cols, grid_rows
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e", line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(7.8, 5.2))
    fig.set_facecolor("#1a1a2e")
    fig.set_dpi(FIG_DPI)
    scale = 7.8 / MAP_REF_WIDTH

    x_bins = np.linspace(0, FIELD_X, cols + 1)
    y_bins = np.linspace(0, FIELD_Y, rows + 1)
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

    for iy in range(rows):
        for ix in range(cols):
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


def render_comparison(
    player_data: dict[str, pd.DataFrame],
    match_selection: str,
    *,
    impact_plays_only: bool = False,
) -> None:
    match_label = _match_scope_label(match_selection)
    map_cols = st.columns(3)

    for col, player in zip(map_cols, PLAYERS):
        with col:
            st.markdown(f'<div class="player-header">{player["name"]}</div>', unsafe_allow_html=True)
            df = filter_by_match(player_data[player["code"]], match_selection)

            if df.empty:
                st.warning(f"Sem dados para {player['name']}.")
                continue

            if impact_plays_only:
                st.markdown('<div class="map-label">Impact Plays</div>', unsafe_allow_html=True)
                _show_map(
                    draw_impact_plays_map, df, player["name"], match_label,
                    "Sem impact plays no recorte.",
                )
            else:
                st.markdown('<div class="map-label">Passes</div>', unsafe_allow_html=True)
                _show_map(draw_pass_map, df, player["name"], match_label, "Sem passes no recorte.")

                st.markdown('<div class="map-label">Conduções</div>', unsafe_allow_html=True)
                _show_map(draw_carry_map, df, player["name"], match_label, "Sem conduções no recorte.")

            st.markdown(
                f'<div class="map-label">Top {TOP_DELTAXT_N} ΔxT</div>',
                unsafe_allow_html=True,
            )
            _show_map(
                lambda d, n, m: draw_top_deltaxt_map(
                    d, n, m, delta_col="delta_xt_v31", model_label="v3.1"
                ),
                df, player["name"], match_label,
                "Sem ações com ΔxT positivo.",
            )


def render_stats_tab(player_data: dict[str, pd.DataFrame], match_selection: str) -> None:
    match_label = _match_scope_label(match_selection)
    st.markdown("### Estatísticas")
    st.caption(
        f"Recorte: **{match_label}** · xT heurístico **v3.1** · "
        "Finalizações, xG, assistências e xA não constam nos CSVs Wyscout exportados."
    )

    stat_cols = st.columns(3)
    for col, player in zip(stat_cols, PLAYERS):
        with col:
            st.markdown(f'<div class="player-header">{player["name"]}</div>', unsafe_allow_html=True)
            df = filter_by_match(player_data[player["code"]], match_selection)
            if df.empty:
                st.warning(f"Sem dados para {player['name']}.")
                continue

            stats = compute_player_stats(df)
            render_general_stats_card(stats, player["tone"])
            render_impact_card(stats, player["tone"])
            render_xt_efficiency_card(stats, player["tone"])


def render_xt_model_comparison(
    player_data: dict[str, pd.DataFrame], match_selection: str
) -> None:
    """Compare xT v3 original vs v3.1 (transições suaves)."""
    match_label = _match_scope_label(match_selection)

    compare_models = [
        {
            "key": "v3",
            "label": "Heurístico v3",
            "grid_fn": compute_heuristic_v3_xt_grid,
            "fine_fn": compute_heuristic_v3_fine_grid,
            "vmax": XT_V3_SURFACE_MAX,
            "delta_col": "delta_xt",
            "xt_end_col": "xt_end",
            "desc": "Original — zonas com blend 22 m e monotonicidade por linha.",
        },
        {
            "key": "v3.1",
            "label": "Heurístico v3.1",
            "grid_fn": compute_heuristic_v31_xt_grid,
            "fine_fn": compute_heuristic_v31_fine_grid,
            "vmax": XT_V3_SURFACE_MAX,
            "delta_col": "delta_xt_v31",
            "xt_end_col": "xt_end_v31",
            "desc": (
                "Blend amplo (48 m) + gaussiana só em X (σx=3.5) + rampa 5.0/7.8 pp por coluna. "
                "Penalização lateral (6%) apenas no campo ofensivo (x≥60)."
            ),
        },
    ]

    st.markdown("### Mapa xT por quadrante")
    st.caption(
        "Comparação do **v3 original** com o **v3.1** — transições mais suaves entre colunas "
        "e menor salto defensivo→ofensivo, preservando valorização central no ataque."
    )

    grids = {m["key"]: m["grid_fn"]() for m in compare_models}
    jump_rows = [
        {
            "Modelo": m["key"],
            "Máx salto col. adj. (%)": round(_max_adjacent_col_jump_pct(grids[m["key"]]), 1),
            "Salto col. 8→9 (%)": round(
                abs(grids[m["key"]][:, 8].mean() - grids[m["key"]][:, 7].mean()) * 100.0, 1
            ),
        }
        for m in compare_models
    ]
    st.dataframe(pd.DataFrame(jump_rows), use_container_width=True, hide_index=True)

    grid_cols = st.columns(2)
    for col, model in zip(grid_cols, compare_models):
        grid = grids[model["key"]]
        with col:
            st.markdown(f'<div class="map-label">{model["label"]}</div>', unsafe_allow_html=True)
            img, fig = draw_xt_grid_map(grid, model["label"], as_percent=True)
            plt.close(fig)
            st.image(img, use_container_width=True)
            st.caption(
                f"16×12 · Máx: {grid.max():.3f} · Média: {grid.mean():.3f} · "
                f"{model['desc']}"
            )

    with st.expander("Superfície contínua xT"):
        surf_cols = st.columns(2)
        for col, model in zip(surf_cols, compare_models):
            with col:
                fine = model["fine_fn"]()
                img, fig = draw_xt_threat_surface(
                    fine, f"Superfície {model['key']}", model["vmax"]
                )
                plt.close(fig)
                st.image(img, use_container_width=True)

    st.markdown("---")
    st.markdown("### Top 10 ΔxT — v3 / v3.1")

    summary_rows = []
    for player in PLAYERS:
        df = filter_by_match(player_data[player["code"]], match_selection)
        st.markdown(f'<div class="player-header">{player["name"]}</div>', unsafe_allow_html=True)

        if df.empty:
            st.warning(f"Sem dados para {player['name']}.")
            continue

        xt_actions = df[df["category"].isin(["passes", "ball-carries"]) & df["has_end"]]
        passes = df[df["category"] == "passes"]
        row = {"Jogador": player["name"]}
        for model in compare_models:
            key = model["key"]
            row[f"Σ ΔxT {key}"] = round(_safe_col_sum(xt_actions, model["delta_col"]), 3)
            row[f"Σ xT final {key}"] = round(_safe_col_sum(passes, model["xt_end_col"]), 3)
        summary_rows.append(row)

        cmp_cols = st.columns(2)
        for col, model in zip(cmp_cols, compare_models):
            with col:
                st.markdown(f'<div class="map-label">Top ΔxT · {model["key"]}</div>', unsafe_allow_html=True)
                _show_map(
                    lambda d, n, m, mc=model["delta_col"], ml=model["key"]: draw_top_deltaxt_map(
                        d, n, m, delta_col=mc, model_label=ml
                    ),
                    df, player["name"], match_label,
                    f"Sem ações com ΔxT positivo ({model['key']}).",
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
        Bruno Guimarães · Casemiro · Lucas Paquetá — xT Heurístico v3.1
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
    st.markdown("---")
    impact_plays_only = st.checkbox(
        "Apenas impact plays nos mapas",
        value=False,
        help="Mostra um mapa unificado só com passes e conduções classificados como impact (xT v3.1).",
    )
    st.caption("xT v3.1 · Progressivos Wyscout · Stats gerais")

tab_analysis, tab_stats, tab_compare = st.tabs(["Análise", "Stats", "Comparar xT v3 / v3.1"])

with tab_analysis:
    render_comparison(player_data, selected_match, impact_plays_only=impact_plays_only)

with tab_stats:
    render_stats_tab(player_data, selected_match)

with tab_compare:
    render_xt_model_comparison(player_data, selected_match)
