from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mplsoccer import Pitch
from PIL import Image

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Player Analysis — Action Map")

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
FIG_W, FIG_H = 6.2, 4.1
FIG_DPI = 160
PASS_START_MARKER_SIZE = 10
CARRY_START_MARKER_SIZE = 10
MAP_REF_WIDTH = 6.2
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

CARD_TITLE_TEXT = "14px"
CARD_LABEL_TEXT = "16px"
CARD_INNER_BORDER = "rgba(107,114,128,0.45)"
PLAYER_TONE = "#5b9bd5"

COLOR_SUCCESS = "#c8c8c8"
COLOR_PROGRESSIVE = "#2F80ED"
COLOR_HIGHLY_PROGRESSIVE = "#1B44A8"
COLOR_FAIL = "#E07070"
ALPHA_SUCCESS = 0.07
COLOR_CARRY = "#a855f7"
COLOR_CARRY_BASE_ALPHA = 0.50

ACTION_COLORS = {
    "passes": "#5b9bd5",
    "dribbles": "#22c55e",
    "ball-carries": "#a855f7",
    "defensive": "#ef4444",
}

CATEGORY_LABELS = {
    "passes": "Passes",
    "dribbles": "Dribles",
    "ball-carries": "Conduções",
    "defensive": "Ações defensivas",
}

ACTION_TYPE_LABELS = {
    "pass": "Passe",
    "dribble": "Drible",
    "ball-carry": "Condução",
    "tackle": "Desarme",
    "interception": "Interceptação",
    "clearance": "Corte",
    "ball-recovery": "Recuperação",
}


# ── COORDINATE HELPERS ───────────────────────────────────────
def wyscout_to_statsbomb(x: float, y: float) -> tuple[float, float]:
    """Wyscout 0–100 → StatsBomb 120×80 (ataque da esquerda para a direita)."""
    return x * FIELD_X / WYSCOUT_PITCH_SIZE, y * FIELD_Y / WYSCOUT_PITCH_SIZE


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

    for col in ("xt_start", "xt_end", "delta_xt"):
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


# ── DATA LOADING ─────────────────────────────────────────────
def discover_csv_files(base_dir: Path | None = None) -> list[Path]:
    root = base_dir or Path(__file__).resolve().parent
    return sorted(root.glob("*.csv"))


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


def load_all_players(base_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    players: dict[str, pd.DataFrame] = {}
    for path in discover_csv_files(base_dir):
        try:
            players[path.stem] = load_player_csv(path)
        except Exception as exc:
            st.warning(f"Não foi possível carregar `{path.name}`: {exc}")
    return players


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


def render_player_cards(stats: dict, tone: str) -> None:
    wyscout = stats["progressive_wyscout"]
    impact = stats["impact_pass"]
    high_impact = stats["high_impact_pass"]
    carry_impact = stats["impact_carry"]
    carry_high = stats["high_impact_carry"]
    stats_section_card(
        "Overview",
        tone,
        [
            ("Total Passes", f"{stats['total_passes']:.0f}"),
            ("% Accuracy", f"{stats['accuracy_pct']:.1f}%"),
        ],
    )
    stats_section_card(
        "Passes",
        tone,
        [
            ("Passes Progressivos", f"{wyscout['successful']:.0f}"),
            ("% Acurácia Progressiva", f"{wyscout['accuracy_pct']:.1f}%"),
            ("Impact Passes", f"{impact['successful']:.0f}"),
            ("% Acurácia Impact Passes", f"{impact['accuracy_pct']:.1f}%"),
            ("High Impact Passes", f"{high_impact['successful']:.0f}"),
            ("% Acurácia High Impact", f"{high_impact['accuracy_pct']:.1f}%"),
        ],
    )
    stats_section_card(
        "Impact",
        tone,
        [
            ("Pass Impact (xT v3)", f"{stats['sum_dxt_passes']:.2f}"),
            ("Carry Impact (xT v3)", f"{stats['sum_dxt_carries']:.2f}"),
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
    ax.axvline(x=OPT_ATTACKING_TWO_THIRDS_X, color="#ffffff", lw=0.9, alpha=0.22, linestyle="--")
    ax.axvline(x=FINAL_THIRD_LINE_X, color="#ffffff", lw=1.2, alpha=0.40, linestyle="--")
    ax.axvline(x=HALF_LINE_X, color="#ffffff", lw=0.7, alpha=0.12, linestyle="--")
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


def draw_pass_map(df: pd.DataFrame):
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
            color, alpha = COLOR_FAIL, 0.72
        elif is_high_impact:
            color, alpha = COLOR_HIGHLY_PROGRESSIVE, 0.95
        elif is_prog:
            color, alpha = COLOR_PROGRESSIVE, 0.88
        else:
            color, alpha = COLOR_SUCCESS, ALPHA_SUCCESS

        pitch.arrows(
            row["x_start"],
            row["y_start"],
            row["x_end"],
            row["y_end"],
            color=color,
            width=1.3 * scale,
            headwidth=2.0 * scale,
            headlength=2.0 * scale,
            ax=ax,
            zorder=3,
            alpha=alpha,
        )
        pitch.scatter(
            row["x_start"],
            row["y_start"],
            s=PASS_START_MARKER_SIZE,
            marker="o",
            color=color,
            edgecolors="white",
            linewidths=0.4,
            ax=ax,
            zorder=6,
            alpha=alpha,
        )

    legend_handles = [
        Line2D([0], [0], color=COLOR_SUCCESS, lw=2.0 * scale, label="Completado", alpha=0.65),
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=2.0 * scale, label="Progressivo (Wyscout)", alpha=0.90),
        Line2D([0], [0], color=COLOR_HIGHLY_PROGRESSIVE, lw=2.0 * scale, label="High Impact (xT v3)", alpha=0.95),
        Line2D([0], [0], color=COLOR_FAIL, lw=2.0 * scale, label="Incompleto", alpha=0.90),
    ]
    _add_map_legend(ax, legend_handles)
    _attack_arrow(fig)
    return _save_fig(fig), fig


def draw_carry_map(df: pd.DataFrame):
    carries = df[df["category"] == "ball-carries"].copy()
    fig, ax, pitch = _base_pitch()
    scale = _map_scale()

    for _, row in carries.iterrows():
        if not row["has_end"]:
            continue
        is_high_impact = bool(row.get("high_impact_carry", False))
        is_impact = bool(row.get("impact_carry", False))
        if is_high_impact:
            color, alpha = COLOR_HIGHLY_PROGRESSIVE, 0.95
        elif is_impact:
            color, alpha = COLOR_PROGRESSIVE, 0.88
        else:
            color, alpha = COLOR_CARRY, COLOR_CARRY_BASE_ALPHA

        pitch.arrows(
            row["x_start"],
            row["y_start"],
            row["x_end"],
            row["y_end"],
            color=color,
            width=1.3 * scale,
            headwidth=2.0 * scale,
            headlength=2.0 * scale,
            ax=ax,
            zorder=3,
            alpha=alpha,
        )
        pitch.scatter(
            row["x_start"],
            row["y_start"],
            s=CARRY_START_MARKER_SIZE,
            marker="o",
            color=color,
            edgecolors="white",
            linewidths=0.4,
            ax=ax,
            zorder=6,
            alpha=alpha,
        )

    legend_handles = [
        Line2D([0], [0], color=COLOR_CARRY, lw=2.0 * scale, label="Condução", alpha=0.65),
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=2.0 * scale, label="Impact (xT v3)", alpha=0.90),
        Line2D([0], [0], color=COLOR_HIGHLY_PROGRESSIVE, lw=2.0 * scale, label="High Impact (xT v3)", alpha=0.95),
    ]
    _add_map_legend(ax, legend_handles)
    _attack_arrow(fig)
    return _save_fig(fig), fig


def render_maps(df: pd.DataFrame):
    col_pass, col_carry = st.columns(2)

    with col_pass:
        st.markdown('<div class="map-label">Mapa de Passes</div>', unsafe_allow_html=True)
        if (df["category"] == "passes").any():
            img_pass, fig_pass = draw_pass_map(df)
            plt.close(fig_pass)
            st.image(img_pass, use_container_width=True)
        else:
            st.info("Sem passes no recorte selecionado.")

    with col_carry:
        st.markdown('<div class="map-label">Mapa de Conduções</div>', unsafe_allow_html=True)
        if (df["category"] == "ball-carries").any():
            img_carry, fig_carry = draw_carry_map(df)
            plt.close(fig_carry)
            st.image(img_carry, use_container_width=True)
        else:
            st.info("Sem conduções no recorte selecionado.")


# ── MAIN ─────────────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div style="text-align:center;">
      <h2 style="margin:0;color:#eef1f7;">Player Analysis</h2>
      <p style="color:#94a3b8;font-size:0.9rem;">Mapa de ações por jogador</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.caption("Impacto: xT Heurístico v3 · Progressivos: Wyscout")

players = load_all_players()

if not players:
    st.error(
        "Nenhum arquivo `.csv` encontrado no diretório do app. "
        "Adicione arquivos como `enzo.csv` com as colunas esperadas."
    )
    st.stop()

player_keys = list(players.keys())
selected_player = st.sidebar.selectbox(
    "Jogador",
    options=player_keys,
    format_func=lambda k: players[k]["player"].iloc[0],
)

df = players[selected_player]
player_name = df["player"].iloc[0]
source_file = df["source_file"].iloc[0]

available_categories = sorted(df["category"].unique())
selected_categories = st.sidebar.multiselect(
    "Categorias",
    options=available_categories,
    default=available_categories,
    format_func=lambda c: CATEGORY_LABELS.get(c, c.title()),
)

outcome_filter = st.sidebar.radio(
    "Resultado",
    options=["all", "success", "fail"],
    format_func=lambda v: {"all": "Todos", "success": "Sucesso", "fail": "Falha"}[v],
    horizontal=True,
)

filtered = df[df["category"].isin(selected_categories)].copy()
if outcome_filter == "success":
    filtered = filtered[filtered["is_success"] | (filtered["category"] == "ball-carries")]
elif outcome_filter == "fail":
    filtered = filtered[~filtered["is_success"] & (filtered["category"] != "ball-carries")]

stats = compute_player_stats(filtered)

st.markdown(f"## {player_name}")
st.caption(
    f"Fonte: `{source_file}` · {stats['total_actions']} ações · "
    f"xT Heurístico v3 em passes e conduções"
)

tab_maps, tab_stats, tab_data = st.tabs(["Mapas", "Estatísticas", "Dados"])

with tab_maps:
    if filtered.empty:
        st.info("Nenhuma ação para os filtros selecionados.")
    else:
        render_maps(filtered)

with tab_stats:
    render_player_cards(stats, PLAYER_TONE)

    st.markdown("### Por tipo de ação")
    type_rows = [
        {
            "Tipo": ACTION_TYPE_LABELS.get(k, k),
            "Categoria": CATEGORY_LABELS.get(
                filtered.loc[filtered["action_type"] == k, "category"].iloc[0]
                if (filtered["action_type"] == k).any()
                else "",
                "",
            ),
            "Quantidade": v,
        }
        for k, v in sorted(stats["by_action_type"].items(), key=lambda item: -item[1])
    ]
    st.dataframe(pd.DataFrame(type_rows), use_container_width=True, hide_index=True)

with tab_data:
    display_cols = [
        "row_id",
        "category",
        "action_type",
        "is_success",
        "is_key_pass",
        "is_long_ball",
        "progressive",
        "impact_pass",
        "high_impact_pass",
        "impact_carry",
        "high_impact_carry",
        "delta_xt",
        "x_start",
        "y_start",
        "x_end",
        "y_end",
    ]
    st.dataframe(
        filtered[display_cols].rename(
            columns={
                "row_id": "#",
                "category": "Categoria",
                "action_type": "Ação",
                "is_success": "Sucesso",
                "is_key_pass": "Key Pass",
                "is_long_ball": "Bola longa",
                "progressive": "Progressivo",
                "impact_pass": "Impact Pass",
                "high_impact_pass": "High Impact",
                "impact_carry": "Impact Carry",
                "high_impact_carry": "High Impact Carry",
                "delta_xt": "ΔxT v3",
                "x_start": "X início",
                "y_start": "Y início",
                "x_end": "X fim",
                "y_end": "Y fim",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
