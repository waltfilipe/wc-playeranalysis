"""External possession-value models: xT Markov and VAEP.

Markov xT and VAEP inference run without socceraction so the app works on
Streamlit Cloud (Python 3.14). Training still uses socceraction — see
scripts/train_external_models.py and scripts/export_vaep_bundle.py.
"""

from __future__ import annotations

import functools
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "models"
XT_MODEL_PATH = MODEL_DIR / "xt_markov_wsl_16x12.json"
VAEP_MODEL_PATH = MODEL_DIR / "vaep_wsl.pkl"
VAEP_XGB_PATH = MODEL_DIR / "vaep_xgb.pkl"

SPADL_FIELD_LENGTH = 105.0
SPADL_FIELD_WIDTH = 68.0
SB_FIELD_X = 120.0
SB_FIELD_Y = 80.0
BRAZIL_TEAM_ID = 1
OPP_TEAM_ID = 2

SPADL_ACTIONTYPES = [
    "pass", "cross", "throw_in", "freekick_crossed", "freekick_short",
    "corner_crossed", "corner_short", "take_on", "foul", "tackle",
    "interception", "shot", "shot_penalty", "shot_freekick", "keeper_save",
    "keeper_claim", "keeper_punch", "keeper_pick_up", "clearance", "bad_touch",
    "non_action", "dribble", "goalkick",
]
SPADL_RESULTS = ["fail", "success", "offside", "owngoal", "yellow_card", "red_card"]
SPADL_BODYPARTS = ["foot", "head", "other", "head/other", "foot_left", "foot_right"]

MOVE_TYPE_IDS = {
    SPADL_ACTIONTYPES.index("pass"),
    SPADL_ACTIONTYPES.index("cross"),
    SPADL_ACTIONTYPES.index("dribble"),
}
RESULT_SUCCESS_ID = SPADL_RESULTS.index("success")
BODYPART_FOOT_ID = SPADL_BODYPARTS.index("foot")

TYPE_MAP: dict[tuple[str, str], str] = {
    ("passes", "pass"): "pass",
    ("passes", "cross"): "cross",
    ("passes", "throw-in"): "throw_in",
    ("ball-carries", "ball-carry"): "dribble",
    ("dribbles", "dribble"): "dribble",
    ("defensive", "tackle"): "tackle",
    ("defensive", "interception"): "interception",
    ("defensive", "clearance"): "clearance",
    ("defensive", "block"): "clearance",
    ("defensive", "ball-recovery"): "pass",
}

_vaep_error: str | None = None


@dataclass(frozen=True)
class MarkovXtGrid:
    """Pre-trained xT surface (socceraction-compatible JSON export)."""

    xT: np.ndarray

    @property
    def l(self) -> int:
        return int(self.xT.shape[1])

    @property
    def w(self) -> int:
        return int(self.xT.shape[0])

    def rate(self, spadl: pd.DataFrame) -> np.ndarray:
        """Rate SPADL actions with ΔxT (no interpolation — same as socceraction default)."""
        ratings = np.full(len(spadl), np.nan, dtype=float)
        if spadl.empty:
            return ratings

        move_mask = spadl["type_id"].isin(MOVE_TYPE_IDS) & (spadl["result_id"] == RESULT_SUCCESS_ID)
        move_actions = spadl.loc[move_mask]
        if move_actions.empty:
            return ratings

        startxc, startyc = _cell_indexes(move_actions["start_x"], move_actions["start_y"], self.l, self.w)
        endxc, endyc = _cell_indexes(move_actions["end_x"], move_actions["end_y"], self.l, self.w)

        xT_start = self.xT[self.w - 1 - startyc, startxc]
        xT_end = self.xT[self.w - 1 - endyc, endxc]
        ratings[move_actions.index.to_numpy()] = xT_end - xT_start
        return ratings


def _cell_indexes(x: pd.Series, y: pd.Series, l: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    xi = (x / SPADL_FIELD_LENGTH * l).astype(np.int64).clip(0, l - 1).to_numpy()
    yj = (y / SPADL_FIELD_WIDTH * w).astype(np.int64).clip(0, w - 1).to_numpy()
    return xi, yj


def statsbomb_to_spadl(x: float, y: float) -> tuple[float, float]:
    return (
        float(np.clip(x * SPADL_FIELD_LENGTH / SB_FIELD_X, 0.0, SPADL_FIELD_LENGTH)),
        float(np.clip(y * SPADL_FIELD_WIDTH / SB_FIELD_Y, 0.0, SPADL_FIELD_WIDTH)),
    )


def _game_id(match: str, source_file: str) -> int:
    return abs(hash(f"{match}|{source_file}")) % 1_000_000_000


def _result_id(row: pd.Series) -> int:
    if row["category"] == "ball-carries":
        return RESULT_SUCCESS_ID if row["has_end"] else SPADL_RESULTS.index("fail")
    return RESULT_SUCCESS_ID if bool(row["is_success"]) else SPADL_RESULTS.index("fail")


def _type_id(category: str, action_type: str) -> int | None:
    name = TYPE_MAP.get((category, action_type))
    if name is None:
        return None
    return SPADL_ACTIONTYPES.index(name)


def match_df_to_spadl(match_df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Convert one match slice to SPADL actions + synthetic game metadata."""
    if match_df.empty:
        return pd.Series(dtype=object), pd.DataFrame()

    ordered = match_df.sort_values("row_id").reset_index(drop=True)
    is_home = bool(ordered["is_home"].iloc[0]) if "is_home" in ordered.columns else True
    home_team_id = BRAZIL_TEAM_ID if is_home else OPP_TEAM_ID
    match_name = str(ordered["match"].iloc[0]) if "match" in ordered.columns else "match"
    source_file = str(ordered["source_file"].iloc[0]) if "source_file" in ordered.columns else "file"
    game_id = _game_id(match_name, source_file)

    rows: list[dict] = []
    for seq, (_, row) in enumerate(ordered.iterrows()):
        type_id = _type_id(str(row["category"]), str(row["action_type"]))
        if type_id is None:
            continue

        sx, sy = statsbomb_to_spadl(float(row["x_start"]), float(row["y_start"]))
        if row["has_end"]:
            ex, ey = statsbomb_to_spadl(float(row["x_end"]), float(row["y_end"]))
        else:
            ex, ey = sx, sy

        rows.append(
            {
                "original_row_id": int(row["row_id"]),
                "game_id": game_id,
                "original_event_id": int(row["row_id"]),
                "action_id": seq,
                "period_id": 1,
                "time_seconds": float(seq * 3.0),
                "team_id": BRAZIL_TEAM_ID,
                "player_id": 1,
                "start_x": sx,
                "start_y": sy,
                "end_x": ex,
                "end_y": ey,
                "bodypart_id": BODYPART_FOOT_ID,
                "type_id": type_id,
                "result_id": _result_id(row),
            }
        )

    if not rows:
        return pd.Series(dtype=object), pd.DataFrame()

    meta = pd.DataFrame(rows)
    row_ids = meta["original_row_id"].tolist()
    spadl = meta.drop(columns=["original_row_id"]).reset_index(drop=True)
    spadl["_original_row_id"] = row_ids
    game = pd.Series(
        {
            "game_id": game_id,
            "home_team_id": home_team_id,
            "away_team_id": OPP_TEAM_ID if home_team_id == BRAZIL_TEAM_ID else BRAZIL_TEAM_ID,
        }
    )
    return game, spadl


@functools.lru_cache(maxsize=1)
def load_xt_markov_model() -> MarkovXtGrid:
    if not XT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Grid xT Markov não encontrado em {XT_MODEL_PATH}. "
            "Execute o script de treino ou inclua o arquivo no diretório models/."
        )
    with open(XT_MODEL_PATH, encoding="utf-8") as handle:
        grid = np.array(json.load(handle), dtype=float)
    return MarkovXtGrid(xT=grid)


@functools.lru_cache(maxsize=1)
def _try_load_vaep_runtime():
    global _vaep_error
    try:
        from vaep_standalone import load_vaep_runtime

        runtime = load_vaep_runtime()
        if runtime is not None:
            return runtime
    except Exception as exc:  # noqa: BLE001
        _vaep_error = f"VAEP standalone falhou: {exc}"
        return None

    if not VAEP_XGB_PATH.exists() and not VAEP_MODEL_PATH.exists():
        _vaep_error = f"Modelo VAEP não encontrado em {MODEL_DIR}."
        return None

    # Legacy fallback: socceraction pickle (Python 3.11/3.12 only)
    if not VAEP_MODEL_PATH.exists():
        _vaep_error = f"Bundle VAEP ausente ({VAEP_XGB_PATH.name})."
        return None
    try:
        import socceraction.vaep.base  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        _vaep_error = (
            f"VAEP indisponível ({exc}). "
            f"Inclua {VAEP_XGB_PATH.name} no repositório."
        )
        return None
    try:
        with open(VAEP_MODEL_PATH, "rb") as handle:
            return pickle.load(handle)
    except Exception as exc:  # noqa: BLE001
        _vaep_error = str(exc)
        return None


@functools.lru_cache(maxsize=1)
def load_vaep_model():
    model = _try_load_vaep_runtime()
    if model is None:
        msg = _vaep_error or "VAEP indisponível neste ambiente."
        raise RuntimeError(msg)
    return model


def vaep_available() -> bool:
    return _try_load_vaep_runtime() is not None


def vaep_status_message() -> str | None:
    if vaep_available():
        return None
    return _vaep_error or "VAEP indisponível."


def rate_match_xt(spadl: pd.DataFrame, xt_model: MarkovXtGrid) -> pd.Series:
    if spadl.empty:
        return pd.Series(dtype=float)
    ratings = xt_model.rate(spadl)
    return pd.Series(ratings, index=spadl.index, dtype=float)


def rate_match_vaep(game: pd.Series, spadl: pd.DataFrame, vaep_model) -> pd.Series:
    if spadl.empty:
        return pd.Series(dtype=float)
    rated = vaep_model.rate(game, spadl)
    if isinstance(rated, pd.Series):
        return rated
    return rated["vaep_value"]


def apply_external_models(df: pd.DataFrame) -> pd.DataFrame:
    """Add delta_xt_markov and vaep_value columns to a player dataframe."""
    out = df.copy()
    out["delta_xt_markov"] = np.nan
    out["vaep_value"] = np.nan

    if df.empty:
        return out

    xt_model = load_xt_markov_model()
    vaep_model = _try_load_vaep_runtime()

    if "match" not in out.columns:
        groups = [("all", out)]
    else:
        groups = list(out.groupby("match", sort=False))

    for match_name, match_df in groups:
        game, spadl = match_df_to_spadl(match_df)
        if spadl.empty:
            continue

        xt_vals = rate_match_xt(spadl, xt_model)
        vaep_vals = (
            rate_match_vaep(game, spadl, vaep_model)
            if vaep_model is not None
            else pd.Series(np.nan, index=spadl.index, dtype=float)
        )
        row_ids = spadl["_original_row_id"].astype(int).tolist()

        for i, rid in enumerate(row_ids):
            if "match" in out.columns and match_name != "all":
                mask = (out["row_id"] == rid) & (out["match"] == match_name)
            else:
                mask = out["row_id"] == rid
            if not mask.any():
                continue
            idx = out.index[mask][0]
            out.at[idx, "delta_xt_markov"] = float(xt_vals.iloc[i])
            if vaep_model is not None:
                out.at[idx, "vaep_value"] = float(vaep_vals.iloc[i])

    return out


def markov_grid_for_display(xt_model: MarkovXtGrid) -> np.ndarray:
    """Return 12×16 grid aligned with app NX_XT/NY_XT orientation."""
    return xt_model.xT.copy()
