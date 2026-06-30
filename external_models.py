"""External possession-value models: xT Markov (socceraction) and VAEP."""

from __future__ import annotations

import functools
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import socceraction.spadl.config as spadlconfig
import socceraction.xthreat as xthreat
from socceraction.spadl.utils import add_names
from socceraction.vaep.base import VAEP

MODEL_DIR = Path(__file__).resolve().parent / "models"
XT_MODEL_PATH = MODEL_DIR / "xt_markov_wsl_16x12.json"
VAEP_MODEL_PATH = MODEL_DIR / "vaep_wsl.pkl"

SPADL_FIELD_LENGTH = spadlconfig.field_length
SPADL_FIELD_WIDTH = spadlconfig.field_width
SB_FIELD_X = 120.0
SB_FIELD_Y = 80.0
BRAZIL_TEAM_ID = 1
OPP_TEAM_ID = 2

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


def statsbomb_to_spadl(x: float, y: float) -> tuple[float, float]:
    return (
        float(np.clip(x * SPADL_FIELD_LENGTH / SB_FIELD_X, 0.0, SPADL_FIELD_LENGTH)),
        float(np.clip(y * SPADL_FIELD_WIDTH / SB_FIELD_Y, 0.0, SPADL_FIELD_WIDTH)),
    )


def _game_id(match: str, source_file: str) -> int:
    return abs(hash(f"{match}|{source_file}")) % 1_000_000_000


def _result_id(row: pd.Series) -> int:
    if row["category"] == "ball-carries":
        return spadlconfig.results.index("success" if row["has_end"] else "fail")
    return spadlconfig.results.index("success" if bool(row["is_success"]) else "fail")


def _type_id(category: str, action_type: str) -> int | None:
    name = TYPE_MAP.get((category, action_type))
    if name is None:
        return None
    return spadlconfig.actiontypes.index(name)


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
                "bodypart_id": spadlconfig.bodyparts.index("foot"),
                "type_id": type_id,
                "result_id": _result_id(row),
            }
        )

    if not rows:
        return pd.Series(dtype=object), pd.DataFrame()

    meta = pd.DataFrame(rows)
    row_ids = meta["original_row_id"].tolist()
    spadl = add_names(meta.drop(columns=["original_row_id"]))
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
def load_xt_markov_model() -> xthreat.ExpectedThreat:
    if not XT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Grid xT Markov não encontrado em {XT_MODEL_PATH}. "
            "Execute o script de treino ou inclua o arquivo no diretório models/."
        )
    return xthreat.load_model(str(XT_MODEL_PATH))


@functools.lru_cache(maxsize=1)
def load_vaep_model() -> VAEP:
    if not VAEP_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo VAEP não encontrado em {VAEP_MODEL_PATH}. "
            "Execute o script de treino ou inclua o arquivo no diretório models/."
        )
    with open(VAEP_MODEL_PATH, "rb") as handle:
        return pickle.load(handle)


def rate_match_xt(spadl: pd.DataFrame, xt_model: xthreat.ExpectedThreat) -> pd.Series:
    if spadl.empty:
        return pd.Series(dtype=float)
    # scipy>=1.14 removed interp2d; socceraction still calls it when use_interpolation=True.
    ratings = xt_model.rate(spadl, use_interpolation=False)
    return pd.Series(ratings, index=spadl.index, dtype=float)


def rate_match_vaep(game: pd.Series, spadl: pd.DataFrame, vaep_model: VAEP) -> pd.Series:
    if spadl.empty:
        return pd.Series(dtype=float)
    rated = vaep_model.rate(game, spadl)
    return rated["vaep_value"]


def apply_external_models(df: pd.DataFrame) -> pd.DataFrame:
    """Add delta_xt_markov and vaep_value columns to a player dataframe."""
    out = df.copy()
    out["delta_xt_markov"] = np.nan
    out["vaep_value"] = np.nan

    if df.empty:
        return out

    xt_model = load_xt_markov_model()
    vaep_model = load_vaep_model()

    if "match" not in out.columns:
        groups = [("all", out)]
    else:
        groups = list(out.groupby("match", sort=False))

    for match_name, match_df in groups:
        game, spadl = match_df_to_spadl(match_df)
        if spadl.empty:
            continue

        xt_vals = rate_match_xt(spadl, xt_model)
        vaep_vals = rate_match_vaep(game, spadl, vaep_model)
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
            out.at[idx, "vaep_value"] = float(vaep_vals.iloc[i])

    return out


def markov_grid_for_display(xt_model: xthreat.ExpectedThreat) -> np.ndarray:
    """Return 12×16 grid aligned with app NX_XT/NY_XT orientation."""
    return xt_model.xT.copy()
