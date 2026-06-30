"""Standalone VAEP inference (no socceraction/pandera dependency)."""

from __future__ import annotations

import functools
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

SPADL_FIELD_LENGTH = 105.0
SPADL_FIELD_WIDTH = 68.0
SPADL_ACTIONTYPES = [
    "pass", "cross", "throw_in", "freekick_crossed", "freekick_short",
    "corner_crossed", "corner_short", "take_on", "foul", "tackle",
    "interception", "shot", "shot_penalty", "shot_freekick", "keeper_save",
    "keeper_claim", "keeper_punch", "keeper_pick_up", "clearance", "bad_touch",
    "non_action", "dribble", "goalkick",
]
SPADL_RESULTS = ["fail", "success", "offside", "owngoal", "yellow_card", "red_card"]
SPADL_BODYPARTS = ["foot", "head", "other", "head/other", "foot_left", "foot_right"]

GOAL_X = SPADL_FIELD_LENGTH
GOAL_Y = SPADL_FIELD_WIDTH / 2
SAMEPHASE_NB = 10

VAEP_XGB_PATH = Path(__file__).resolve().parent / "models" / "vaep_xgb.pkl"


def add_spadl_names(actions: pd.DataFrame) -> pd.DataFrame:
    out = actions.copy()
    out["type_name"] = out["type_id"].map(dict(enumerate(SPADL_ACTIONTYPES)))
    out["result_name"] = out["result_id"].map(dict(enumerate(SPADL_RESULTS)))
    out["bodypart_name"] = out["bodypart_id"].map(dict(enumerate(SPADL_BODYPARTS)))
    return out


def gamestates(actions: pd.DataFrame, nb_prev_actions: int = 3) -> list[pd.DataFrame]:
    """Previous-action windows per game/period (socceraction-compatible)."""
    states = [actions]
    first_row = actions.iloc[0]
    group_cols = ["game_id", "period_id"]
    value_cols = [c for c in actions.columns if c not in group_cols]

    for i in range(1, nb_prev_actions):
        prev_actions = actions.copy()
        grouped = actions.groupby(group_cols, sort=False)
        for col in value_cols:
            prev_actions[col] = grouped[col].shift(i)
        for col in group_cols:
            prev_actions[col] = actions[col].values
        prev_actions = prev_actions.fillna(first_row)
        prev_actions.index = actions.index.copy()
        states.append(prev_actions)
    return states


def play_left_to_right(gamestates_list: list[pd.DataFrame], home_team_id: int) -> list[pd.DataFrame]:
    a0 = gamestates_list[0]
    away_idx = a0.team_id != home_team_id
    for actions in gamestates_list:
        for col in ["start_x", "end_x"]:
            actions.loc[away_idx, col] = SPADL_FIELD_LENGTH - actions.loc[away_idx, col].values
        for col in ["start_y", "end_y"]:
            actions.loc[away_idx, col] = SPADL_FIELD_WIDTH - actions.loc[away_idx, col].values
    return gamestates_list


def _simple(actionfn: Callable[[pd.DataFrame], pd.DataFrame]) -> Callable[[list[pd.DataFrame]], pd.DataFrame]:
    def _wrapper(gamestates_list: list[pd.DataFrame]) -> pd.DataFrame:
        parts = []
        for i, actions in enumerate(gamestates_list):
            xi = actionfn(actions)
            xi.columns = [f"{c}_a{i}" for c in xi.columns]
            parts.append(xi)
        return pd.concat(parts, axis=1)

    return _wrapper


@_simple
def actiontype_onehot(actions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {f"actiontype_{name}": actions["type_id"] == type_id for type_id, name in enumerate(SPADL_ACTIONTYPES)},
        index=actions.index,
    )


@_simple
def result_onehot(actions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {f"result_{name}": actions["result_id"] == result_id for result_id, name in enumerate(SPADL_RESULTS)},
        index=actions.index,
    )


@_simple
def actiontype_result_onehot(actions: pd.DataFrame) -> pd.DataFrame:
    res_cols = {f"result_{name}": actions["result_id"] == rid for rid, name in enumerate(SPADL_RESULTS)}
    tys_cols = {f"actiontype_{name}": actions["type_id"] == tid for tid, name in enumerate(SPADL_ACTIONTYPES)}
    res = pd.DataFrame(res_cols, index=actions.index)
    tys = pd.DataFrame(tys_cols, index=actions.index)
    return pd.DataFrame(
        {f"{tyscol}_{rescol}": tys[tyscol] & res[rescol] for tyscol in tys.columns for rescol in res.columns},
        index=actions.index,
    )


@_simple
def bodypart_onehot(actions: pd.DataFrame) -> pd.DataFrame:
    foot_id = SPADL_BODYPARTS.index("foot")
    left_foot_id = SPADL_BODYPARTS.index("foot_left")
    right_foot_id = SPADL_BODYPARTS.index("foot_right")
    head_id = SPADL_BODYPARTS.index("head")
    other_id = SPADL_BODYPARTS.index("other")
    head_other_id = SPADL_BODYPARTS.index("head/other")
    return pd.DataFrame(
        {
            "bodypart_foot": actions["bodypart_id"].isin([foot_id, left_foot_id, right_foot_id]),
            "bodypart_head": actions["bodypart_id"] == head_id,
            "bodypart_other": actions["bodypart_id"] == other_id,
            "bodypart_head/other": actions["bodypart_id"].isin([head_id, other_id, head_other_id]),
        },
        index=actions.index,
    )


@_simple
def time(actions: pd.DataFrame) -> pd.DataFrame:
    match_time_at_period_start = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}
    timedf = actions[["period_id", "time_seconds"]].copy()
    timedf["time_seconds_overall"] = (
        timedf.period_id.map(match_time_at_period_start) * 60
    ) + timedf.time_seconds
    return timedf


@_simple
def startlocation(actions: pd.DataFrame) -> pd.DataFrame:
    return actions[["start_x", "start_y"]]


@_simple
def endlocation(actions: pd.DataFrame) -> pd.DataFrame:
    return actions[["end_x", "end_y"]]


@_simple
def startpolar(actions: pd.DataFrame) -> pd.DataFrame:
    polardf = pd.DataFrame(index=actions.index)
    dx = (GOAL_X - actions["start_x"]).abs().values
    dy = (GOAL_Y - actions["start_y"]).abs().values
    polardf["start_dist_to_goal"] = np.sqrt(dx**2 + dy**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        polardf["start_angle_to_goal"] = np.nan_to_num(np.arctan(dy / dx))
    return polardf


@_simple
def endpolar(actions: pd.DataFrame) -> pd.DataFrame:
    polardf = pd.DataFrame(index=actions.index)
    dx = (GOAL_X - actions["end_x"]).abs().values
    dy = (GOAL_Y - actions["end_y"]).abs().values
    polardf["end_dist_to_goal"] = np.sqrt(dx**2 + dy**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        polardf["end_angle_to_goal"] = np.nan_to_num(np.arctan(dy / dx))
    return polardf


@_simple
def movement(actions: pd.DataFrame) -> pd.DataFrame:
    mov = pd.DataFrame(index=actions.index)
    mov["dx"] = actions.end_x - actions.start_x
    mov["dy"] = actions.end_y - actions.start_y
    mov["movement"] = np.sqrt(mov.dx**2 + mov.dy**2)
    return mov


def team(gamestates_list: list[pd.DataFrame]) -> pd.DataFrame:
    a0 = gamestates_list[0]
    teamdf = pd.DataFrame(index=a0.index)
    for i, actions in enumerate(gamestates_list[1:], start=1):
        teamdf[f"team_{i}"] = actions.team_id == a0.team_id
    return teamdf


def time_delta(gamestates_list: list[pd.DataFrame]) -> pd.DataFrame:
    a0 = gamestates_list[0]
    dt = pd.DataFrame(index=a0.index)
    for i, actions in enumerate(gamestates_list[1:], start=1):
        dt[f"time_delta_{i}"] = a0.time_seconds - actions.time_seconds
    return dt


def space_delta(gamestates_list: list[pd.DataFrame]) -> pd.DataFrame:
    a0 = gamestates_list[0]
    spaced = pd.DataFrame(index=a0.index)
    for i, actions in enumerate(gamestates_list[1:], start=1):
        dx = actions.end_x - a0.start_x
        dy = actions.end_y - a0.start_y
        spaced[f"dx_a0{i}"] = dx
        spaced[f"dy_a0{i}"] = dy
        spaced[f"mov_a0{i}"] = np.sqrt(dx**2 + dy**2)
    return spaced


def goalscore(gamestates_list: list[pd.DataFrame]) -> pd.DataFrame:
    actions = gamestates_list[0]
    team_a = actions["team_id"].values[0]
    goals = actions["type_name"].str.contains("shot") & (
        actions["result_id"] == SPADL_RESULTS.index("success")
    )
    owngoals = actions["type_name"].str.contains("shot") & (
        actions["result_id"] == SPADL_RESULTS.index("owngoal")
    )
    team_is_a = actions["team_id"] == team_a
    team_is_b = ~team_is_a
    goals_team_a = (goals & team_is_a) | (owngoals & team_is_b)
    goals_team_b = (goals & team_is_b) | (owngoals & team_is_a)
    goalscore_team_a = goals_team_a.cumsum() - goals_team_a
    goalscore_team_b = goals_team_b.cumsum() - goals_team_b

    scoredf = pd.DataFrame(index=actions.index)
    scoredf["goalscore_team"] = (goalscore_team_a * team_is_a) + (goalscore_team_b * team_is_b)
    scoredf["goalscore_opponent"] = (goalscore_team_b * team_is_a) + (goalscore_team_a * team_is_b)
    scoredf["goalscore_diff"] = scoredf["goalscore_team"] - scoredf["goalscore_opponent"]
    return scoredf


VAEP_XFNS = [
    actiontype_onehot,
    result_onehot,
    actiontype_result_onehot,
    bodypart_onehot,
    time,
    startlocation,
    endlocation,
    startpolar,
    endpolar,
    movement,
    team,
    time_delta,
    space_delta,
    goalscore,
]


def _prev(x: pd.Series) -> pd.Series:
    prev_x = x.shift(1)
    prev_x.iloc[:1] = x.values[0]
    return prev_x


def offensive_value(actions: pd.DataFrame, scores: pd.Series, concedes: pd.Series) -> pd.Series:
    sameteam = _prev(actions.team_id) == actions.team_id
    prev_scores = (_prev(scores) * sameteam + _prev(concedes) * (~sameteam)).astype(float)

    toolong_idx = abs(actions.time_seconds - _prev(actions.time_seconds)) > SAMEPHASE_NB
    prev_scores.loc[toolong_idx] = 0.0

    prevgoal_idx = (_prev(actions.type_name).isin(["shot", "shot_freekick", "shot_penalty"])) & (
        _prev(actions.result_name) == "success"
    )
    prev_scores.loc[prevgoal_idx] = 0.0

    penalty_idx = actions.type_name == "shot_penalty"
    prev_scores.loc[penalty_idx] = 0.792453

    corner_idx = actions.type_name.isin(["corner_crossed", "corner_short"])
    prev_scores.loc[corner_idx] = 0.046500

    return scores - prev_scores


def defensive_value(actions: pd.DataFrame, scores: pd.Series, concedes: pd.Series) -> pd.Series:
    sameteam = _prev(actions.team_id) == actions.team_id
    prev_concedes = (_prev(concedes) * sameteam + _prev(scores) * (~sameteam)).astype(float)

    toolong_idx = abs(actions.time_seconds - _prev(actions.time_seconds)) > SAMEPHASE_NB
    prev_concedes.loc[toolong_idx] = 0.0

    prevgoal_idx = (_prev(actions.type_name).isin(["shot", "shot_freekick", "shot_penalty"])) & (
        _prev(actions.result_name) == "success"
    )
    prev_concedes.loc[prevgoal_idx] = 0.0

    return -(concedes - prev_concedes)


def vaep_value(actions: pd.DataFrame, p_scores: pd.Series, p_concedes: pd.Series) -> pd.DataFrame:
    offensive = offensive_value(actions, p_scores, p_concedes)
    defensive = defensive_value(actions, p_scores, p_concedes)
    return pd.DataFrame(
        {
            "offensive_value": offensive,
            "defensive_value": defensive,
            "vaep_value": offensive + defensive,
        },
        index=actions.index,
    )


@dataclass
class VaepRuntime:
    nb_prev_actions: int
    feature_columns: list[str]
    models: dict

    def compute_features(self, game: pd.Series, actions: pd.DataFrame) -> pd.DataFrame:
        named = add_spadl_names(actions)
        states = gamestates(named, self.nb_prev_actions)
        states = play_left_to_right(states, int(game.home_team_id))
        return pd.concat([fn(states) for fn in VAEP_XFNS], axis=1)

    def rate(self, game: pd.Series, actions: pd.DataFrame) -> pd.Series:
        core = actions.drop(columns=["_original_row_id"], errors="ignore")
        named = add_spadl_names(core)
        features = self.compute_features(game, core)
        missing = set(self.feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"VAEP features ausentes: {sorted(missing)[:5]}…")

        probs = pd.DataFrame(index=features.index)
        x = features[self.feature_columns]
        for col, model in self.models.items():
            probs[col] = model.predict_proba(x)[:, 1]

        return vaep_value(named, probs["scores"], probs["concedes"])["vaep_value"]


@functools.lru_cache(maxsize=1)
def load_vaep_runtime() -> VaepRuntime | None:
    if not VAEP_XGB_PATH.exists():
        return None
    with open(VAEP_XGB_PATH, "rb") as handle:
        bundle = pickle.load(handle)
    return VaepRuntime(
        nb_prev_actions=int(bundle["nb_prev_actions"]),
        feature_columns=list(bundle["feature_columns"]),
        models=bundle["models"],
    )
