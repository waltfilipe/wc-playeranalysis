"""Train xT Markov variants, Bayesian smoothing, and hold-out validation."""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from socceraction.data.statsbomb import StatsBombLoader
from socceraction.spadl import play_left_to_right
from socceraction.spadl.statsbomb import convert_to_actions
from socceraction.spadl.utils import add_names
from socceraction.xthreat import ExpectedThreat, get_successful_move_actions

from external_models import (
    GRID_L,
    GRID_W,
    MARKOV_MODEL_SPECS,
    MODEL_DIR,
    SPADL_FIELD_LENGTH,
    SPADL_FIELD_WIDTH,
    VALIDATION_REPORT_PATH,
    MarkovXtGrid,
    _align_markov_grid,
    save_markov_model,
)

COMPETITION_POOLS: dict[str, list[tuple[int, int, str]]] = {
    "wsl": [
        (37, 4, "FA WSL 2018/19"),
    ],
    "womens": [
        (37, 4, "FA WSL 2018/19"),
        (37, 42, "FA WSL 2019/20"),
        (37, 90, "FA WSL 2020/21"),
        (72, 30, "WWC 2019"),
    ],
    "top5": [
        (11, 4, "La Liga 2018/19"),
        (11, 42, "La Liga 2019/20"),
        (11, 90, "La Liga 2020/21"),
        (2, 27, "Premier League 2015/16"),
        (12, 27, "Serie A 2015/16"),
        (16, 4, "Champions League 2018/19"),
    ],
}

SHOT_TYPE_IDS = {11, 12, 13}
HOLDOUT_FRACTION = 0.30
BAYES_K = 35.0
LOOKAHEAD_ACTIONS = 8
RANDOM_SEED = 42


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_competition_actions(competition_id: int, season_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = StatsBombLoader()
    games = loader.games(competition_id=competition_id, season_id=season_id)
    action_frames: list[pd.DataFrame] = []
    for _, game in games.iterrows():
        events = loader.events(int(game["game_id"]))
        actions = add_names(convert_to_actions(events, int(game["home_team_id"])))
        actions = play_left_to_right(actions, int(game["home_team_id"]))
        action_frames.append(actions)
    if not action_frames:
        return games, games.iloc[0:0].copy()
    return games, pd.concat(action_frames, ignore_index=True)


def orient_actions_left_to_right(games: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Actions are already LTR-oriented during load; keep API for clarity."""
    return actions.copy()


def load_pool_actions(pool_name: str) -> tuple[pd.DataFrame, list[str]]:
    specs = COMPETITION_POOLS[pool_name]
    action_frames: list[pd.DataFrame] = []
    labels: list[str] = []

    for comp_id, season_id, label in specs:
        print(f"  · {label} ({comp_id}/{season_id})…", flush=True)
        try:
            games, actions = load_competition_actions(comp_id, season_id)
            ltr = orient_actions_left_to_right(games, actions)
            if ltr.empty:
                print("    (vazio, ignorado)")
                continue
            action_frames.append(ltr)
            labels.append(label)
            print(f"    {ltr['game_id'].nunique()} jogos · {len(ltr):,} ações")
        except Exception as exc:  # noqa: BLE001
            print(f"    ERRO: {exc}")

    if not action_frames:
        raise RuntimeError(f"Nenhuma ação carregada para pool '{pool_name}'.")

    combined = pd.concat(action_frames, ignore_index=True)
    return combined, labels


def split_actions_by_game(actions: pd.DataFrame, holdout_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    game_ids = actions["game_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(game_ids)
    n_test = max(1, int(len(game_ids) * holdout_fraction))
    test_ids = set(game_ids[:n_test])
    train = actions[~actions["game_id"].isin(test_ids)].copy()
    test = actions[actions["game_id"].isin(test_ids)].copy()
    return train, test


def train_markov(actions: pd.DataFrame) -> ExpectedThreat:
    xt = ExpectedThreat(l=GRID_L, w=GRID_W)
    xt.fit(actions)
    return xt


def aligned_grid_from_model(xt: ExpectedThreat) -> np.ndarray:
    return _align_markov_grid(np.array(xt.xT, dtype=float))


def saved_grid_from_model(xt: ExpectedThreat) -> np.ndarray:
    """StatsBomb LTR grid as written to JSON (attack toward +x)."""
    return aligned_grid_from_model(xt)[::-1, ::-1]


def cell_counts(actions: pd.DataFrame) -> np.ndarray:
    moves = get_successful_move_actions(actions.reset_index(drop=True))
    counts = np.zeros((GRID_W, GRID_L), dtype=float)
    if moves.empty:
        return counts
    xi = (moves["start_x"] / SPADL_FIELD_LENGTH * GRID_L).astype(int).clip(0, GRID_L - 1)
    yj = (moves["start_y"] / SPADL_FIELD_WIDTH * GRID_W).astype(int).clip(0, GRID_W - 1)
    for x_idx, y_idx in zip(xi, yj):
        counts[int(y_idx), int(x_idx)] += 1.0
    return counts


def bayesian_smooth_grid(
    observed: np.ndarray,
    prior: np.ndarray,
    counts: np.ndarray,
    k: float = BAYES_K,
) -> np.ndarray:
    return (counts * observed + k * prior) / (counts + k)


def label_shot_within_lookahead(actions: pd.DataFrame, lookahead: int = LOOKAHEAD_ACTIONS) -> pd.Series:
    labels = pd.Series(False, index=actions.index, dtype=bool)
    for _game_id, game_actions in actions.groupby("game_id"):
        game_actions = game_actions.sort_values(["period_id", "time_seconds", "action_id"])
        idxs = game_actions.index.to_list()
        teams = game_actions["team_id"].to_numpy()
        types = game_actions["type_id"].to_numpy()
        for pos, idx in enumerate(idxs):
            team = teams[pos]
            dangerous = False
            for nxt in range(pos + 1, min(pos + 1 + lookahead, len(idxs))):
                if teams[nxt] != team:
                    break
                if int(types[nxt]) in SHOT_TYPE_IDS:
                    dangerous = True
                    break
            labels.at[idx] = dangerous
    return labels


def evaluate_markov_on_holdout(
    model_key: str,
    grid: np.ndarray,
    test_actions: pd.DataFrame,
) -> dict[str, float | int]:
    moves = get_successful_move_actions(test_actions.reset_index(drop=True))
    if moves.empty:
        return {
            "n_moves": 0,
            "auc_shot_8": float("nan"),
            "corr_shot_8": float("nan"),
            "mean_delta_dangerous": float("nan"),
            "mean_delta_safe": float("nan"),
        }

    xt_model = MarkovXtGrid(xT=grid.copy(), model_key=model_key)
    deltas = xt_model.rate(moves.reset_index(drop=True))
    dangerous = label_shot_within_lookahead(test_actions).reindex(moves.index).fillna(False).to_numpy()

    valid = np.isfinite(deltas)
    deltas_v = deltas[valid]
    danger_v = dangerous[valid].astype(int)

    auc = float("nan")
    if len(np.unique(danger_v)) > 1:
        auc = float(roc_auc_score(danger_v, deltas_v))

    corr = float("nan")
    if len(deltas_v) > 2 and np.std(deltas_v) > 0 and np.std(danger_v) > 0:
        corr = float(np.corrcoef(deltas_v, danger_v)[0, 1])

    if danger_v.any():
        mean_danger = float(deltas_v[danger_v.astype(bool)].mean())
    else:
        mean_danger = float("nan")
    if (~danger_v.astype(bool)).any():
        mean_safe = float(deltas_v[~danger_v.astype(bool)].mean())
    else:
        mean_safe = float("nan")

    return {
        "n_moves": int(len(deltas_v)),
        "auc_shot_8": auc,
        "corr_shot_8": corr,
        "mean_delta_dangerous": mean_danger,
        "mean_delta_safe": mean_safe,
    }


def pick_validation_winner(metrics: dict[str, dict[str, float | int]]) -> tuple[str, str]:
    ranked: list[tuple[str, float]] = []
    for key, vals in metrics.items():
        auc = vals.get("auc_shot_8")
        if auc is not None and np.isfinite(auc):
            ranked.append((key, float(auc)))
    if not ranked:
        return "wsl", "fallback_wsl"
    ranked.sort(key=lambda item: item[1], reverse=True)
    winner, best_auc = ranked[0]
    if len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) < 0.005:
        return winner, f"highest_auc_tiebreak_{best_auc:.4f}"
    return winner, f"highest_auc_{best_auc:.4f}"


def save_validation_report(
    metrics: dict[str, dict[str, float | int]],
    winner: str,
    winner_reason: str,
    *,
    train_sizes: dict[str, int],
    test_moves: int,
) -> None:
    report = {
        "generated_at": _utc_now(),
        "holdout_fraction": HOLDOUT_FRACTION,
        "lookahead_actions": LOOKAHEAD_ACTIONS,
        "metrics": metrics,
        "winner": winner,
        "winner_reason": winner_reason,
        "v33_bonus_source": winner,
        "train_sizes": train_sizes,
        "test_moves_evaluated": test_moves,
    }
    VALIDATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved {VALIDATION_REPORT_PATH} · winner={winner} ({winner_reason})")


def train_vaep_models(actions: pd.DataFrame, games: pd.DataFrame) -> None:
    from socceraction.vaep import features as vaep_features
    from socceraction.vaep.base import VAEP

    print("Training VAEP…")
    xfns = [
        vaep_features.actiontype,
        vaep_features.actiontype_onehot,
        vaep_features.bodypart_onehot,
        vaep_features.goalscore,
        vaep_features.location,
        vaep_features.polar,
        vaep_features.team,
        vaep_features.time,
    ]
    yfns = [vaep_features.scores, vaep_features.concedes]
    vaep = VAEP(xfns, yfns)
    vaep.fit(games, actions)
    vaep_path = MODEL_DIR / "vaep_wsl.pkl"
    with open(vaep_path, "wb") as handle:
        pickle.dump(vaep, handle)
    print(f"Saved {vaep_path} · {len(actions):,} actions")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Pool WSL (baseline LTR) ===")
    wsl_actions, wsl_labels = load_pool_actions("wsl")
    wsl_train, wsl_test = split_actions_by_game(wsl_actions, HOLDOUT_FRACTION)

    print("=== Pool Womens ===")
    womens_actions, womens_labels = load_pool_actions("womens")
    womens_train, womens_test = split_actions_by_game(womens_actions, HOLDOUT_FRACTION)

    print("=== Pool Top5 ===")
    top5_actions, top5_labels = load_pool_actions("top5")
    top5_train, top5_test = split_actions_by_game(top5_actions, HOLDOUT_FRACTION)

    trained_grids: dict[str, np.ndarray] = {}
    train_sizes: dict[str, int] = {}

    for key, train_df, labels in (
        ("wsl", wsl_train, wsl_labels),
        ("womens", womens_train, womens_labels),
        ("top5", top5_train, top5_labels),
    ):
        print(f"Training Markov '{key}'…")
        xt = train_markov(train_df)
        grid = saved_grid_from_model(xt)
        trained_grids[key] = grid
        train_sizes[key] = int(len(train_df))
        meta = {
            "competitions": labels,
            "n_games_train": int(train_df["game_id"].nunique()),
            "n_actions_train": int(len(train_df)),
            "ltr_corrected": True,
            "max_xt": float(grid.max()),
        }
        out_path = MODEL_DIR / MARKOV_MODEL_SPECS[key]["filename"]
        save_markov_model(out_path, grid, model_key=key, metadata=meta)
        print(f"  Saved {out_path} · max xT={grid.max():.4f}")

    print("Training Markov 'bayesian'…")
    counts = cell_counts(womens_train)
    bayes_grid = bayesian_smooth_grid(
        observed=trained_grids["womens"],
        prior=trained_grids["wsl"],
        counts=counts,
        k=BAYES_K,
    )
    trained_grids["bayesian"] = bayes_grid
    train_sizes["bayesian"] = train_sizes["womens"]
    bayes_meta = {
        "competitions": womens_labels,
        "prior_model": "wsl",
        "observed_model": "womens",
        "bayes_k": BAYES_K,
        "ltr_corrected": True,
        "max_xt": float(bayes_grid.max()),
    }
    bayes_path = MODEL_DIR / MARKOV_MODEL_SPECS["bayesian"]["filename"]
    save_markov_model(bayes_path, bayes_grid, model_key="bayesian", metadata=bayes_meta)
    print(f"  Saved {bayes_path} · max xT={bayes_grid.max():.4f}")

    print("=== Hold-out validation (shot em 8 ações) ===")
    test_sets = {
        "wsl": wsl_test,
        "womens": womens_test,
        "top5": top5_test,
        "bayesian": womens_test,
    }
    metrics: dict[str, dict[str, float | int]] = {}
    for key, test_df in test_sets.items():
        vals = evaluate_markov_on_holdout(key, trained_grids[key], test_df)
        metrics[key] = vals
        print(
            f"  {key:8s} · n={vals['n_moves']:6d} · "
            f"AUC={vals['auc_shot_8']:.4f} · r={vals['corr_shot_8']:.4f}"
        )

    winner, reason = pick_validation_winner(metrics)
    test_moves = int(metrics.get(winner, {}).get("n_moves", 0))
    save_validation_report(metrics, winner, reason, train_sizes=train_sizes, test_moves=test_moves)

    try:
        games, _ = load_competition_actions(37, 4)
        train_vaep_models(wsl_train, games)
    except Exception as exc:  # noqa: BLE001
        print(f"VAEP skip: {exc}")


if __name__ == "__main__":
    main()
