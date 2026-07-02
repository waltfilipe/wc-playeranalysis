"""Build per-player heuristic v4 stats from StatsBomb Top 5 league pool."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from socceraction.data.statsbomb import StatsBombLoader
from socceraction.spadl import play_left_to_right
from socceraction.spadl.statsbomb import convert_to_actions
from socceraction.spadl.utils import add_names

from heuristic_scoring import MOVE_TYPE_NAMES, score_move_actions_raw_delta, shorten_position
from scripts.train_external_models import COMPETITION_POOLS, load_competition_actions

MODEL_DIR = ROOT / "models"
OUTPUT_PATH = MODEL_DIR / "top5_player_stats_v4.json"
MIN_PASSES = 250
TOP_N = 40


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_v4_fine_grid() -> np.ndarray:
  from app import FIELD_X, FIELD_Y, XT_V3_FINE_NX, XT_V3_FINE_NY, _build_heuristic_v4_threat_surface

  xe = np.linspace(0.0, FIELD_X, XT_V3_FINE_NX)
  ye = np.linspace(0.0, FIELD_Y, XT_V3_FINE_NY)
  Xc, Yc = np.meshgrid(xe, ye)
  return _build_heuristic_v4_threat_surface(Xc, Yc)


def _load_top5_games() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for comp_id, season_id, label in COMPETITION_POOLS["top5"]:
        try:
            games, _ = load_competition_actions(comp_id, season_id)
            games = games.copy()
            games["competition_label"] = label
            frames.append(games)
            print(f"  · {label}: {len(games)} jogos", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  · {label}: ERRO {exc}", flush=True)
    if not frames:
        raise RuntimeError("Nenhum jogo Top5 carregado.")
    return pd.concat(frames, ignore_index=True).drop_duplicates("game_id")


def _player_metadata_for_game(loader: StatsBombLoader, game_id: int) -> dict[int, dict]:
    meta: dict[int, dict] = {}
    for team in loader._lineups(game_id):
        for player in team["lineup"]:
            pid = int(player["player_id"])
            name = player.get("player_nickname") or player.get("player_name") or str(pid)
            positions = player.get("positions") or []
            primary_pos = positions[0]["position"] if positions else None
            entry = meta.setdefault(pid, {"player_name": name, "positions": []})
            if primary_pos:
                entry["positions"].append(primary_pos)
    return meta


def _merge_player_metadata(
    store: dict[int, dict],
    game_meta: dict[int, dict],
) -> None:
    for pid, vals in game_meta.items():
        entry = store.setdefault(pid, {"player_name": vals["player_name"], "positions": []})
        if vals["player_name"]:
            entry["player_name"] = vals["player_name"]
        entry["positions"].extend(vals["positions"])


def _primary_position(positions: list[str]) -> str:
    if not positions:
        return "—"
    return shorten_position(Counter(positions).most_common(1)[0][0])


def build_top5_player_stats() -> dict:
    print("Carregando jogos Top5…", flush=True)
    games = _load_top5_games()
    print(f"Total: {games['game_id'].nunique()} jogos", flush=True)

    print("Construindo grade heurística v4…", flush=True)
    fine = _build_v4_fine_grid()

    loader = StatsBombLoader()
    player_meta: dict[int, dict] = {}
    agg: dict[int, dict[str, float | int]] = defaultdict(
        lambda: {
            "sum_delta": 0.0,
            "sum_xt_end_passes": 0.0,
            "passes": 0,
            "carries": 0,
            "positive_delta": 0,
        }
    )

    for idx, game in games.iterrows():
        game_id = int(game["game_id"])
        home_team_id = int(game["home_team_id"])
        if idx % 25 == 0:
            print(f"  Processando jogo {idx + 1}/{len(games)} (id={game_id})…", flush=True)

        _merge_player_metadata(player_meta, _player_metadata_for_game(loader, game_id))

        events = loader.events(game_id)
        actions = add_names(convert_to_actions(events, home_team_id))
        actions = play_left_to_right(actions, home_team_id)
        moves = actions[
            (actions["result_name"] == "success")
            & actions["type_name"].isin(MOVE_TYPE_NAMES)
        ].copy()
        if moves.empty:
            continue

        start_x = moves["start_x"].to_numpy(dtype=float)
        start_y = moves["start_y"].to_numpy(dtype=float)
        end_x = moves["end_x"].to_numpy(dtype=float)
        end_y = moves["end_y"].to_numpy(dtype=float)
        _, xt_end, delta = score_move_actions_raw_delta(
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
            fine_grid=fine,
        )

        moves = moves.assign(delta_xt_v4=delta, xt_end_v4=xt_end)
        is_pass = moves["type_name"].isin(["pass", "cross"])

        for pid, group in moves.groupby("player_id"):
            pid_int = int(pid)
            bucket = agg[pid_int]
            bucket["sum_delta"] = float(bucket["sum_delta"]) + float(group["delta_xt_v4"].sum())
            bucket["positive_delta"] = int(bucket["positive_delta"]) + int((group["delta_xt_v4"] > 0).sum())
            passes = group[is_pass.loc[group.index]]
            bucket["passes"] = int(bucket["passes"]) + len(passes)
            bucket["sum_xt_end_passes"] = float(bucket["sum_xt_end_passes"]) + float(
                passes["xt_end_v4"].sum()
            )
            bucket["carries"] = int(bucket["carries"]) + int((~is_pass.loc[group.index]).sum())

    rows: list[dict] = []
    for pid, bucket in agg.items():
        passes = int(bucket["passes"])
        if passes < MIN_PASSES:
            continue
        meta = player_meta.get(pid, {})
        positions = meta.get("positions", [])
        rows.append(
            {
                "player_id": pid,
                "player_name": meta.get("player_name", str(pid)),
                "position": _primary_position(positions),
                "sum_delta_xt": round(float(bucket["sum_delta"]), 3),
                "sum_xt_end_passes": round(float(bucket["sum_xt_end_passes"]), 3),
                "passes": passes,
                "carries": int(bucket["carries"]),
                "dxt_per_pass": round(float(bucket["sum_delta"]) / passes, 4),
                "xt_per_pass": round(float(bucket["sum_xt_end_passes"]) / passes, 4),
                "positive_delta_actions": int(bucket["positive_delta"]),
            }
        )

    rows.sort(key=lambda r: r["sum_delta_xt"], reverse=True)
    top_rows = rows[:TOP_N]

    return {
        "metadata": {
            "model": "heuristic_v4",
            "pool": "top5",
            "built_at": _utc_now(),
            "n_games": int(games["game_id"].nunique()),
            "n_players": len(rows),
            "min_passes": MIN_PASSES,
            "competitions": [label for _, _, label in COMPETITION_POOLS["top5"]],
            "note": "ΔxT bruto (xt_end - xt_start) na grade v4; passes com type pass/cross.",
        },
        "players": top_rows,
        "all_players_count": len(rows),
    }


def main() -> None:
    payload = build_top5_player_stats()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Salvo {OUTPUT_PATH} · {len(payload['players'])} jogadores no top {TOP_N}")


if __name__ == "__main__":
    main()
