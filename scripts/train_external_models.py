"""Train xT Markov and VAEP models on StatsBomb Open Data (FA WSL 2018/19)."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from statsbombpy import sb
from socceraction.spadl.statsbomb import convert_to_actions
from socceraction.spadl.utils import add_names
from socceraction.xthreat import ExpectedThreat
from socceraction.vaep import features as vaep_features
from socceraction.vaep.base import VAEP

COMPETITION_ID = 37
SEASON_ID = 4
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading StatsBomb events…")
    events = sb.events(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    games = sb.games(competition_id=COMPETITION_ID, season_id=SEASON_ID)

    print("Converting to SPADL…")
    actions = convert_to_actions(events, games)
    actions = add_names(actions)

    print("Training xT Markov (16×12)…")
    xt = ExpectedThreat(l=16, w=12)
    xt.fit(actions)
    xt_path = MODEL_DIR / "xt_markov_wsl_16x12.json"
    xt.save_model(str(xt_path))
    print(f"Saved {xt_path} · max xT={xt.xT.max():.4f}")

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


if __name__ == "__main__":
    main()
