"""Export VAEP XGBoost classifiers for standalone inference (no socceraction at runtime)."""

from __future__ import annotations

import pickle
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
VAEP_PKL = MODEL_DIR / "vaep_wsl.pkl"
VAEP_XGB = MODEL_DIR / "vaep_xgb.pkl"


def main() -> None:
    if not VAEP_PKL.exists():
        raise FileNotFoundError(f"Treine VAEP primeiro: {VAEP_PKL} ausente.")

    with open(VAEP_PKL, "rb") as handle:
        vaep = pickle.load(handle)

    models = vaep._VAEP__models  # noqa: SLF001
    scores_model = models["scores"]
    bundle = {
        "nb_prev_actions": int(vaep.nb_prev_actions),
        "feature_columns": list(scores_model.feature_names_in_),
        "models": models,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(VAEP_XGB, "wb") as handle:
        pickle.dump(bundle, handle)

    print(f"Saved {VAEP_XGB} · {len(bundle['feature_columns'])} features")


if __name__ == "__main__":
    main()
