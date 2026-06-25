import os
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
from matplotlib.colors import Normalize, LinearSegmentedColormap
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
        font-size: 0.95rem;
        font-weight: 600;
        color: #c7cdda;
        margin: 0.6rem 0 0.25rem 0;
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
LANE_LEFT_MIN = 53.33
LANE_RIGHT_MAX = 26.67
FIG_W, FIG_H = 7.0, 4.7
FIG_DPI = 180
WYSCOUT_PITCH_SIZE = 100.0
OPT_ATTACKING_TWO_THIRDS_X = 40.0
WYSCOUT_PROG_OWN_HALF = 30.0
WYSCOUT_PROG_CROSS_HALF = 15.0
WYSCOUT_PROG_OPP_HALF = 10.0

COLOR_SUCCESS = "#c8c8c8"
COLOR_PROGRESSIVE = "#2F80ED"
COLOR_KEY_PASS = "#f59e0b"
COLOR_FAIL = "#E07070"
ALPHA_SUCCESS = 0.07

ACTION_COLORS = {
    "passes": "#5b9bd5",
    "dribbles": "#22c55e",
    "ball-carries": "#a855f7",
    "defensive": "#ef4444",
}

DEFENSIVE_MARKERS = {
    "tackle": "s",
    "interception": "D",
    "clearance": "^",
    "ball-recovery": "o",
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
    """Convert Wyscout 0–100 pitch coords to StatsBomb 120×80 (attack → right)."""
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

    df = pd.DataFrame(rows)
    df["progressive"] = False
    pass_mask = df["category"] == "passes"
    if pass_mask.any():
        df.loc[pass_mask, "progressive"] = df.loc[pass_mask].apply(
            lambda r: r["is_success"]
            and r["has_end"]
            and is_progressive_wyscout(r["x_start"], r["y_start"], r["x_end"], r["y_end"]),
            axis=1,
        )
    return df


def load_all_players(base_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    players: dict[str, pd.DataFrame] = {}
    for path in discover_csv_files(base_dir):
        try:
            players[path.stem] = load_player_csv(path)
        except Exception as exc:
            st.warning(f"Não foi possível carregar `{path.name}`: {exc}")
    return players


# ── STATS ────────────────────────────────────────────────────
def compute_action_stats(df: pd.DataFrame) -> dict:
    passes = df[df["category"] == "passes"]
    dribbles = df[df["category"] == "dribbles"]
    carries = df[df["category"] == "ball-carries"]
    defensive = df[df["category"] == "defensive"]

    pass_total = len(passes)
    pass_success = int(passes["is_success"].sum()) if pass_total else 0
    pass_prog = int(passes["progressive"].sum()) if pass_total else 0
    pass_key = int(passes["is_key_pass"].sum()) if pass_total else 0

    return {
        "total_actions": len(df),
        "passes_total": pass_total,
        "passes_success": pass_success,
        "passes_accuracy": (pass_success / pass_total * 100.0) if pass_total else 0.0,
        "passes_progressive": pass_prog,
        "passes_key": pass_key,
        "dribbles_total": len(dribbles),
        "dribbles_success": int(dribbles["is_success"].sum()) if len(dribbles) else 0,
        "carries_total": len(carries),
        "defensive_total": len(defensive),
        "defensive_success": int(defensive["is_success"].sum()) if len(defensive) else 0,
        "by_category": df.groupby("category").size().to_dict(),
        "by_action_type": df.groupby("action_type").size().to_dict(),
    }


def stats_card(title: str, border_color: str, items: list[tuple[str, str]]) -> None:
    rows = "".join(
        f"""
        <div style="display:flex;justify-content:space-between;padding:0.35rem 0;
                    border-bottom:1px solid rgba(255,255,255,0.06);">
            <span style="color:#94a3b8;font-size:0.82rem;">{label}</span>
            <span style="color:#eef1f7;font-weight:600;font-size:0.85rem;">{value}</span>
        </div>
        """
        for label, value in items
    )
    st.markdown(
        f"""
        <div style="background:#1a1a2e;border:1px solid {border_color};border-radius:10px;
                    padding:0.85rem 1rem;margin-bottom:0.75rem;">
            <div style="color:{border_color};font-weight:700;font-size:0.9rem;margin-bottom:0.5rem;">
                {title}
            </div>
            {rows}
        </div>
        """,
        unsafe_allow_html=True,
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


def _attack_arrow(fig, has_cbar: bool = False):
    ox = -0.04 if has_cbar else 0.0
    fig.patches.append(
        FancyArrowPatch(
            (0.44 + ox, 0.045),
            (0.56 + ox, 0.045),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.6,
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
        fontsize=7.5,
        color="#aaaaaa",
    )


def _save_fig(fig):
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)


def draw_action_map(df: pd.DataFrame, categories: set[str] | None = None):
    fig, ax, pitch = _base_pitch()
    subset = df if categories is None else df[df["category"].isin(categories)]

    for _, row in subset.iterrows():
        category = row["category"]
        color = ACTION_COLORS.get(category, "#94a3b8")
        alpha = 0.90 if row["is_success"] else 0.55
        if not row["is_success"] and category == "passes":
            color = COLOR_FAIL
            alpha = 0.72

        if row["has_end"]:
            pitch.arrows(
                row["x_start"],
                row["y_start"],
                row["x_end"],
                row["y_end"],
                color=color,
                width=1.2,
                headwidth=2.0,
                headlength=2.0,
                ax=ax,
                zorder=3,
                alpha=alpha,
            )
        else:
            marker = DEFENSIVE_MARKERS.get(row["action_type"], "o")
            pitch.scatter(
                row["x_start"],
                row["y_start"],
                s=90,
                marker=marker,
                color=color,
                edgecolors="white",
                linewidths=0.6,
                ax=ax,
                zorder=5,
                alpha=alpha,
            )

    legend_handles = [
        Line2D([0], [0], color=ACTION_COLORS["passes"], lw=2.0, label="Passes"),
        Line2D([0], [0], color=ACTION_COLORS["dribbles"], lw=2.0, label="Dribles"),
        Line2D([0], [0], color=ACTION_COLORS["ball-carries"], lw=2.0, label="Conduções"),
        Line2D([0], [0], color=ACTION_COLORS["defensive"], lw=2.0, label="Defensivas"),
        Line2D([0], [0], color=COLOR_FAIL, lw=2.0, label="Passe incompleto"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="#1a1a2e",
        edgecolor="#444466",
        fontsize=6.5,
        labelspacing=0.35,
        borderpad=0.4,
    )
    for text in leg.get_texts():
        text.set_color("white")
    leg.get_frame().set_alpha(0.90)
    _attack_arrow(fig)
    return _save_fig(fig), fig


def draw_pass_map(df: pd.DataFrame):
    passes = df[df["category"] == "passes"].copy()
    fig, ax, pitch = _base_pitch()

    for _, row in passes.iterrows():
        is_lost = not row["is_success"]
        is_key = bool(row["is_key_pass"])
        is_prog = bool(row["progressive"])
        if is_lost:
            color, alpha = COLOR_FAIL, 0.72
        elif is_key:
            color, alpha = COLOR_KEY_PASS, 0.95
        elif is_prog:
            color, alpha = COLOR_PROGRESSIVE, 0.88
        else:
            color, alpha = COLOR_SUCCESS, ALPHA_SUCCESS

        if not row["has_end"]:
            continue

        pitch.arrows(
            row["x_start"],
            row["y_start"],
            row["x_end"],
            row["y_end"],
            color=color,
            width=1.3,
            headwidth=2.0,
            headlength=2.0,
            ax=ax,
            zorder=3,
            alpha=alpha,
        )
        pitch.scatter(
            row["x_start"],
            row["y_start"],
            s=32,
            marker="o",
            color=color,
            edgecolors="white",
            linewidths=0.6,
            ax=ax,
            zorder=6,
            alpha=alpha,
        )

    legend_handles = [
        Line2D([0], [0], color=COLOR_SUCCESS, lw=2.0, label="Completado", alpha=0.65),
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=2.0, label="Progressivo (Wyscout)", alpha=0.90),
        Line2D([0], [0], color=COLOR_KEY_PASS, lw=2.0, label="Key Pass", alpha=0.95),
        Line2D([0], [0], color=COLOR_FAIL, lw=2.0, label="Incompleto", alpha=0.90),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="#1a1a2e",
        edgecolor="#444466",
        fontsize=6.5,
        labelspacing=0.35,
        borderpad=0.4,
    )
    for text in leg.get_texts():
        text.set_color("white")
    leg.get_frame().set_alpha(0.90)
    _attack_arrow(fig)
    return _save_fig(fig), fig


def draw_destination_heatmap(df: pd.DataFrame):
    df_s = df[(df["category"] == "passes") & df["is_success"] & df["has_end"]].copy()
    x_bins = np.linspace(0.0, FIELD_X, 7)
    corridors = {
        "left": (LANE_LEFT_MIN, FIELD_Y),
        "center": (LANE_RIGHT_MAX, LANE_LEFT_MIN),
        "right": (0.0, LANE_RIGHT_MAX),
    }
    counts = {}
    for cname, (y0, y1) in corridors.items():
        arr = np.zeros(6, dtype=int)
        for i in range(6):
            x0_, x1_ = x_bins[i], x_bins[i + 1]
            arr[i] = int(
                (
                    (df_s["x_end"] >= x0_)
                    & (df_s["x_end"] < x1_)
                    & (df_s["y_end"] >= y0)
                    & (df_s["y_end"] < y1_)
                ).sum()
            )
        counts[cname] = arr

    all_vals = np.concatenate([counts[c] for c in counts]) if counts else np.array([0])
    vmax = max(1, int(all_vals.max()))
    cmap = LinearSegmentedColormap.from_list("wr", ["#ffffff", "#ffecec", "#ffbfbf", "#ff8080", "#ff3b3b", "#ff0000"])
    norm = Normalize(vmin=0, vmax=vmax)
    threshold = max(1, vmax * 0.35)

    fig, ax, pitch = _base_pitch()
    for cname, (y0, y1) in corridors.items():
        for i in range(6):
            x0_, x1_ = x_bins[i], x_bins[i + 1]
            value = counts[cname][i]
            ax.add_patch(
                Rectangle(
                    (x0_, y0),
                    x1_ - x0_,
                    y1 - y0,
                    facecolor=cmap(norm(value)),
                    edgecolor=(1, 1, 1, 0.12),
                    lw=0.5,
                    alpha=0.95,
                    zorder=2,
                )
            )
            ax.text(
                (x0_ + x1_) / 2,
                (y0 + y1) / 2,
                str(value),
                ha="center",
                va="center",
                color="#000000" if value <= threshold else "#ffffff",
                fontsize=9,
                fontweight="700" if value >= vmax * 0.5 else "600",
                zorder=4,
            )

    ax.axhline(y=LANE_LEFT_MIN, color="#ffffff", lw=0.5, alpha=0.15, linestyle="--", zorder=3)
    ax.axhline(y=LANE_RIGHT_MAX, color="#ffffff", lw=0.5, alpha=0.15, linestyle="--", zorder=3)
    _attack_arrow(fig)
    return _save_fig(fig), fig


def render_maps(df: pd.DataFrame, categories: set[str] | None = None):
    img_action, fig_action = draw_action_map(df, categories)
    plt.close(fig_action)
    st.markdown('<div class="map-label">Mapa de Ações</div>', unsafe_allow_html=True)
    st.image(img_action, use_container_width=True)

    if categories is None or "passes" in categories:
        img_pass, fig_pass = draw_pass_map(df)
        plt.close(fig_pass)
        st.markdown('<div class="map-label">Mapa de Passes</div>', unsafe_allow_html=True)
        st.image(img_pass, use_container_width=True)

        img_heat, fig_heat = draw_destination_heatmap(df)
        plt.close(fig_heat)
        st.markdown('<div class="map-label">Heatmap de Destino (Passes)</div>', unsafe_allow_html=True)
        st.image(img_heat, use_container_width=True)


# ── MAIN ─────────────────────────────────────────────────────
players = load_all_players()

if not players:
    st.error(
        "Nenhum arquivo `.csv` encontrado no diretório do app. "
        "Adicione arquivos como `enzo.csv` com as colunas esperadas."
    )
    st.stop()

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
    filtered = filtered[filtered["is_success"]]
elif outcome_filter == "fail":
    filtered = filtered[~filtered["is_success"]]

stats = compute_action_stats(filtered)

st.markdown(f"## {player_name}")
st.caption(f"Fonte: `{source_file}` · {stats['total_actions']} ações no recorte")

tab_maps, tab_stats, tab_data = st.tabs(["Mapas", "Estatísticas", "Dados"])

with tab_maps:
    if filtered.empty:
        st.info("Nenhuma ação para os filtros selecionados.")
    else:
        render_maps(filtered, set(selected_categories))

with tab_stats:
    col1, col2, col3 = st.columns(3)
    with col1:
        stats_card(
            "Passes",
            ACTION_COLORS["passes"],
            [
                ("Total", f"{stats['passes_total']}"),
                ("Completados", f"{stats['passes_success']}"),
                ("Acurácia", f"{stats['passes_accuracy']:.1f}%"),
                ("Progressivos", f"{stats['passes_progressive']}"),
                ("Key Passes", f"{stats['passes_key']}"),
            ],
        )
    with col2:
        stats_card(
            "Condução",
            ACTION_COLORS["ball-carries"],
            [
                ("Dribles", f"{stats['dribbles_total']}"),
                ("Dribles c/ sucesso", f"{stats['dribbles_success']}"),
                ("Conduções", f"{stats['carries_total']}"),
            ],
        )
    with col3:
        stats_card(
            "Defesa",
            ACTION_COLORS["defensive"],
            [
                ("Ações defensivas", f"{stats['defensive_total']}"),
                ("Com sucesso", f"{stats['defensive_success']}"),
            ],
        )

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
                "x_start": "X início",
                "y_start": "Y início",
                "x_end": "X fim",
                "y_end": "Y fim",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
