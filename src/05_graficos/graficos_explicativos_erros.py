#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
graficos_explicativos_erros.py

Gera gráficos explicativos para análise de erros por pergunta e por modelo.

Entradas esperadas:
    01_resumo_por_modelo.csv
    02_ranking_perguntas_mais_erradas.csv
    04_matriz_modelo_pergunta_categoria.csv
    05_concentracao_erros.csv

Como rodar:

    python graficos_explicativos_erros.py \
      --input-dir results/erros \
      --out-dir results/figuras \
      --top-n 14

Dependências:
    pip install pandas matplotlib numpy
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# ============================================================
# Paleta de cores
# ============================================================

COLOR_CORRETO = "#2E7D32"       # verde
COLOR_PARCIAL = "#F9A825"       # amarelo/laranja
COLOR_INCORRETO = "#C62828"     # vermelho
COLOR_LINHA = "#1565C0"         # azul
COLOR_NEUTRO = "#616161"        # cinza
COLOR_GRID = "#E0E0E0"


# ============================================================
# Utilidades
# ============================================================

def wrap_text(text: str, width: int = 45, max_lines: int = 3) -> str:
    text = " ".join(str(text).split())
    lines_full = textwrap.wrap(text, width=width)

    if not lines_full:
        return ""

    lines = lines_full[:max_lines]

    if len(lines_full) > max_lines:
        lines[-1] = lines[-1] + "..."

    return "\n".join(lines)


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: {path}")


def ensure_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"Colunas ausentes em {name}: {missing}\n"
            f"Colunas encontradas: {list(df.columns)}"
        )


def apply_grid(ax) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4, color=COLOR_GRID)
    ax.set_axisbelow(True)


# ============================================================
# Gráfico 1 — Pareto dos erros por pergunta
# ============================================================

def plot_pareto_erros(ranking: pd.DataFrame, out_dir: Path, top_n: int = 20) -> None:
    df = ranking.sort_values(
        ["n_incorreto", "n_parcialmente_correto", "score_medio_0_5"],
        ascending=[False, False, True],
    ).head(top_n).copy()

    total_erros = ranking["n_incorreto"].sum()

    df["erros_acumulados"] = df["n_incorreto"].cumsum()
    df["perc_acumulado"] = df["erros_acumulados"] / total_erros * 100
    df["pergunta_curta"] = [f"Q{i + 1}" for i in range(len(df))]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    bars = ax1.bar(
        df["pergunta_curta"],
        df["n_incorreto"],
        color=COLOR_INCORRETO,
        edgecolor="black",
        linewidth=0.4,
    )

    ax1.set_title("Pareto dos erros: poucas perguntas concentram grande parte das falhas")
    ax1.set_xlabel("Perguntas mais difíceis")
    ax1.set_ylabel("Número de respostas incorretas")
    ax1.set_ylim(0, max(df["n_incorreto"]) + 2)
    apply_grid(ax1)

    for bar, value in zip(bars, df["n_incorreto"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax2 = ax1.twinx()

    ax2.plot(
        df["pergunta_curta"],
        df["perc_acumulado"],
        marker="o",
        color=COLOR_LINHA,
        linewidth=2,
    )

    ax2.set_ylabel("Percentual acumulado dos erros (%)")
    ax2.set_ylim(0, 100)

    for x, y in zip(df["pergunta_curta"], df["perc_acumulado"]):
        ax2.text(
            x,
            y + 2,
            f"{y:.1f}%",
            ha="center",
            fontsize=8,
            color=COLOR_LINHA,
        )

    fig.tight_layout()

    save_fig(fig, out_dir / "01_pareto_erros_por_pergunta.png")

    legenda = df[
        [
            "pergunta_curta",
            "pergunta",
            "n_incorreto",
            "n_parcialmente_correto",
            "score_medio_0_5",
            "perc_acumulado",
        ]
    ].copy()

    legenda.to_csv(
        out_dir / "01_pareto_erros_legenda_perguntas.csv",
        index=False,
        encoding="utf-8",
    )


# ============================================================
# Gráfico 2 — Composição das perguntas difíceis
# ============================================================

def plot_top_perguntas_composicao(
    ranking: pd.DataFrame,
    out_dir: Path,
    top_n: int = 10,
) -> None:
    df = ranking.sort_values(
        ["n_incorreto", "n_parcialmente_correto", "score_medio_0_5"],
        ascending=[False, False, True],
    ).head(top_n).copy()

    df["pergunta_curta"] = df["pergunta"].apply(
        lambda x: wrap_text(x, width=50, max_lines=3)
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    y = np.arange(len(df))

    ax.barh(
        y,
        df["n_incorreto"],
        label="Incorretas",
        color=COLOR_INCORRETO,
        edgecolor="black",
        linewidth=0.3,
    )

    ax.barh(
        y,
        df["n_parcialmente_correto"],
        left=df["n_incorreto"],
        label="Parcialmente corretas",
        color=COLOR_PARCIAL,
        edgecolor="black",
        linewidth=0.3,
    )

    ax.barh(
        y,
        df["n_correto"],
        left=df["n_incorreto"] + df["n_parcialmente_correto"],
        label="Corretas",
        color=COLOR_CORRETO,
        edgecolor="black",
        linewidth=0.3,
    )

    ax.set_title(f"Composição das {top_n} perguntas mais difíceis")
    ax.set_xlabel("Quantidade de modelos")
    ax.set_ylabel("Pergunta")
    ax.set_yticks(y)
    ax.set_yticklabels(df["pergunta_curta"])
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    apply_grid(ax)

    for pos, (_, row) in enumerate(df.iterrows()):
        total = row["n_correto"] + row["n_parcialmente_correto"] + row["n_incorreto"]
        ax.text(
            total + 0.2,
            pos,
            f"score={row['score_medio_0_5']:.2f}",
            va="center",
            fontsize=8,
            color=COLOR_NEUTRO,
        )

    fig.tight_layout()

    save_fig(fig, out_dir / f"02_top{top_n}_perguntas_composicao.png")


# ============================================================
# Gráfico 3 — Percentual de categorias por modelo
# ============================================================

def plot_percentual_categorias_por_modelo(
    resumo: pd.DataFrame,
    out_dir: Path,
) -> None:
    df = resumo.sort_values("score_medio_0_5", ascending=True).copy()

    df["perc_correto"] = df["n_correto"] / df["n_total"] * 100
    df["perc_parcial"] = df["n_parcialmente_correto"] / df["n_total"] * 100
    df["perc_incorreto"] = df["n_incorreto"] / df["n_total"] * 100

    fig, ax = plt.subplots(figsize=(12, 7))

    y = np.arange(len(df))

    ax.barh(
        y,
        df["perc_correto"],
        label="Corretas",
        color=COLOR_CORRETO,
        edgecolor="black",
        linewidth=0.3,
    )

    ax.barh(
        y,
        df["perc_parcial"],
        left=df["perc_correto"],
        label="Parcialmente corretas",
        color=COLOR_PARCIAL,
        edgecolor="black",
        linewidth=0.3,
    )

    ax.barh(
        y,
        df["perc_incorreto"],
        left=df["perc_correto"] + df["perc_parcial"],
        label="Incorretas",
        color=COLOR_INCORRETO,
        edgecolor="black",
        linewidth=0.3,
    )

    ax.set_title("Perfil percentual das respostas por modelo")
    ax.set_xlabel("Percentual das 101 perguntas (%)")
    ax.set_ylabel("Modelo")
    ax.set_yticks(y)
    ax.set_yticklabels(df["modelo_avaliado"])
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right")
    apply_grid(ax)

    for i, row in enumerate(df.itertuples()):
        if row.perc_correto > 7:
            ax.text(
                row.perc_correto / 2,
                i,
                f"{row.perc_correto:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )

        if row.perc_incorreto > 5:
            ax.text(
                row.perc_correto + row.perc_parcial + row.perc_incorreto / 2,
                i,
                f"{row.perc_incorreto:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )

    fig.tight_layout()

    save_fig(fig, out_dir / "03_percentual_categorias_por_modelo.png")


# ============================================================
# Gráfico 4 — Score médio vs taxa de erro
# ============================================================

def plot_score_vs_taxa_erro(resumo: pd.DataFrame, out_dir: Path) -> None:
    df = resumo.copy()

    df["taxa_erro_percentual"] = df["taxa_erro"] * 100

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(
        df["score_medio_0_5"],
        df["taxa_erro_percentual"],
        color=COLOR_INCORRETO,
        edgecolor="black",
        s=80,
        alpha=0.9,
    )

    for _, row in df.iterrows():
        ax.annotate(
            row["modelo_avaliado"],
            (row["score_medio_0_5"], row["taxa_erro_percentual"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title("Relação entre score médio e taxa de erro")
    ax.set_xlabel("Score médio (0 a 5)")
    ax.set_ylabel("Taxa de respostas incorretas (%)")

    ax.axvline(
        df["score_medio_0_5"].mean(),
        linestyle="--",
        linewidth=1,
        color=COLOR_NEUTRO,
    )

    ax.axhline(
        df["taxa_erro_percentual"].mean(),
        linestyle="--",
        linewidth=1,
        color=COLOR_NEUTRO,
    )

    apply_grid(ax)
    fig.tight_layout()

    save_fig(fig, out_dir / "04_score_vs_taxa_erro.png")


# ============================================================
# Gráfico 5 — Heatmap modelo x perguntas difíceis
# ============================================================

def plot_heatmap_modelos_perguntas(
    matriz: pd.DataFrame,
    ranking: pd.DataFrame,
    out_dir: Path,
    top_n: int = 14,
) -> None:
    top = ranking.sort_values(
        ["n_incorreto", "n_parcialmente_correto", "score_medio_0_5"],
        ascending=[False, False, True],
    ).head(top_n).copy()

    top_norm = top["pergunta_norm"].tolist()

    df = matriz[matriz["pergunta_norm"].isin(top_norm)].copy()

    df["ordem"] = df["pergunta_norm"].apply(lambda x: top_norm.index(x))
    df = df.sort_values("ordem")

    meta_cols = ["pergunta_norm", "pergunta", "ordem"]
    model_cols = [c for c in df.columns if c not in meta_cols]

    mapa = {
        "I": 0,
        "P": 1,
        "C": 2,
    }

    data = df[model_cols].replace(mapa).to_numpy(dtype=float)

    cmap = ListedColormap(
        [
            COLOR_INCORRETO,
            COLOR_PARCIAL,
            COLOR_CORRETO,
        ]
    )

    fig, ax = plt.subplots(figsize=(13, max(6, top_n * 0.5)))

    im = ax.imshow(data, aspect="auto", vmin=0, vmax=2, cmap=cmap)

    ax.set_title(f"Mapa de acertos e erros nas {top_n} perguntas mais difíceis")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Pergunta")

    ax.set_xticks(np.arange(len(model_cols)))
    ax.set_xticklabels(model_cols, rotation=45, ha="right")

    y_labels = [f"Q{i + 1}" for i in range(len(df))]
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            original = df.iloc[i][model_cols[j]]
            color_text = "white" if original in ["I", "C"] else "black"
            ax.text(
                j,
                i,
                original,
                ha="center",
                va="center",
                fontsize=9,
                color=color_text,
                fontweight="bold",
            )

    legend_elements = [
        Patch(facecolor=COLOR_INCORRETO, edgecolor="black", label="I = Incorreto"),
        Patch(facecolor=COLOR_PARCIAL, edgecolor="black", label="P = Parcial"),
        Patch(facecolor=COLOR_CORRETO, edgecolor="black", label="C = Correto"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout()

    save_fig(fig, out_dir / f"05_heatmap_top{top_n}_perguntas_dificeis.png")

    legenda = top[
        [
            "pergunta",
            "n_correto",
            "n_parcialmente_correto",
            "n_incorreto",
            "score_medio_0_5",
        ]
    ].copy()

    legenda.insert(0, "id_pergunta", [f"Q{i + 1}" for i in range(len(legenda))])

    legenda.to_csv(
        out_dir / f"05_heatmap_top{top_n}_legenda_perguntas.csv",
        index=False,
        encoding="utf-8",
    )


# ============================================================
# Gráfico 6 — Robustez nas perguntas difíceis
# ============================================================

def plot_robustez_perguntas_dificeis(
    matriz: pd.DataFrame,
    ranking: pd.DataFrame,
    out_dir: Path,
    top_n: int = 14,
) -> None:
    top = ranking.sort_values(
        ["n_incorreto", "n_parcialmente_correto", "score_medio_0_5"],
        ascending=[False, False, True],
    ).head(top_n).copy()

    df = matriz[matriz["pergunta_norm"].isin(top["pergunta_norm"])].copy()

    model_cols = [c for c in df.columns if c not in ["pergunta_norm", "pergunta"]]

    rows = []

    for model in model_cols:
        rows.append(
            {
                "modelo": model,
                "corretas_top": (df[model] == "C").sum(),
                "parciais_top": (df[model] == "P").sum(),
                "incorretas_top": (df[model] == "I").sum(),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["corretas_top", "incorretas_top"],
        ascending=[False, True],
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(out))
    width = 0.25

    ax.bar(
        x - width,
        out["corretas_top"],
        width=width,
        label="Corretas",
        color=COLOR_CORRETO,
        edgecolor="black",
        linewidth=0.3,
    )

    ax.bar(
        x,
        out["parciais_top"],
        width=width,
        label="Parciais",
        color=COLOR_PARCIAL,
        edgecolor="black",
        linewidth=0.3,
    )

    ax.bar(
        x + width,
        out["incorretas_top"],
        width=width,
        label="Incorretas",
        color=COLOR_INCORRETO,
        edgecolor="black",
        linewidth=0.3,
    )

    ax.set_title(f"Desempenho dos modelos nas {top_n} perguntas mais difíceis")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Quantidade de perguntas")
    ax.set_xticks(x)
    ax.set_xticklabels(out["modelo"], rotation=45, ha="right")
    ax.legend()
    apply_grid(ax)

    for bars in ax.containers:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.1,
                    str(int(height)),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    fig.tight_layout()

    save_fig(fig, out_dir / f"06_robustez_top{top_n}_perguntas_dificeis.png")

    out.to_csv(
        out_dir / f"06_robustez_top{top_n}_perguntas_dificeis.csv",
        index=False,
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=str,
        default=".",
        help="Pasta onde estão os CSVs da análise de erros.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/figuras",
        help="Pasta de saída dos gráficos.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=14,
        help="Número de perguntas difíceis usadas nos gráficos.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    resumo_path = input_dir / "01_resumo_por_modelo.csv"
    ranking_path = input_dir / "02_ranking_perguntas_mais_erradas.csv"
    matriz_path = input_dir / "04_matriz_modelo_pergunta_categoria.csv"
    concentracao_path = input_dir / "05_concentracao_erros.csv"

    resumo = pd.read_csv(resumo_path)
    ranking = pd.read_csv(ranking_path)
    matriz = pd.read_csv(matriz_path)
    concentracao = pd.read_csv(concentracao_path)

    ensure_columns(
        resumo,
        [
            "modelo_avaliado",
            "n_total",
            "n_correto",
            "n_parcialmente_correto",
            "n_incorreto",
            "score_medio_0_5",
            "taxa_erro",
            "taxa_nao_correto",
        ],
        resumo_path.name,
    )

    ensure_columns(
        ranking,
        [
            "pergunta_norm",
            "pergunta",
            "n_correto",
            "n_parcialmente_correto",
            "n_incorreto",
            "score_medio_0_5",
        ],
        ranking_path.name,
    )

    ensure_columns(
        matriz,
        [
            "pergunta_norm",
            "pergunta",
        ],
        matriz_path.name,
    )

    ensure_columns(
        concentracao,
        [
            "top_n_perguntas",
            "erros_acumulados",
            "percentual_dos_erros",
        ],
        concentracao_path.name,
    )

    plot_pareto_erros(ranking, out_dir, top_n=20)
    plot_top_perguntas_composicao(ranking, out_dir, top_n=10)
    plot_percentual_categorias_por_modelo(resumo, out_dir)
    plot_score_vs_taxa_erro(resumo, out_dir)
    plot_heatmap_modelos_perguntas(matriz, ranking, out_dir, top_n=args.top_n)
    plot_robustez_perguntas_dificeis(matriz, ranking, out_dir, top_n=args.top_n)

    print("\nFinalizado.")
    print(f"Gráficos salvos em: {out_dir.resolve()}")


if __name__ == "__main__":
    main()