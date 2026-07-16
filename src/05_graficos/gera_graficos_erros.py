#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gera_graficos_erros.py

Gera gráficos para análise dos resultados por modelo e por pergunta.

Entradas esperadas:
    01_resumo_por_modelo.csv
    02_ranking_perguntas_mais_erradas.csv
    05_concentracao_erros.csv

Como rodar:

    python gera_graficos_erros.py \
      --input-dir results/erros \
      --out-dir ./graficos_artigo

Se os CSVs estiverem na pasta atual:

    python gera_graficos_erros.py \
      --input-dir . \
      --out-dir ./graficos_artigo
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Utilidades
# ============================================================

def wrap_text(text: str, width: int = 55, max_lines: int = 3) -> str:
    """
    Quebra texto longo em múltiplas linhas para caber no gráfico.
    """
    text = " ".join(str(text).split())
    wrapped_full = textwrap.wrap(text, width=width)

    if not wrapped_full:
        return ""

    wrapped = wrapped_full[:max_lines]

    if len(wrapped_full) > max_lines:
        wrapped[-1] = wrapped[-1] + "..."

    return "\n".join(wrapped)


def ensure_columns(df: pd.DataFrame, required: list[str], file_name: str) -> None:
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Colunas ausentes em {file_name}: {missing}\n"
            f"Colunas encontradas: {list(df.columns)}"
        )


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico salvo: {path}")


# ============================================================
# Gráfico 1 — Score médio por modelo
# ============================================================

def plot_score_medio_por_modelo(resumo: pd.DataFrame, out_dir: Path) -> None:
    resumo = resumo.sort_values("score_medio_0_5", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        resumo["modelo_avaliado"],
        resumo["score_medio_0_5"],
    )

    ax.set_title("Pontuação média por modelo")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Score médio (0 a 5)")
    ax.set_ylim(0, 5)
    ax.tick_params(axis="x", rotation=45)

    for bar, value in zip(bars, resumo["score_medio_0_5"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    save_figure(
        fig,
        out_dir / "01_pontuacao_media_por_modelo.png",
    )


# ============================================================
# Gráfico 2 — Corretas, parciais e incorretas por modelo
# ============================================================

def plot_distribuicao_respostas_por_modelo(resumo: pd.DataFrame, out_dir: Path) -> None:
    resumo = resumo.sort_values("score_medio_0_5", ascending=False).reset_index(drop=True)

    x = list(range(len(resumo)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))

    bars_corretas = ax.bar(
        [i - width for i in x],
        resumo["n_correto"],
        width=width,
        label="Corretas",
    )

    bars_parciais = ax.bar(
        x,
        resumo["n_parcialmente_correto"],
        width=width,
        label="Parcialmente corretas",
    )

    bars_incorretas = ax.bar(
        [i + width for i in x],
        resumo["n_incorreto"],
        width=width,
        label="Incorretas",
    )

    ax.set_title("Distribuição de respostas por modelo")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Quantidade de respostas")
    ax.set_xticks(x)
    ax.set_xticklabels(resumo["modelo_avaliado"], rotation=45, ha="right")
    ax.legend()

    for bars in [bars_corretas, bars_parciais, bars_incorretas]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.2,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()

    save_figure(
        fig,
        out_dir / "02_distribuicao_respostas_por_modelo.png",
    )


# ============================================================
# Gráfico 3 — Top perguntas mais erradas
# ============================================================

def plot_top_perguntas_mais_erradas(
    ranking: pd.DataFrame,
    out_dir: Path,
    top_n: int = 10,
) -> None:
    ranking = ranking.sort_values(
        ["n_incorreto", "n_parcialmente_correto", "score_medio_0_5"],
        ascending=[False, False, True],
    ).head(top_n).copy()

    ranking["pergunta_curta"] = ranking["pergunta"].apply(
        lambda x: wrap_text(x, width=55, max_lines=3)
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.barh(
        ranking["pergunta_curta"],
        ranking["n_incorreto"],
    )

    ax.set_title(f"Top {top_n} perguntas com maior número de respostas incorretas")
    ax.set_xlabel("Número de respostas incorretas")
    ax.set_ylabel("Pergunta")
    ax.invert_yaxis()

    for bar, value in zip(bars, ranking["n_incorreto"]):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value)}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()

    save_figure(
        fig,
        out_dir / f"03_top{top_n}_perguntas_mais_erradas.png",
    )


# ============================================================
# Gráfico 4 — Concentração dos erros
# ============================================================

def plot_concentracao_erros(concentracao: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    percentual = concentracao["percentual_dos_erros"] * 100

    bars = ax.bar(
        concentracao["top_n_perguntas"].astype(str),
        percentual,
    )

    ax.set_title("Concentração dos erros nas perguntas mais difíceis")
    ax.set_xlabel("Número de perguntas consideradas")
    ax.set_ylabel("Percentual acumulado dos erros (%)")
    ax.set_ylim(0, 100)

    for bar, value in zip(bars, percentual):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    save_figure(
        fig,
        out_dir / "04_concentracao_dos_erros.png",
    )


# ============================================================
# Tabela auxiliar para o artigo
# ============================================================

def salvar_resumo_para_artigo(resumo: pd.DataFrame, out_dir: Path) -> None:
    cols = [
        "posicao",
        "modelo_avaliado",
        "score_medio_0_5",
        "score_normalizado",
        "n_correto",
        "n_parcialmente_correto",
        "n_incorreto",
    ]

    df = resumo[cols].copy()
    df["score_normalizado_percentual"] = df["score_normalizado"] * 100
    df = df.drop(columns=["score_normalizado"])

    out_path = out_dir / "resumo_para_artigo.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Tabela salva: {out_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=str,
        default=".",
        help="Pasta onde estão os CSVs de entrada.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="./graficos_artigo",
        help="Pasta onde os gráficos serão salvos.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Quantidade de perguntas mais erradas no gráfico.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    resumo_path = input_dir / "01_resumo_por_modelo.csv"
    ranking_path = input_dir / "02_ranking_perguntas_mais_erradas.csv"
    concentracao_path = input_dir / "05_concentracao_erros.csv"

    if not resumo_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {resumo_path}")

    if not ranking_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {ranking_path}")

    if not concentracao_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {concentracao_path}")

    resumo = pd.read_csv(resumo_path)
    ranking = pd.read_csv(ranking_path)
    concentracao = pd.read_csv(concentracao_path)

    ensure_columns(
        resumo,
        [
            "posicao",
            "modelo_avaliado",
            "score_medio_0_5",
            "score_normalizado",
            "n_correto",
            "n_parcialmente_correto",
            "n_incorreto",
        ],
        resumo_path.name,
    )

    ensure_columns(
        ranking,
        [
            "pergunta",
            "n_correto",
            "n_parcialmente_correto",
            "n_incorreto",
            "score_medio_0_5",
        ],
        ranking_path.name,
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

    out_dir.mkdir(parents=True, exist_ok=True)

    plot_score_medio_por_modelo(resumo, out_dir)
    plot_distribuicao_respostas_por_modelo(resumo, out_dir)
    plot_top_perguntas_mais_erradas(ranking, out_dir, top_n=args.top_n)
    plot_concentracao_erros(concentracao, out_dir)
    salvar_resumo_para_artigo(resumo, out_dir)

    print("\nFinalizado.")
    print(f"Arquivos salvos em: {out_dir.resolve()}")


if __name__ == "__main__":
    main()