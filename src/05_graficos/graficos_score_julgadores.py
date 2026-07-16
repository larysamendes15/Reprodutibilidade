"""
graficos_score_julgadores.py

Gera gráficos para avaliar o resultado do julgamento escalar 0-5
feito por GPT, Gemini e Claude.

Entrada principal:
    summary_101_3_julgadores_score_batch.csv

Também aceita, opcionalmente, o arquivo detalhado:
    all_results_101_3_julgadores_score_batch.csv

Exemplo de uso:
    python graficos_score_julgadores.py \
      --summary results/legacy/result_judges_score/summary_101_3_julgadores_score_batch.csv \
      --out results/legacy/result_judges_score/graficos_score

Com arquivo detalhado:
    python graficos_score_julgadores.py \
      --summary results/legacy/result_judges_score/summary_101_3_julgadores_score_batch.csv \
      --details results/legacy/result_judges_score/all_results_101_3_julgadores_score_batch.csv \
      --out results/legacy/result_judges_score/graficos_score

Dependências:
    pip install pandas matplotlib numpy
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_summary(summary_path: Path) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(f"Arquivo summary não encontrado: {summary_path}")

    df = pd.read_csv(summary_path)

    required = [
        "modelo",
        "n_total",
        "score_medio_0_5",
        "score_medio_normalizado",
        "n_correto",
        "n_parcialmente_correto",
        "n_incorreto",
        "n_erros",
        "n_unanimidade_categoria",
        "n_divergencia_categoria",
        "taxa_unanimidade_categoria",
        "taxa_divergencia_categoria",
        "desvio_medio_score_entre_julgadores",
        "media_gpt_score_0_5",
        "media_gemini_score_0_5",
        "media_claude_score_0_5",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no summary: {missing}")

    numeric_cols = [c for c in required if c != "modelo"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("score_medio_0_5", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    df["gap_gemini_gpt"] = df["media_gemini_score_0_5"] - df["media_gpt_score_0_5"]
    df["gap_gemini_claude"] = df["media_gemini_score_0_5"] - df["media_claude_score_0_5"]
    df["gap_gpt_claude"] = df["media_gpt_score_0_5"] - df["media_claude_score_0_5"]

    return df


def save_extra_table(df: pd.DataFrame, out_dir: Path) -> None:
    cols = [
        "rank",
        "modelo",
        "n_total",
        "score_medio_0_5",
        "score_medio_normalizado",
        "n_correto",
        "n_parcialmente_correto",
        "n_incorreto",
        "n_erros",
        "taxa_unanimidade_categoria",
        "taxa_divergencia_categoria",
        "desvio_medio_score_entre_julgadores",
        "media_gpt_score_0_5",
        "media_gemini_score_0_5",
        "media_claude_score_0_5",
        "gap_gemini_gpt",
        "gap_gemini_claude",
        "gap_gpt_claude",
    ]

    df[cols].to_csv(out_dir / "summary_score_com_metricas_extras.csv", index=False, encoding="utf-8")


def plot_score_ranking(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.sort_values("score_medio_0_5", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(plot_df["modelo"], plot_df["score_medio_0_5"])
    ax.set_title("Ranking dos modelos por score médio oficial")
    ax.set_xlabel("Score médio oficial (0 a 5)")
    ax.set_ylabel("Modelo avaliado")
    ax.set_xlim(0, 5)

    for i, value in enumerate(plot_df["score_medio_0_5"]):
        ax.text(value + 0.05, i, f"{value:.2f}", va="center")

    fig.tight_layout()
    fig.savefig(out_dir / "01_ranking_score_medio_0_5.png", dpi=200)
    plt.close(fig)


def plot_score_normalizado(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.sort_values("score_medio_normalizado", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(plot_df["modelo"], plot_df["score_medio_normalizado"])
    ax.set_title("Score médio normalizado por modelo")
    ax.set_xlabel("Score normalizado (0 a 1)")
    ax.set_ylabel("Modelo avaliado")
    ax.set_xlim(0, 1)

    for i, value in enumerate(plot_df["score_medio_normalizado"]):
        ax.text(value + 0.01, i, f"{value:.3f}", va="center")

    fig.tight_layout()
    fig.savefig(out_dir / "02_score_medio_normalizado.png", dpi=200)
    plt.close(fig)


def plot_categorias_empilhadas(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.sort_values("score_medio_0_5", ascending=False)

    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(15, 7))

    bottom = np.zeros(len(plot_df))
    ax.bar(x, plot_df["n_correto"], bottom=bottom, label="Correto")
    bottom += plot_df["n_correto"].to_numpy()

    ax.bar(x, plot_df["n_parcialmente_correto"], bottom=bottom, label="Parcialmente correto")
    bottom += plot_df["n_parcialmente_correto"].to_numpy()

    ax.bar(x, plot_df["n_incorreto"], bottom=bottom, label="Incorreto")
    bottom += plot_df["n_incorreto"].to_numpy()

    if "n_erros" in plot_df.columns and plot_df["n_erros"].sum() > 0:
        ax.bar(x, plot_df["n_erros"], bottom=bottom, label="Erro")

    ax.set_title("Distribuição das categorias finais por modelo")
    ax.set_xlabel("Modelo avaliado")
    ax.set_ylabel("Quantidade de perguntas")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["modelo"], rotation=45, ha="right")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "03_categorias_finais_empilhadas.png", dpi=200)
    plt.close(fig)


def plot_scores_por_julgador(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.sort_values("score_medio_0_5", ascending=False)

    x = np.arange(len(plot_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(x - width, plot_df["media_gpt_score_0_5"], width, label="GPT julgador")
    ax.bar(x, plot_df["media_gemini_score_0_5"], width, label="Gemini julgador")
    ax.bar(x + width, plot_df["media_claude_score_0_5"], width, label="Claude julgador")

    ax.set_title("Score médio atribuído por cada julgador")
    ax.set_xlabel("Modelo avaliado")
    ax.set_ylabel("Score médio (0 a 5)")
    ax.set_ylim(0, 5)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["modelo"], rotation=45, ha="right")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "04_score_medio_por_julgador.png", dpi=200)
    plt.close(fig)


def plot_permissividade_julgadores(df: pd.DataFrame, out_dir: Path) -> None:
    medias = pd.Series(
        {
            "GPT julgador": df["media_gpt_score_0_5"].mean(),
            "Gemini julgador": df["media_gemini_score_0_5"].mean(),
            "Claude julgador": df["media_claude_score_0_5"].mean(),
        }
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(medias.index, medias.values)
    ax.set_title("Permissividade média dos julgadores")
    ax.set_ylabel("Média dos scores atribuídos (0 a 5)")
    ax.set_ylim(0, 5)

    for i, value in enumerate(medias.values):
        ax.text(i, value + 0.05, f"{value:.2f}", ha="center")

    fig.tight_layout()
    fig.savefig(out_dir / "05_permissividade_media_julgadores.png", dpi=200)
    plt.close(fig)


def plot_unanimidade_divergencia(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.sort_values("taxa_divergencia_categoria", ascending=False)

    x = np.arange(len(plot_df))
    width = 0.40

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(x - width / 2, plot_df["n_unanimidade_categoria"], width, label="Unanimidade")
    ax.bar(x + width / 2, plot_df["n_divergencia_categoria"], width, label="Divergência")

    ax.set_title("Unanimidade e divergência entre julgadores")
    ax.set_xlabel("Modelo avaliado")
    ax.set_ylabel("Quantidade de perguntas")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["modelo"], rotation=45, ha="right")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "06_unanimidade_vs_divergencia.png", dpi=200)
    plt.close(fig)


def plot_taxa_divergencia(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.sort_values("taxa_divergencia_categoria", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(plot_df["modelo"], plot_df["taxa_divergencia_categoria"])
    ax.set_title("Taxa de divergência categórica entre julgadores")
    ax.set_xlabel("Divergência / total de perguntas")
    ax.set_ylabel("Modelo avaliado")
    ax.set_xlim(0, 1)

    for i, value in enumerate(plot_df["taxa_divergencia_categoria"]):
        ax.text(value + 0.01, i, f"{value:.1%}", va="center")

    fig.tight_layout()
    fig.savefig(out_dir / "07_taxa_divergencia_categoria.png", dpi=200)
    plt.close(fig)


def plot_desvio_score(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.sort_values("desvio_medio_score_entre_julgadores", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(plot_df["modelo"], plot_df["desvio_medio_score_entre_julgadores"])
    ax.set_title("Desvio médio dos scores entre julgadores")
    ax.set_xlabel("Desvio padrão médio dos scores")
    ax.set_ylabel("Modelo avaliado")

    for i, value in enumerate(plot_df["desvio_medio_score_entre_julgadores"]):
        ax.text(value + 0.01, i, f"{value:.2f}", va="center")

    fig.tight_layout()
    fig.savefig(out_dir / "08_desvio_score_entre_julgadores.png", dpi=200)
    plt.close(fig)


def plot_gaps_julgadores(df: pd.DataFrame, out_dir: Path) -> None:
    gap_cols = {
        "gap_gemini_gpt": "Gemini - GPT",
        "gap_gemini_claude": "Gemini - Claude",
        "gap_gpt_claude": "GPT - Claude",
    }

    for col, title in gap_cols.items():
        plot_df = df.sort_values(col, ascending=True)

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.barh(plot_df["modelo"], plot_df[col])
        ax.axvline(0, linewidth=1)
        ax.set_title(f"Diferença de score médio entre julgadores: {title}")
        ax.set_xlabel(title)
        ax.set_ylabel("Modelo avaliado")

        for i, value in enumerate(plot_df[col]):
            ax.text(value + 0.02, i, f"{value:.2f}", va="center")

        fig.tight_layout()
        safe_title = title.lower().replace(" ", "_").replace("-", "menos")
        fig.savefig(out_dir / f"09_gap_{safe_title}.png", dpi=200)
        plt.close(fig)


def plot_score_vs_divergencia(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(df["score_medio_0_5"], df["taxa_divergencia_categoria"])

    for _, row in df.iterrows():
        ax.annotate(
            row["modelo"],
            (row["score_medio_0_5"], row["taxa_divergencia_categoria"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title("Relação entre score médio e divergência entre julgadores")
    ax.set_xlabel("Score médio oficial (0 a 5)")
    ax.set_ylabel("Taxa de divergência categórica")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, max(0.55, df["taxa_divergencia_categoria"].max() + 0.05))

    fig.tight_layout()
    fig.savefig(out_dir / "10_score_vs_divergencia.png", dpi=200)
    plt.close(fig)


def plot_heatmap_scores(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.sort_values("score_medio_0_5", ascending=False).copy()

    data = plot_df[
        [
            "media_gpt_score_0_5",
            "media_gemini_score_0_5",
            "media_claude_score_0_5",
        ]
    ].to_numpy()

    fig, ax = plt.subplots(figsize=(9, max(6, 0.45 * len(plot_df))))
    im = ax.imshow(data, aspect="auto")

    ax.set_title("Mapa de calor dos scores médios por julgador")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["GPT", "Gemini", "Claude"])
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["modelo"])

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, label="Score médio (0 a 5)")
    fig.tight_layout()
    fig.savefig(out_dir / "11_heatmap_scores_julgadores.png", dpi=200)
    plt.close(fig)


def read_details(details_path: Path | None) -> pd.DataFrame | None:
    if details_path is None:
        return None

    if not details_path.exists():
        print(f"Arquivo detalhado não encontrado, ignorando: {details_path}")
        return None

    df = pd.read_csv(details_path)
    return df


def plot_distribuicao_scores_detalhado(details_df: pd.DataFrame, out_dir: Path) -> None:
    required = [
        "modelo_avaliado",
        "score_oficial_0_5",
        "gpt_judge_score_0_5",
        "gemini_judge_score_0_5",
        "claude_judge_score_0_5",
    ]

    missing = [c for c in required if c not in details_df.columns]
    if missing:
        print(f"Arquivo detalhado sem colunas para distribuição de scores: {missing}")
        return

    score_cols = [
        "score_oficial_0_5",
        "gpt_judge_score_0_5",
        "gemini_judge_score_0_5",
        "claude_judge_score_0_5",
    ]

    for c in score_cols:
        details_df[c] = pd.to_numeric(details_df[c], errors="coerce")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        [
            details_df["gpt_judge_score_0_5"].dropna(),
            details_df["gemini_judge_score_0_5"].dropna(),
            details_df["claude_judge_score_0_5"].dropna(),
            details_df["score_oficial_0_5"].dropna(),
        ],
        bins=np.arange(-0.5, 6.5, 1),
        label=["GPT", "Gemini", "Claude", "Oficial"],
    )

    ax.set_title("Distribuição dos scores no arquivo detalhado")
    ax.set_xlabel("Score")
    ax.set_ylabel("Quantidade de avaliações")
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "12_distribuicao_scores_detalhado.png", dpi=200)
    plt.close(fig)


def plot_boxplot_scores_por_modelo(details_df: pd.DataFrame, out_dir: Path) -> None:
    required = ["modelo_avaliado", "score_oficial_0_5"]

    missing = [c for c in required if c not in details_df.columns]
    if missing:
        print(f"Arquivo detalhado sem colunas para boxplot: {missing}")
        return

    details_df["score_oficial_0_5"] = pd.to_numeric(details_df["score_oficial_0_5"], errors="coerce")

    grouped = []
    labels = []

    order = (
        details_df.groupby("modelo_avaliado")["score_oficial_0_5"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    for model in order:
        values = details_df.loc[
            details_df["modelo_avaliado"] == model,
            "score_oficial_0_5",
        ].dropna()

        if len(values) > 0:
            grouped.append(values)
            labels.append(model)

    if not grouped:
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.boxplot(grouped, tick_labels=labels)
    ax.set_title("Distribuição do score oficial por modelo")
    ax.set_xlabel("Modelo avaliado")
    ax.set_ylabel("Score oficial (0 a 5)")
    ax.set_ylim(0, 5)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(out_dir / "13_boxplot_score_oficial_por_modelo.png", dpi=200)
    plt.close(fig)


def print_analysis(df: pd.DataFrame) -> None:
    print("\n=== RANKING POR SCORE MÉDIO OFICIAL ===")
    cols = [
        "rank",
        "modelo",
        "n_total",
        "score_medio_0_5",
        "score_medio_normalizado",
        "n_correto",
        "n_parcialmente_correto",
        "n_incorreto",
    ]
    print(df[cols].to_string(index=False))

    print("\n=== MÉDIA GERAL DOS JULGADORES ===")
    means = {
        "GPT": df["media_gpt_score_0_5"].mean(),
        "Gemini": df["media_gemini_score_0_5"].mean(),
        "Claude": df["media_claude_score_0_5"].mean(),
    }

    for name, value in sorted(means.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {value:.3f}")

    print(f"\nJulgador mais permissivo: {max(means, key=means.get)}")
    print(f"Julgador mais rígido: {min(means, key=means.get)}")

    print("\n=== MAIOR DIVERGÊNCIA CATEGÓRICA ===")
    cols = [
        "modelo",
        "n_divergencia_categoria",
        "taxa_divergencia_categoria",
        "desvio_medio_score_entre_julgadores",
    ]
    print(df.sort_values("taxa_divergencia_categoria", ascending=False)[cols].head(5).to_string(index=False))

    print("\n=== MAIOR DESVIO ENTRE JULGADORES ===")
    print(df.sort_values("desvio_medio_score_entre_julgadores", ascending=False)[cols].head(5).to_string(index=False))

    print("\n=== TOP 5 MODELOS ===")
    print(df.head(5)[["rank", "modelo", "score_medio_0_5", "n_correto", "n_parcialmente_correto", "n_incorreto"]].to_string(index=False))

    n_values = sorted(df["n_total"].dropna().unique())
    print("\n=== TAMANHO DA AMOSTRA ===")
    print(f"Valores encontrados em n_total: {n_values}")
    if len(n_values) == 1 and n_values[0] < 101:
        print("Atenção: parece ser uma rodada de teste, não a avaliação completa com 101 perguntas.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Caminho do summary CSV com score 0-5.",
    )
    parser.add_argument(
        "--details",
        type=str,
        default=None,
        help="Opcional: caminho do all_results detalhado.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./graficos_score_julgadores",
        help="Pasta onde os gráficos serão salvos.",
    )

    args = parser.parse_args()

    summary_path = Path(args.summary)
    details_path = Path(args.details) if args.details else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_summary(summary_path)
    details_df = read_details(details_path)

    save_extra_table(df, out_dir)

    plot_score_ranking(df, out_dir)
    plot_score_normalizado(df, out_dir)
    plot_categorias_empilhadas(df, out_dir)
    plot_scores_por_julgador(df, out_dir)
    plot_permissividade_julgadores(df, out_dir)
    plot_unanimidade_divergencia(df, out_dir)
    plot_taxa_divergencia(df, out_dir)
    plot_desvio_score(df, out_dir)
    plot_gaps_julgadores(df, out_dir)
    plot_score_vs_divergencia(df, out_dir)
    plot_heatmap_scores(df, out_dir)

    if details_df is not None:
        plot_distribuicao_scores_detalhado(details_df, out_dir)
        plot_boxplot_scores_por_modelo(details_df, out_dir)

    print_analysis(df)

    print(f"\nGráficos salvos em: {out_dir.resolve()}")
    print(f"Tabela extra salva em: {(out_dir / 'summary_score_com_metricas_extras.csv').resolve()}")


if __name__ == "__main__":
    main()
