"""
analisa_summary_julgadores.py

Gera gráficos e tabelas para analisar o arquivo:
summary_101_3_julgadores_score_batch.csv

Este script foi adaptado para o CSV de avaliação por score de 0 a 5,
com colunas como:

modelo,
n_total,
score_medio_0_5,
score_medio_normalizado,
n_correto,
n_parcialmente_correto,
n_incorreto,
n_unanimidade_categoria,
n_divergencia_categoria,
taxa_unanimidade_categoria,
taxa_divergencia_categoria,
desvio_medio_score_entre_julgadores,
media_gpt_score_0_5,
media_gemini_score_0_5,
media_claude_score_0_5

Como usar:

python analisa_summary_julgadores.py \
  --csv results/legacy/result_judges_score/summary_101_3_julgadores_score_batch.csv \
  --out results/legacy/result_judges_score/graficos_summary

Dependências:

pip install pandas matplotlib numpy
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Preparação dos dados
# ============================================================

def preparar_dados(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {csv_path}\n"
            "Verifique o caminho passado em --csv.\n\n"
            "Exemplo:\n"
            "python analisa_summary_julgadores.py "
            "--csv results/legacy/result_judges_score/summary_101_3_julgadores_score_batch.csv"
        )

    df = pd.read_csv(csv_path)

    # Compatibilidade com nomes alternativos
    renames = {
        "n_errado": "n_incorreto",

        # Caso o CSV antigo/binário seja usado por engano
        "n_unanimidade": "n_unanimidade_categoria",
        "n_divergencia_entre_julgadores": "n_divergencia_categoria",
        "acuracia_binaria_101": "score_medio_normalizado",

        # Caso médias antigas venham sem sufixo
        "media_gpt_score": "media_gpt_score_original",
        "media_gemini_score": "media_gemini_score_original",
        "media_claude_score": "media_claude_score_original",
    }

    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

    # Se não existir coluna de parcialmente correto, assume 0
    if "n_parcialmente_correto" not in df.columns:
        df["n_parcialmente_correto"] = 0

    # Se não existir score_medio_0_5, tenta derivar do score normalizado
    if "score_medio_0_5" not in df.columns:
        if "score_medio_normalizado" in df.columns:
            df["score_medio_0_5"] = df["score_medio_normalizado"] * 5
        else:
            raise ValueError(
                "O CSV não possui 'score_medio_0_5' nem 'score_medio_normalizado'."
            )

    # Se não existir score normalizado, deriva do score 0 a 5
    if "score_medio_normalizado" not in df.columns:
        df["score_medio_normalizado"] = df["score_medio_0_5"] / 5

    # Compatibilidade com médias por julgador
    preparar_coluna_julgador(df, "gpt")
    preparar_coluna_julgador(df, "gemini")
    preparar_coluna_julgador(df, "claude")

    # Se não houver taxas, calcula pelas contagens
    if "taxa_unanimidade_categoria" not in df.columns:
        if "n_unanimidade_categoria" in df.columns:
            df["taxa_unanimidade_categoria"] = (
                df["n_unanimidade_categoria"] / df["n_total"]
            )
        else:
            df["n_unanimidade_categoria"] = 0
            df["taxa_unanimidade_categoria"] = 0

    if "taxa_divergencia_categoria" not in df.columns:
        if "n_divergencia_categoria" in df.columns:
            df["taxa_divergencia_categoria"] = (
                df["n_divergencia_categoria"] / df["n_total"]
            )
        else:
            df["n_divergencia_categoria"] = 0
            df["taxa_divergencia_categoria"] = 0

    # Se não houver contagens absolutas, calcula a partir das taxas
    if "n_unanimidade_categoria" not in df.columns:
        df["n_unanimidade_categoria"] = (
            df["taxa_unanimidade_categoria"] * df["n_total"]
        ).round().astype(int)

    if "n_divergencia_categoria" not in df.columns:
        df["n_divergencia_categoria"] = (
            df["taxa_divergencia_categoria"] * df["n_total"]
        ).round().astype(int)

    colunas_obrigatorias = [
        "modelo",
        "n_total",
        "n_correto",
        "n_parcialmente_correto",
        "n_incorreto",
        "score_medio_0_5",
        "score_medio_normalizado",
        "n_unanimidade_categoria",
        "n_divergencia_categoria",
        "taxa_unanimidade_categoria",
        "taxa_divergencia_categoria",
        "media_gpt_score_0_5",
        "media_gemini_score_0_5",
        "media_claude_score_0_5",
        "media_gpt_score_normalizado",
        "media_gemini_score_normalizado",
        "media_claude_score_normalizado",
    ]

    faltando = [c for c in colunas_obrigatorias if c not in df.columns]

    if faltando:
        raise ValueError(
            f"Colunas ausentes no CSV: {faltando}\n\n"
            f"Colunas encontradas:\n{list(df.columns)}"
        )

    # Ordena pelo score médio agregado
    df = df.sort_values("score_medio_0_5", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    # Gaps entre julgadores usando score 0 a 5
    df["gap_gemini_gpt"] = (
        df["media_gemini_score_0_5"] - df["media_gpt_score_0_5"]
    )
    df["gap_gemini_claude"] = (
        df["media_gemini_score_0_5"] - df["media_claude_score_0_5"]
    )
    df["gap_gpt_claude"] = (
        df["media_gpt_score_0_5"] - df["media_claude_score_0_5"]
    )

    # Desvio entre julgadores
    if "desvio_medio_score_entre_julgadores" in df.columns:
        df["desvio_entre_julgadores"] = df["desvio_medio_score_entre_julgadores"]
    else:
        judge_cols = [
            "media_gpt_score_0_5",
            "media_gemini_score_0_5",
            "media_claude_score_0_5",
        ]
        df["desvio_entre_julgadores"] = df[judge_cols].std(axis=1)

    return df


def preparar_coluna_julgador(df: pd.DataFrame, julgador: str) -> None:
    """
    Garante que existam duas colunas por julgador:
    - media_<julgador>_score_0_5
    - media_<julgador>_score_normalizado
    """

    col_0_5 = f"media_{julgador}_score_0_5"
    col_norm = f"media_{julgador}_score_normalizado"
    col_original = f"media_{julgador}_score_original"

    if col_0_5 in df.columns and col_norm in df.columns:
        return

    if col_0_5 in df.columns and col_norm not in df.columns:
        df[col_norm] = df[col_0_5] / 5
        return

    if col_norm in df.columns and col_0_5 not in df.columns:
        df[col_0_5] = df[col_norm] * 5
        return

    if col_original in df.columns:
        max_val = df[col_original].max()

        if max_val <= 1:
            df[col_norm] = df[col_original]
            df[col_0_5] = df[col_original] * 5
        else:
            df[col_0_5] = df[col_original]
            df[col_norm] = df[col_original] / 5

        return

    raise ValueError(
        f"Não encontrei as colunas de média do julgador {julgador}.\n"
        f"Esperado algo como: {col_0_5} ou {col_norm}"
    )


# ============================================================
# Utilitários
# ============================================================

def salvar_figura(fig, out_dir: Path, nome: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / nome, dpi=200, bbox_inches="tight")
    plt.close(fig)


def formatar_percentual(valor: float) -> str:
    return f"{valor:.1%}"


def abreviar_modelo(nome: str) -> str:
    """
    Abrevia nomes longos apenas para gráficos de dispersão.
    """
    substituicoes = {
        "google_gemma-4-31b-it": "gemma-4-31b",
        "meta_llama-3.3-70b-instruct": "llama-3.3-70b",
        "mistralai_mixtral-8x22b-instruct-v0.1": "mixtral-8x22b",
        "deepseek-ai_deepseek-v4-flash": "deepseek-v4",
        "openai_gpt-oss-120b": "gpt-oss-120b",
        "qwen_qwen3.5-122b-a10b": "qwen3.5-122b",
        "qwen2-72B-Instruct_old": "qwen2-72b",
        "gemini-3-pro-preview": "gemini-3-pro",
        "claude-opus-4-6": "claude-opus-4.6",
        "z-ai_glm-5.1": "glm-5.1",
        "gpt-5.1": "gpt-5.1",
    }

    return substituicoes.get(nome, nome)


# ============================================================
# Tabela resumo
# ============================================================

def salvar_tabela_resumo(df: pd.DataFrame, out_dir: Path) -> None:
    colunas = [
        "rank",
        "modelo",
        "n_total",
        "score_medio_0_5",
        "score_medio_normalizado",
        "n_correto",
        "n_parcialmente_correto",
        "n_incorreto",
        "n_unanimidade_categoria",
        "n_divergencia_categoria",
        "taxa_unanimidade_categoria",
        "taxa_divergencia_categoria",
        "desvio_entre_julgadores",
        "media_gpt_score_0_5",
        "media_gemini_score_0_5",
        "media_claude_score_0_5",
        "gap_gemini_gpt",
        "gap_gemini_claude",
        "gap_gpt_claude",
    ]

    tabela = df[colunas].copy()
    tabela.to_csv(
        out_dir / "summary_com_metricas_extras.csv",
        index=False,
        encoding="utf-8",
    )


# ============================================================
# Gráficos
# ============================================================

def grafico_ranking_score_medio(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.sort_values("score_medio_0_5", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(plot_df["modelo"], plot_df["score_medio_0_5"])

    ax.set_title("Ranking dos modelos por score médio agregado")
    ax.set_xlabel("Score médio agregado (0 a 5)")
    ax.set_ylabel("Modelo avaliado")
    ax.set_xlim(0, 5)

    for i, value in enumerate(plot_df["score_medio_0_5"]):
        ax.text(value + 0.03, i, f"{value:.3f}", va="center")

    salvar_figura(fig, out_dir, "01_ranking_score_medio_0_5.png")


def grafico_score_normalizado(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.sort_values("score_medio_normalizado", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(plot_df["modelo"], plot_df["score_medio_normalizado"])

    ax.set_title("Score médio normalizado por modelo")
    ax.set_xlabel("Score normalizado (0 a 1)")
    ax.set_ylabel("Modelo avaliado")
    ax.set_xlim(0, 1)

    for i, value in enumerate(plot_df["score_medio_normalizado"]):
        ax.text(value + 0.01, i, f"{value:.3f}", va="center")

    salvar_figura(fig, out_dir, "02_score_medio_normalizado.png")


def grafico_categorias_finais(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.sort_values("score_medio_0_5", ascending=False)

    x = np.arange(len(plot_df))

    corretas = plot_df["n_correto"].values
    parciais = plot_df["n_parcialmente_correto"].values
    incorretas = plot_df["n_incorreto"].values

    fig, ax = plt.subplots(figsize=(15, 7))

    ax.bar(x, corretas, label="Correto")
    ax.bar(x, parciais, bottom=corretas, label="Parcialmente correto")
    ax.bar(x, incorretas, bottom=corretas + parciais, label="Incorreto")

    ax.set_title("Distribuição das categorias finais por modelo")
    ax.set_xlabel("Modelo avaliado")
    ax.set_ylabel("Quantidade de perguntas")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["modelo"], rotation=45, ha="right")
    ax.legend()

    salvar_figura(fig, out_dir, "03_categorias_finais_empilhadas.png")


def grafico_scores_por_julgador(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.sort_values("score_medio_0_5", ascending=False)

    x = np.arange(len(plot_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(15, 7))

    ax.bar(
        x - width,
        plot_df["media_gpt_score_0_5"],
        width,
        label="GPT julgador",
    )
    ax.bar(
        x,
        plot_df["media_gemini_score_0_5"],
        width,
        label="Gemini julgador",
    )
    ax.bar(
        x + width,
        plot_df["media_claude_score_0_5"],
        width,
        label="Claude julgador",
    )

    ax.set_title("Score médio atribuído por cada julgador")
    ax.set_xlabel("Modelo avaliado")
    ax.set_ylabel("Score médio (0 a 5)")
    ax.set_ylim(0, 5)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["modelo"], rotation=45, ha="right")
    ax.legend()

    salvar_figura(fig, out_dir, "04_score_medio_por_julgador.png")


def grafico_permissividade_julgadores(df: pd.DataFrame, out_dir: Path) -> None:
    medias = pd.Series({
        "Gemini julgador": df["media_gemini_score_0_5"].mean(),
        "GPT julgador": df["media_gpt_score_0_5"].mean(),
        "Claude julgador": df["media_claude_score_0_5"].mean(),
    }).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(medias.index, medias.values)

    ax.set_title("Permissividade média dos julgadores")
    ax.set_ylabel("Média dos scores atribuídos (0 a 5)")
    ax.set_ylim(0, 5)

    for i, value in enumerate(medias.values):
        ax.text(i, value + 0.05, f"{value:.3f}", ha="center")

    salvar_figura(fig, out_dir, "05_permissividade_media_julgadores.png")


def grafico_unanimidade_divergencia(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.sort_values("taxa_divergencia_categoria", ascending=False)

    x = np.arange(len(plot_df))
    width = 0.40

    fig, ax = plt.subplots(figsize=(15, 7))

    ax.bar(
        x - width / 2,
        plot_df["n_unanimidade_categoria"],
        width,
        label="Unanimidade",
    )
    ax.bar(
        x + width / 2,
        plot_df["n_divergencia_categoria"],
        width,
        label="Divergência",
    )

    ax.set_title("Unanimidade e divergência entre julgadores")
    ax.set_xlabel("Modelo avaliado")
    ax.set_ylabel("Quantidade de perguntas")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["modelo"], rotation=45, ha="right")
    ax.legend()

    salvar_figura(fig, out_dir, "06_unanimidade_vs_divergencia.png")


def grafico_taxa_divergencia(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.sort_values("taxa_divergencia_categoria", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(plot_df["modelo"], plot_df["taxa_divergencia_categoria"])

    ax.set_title("Taxa de divergência categórica entre julgadores")
    ax.set_xlabel("Divergência / total de perguntas")
    ax.set_ylabel("Modelo avaliado")
    ax.set_xlim(0, 1)

    for i, value in enumerate(plot_df["taxa_divergencia_categoria"]):
        ax.text(value + 0.01, i, formatar_percentual(value), va="center")

    salvar_figura(fig, out_dir, "07_taxa_divergencia_categoria.png")


def grafico_desvio_entre_julgadores(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.sort_values("desvio_entre_julgadores", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(plot_df["modelo"], plot_df["desvio_entre_julgadores"])

    ax.set_title("Desvio médio dos scores entre julgadores")
    ax.set_xlabel("Desvio médio dos scores")
    ax.set_ylabel("Modelo avaliado")

    limite = max(0.1, plot_df["desvio_entre_julgadores"].max() + 0.1)
    ax.set_xlim(0, limite)

    for i, value in enumerate(plot_df["desvio_entre_julgadores"]):
        ax.text(value + 0.01, i, f"{value:.3f}", va="center")

    salvar_figura(fig, out_dir, "08_desvio_score_entre_julgadores.png")


def grafico_gap(
    df: pd.DataFrame,
    out_dir: Path,
    coluna: str,
    titulo: str,
    xlabel: str,
    nome_arquivo: str,
) -> None:
    plot_df = df.sort_values(coluna, ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(plot_df["modelo"], plot_df[coluna])
    ax.axvline(0, linewidth=1)

    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Modelo avaliado")

    for i, value in enumerate(plot_df[coluna]):
        deslocamento = 0.02 if value >= 0 else -0.02
        alinhamento = "left" if value >= 0 else "right"
        ax.text(
            value + deslocamento,
            i,
            f"{value:.3f}",
            va="center",
            ha=alinhamento,
        )

    salvar_figura(fig, out_dir, nome_arquivo)


def grafico_score_vs_divergencia(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    x = df["score_medio_0_5"]
    y = df["taxa_divergencia_categoria"]

    ax.scatter(x, y)

    for idx, row in df.iterrows():
        # Alterna deslocamentos para reduzir sobreposição
        dx = 5 if idx % 2 == 0 else -5
        dy = 5 if idx % 3 == 0 else -8

        ax.annotate(
            abreviar_modelo(row["modelo"]),
            (row["score_medio_0_5"], row["taxa_divergencia_categoria"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title("Relação entre score médio e divergência entre julgadores")
    ax.set_xlabel("Score médio agregado (0 a 5)")
    ax.set_ylabel("Taxa de divergência categórica")
    ax.set_xlim(
        max(0, df["score_medio_0_5"].min() - 0.3),
        min(5, df["score_medio_0_5"].max() + 0.3),
    )
    ax.set_ylim(
        0,
        min(1, max(0.1, df["taxa_divergencia_categoria"].max() + 0.1)),
    )

    salvar_figura(fig, out_dir, "10_score_vs_divergencia.png")


def grafico_heatmap_scores_julgadores(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.sort_values("score_medio_0_5", ascending=False)

    matriz = plot_df[
        [
            "media_gpt_score_0_5",
            "media_gemini_score_0_5",
            "media_claude_score_0_5",
        ]
    ].values

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matriz, aspect="auto")

    ax.set_title("Mapa de calor dos scores médios por julgador")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["GPT", "Gemini", "Claude"])
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["modelo"])

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(
                j,
                i,
                f"{matriz[i, j]:.2f}",
                ha="center",
                va="center",
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Score médio (0 a 5)")

    salvar_figura(fig, out_dir, "11_heatmap_scores_julgadores.png")


# ============================================================
# Impressão no terminal
# ============================================================

def imprimir_leitura(df: pd.DataFrame) -> None:
    print("\n=== RANKING POR SCORE MÉDIO AGREGADO ===")
    print(
        df[
            [
                "rank",
                "modelo",
                "n_total",
                "score_medio_0_5",
                "score_medio_normalizado",
                "n_correto",
                "n_parcialmente_correto",
                "n_incorreto",
            ]
        ].to_string(index=False)
    )

    print("\n=== MÉDIA DOS JULGADORES ===")

    medias = {
        "GPT": df["media_gpt_score_0_5"].mean(),
        "Gemini": df["media_gemini_score_0_5"].mean(),
        "Claude": df["media_claude_score_0_5"].mean(),
    }

    for nome, valor in sorted(medias.items(), key=lambda x: x[1], reverse=True):
        print(f"{nome}: {valor:.4f}")

    mais_permissivo = max(medias, key=medias.get)
    mais_rigido = min(medias, key=medias.get)

    print(f"\nJulgador mais permissivo: {mais_permissivo}")
    print(f"Julgador mais rígido: {mais_rigido}")

    print("\n=== MODELOS COM MAIOR DIVERGÊNCIA CATEGÓRICA ===")
    cols_div = [
        "modelo",
        "n_divergencia_categoria",
        "taxa_divergencia_categoria",
        "desvio_entre_julgadores",
    ]

    print(
        df.sort_values("taxa_divergencia_categoria", ascending=False)[cols_div]
        .head(5)
        .to_string(index=False)
    )

    print("\n=== MODELOS COM MENOR DESVIO ENTRE JULGADORES ===")
    cols_desvio = [
        "modelo",
        "score_medio_0_5",
        "desvio_entre_julgadores",
    ]

    print(
        df.sort_values("desvio_entre_julgadores", ascending=True)[cols_desvio]
        .head(5)
        .to_string(index=False)
    )

    print("\n=== TOP 5 MODELOS ===")
    print(
        df.head(5)[
            [
                "rank",
                "modelo",
                "score_medio_0_5",
                "score_medio_normalizado",
                "n_correto",
                "n_parcialmente_correto",
                "n_incorreto",
            ]
        ].to_string(index=False)
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        default="results/legacy/result_judges_score/summary_101_3_julgadores_score_batch.csv",
        help="Caminho do summary CSV.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="results/legacy/result_judges_score/graficos_summary",
        help="Pasta onde os gráficos e a tabela serão salvos.",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)

    out_dir.mkdir(parents=True, exist_ok=True)

    df = preparar_dados(csv_path)

    salvar_tabela_resumo(df, out_dir)

    grafico_ranking_score_medio(df, out_dir)
    grafico_score_normalizado(df, out_dir)
    grafico_categorias_finais(df, out_dir)
    grafico_scores_por_julgador(df, out_dir)
    grafico_permissividade_julgadores(df, out_dir)
    grafico_unanimidade_divergencia(df, out_dir)
    grafico_taxa_divergencia(df, out_dir)
    grafico_desvio_entre_julgadores(df, out_dir)

    grafico_gap(
        df=df,
        out_dir=out_dir,
        coluna="gap_gemini_claude",
        titulo="Diferença de score médio entre julgadores: Gemini - Claude",
        xlabel="Gemini - Claude",
        nome_arquivo="09_gap_gemini_menos_claude.png",
    )

    grafico_gap(
        df=df,
        out_dir=out_dir,
        coluna="gap_gemini_gpt",
        titulo="Diferença de score médio entre julgadores: Gemini - GPT",
        xlabel="Gemini - GPT",
        nome_arquivo="09_gap_gemini_menos_gpt.png",
    )

    grafico_gap(
        df=df,
        out_dir=out_dir,
        coluna="gap_gpt_claude",
        titulo="Diferença de score médio entre julgadores: GPT - Claude",
        xlabel="GPT - Claude",
        nome_arquivo="09_gap_gpt_menos_claude.png",
    )

    grafico_score_vs_divergencia(df, out_dir)
    grafico_heatmap_scores_julgadores(df, out_dir)

    imprimir_leitura(df)

    print(f"\nGráficos salvos em: {out_dir.resolve()}")
    print("Tabela extra salva como: summary_com_metricas_extras.csv")


if __name__ == "__main__":
    main()