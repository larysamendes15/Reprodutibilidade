from pathlib import Path
import textwrap
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CAMINHOS
# ============================================================

ARQUIVO_MODELOS = Path("results/erros/01_resumo_por_modelo.csv")
ARQUIVO_PERGUNTAS = Path("results/ranking_erros_perguntas/ranking_perguntas_erros_com_respostas.csv")
ARQUIVO_PARETO = Path("results/figuras/01_pareto_erros_legenda_perguntas.csv")

PASTA_SAIDA = Path("results/graficos_tcc_final")
PASTA_SAIDA.mkdir(exist_ok=True)


# ============================================================
# CORES
# ============================================================

VERDE = "#2E7D32"
LARANJA = "#F9A825"
VERMELHO = "#C62828"
AZUL = "#1565C0"
CINZA = "#666666"
CINZA_CLARO = "#E0E0E0"
PRETO = "#222222"


# ============================================================
# FUNÇÕES GERAIS
# ============================================================

def ler_csv(caminho):
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    try:
        return pd.read_csv(caminho, sep=None, engine="python")
    except Exception:
        return pd.read_csv(caminho)


def normalizar_colunas(df):
    df = df.copy()

    novas = []
    for c in df.columns:
        c = str(c).strip().lower()
        c = unicodedata.normalize("NFKD", c)
        c = "".join(ch for ch in c if not unicodedata.combining(ch))
        c = re.sub(r"[^a-z0-9]+", "_", c)
        c = re.sub(r"_+", "_", c).strip("_")
        novas.append(c)

    df.columns = novas
    return df


def nome_modelo_curto(nome):
    mapa = {
        "gpt-5.1": "GPT-5.1",
        "claude-opus-4-6": "Claude-Opus-4.6",
        "z-ai_glm-5.1": "GLM-5.1",
        "deepseek-ai_deepseek-v4-flash": "DeepSeek-V4-Flash",
        "google_gemma-4-31b-it": "Gemma-4-31B",
        "qwen_qwen3.5-122b-a10b": "Qwen3.5-122B",
        "gemini-3-pro-preview": "Gemini-3-Pro",
        "mistralai_mixtral-8x22b-instruct-v0.1": "Mixtral-8x22B",
        "openai_gpt-oss-120b": "GPT-OSS-120B",
        "qwen2-72B-Instruct": "Qwen2-72B",
        "qwen2-72B-Instruct_old": "Qwen2-72B",
        "meta_llama-3.3-70b-instruct": "Llama-3.3-70B",
    }

    return mapa.get(str(nome), str(nome).replace("_", "-"))


def salvar(fig, nome):
    png = PASTA_SAIDA / f"{nome}.png"
    pdf = PASTA_SAIDA / f"{nome}.pdf"

    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")

    plt.close(fig)

    print(f"Gerado: {png}")
    print(f"Gerado: {pdf}")


def estilo_limpo(ax):
    ax.grid(axis="x", linestyle="--", linewidth=0.7, color=CINZA_CLARO)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAAAAA")
    ax.spines["bottom"].set_color("#AAAAAA")


def corrigir_score(valor):
    """
    Corrige casos em que o CSV ficou com score_medio_0_5 errado:
    exemplo: 303 vira 0.303; 9.697 vira 0.9697.
    """
    try:
        v = float(valor)
    except Exception:
        return np.nan

    if v > 100:
        return v / 1000

    if v > 5:
        return v / 10

    return v


def resumo_pergunta(texto, limite=70):
    texto = str(texto).replace("\n", " ").strip()
    texto = re.sub(r"\s+", " ", texto)
    return textwrap.shorten(texto, width=limite, placeholder="...")


def tipo_falha(id_pergunta):
    mapa = {
        "Q1": "Inversão de conclusão",
        "Q2": "Procedimento contábil-fiscal específico",
        "Q3": "Fórmula normativa específica",
        "Q4": "Referência legal específica",
        "Q5": "Condição normativa",
        "Q6": "Condição temporal/regime",
        "Q7": "Marco legal e data-corte",
        "Q8": "Omissão de exceção",
        "Q9": "Definição técnica restrita",
        "Q10": "Procedimento específico",
        "Q11": "Exceção normativa",
        "Q12": "Procedimento formal",
        "Q13": "Generalização da retificação",
        "Q14": "Registro contábil vs. cálculo fiscal",
    }

    return mapa.get(str(id_pergunta), "Erro jurídico específico")


# ============================================================
# 1. RANKING DOS MODELOS
# ============================================================

def grafico_ranking_modelos(df):
    df = df.copy()

    df["modelo_limpo"] = df["modelo_avaliado"].apply(nome_modelo_curto)
    df["score"] = df["score_medio_0_5"].apply(corrigir_score)

    df = df.sort_values("score", ascending=True)

    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    ax.barh(y, df["n_correto"], color=VERDE, label="Corretas")
    ax.barh(
        y,
        df["n_parcialmente_correto"],
        left=df["n_correto"],
        color=LARANJA,
        label="Parcialmente corretas",
    )
    ax.barh(
        y,
        df["n_incorreto"],
        left=df["n_correto"] + df["n_parcialmente_correto"],
        color=VERMELHO,
        label="Incorretas",
    )

    for i, row in enumerate(df.itertuples()):
        ax.text(
            103,
            i,
            f"{row.score:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            color=PRETO,
        )

        if row.n_correto >= 8:
            ax.text(
                row.n_correto / 2,
                i,
                int(row.n_correto),
                va="center",
                ha="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

        if row.n_incorreto >= 5:
            ax.text(
                row.n_correto + row.n_parcialmente_correto + row.n_incorreto / 2,
                i,
                int(row.n_incorreto),
                va="center",
                ha="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(df["modelo_limpo"], fontsize=9)

    ax.set_xlim(0, 112)
    ax.set_xlabel("Quantidade de respostas")
    ax.set_title(
        "Ranking dos modelos por categorias de resposta",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    ax.text(103, len(df) - 0.2, "Score", fontsize=9, fontweight="bold")

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )

    estilo_limpo(ax)

    salvar(fig, "01_ranking_modelos")


# ============================================================
# 2. PARETO DOS ERROS
# ============================================================

def grafico_pareto(df):
    df = df.copy()

    if "pergunta_curta" in df.columns:
        col_id = "pergunta_curta"
    else:
        col_id = "id_pergunta"

    df = df.sort_values("n_incorreto", ascending=False).head(20).reset_index(drop=True)

    total_erros = df["n_incorreto"].sum()
    df["perc_acumulado_calc"] = df["n_incorreto"].cumsum() / total_erros * 100

    x = np.arange(len(df))

    fig, ax1 = plt.subplots(figsize=(11, 5.8))

    barras = ax1.bar(x, df["n_incorreto"], color=VERMELHO, alpha=0.92)

    for bar, valor in zip(barras, df["n_incorreto"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            valor + 0.25,
            str(int(valor)),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax1.set_ylabel("Número de respostas incorretas")
    ax1.set_xlabel("Perguntas mais difíceis")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df[col_id], fontsize=9)
    ax1.set_ylim(0, max(df["n_incorreto"]) + 2)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        df["perc_acumulado_calc"],
        color=AZUL,
        marker="o",
        linewidth=2.2,
    )

    ax2.set_ylabel("Percentual acumulado dos erros (%)")
    ax2.set_ylim(0, 105)

    for i, valor in enumerate(df["perc_acumulado_calc"]):
        if i in [0, 1, 2, 3, 4, 6, 9, 13, 19]:
            ax2.text(
                i,
                valor + 2.4,
                f"{valor:.1f}%",
                color=AZUL,
                fontsize=8,
                ha="center",
            )

    ax1.set_title(
        "Pareto dos erros por pergunta",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    ax1.grid(axis="y", linestyle="--", linewidth=0.7, color=CINZA_CLARO)
    ax1.set_axisbelow(True)

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    salvar(fig, "02_pareto_erros")


# ============================================================
# 3. TOP 12 PERGUNTAS — COMPOSIÇÃO
# ============================================================

def grafico_top12_composicao(df):
    df = df.copy()

    df = df.sort_values(
        ["n_incorretas", "n_parcialmente_corretas"],
        ascending=[False, False],
    ).head(12)

    df = df.sort_values(
        ["n_incorretas", "n_parcialmente_corretas"],
        ascending=[True, True],
    )

    labels = [
        f"{row.id_pergunta} — {resumo_pergunta(row.pergunta, 62)}"
        for row in df.itertuples()
    ]

    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.barh(y, df["n_corretas"], color=VERDE, label="Corretas")
    ax.barh(
        y,
        df["n_parcialmente_corretas"],
        left=df["n_corretas"],
        color=LARANJA,
        label="Parcialmente corretas",
    )
    ax.barh(
        y,
        df["n_incorretas"],
        left=df["n_corretas"] + df["n_parcialmente_corretas"],
        color=VERMELHO,
        label="Incorretas",
    )

    for i, row in enumerate(df.itertuples()):
        total = row.n_corretas + row.n_parcialmente_corretas + row.n_incorretas

        ax.text(
            total + 0.15,
            i,
            f"{int(row.n_incorretas)}/11 incorretas",
            va="center",
            ha="left",
            fontsize=8.5,
            color=PRETO,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)

    ax.set_xlim(0, 12.6)
    ax.set_xlabel("Quantidade de modelos")
    ax.set_title(
        "Composição das respostas nas 12 perguntas mais difíceis",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )

    estilo_limpo(ax)

    salvar(fig, "03_top12_perguntas_composicao")


# ============================================================
# 4. CONCENTRAÇÃO DOS ERROS
# ============================================================

def grafico_concentracao(df):
    df = df.copy()

    contagem = (
        df["n_incorretas"]
        .value_counts()
        .reindex(range(0, 12), fill_value=0)
        .sort_index()
    )

    x = np.arange(0, 12)

    cores = []
    for valor in x:
        if valor <= 2:
            cores.append(VERDE)
        elif valor <= 5:
            cores.append(LARANJA)
        else:
            cores.append(VERMELHO)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    barras = ax.bar(x, contagem.values, color=cores, width=0.68)

    for bar, valor in zip(barras, contagem.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            valor + 0.35,
            str(int(valor)),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title(
        "Concentração dos erros por pergunta",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    ax.set_xlabel("Número de modelos com resposta incorreta")
    ax.set_ylabel("Quantidade de perguntas")
    ax.set_xticks(x)
    ax.set_xlim(-0.6, 11.6)

    ax.grid(axis="y", linestyle="--", linewidth=0.7, color=CINZA_CLARO)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    salvar(fig, "04_concentracao_erros")


# ============================================================
# 5. TABELA DAS 12 PERGUNTAS MAIS DIFÍCEIS
# ============================================================

def grafico_tabela_top12(df_perguntas, df_pareto=None):
    df = df_perguntas.copy()

    # usa score correto do arquivo pareto, se existir
    if df_pareto is not None:
        pareto = df_pareto.copy()

        if "pergunta_curta" in pareto.columns:
            pareto = pareto.rename(columns={"pergunta_curta": "id_pergunta"})

        if "score_medio_0_5" in pareto.columns:
            pareto["score_corrigido"] = pareto["score_medio_0_5"].apply(corrigir_score)
            df = df.drop(columns=["score_medio_0_5"], errors="ignore")
            df = df.merge(
                pareto[["id_pergunta", "score_corrigido"]],
                on="id_pergunta",
                how="left",
            )
        else:
            df["score_corrigido"] = df["score_medio_0_5"].apply(corrigir_score)
    else:
        df["score_corrigido"] = df["score_medio_0_5"].apply(corrigir_score)

    df = df.sort_values(
        ["n_incorretas", "score_corrigido"],
        ascending=[False, True],
    ).head(12)

    linhas = []

    for i, row in enumerate(df.itertuples(), start=1):
        score = row.score_corrigido

        if pd.isna(score):
            score_txt = "-"
        else:
            score_txt = f"{score:.2f}".replace(".", ",")

        linhas.append([
            str(i),
            f"{row.id_pergunta} — {resumo_pergunta(row.pergunta, 58)}",
            f"{int(row.n_incorretas)}/11",
            score_txt,
            tipo_falha(row.id_pergunta),
        ])

    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.axis("off")

    ax.text(
        0,
        1.04,
        "Top 12 perguntas mais difíceis — taxa de erro e score médio",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        color=PRETO,
    )

    colunas = ["#", "Pergunta (resumo)", "Erros/11", "Score", "Tipo de falha"]

    tabela = ax.table(
        cellText=linhas,
        colLabels=colunas,
        cellLoc="left",
        colLoc="left",
        bbox=[0, 0, 1, 0.96],
        colWidths=[0.05, 0.55, 0.10, 0.09, 0.21],
    )

    tabela.auto_set_font_size(False)
    tabela.set_fontsize(9.0)

    for (r, c), cell in tabela.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        cell.set_linewidth(0.5)

        if r == 0:
            cell.set_facecolor("#F2F2F2")
            cell.set_text_props(weight="bold", color=PRETO)
        else:
            cell.set_facecolor("white")

            if c == 0:
                cell.set_text_props(weight="bold", color=VERDE, ha="center")

            if c == 2:
                erros = int(cell.get_text().get_text().split("/")[0])

                if erros >= 10:
                    cell.set_facecolor("#FDEAEA")
                    cell.set_text_props(weight="bold", color=VERMELHO, ha="center")
                elif erros >= 7:
                    cell.set_facecolor("#FFF4DD")
                    cell.set_text_props(weight="bold", color="#E65100", ha="center")
                else:
                    cell.set_facecolor("#F1F8E9")
                    cell.set_text_props(weight="bold", color=VERDE, ha="center")

            if c == 3:
                cell.set_text_props(weight="bold", ha="center")

            if c == 4:
                erros = int(linhas[r - 1][2].split("/")[0])

                if erros >= 10:
                    cell.set_text_props(color=VERMELHO)
                elif erros >= 7:
                    cell.set_text_props(color="#E65100")
                else:
                    cell.set_text_props(color=VERDE)

    salvar(fig, "05_tabela_top12_perguntas")


# ============================================================
# EXECUÇÃO
# ============================================================

def main():
    modelos = normalizar_colunas(ler_csv(ARQUIVO_MODELOS))
    perguntas = normalizar_colunas(ler_csv(ARQUIVO_PERGUNTAS))

    pareto = None
    if ARQUIVO_PARETO.exists():
        pareto = normalizar_colunas(ler_csv(ARQUIVO_PARETO))

    print("Colunas do resumo por modelo:")
    print(modelos.columns.tolist())

    print("\nColunas do ranking por pergunta:")
    print(perguntas.columns.tolist())

    grafico_ranking_modelos(modelos)

    if pareto is not None:
        grafico_pareto(pareto)
    else:
        grafico_pareto(perguntas.rename(columns={"n_incorretas": "n_incorreto"}))

    grafico_top12_composicao(perguntas)
    grafico_concentracao(perguntas)
    grafico_tabela_top12(perguntas, pareto)

    print("\nFinalizado.")
    print(f"Arquivos gerados em: {PASTA_SAIDA.resolve()}")


if __name__ == "__main__":
    main()