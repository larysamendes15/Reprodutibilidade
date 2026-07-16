from pathlib import Path
import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÃO
# ============================================================

# Caminho do CSV gerado pelo ranking_erros_perguntas.py
ARQUIVO_ENTRADA = Path("results/ranking_erros_perguntas/ranking_perguntas_erros_com_respostas.csv")

# Pasta onde os gráficos serão salvos
PASTA_SAIDA = Path("results/graficos_top14_erros")
PASTA_SAIDA.mkdir(exist_ok=True)

TOTAL_MODELOS = 11

# Cores
COR_CORRETA = "#2E7D32"      # verde
COR_PARCIAL = "#F9A825"      # amarelo/laranja
COR_INCORRETA = "#C62828"    # vermelho
COR_AZUL = "#1565C0"
COR_GRID = "#E5E7EB"
COR_TEXTO = "#222222"


# ============================================================
# PADRÕES DE ERRO ASSOCIADOS ÀS 14 PERGUNTAS
# Ajuste os rótulos se quiser mudar a interpretação.
# ============================================================

PADROES_ERRO = {
    "Q1": "Inversão da conclusão normativa",
    "Q2": "Procedimento técnico específico",
    "Q3": "Fórmula/regra normativa específica",
    "Q4": "Referência legal específica",
    "Q5": "Omissão de condição normativa",
    "Q6": "Condição temporal/regime",
    "Q7": "Marco temporal/data-corte",
    "Q8": "Omissão de exceção",
    "Q9": "Definição técnica restrita",
    "Q10": "Procedimento declaratório específico",
    "Q11": "Procedimento formal",
    "Q12": "Exceção normativa",
    "Q13": "Generalização indevida",
    "Q14": "Contábil vs. fiscal",
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def ler_csv(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}\n"
            "Confira o caminho em ARQUIVO_ENTRADA."
        )

    return pd.read_csv(caminho, sep=None, engine="python")


def encurtar(texto, largura=75):
    texto = re.sub(r"\s+", " ", str(texto)).strip()
    return textwrap.shorten(texto, width=largura, placeholder="...")


def salvar(fig, nome):
    caminho_png = PASTA_SAIDA / f"{nome}.png"
    caminho_pdf = PASTA_SAIDA / f"{nome}.pdf"

    fig.savefig(caminho_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(caminho_pdf, bbox_inches="tight", facecolor="white")

    plt.close(fig)

    print(f"Gerado: {caminho_png}")
    print(f"Gerado: {caminho_pdf}")


def limpar_eixo(ax, grid_axis="x"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.7, color=COR_GRID)
    ax.set_axisbelow(True)


def preparar_dados(df):
    colunas_necessarias = [
        "id_pergunta",
        "pergunta",
        "n_corretas",
        "n_parcialmente_corretas",
        "n_incorretas",
        "total_modelos_avaliados",
    ]

    faltando = [c for c in colunas_necessarias if c not in df.columns]

    if faltando:
        raise ValueError(
            f"Colunas ausentes no CSV: {faltando}\n"
            f"Colunas disponíveis: {list(df.columns)}"
        )

    df = df.copy()

    df["taxa_erro_%"] = 100 * df["n_incorretas"] / df["total_modelos_avaliados"]

    df["taxa_nao_correta_%"] = 100 * (
        df["n_incorretas"] + df["n_parcialmente_corretas"]
    ) / df["total_modelos_avaliados"]

    # Índice simples: erro completo pesa 1; parcial pesa 0,5
    df["indice_severidade"] = (
        df["n_incorretas"] + 0.5 * df["n_parcialmente_corretas"]
    )

    df["padrao_erro"] = df["id_pergunta"].map(PADROES_ERRO).fillna(
        "Erro jurídico específico"
    )

    df = df.sort_values(
        ["n_incorretas", "n_parcialmente_corretas", "indice_severidade"],
        ascending=[False, False, False],
    )

    return df.head(14).reset_index(drop=True)


# ============================================================
# GRÁFICO 1 — COMPOSIÇÃO DAS RESPOSTAS NAS 14 PERGUNTAS
# ============================================================

def grafico_composicao_top14(df):
    plot_df = df.iloc[::-1].copy()
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(12.5, 7.2))

    ax.barh(y, plot_df["n_corretas"], color=COR_CORRETA, label="Corretas")

    ax.barh(
        y,
        plot_df["n_parcialmente_corretas"],
        left=plot_df["n_corretas"],
        color=COR_PARCIAL,
        label="Parcialmente corretas",
    )

    ax.barh(
        y,
        plot_df["n_incorretas"],
        left=plot_df["n_corretas"] + plot_df["n_parcialmente_corretas"],
        color=COR_INCORRETA,
        label="Incorretas",
    )

    labels = [
        f"{row.id_pergunta} — {encurtar(row.pergunta, 80)}"
        for row in plot_df.itertuples()
    ]

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)

    for i, row in enumerate(plot_df.itertuples()):
        total = (
            row.n_corretas
            + row.n_parcialmente_corretas
            + row.n_incorretas
        )

        ax.text(
            total + 0.15,
            i,
            f"{int(row.n_incorretas)}/11 incorretas",
            va="center",
            ha="left",
            fontsize=8.5,
            color=COR_TEXTO,
        )

    ax.set_xlim(0, 12.8)
    ax.set_xlabel("Quantidade de modelos")
    ax.set_title(
        "Composição das respostas nas 14 perguntas mais difíceis",
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

    limpar_eixo(ax, "x")
    salvar(fig, "01_composicao_top14")


# ============================================================
# GRÁFICO 2 — PARETO DAS 14 PERGUNTAS
# ============================================================

def grafico_pareto_top14(df):
    plot_df = df.copy()

    total_erros = plot_df["n_incorretas"].sum()
    plot_df["percentual_acumulado"] = (
        100 * plot_df["n_incorretas"].cumsum() / total_erros
    )

    x = np.arange(len(plot_df))

    fig, ax1 = plt.subplots(figsize=(10.8, 5.8))

    barras = ax1.bar(
        x,
        plot_df["n_incorretas"],
        color=COR_INCORRETA,
        alpha=0.93,
    )

    for bar, valor in zip(barras, plot_df["n_incorretas"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            valor + 0.25,
            str(int(valor)),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax1.set_ylabel("Número de respostas incorretas")
    ax1.set_xlabel("Perguntas")
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_df["id_pergunta"])
    ax1.set_ylim(0, max(plot_df["n_incorretas"]) + 2)
    ax1.grid(axis="y", linestyle="--", linewidth=0.7, color=COR_GRID)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()

    ax2.plot(
        x,
        plot_df["percentual_acumulado"],
        color=COR_AZUL,
        marker="o",
        linewidth=2.2,
    )

    ax2.set_ylabel("Percentual acumulado dos erros (%)")
    ax2.set_ylim(0, 105)

    for i, valor in enumerate(plot_df["percentual_acumulado"]):
        ax2.text(
            i,
            valor + 2.2,
            f"{valor:.1f}%",
            color=COR_AZUL,
            fontsize=8,
            ha="center",
        )

    ax1.set_title(
        "Pareto das 14 perguntas mais difíceis",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    salvar(fig, "02_pareto_top14")


# ============================================================
# GRÁFICO 3 — ERRO TOTAL VS RESPOSTA PARCIAL
# ============================================================

def grafico_incorretas_vs_parciais(df):
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    tamanhos = 180 + 45 * df["indice_severidade"]

    ax.scatter(
        df["n_incorretas"],
        df["n_parcialmente_corretas"],
        s=tamanhos,
        color=COR_INCORRETA,
        alpha=0.75,
        edgecolor="white",
        linewidth=1.2,
    )

    for row in df.itertuples():
        ax.text(
            row.n_incorretas + 0.06,
            row.n_parcialmente_corretas + 0.06,
            row.id_pergunta,
            fontsize=9,
            fontweight="bold",
        )

    ax.axvspan(9.5, 11.5, color=COR_INCORRETA, alpha=0.08)
    ax.axhspan(3.5, 6.5, color=COR_PARCIAL, alpha=0.10)

    ax.text(
        10.0,
        max(df["n_parcialmente_corretas"]) + 0.25,
        "Falha quase total",
        color=COR_INCORRETA,
        fontsize=10,
        fontweight="bold",
    )

    ax.text(
        3.0,
        max(df["n_parcialmente_corretas"]) + 0.25,
        "Respostas incompletas",
        color="#A16207",
        fontsize=10,
        fontweight="bold",
    )

    ax.set_title(
        "Erro total vs. resposta parcialmente correta",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    ax.set_xlabel("Número de respostas incorretas")
    ax.set_ylabel("Número de respostas parcialmente corretas")
    ax.set_xlim(-0.5, 11.8)
    ax.set_ylim(-0.5, max(df["n_parcialmente_corretas"]) + 1.2)
    ax.set_xticks(range(0, 12))
    ax.set_yticks(range(0, int(max(df["n_parcialmente_corretas"])) + 2))

    limpar_eixo(ax, "both")
    salvar(fig, "03_incorretas_vs_parciais_top14")


# ============================================================
# GRÁFICO 4 — MAPA DE SEVERIDADE
# ============================================================

def grafico_mapa_severidade(df):
    matriz = df[
        ["n_corretas", "n_parcialmente_corretas", "n_incorretas"]
    ].to_numpy()

    colunas = ["Corretas", "Parciais", "Incorretas"]

    fig, ax = plt.subplots(figsize=(7.5, 7.2))

    im = ax.imshow(
        matriz,
        aspect="auto",
        cmap="Reds",
        vmin=0,
        vmax=11,
    )

    ax.set_title(
        "Mapa de severidade das 14 perguntas mais difíceis",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    ax.set_xticks(np.arange(len(colunas)))
    ax.set_xticklabels(colunas)

    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["id_pergunta"])

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            valor = matriz[i, j]

            ax.text(
                j,
                i,
                str(int(valor)),
                ha="center",
                va="center",
                color="white" if valor >= 6 else "black",
                fontweight="bold",
                fontsize=10,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Quantidade de modelos")

    ax.set_xlabel("Categoria da resposta")
    ax.set_ylabel("Pergunta")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    salvar(fig, "04_mapa_severidade_top14")


# ============================================================
# GRÁFICO 5 — PADRÕES DE ERRO
# ============================================================

def grafico_padroes_erro(df):
    padrao = (
        df.groupby("padrao_erro", as_index=False)
        .agg(
            incorretas=("n_incorretas", "sum"),
            parciais=("n_parcialmente_corretas", "sum"),
            perguntas=("id_pergunta", "count"),
            severidade=("indice_severidade", "sum"),
        )
        .sort_values("severidade", ascending=True)
    )

    y = np.arange(len(padrao))

    fig, ax = plt.subplots(figsize=(11, 6.7))

    ax.barh(
        y,
        padrao["incorretas"],
        color=COR_INCORRETA,
        label="Incorretas",
    )

    ax.barh(
        y,
        padrao["parciais"],
        left=padrao["incorretas"],
        color=COR_PARCIAL,
        label="Parcialmente corretas",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(padrao["padrao_erro"], fontsize=8.8)

    for i, row in enumerate(padrao.itertuples()):
        total = row.incorretas + row.parciais

        ax.text(
            total + 0.3,
            i,
            f"{int(total)}",
            va="center",
            ha="left",
            fontsize=8.5,
            color=COR_TEXTO,
        )

    ax.set_xlabel("Quantidade de respostas problemáticas")
    ax.set_title(
        "Padrões de erro nas 14 perguntas mais difíceis",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )

    limpar_eixo(ax, "x")
    salvar(fig, "05_padroes_erro_top14")


# ============================================================
# GRÁFICO 6 — TABELA DAS 14 PERGUNTAS
# ============================================================

def grafico_tabela_top14(df):
    linhas = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        linhas.append([
            str(i),
            f"{row['id_pergunta']} — {encurtar(row['pergunta'], 58)}",
            f"{int(row['n_incorretas'])}/11",
            f"{row['taxa_erro_%']:.1f}%".replace(".", ","),
            row["padrao_erro"],
        ])

    fig, ax = plt.subplots(figsize=(13.5, 8.2))
    ax.axis("off")

    ax.text(
        0,
        1.04,
        "Top 14 perguntas mais difíceis — erro observado e padrão associado",
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        color=COR_TEXTO,
    )

    tabela = ax.table(
        cellText=linhas,
        colLabels=["#", "Pergunta", "Erros", "Taxa", "Padrão associado"],
        cellLoc="left",
        colLoc="left",
        bbox=[0, 0, 1, 0.96],
        colWidths=[0.045, 0.55, 0.08, 0.08, 0.245],
    )

    tabela.auto_set_font_size(False)
    tabela.set_fontsize(8.6)

    for (r, c), cell in tabela.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.5)

        if r == 0:
            cell.set_facecolor("#F3F4F6")
            cell.set_text_props(weight="bold", color="#111827")
        else:
            cell.set_facecolor("white")

            if c == 0:
                cell.set_text_props(
                    weight="bold",
                    color=COR_CORRETA,
                    ha="center",
                )

            if c in [2, 3]:
                cell.set_text_props(
                    weight="bold",
                    color=COR_INCORRETA,
                    ha="center",
                )

                if c == 2:
                    cell.set_facecolor("#FEE2E2")

            if c == 4:
                erros = int(linhas[r - 1][2].split("/")[0])

                if erros >= 10:
                    cell.set_text_props(color=COR_INCORRETA)
                else:
                    cell.set_text_props(color="#B45309")

    salvar(fig, "06_tabela_top14_erros")


# ============================================================
# EXECUÇÃO
# ============================================================

def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 130,
    })

    df = ler_csv(ARQUIVO_ENTRADA)
    top14 = preparar_dados(df)

    print("Top 14 perguntas analisadas:")
    print(top14[[
        "id_pergunta",
        "n_corretas",
        "n_parcialmente_corretas",
        "n_incorretas",
        "taxa_erro_%",
        "padrao_erro",
    ]])

    grafico_composicao_top14(top14)
    grafico_pareto_top14(top14)
    grafico_incorretas_vs_parciais(top14)
    grafico_mapa_severidade(top14)
    grafico_padroes_erro(top14)
    grafico_tabela_top14(top14)

    top14.to_csv(
        PASTA_SAIDA / "top14_perguntas_analisadas.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFinalizado.")
    print(f"Arquivos salvos em: {PASTA_SAIDA.resolve()}")


if __name__ == "__main__":
    main()
