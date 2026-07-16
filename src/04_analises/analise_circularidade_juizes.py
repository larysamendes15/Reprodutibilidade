#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Análise de circularidade juiz–respondente / self-preference bias.

O que o script calcula:
1) Para cada juiz (GPT, Gemini, Claude):
   - nota média que ele dá aos modelos da própria família;
   - nota média que ele dá aos demais modelos;
   - diferença: própria família - demais.

2) Para cada família de modelo avaliado:
   - nota média dada pelo juiz da mesma família;
   - nota média dada pelos outros dois juízes ao mesmo conjunto de respostas;
   - diferença pareada: juiz da família - outros juízes.

Esse segundo cálculo é o mais forte para responder à crítica de self-preference,
porque compara os juízes avaliando exatamente as mesmas respostas.

Como usar:
    python analise_circularidade_juizes.py caminho/do/arquivo_ou_pasta

Exemplos:
    python analise_circularidade_juizes.py all_results_101_3_julgadores_score_batch.csv
    python analise_circularidade_juizes.py results/judges_scores
    python analise_circularidade_juizes.py .

Saídas:
    results/circularidade/
        01_self_family_vs_outros_modelos.csv
        02_mesmo_modelo_julgador_familia_vs_outros_julgadores.csv
        03_media_score_por_modelo_e_julgador.csv
        04_resumo_textual.md
        05_grafico_self_vs_outros.png/pdf
        06_grafico_vies_pareado.png/pdf
"""

from pathlib import Path
import sys
import re
import math
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÃO
# ============================================================

JUDGES = {
    "gpt": "gpt_judge_score_0_5",
    "gemini": "gemini_judge_score_0_5",
    "claude": "claude_judge_score_0_5",
}

# Famílias usadas na análise de circularidade.
# Ajuste se você quiser considerar outros modelos como "família GPT", "família Claude", etc.
FAMILY_PATTERNS = {
    "gpt": [
        r"\bgpt\b",
        r"gpt-",
        r"gpt_",
        r"openai",
    ],
    "gemini": [
        r"gemini",
        r"google_gemini",
    ],
    "claude": [
        r"claude",
        r"anthropic",
    ],
}

# Se quiser analisar apenas os três modelos principais, deixe True.
# Se quiser incluir openai_gpt-oss-120b como família GPT, deixe False.
USAR_APENAS_MODELOS_PRINCIPAIS = False

MODELOS_PRINCIPAIS = {
    "gpt": [
        "gpt-5.1",
    ],
    "gemini": [
        "gemini-3-pro-preview",
    ],
    "claude": [
        "claude-opus-4-6",
    ],
}

OUT = Path("results/circularidade")
OUT.mkdir(exist_ok=True)


# ============================================================
# LEITURA DOS DADOS
# ============================================================

def ler_csv_robusto(path: Path) -> pd.DataFrame:
    """
    Lê CSV tentando separadores comuns.
    """
    for sep in [None, ",", ";", "\t"]:
        try:
            if sep is None:
                return pd.read_csv(path, sep=None, engine="python")
            return pd.read_csv(path, sep=sep)
        except Exception:
            continue
    raise ValueError(f"Não foi possível ler o arquivo CSV: {path}")


def arquivo_tem_colunas_validas(path: Path) -> bool:
    """
    Verifica se o CSV parece ser de julgamentos com GPT/Gemini/Claude.
    """
    try:
        df_head = ler_csv_robusto(path).head(2)
    except Exception:
        return False

    cols = set(df_head.columns)

    obrigatorias = {"modelo_avaliado"}
    scores = set(JUDGES.values())

    return obrigatorias.issubset(cols) and scores.issubset(cols)


def carregar_dados(input_path: Path) -> pd.DataFrame:
    """
    Aceita tanto um arquivo CSV único quanto uma pasta com vários CSVs.
    """
    input_path = Path(input_path)

    if input_path.is_file():
        df = ler_csv_robusto(input_path)
        print(f"Lido arquivo único: {input_path} | linhas={len(df)}")
        return df

    if not input_path.exists():
        raise FileNotFoundError(f"Caminho não encontrado: {input_path}")

    arquivos = sorted(input_path.rglob("*.csv"))

    dfs = []
    usados = []

    for arq in arquivos:
        if arquivo_tem_colunas_validas(arq):
            df = ler_csv_robusto(arq)
            df["arquivo_fonte"] = str(arq)
            dfs.append(df)
            usados.append(arq)

    if not dfs:
        raise FileNotFoundError(
            f"Nenhum CSV válido encontrado em {input_path}.\n"
            "O arquivo precisa ter as colunas: modelo_avaliado, "
            "gpt_judge_score_0_5, gemini_judge_score_0_5, claude_judge_score_0_5."
        )

    print("Arquivos usados:")
    for arq in usados:
        print(f"  - {arq}")

    dados = pd.concat(dfs, ignore_index=True)
    print(f"Total carregado: {len(dados)} linhas")
    return dados


# ============================================================
# NORMALIZAÇÃO E FAMÍLIAS
# ============================================================

def norm(s) -> str:
    return str(s).strip().lower()


def modelo_principal_family(modelo: str):
    m = norm(modelo)

    for familia, nomes in MODELOS_PRINCIPAIS.items():
        for nome in nomes:
            if m == norm(nome):
                return familia

    return "outros"


def detectar_familia_modelo(modelo: str) -> str:
    """
    Classifica o modelo avaliado em gpt, gemini, claude ou outros.
    """
    if USAR_APENAS_MODELOS_PRINCIPAIS:
        return modelo_principal_family(modelo)

    m = norm(modelo)

    for familia, patterns in FAMILY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, m):
                return familia

    return "outros"


def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    colunas = ["modelo_avaliado"] + list(JUDGES.values())
    faltando = [c for c in colunas if c not in df.columns]

    if faltando:
        raise ValueError(
            f"Colunas ausentes: {faltando}\n"
            f"Colunas disponíveis: {list(df.columns)}"
        )

    df = df.copy()

    for judge, col in JUDGES.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["familia_modelo"] = df["modelo_avaliado"].apply(detectar_familia_modelo)

    return df


# ============================================================
# ESTATÍSTICAS
# ============================================================

def media(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return float(x.mean())


def desvio(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if len(x) <= 1:
        return np.nan
    return float(x.std(ddof=1))


def bootstrap_ci(values, n_boot=5000, seed=42):
    """
    IC bootstrap 95% da média.
    Sem scipy, para ser fácil de rodar.
    """
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()

    if len(vals) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    means = []

    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        means.append(sample.mean())

    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cohen_d_indep(a, b):
    """
    Cohen's d para dois grupos independentes.
    """
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().to_numpy()

    if len(a) < 2 or len(b) < 2:
        return np.nan

    pooled = math.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )

    if pooled == 0:
        return np.nan

    return float((np.mean(a) - np.mean(b)) / pooled)


def analisar_juiz_self_vs_outros_modelos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada juiz:
    compara a média que ele dá para modelos da própria família
    contra a média que ele dá aos demais modelos.
    """
    linhas = []

    for judge, score_col in JUDGES.items():
        self_mask = df["familia_modelo"] == judge

        self_scores = df.loc[self_mask, score_col].dropna()
        outros_scores = df.loc[~self_mask, score_col].dropna()

        delta = media(self_scores) - media(outros_scores)

        ci_low, ci_high = bootstrap_ci(
            np.concatenate([
                self_scores.to_numpy() - media(outros_scores)
            ]) if len(self_scores) > 0 else [],
            n_boot=5000,
            seed=42,
        )

        linhas.append({
            "juiz": judge,
            "familia_propria": judge,
            "n_respostas_familia_propria": len(self_scores),
            "media_nota_familia_propria": media(self_scores),
            "dp_nota_familia_propria": desvio(self_scores),
            "n_respostas_demais_modelos": len(outros_scores),
            "media_nota_demais_modelos": media(outros_scores),
            "dp_nota_demais_modelos": desvio(outros_scores),
            "delta_self_menos_outros": delta,
            "cohen_d": cohen_d_indep(self_scores, outros_scores),
        })

    return pd.DataFrame(linhas)


def analisar_mesmo_modelo_julgador_familia_vs_outros_julgadores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada família de modelo avaliado:
    compara, nas mesmas respostas, a nota do juiz da própria família
    contra a média dos outros dois juízes.

    Exemplo:
    - Linhas em que familia_modelo == 'gpt'
    - compara gpt_judge_score_0_5 contra média de gemini_judge_score_0_5 e claude_judge_score_0_5
    """
    linhas = []

    for familia in ["gpt", "gemini", "claude"]:
        sub = df[df["familia_modelo"] == familia].copy()

        if len(sub) == 0:
            linhas.append({
                "familia_modelo_avaliado": familia,
                "modelos_incluidos": "",
                "n_respostas": 0,
                "media_juiz_mesma_familia": np.nan,
                "media_outros_julgadores": np.nan,
                "delta_mesma_familia_menos_outros_julgadores": np.nan,
                "ic95_delta_low": np.nan,
                "ic95_delta_high": np.nan,
                "proporcao_delta_positivo": np.nan,
            })
            continue

        own_col = JUDGES[familia]
        other_cols = [col for j, col in JUDGES.items() if j != familia]

        sub["nota_juiz_mesma_familia"] = sub[own_col]
        sub["media_outros_julgadores"] = sub[other_cols].mean(axis=1)
        sub["delta_pareado"] = (
            sub["nota_juiz_mesma_familia"] - sub["media_outros_julgadores"]
        )

        deltas = sub["delta_pareado"].dropna()
        ci_low, ci_high = bootstrap_ci(deltas, n_boot=5000, seed=42)

        modelos = ", ".join(sorted(sub["modelo_avaliado"].dropna().unique()))

        linhas.append({
            "familia_modelo_avaliado": familia,
            "modelos_incluidos": modelos,
            "n_respostas": len(sub),
            "media_juiz_mesma_familia": media(sub["nota_juiz_mesma_familia"]),
            "media_outros_julgadores": media(sub["media_outros_julgadores"]),
            "delta_mesma_familia_menos_outros_julgadores": media(deltas),
            "ic95_delta_low": ci_low,
            "ic95_delta_high": ci_high,
            "proporcao_delta_positivo": float((deltas > 0).mean()) if len(deltas) else np.nan,
            "proporcao_delta_zero": float((deltas == 0).mean()) if len(deltas) else np.nan,
            "proporcao_delta_negativo": float((deltas < 0).mean()) if len(deltas) else np.nan,
        })

    return pd.DataFrame(linhas)


def resumo_por_modelo_e_julgador(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []

    for modelo, sub in df.groupby("modelo_avaliado"):
        linha = {
            "modelo_avaliado": modelo,
            "familia_modelo": detectar_familia_modelo(modelo),
            "n": len(sub),
        }

        for judge, col in JUDGES.items():
            linha[f"media_{judge}_judge"] = media(sub[col])
            linha[f"dp_{judge}_judge"] = desvio(sub[col])

        linha["media_tres_julgadores"] = media(
            pd.concat([sub[col] for col in JUDGES.values()], ignore_index=True)
        )

        linhas.append(linha)

    return pd.DataFrame(linhas).sort_values("media_tres_julgadores", ascending=False)


# ============================================================
# GRÁFICOS
# ============================================================

def salvar_fig(fig, nome):
    png = OUT / f"{nome}.png"
    pdf = OUT / f"{nome}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Gerado: {png}")
    print(f"Gerado: {pdf}")


def grafico_self_vs_outros(tabela: pd.DataFrame):
    if tabela.empty:
        return

    judges = tabela["juiz"].tolist()
    x = np.arange(len(judges))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        x - width / 2,
        tabela["media_nota_familia_propria"],
        width,
        label="Modelo da própria família",
        color="#C62828",
        alpha=0.85,
    )

    ax.bar(
        x + width / 2,
        tabela["media_nota_demais_modelos"],
        width,
        label="Demais modelos",
        color="#1565C0",
        alpha=0.85,
    )

    for i, row in tabela.iterrows():
        ax.text(
            i,
            max(row["media_nota_familia_propria"], row["media_nota_demais_modelos"]) + 0.05,
            f"Δ={row['delta_self_menos_outros']:.2f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([j.upper() for j in judges])
    ax.set_ylabel("Nota média atribuída pelo juiz (0–5)")
    ax.set_ylim(0, 5.4)
    ax.set_title(
        "Circularidade: nota do juiz para sua própria família vs. demais modelos",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )

    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    salvar_fig(fig, "05_grafico_self_vs_outros")


def grafico_vies_pareado(tabela: pd.DataFrame):
    if tabela.empty:
        return

    tab = tabela[tabela["n_respostas"] > 0].copy()

    if tab.empty:
        print("Sem dados para gráfico pareado.")
        return

    x = np.arange(len(tab))
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axhline(0, color="#333333", linewidth=1)

    colors = [
        "#C62828" if v > 0 else "#2E7D32"
        for v in tab["delta_mesma_familia_menos_outros_julgadores"]
    ]

    bars = ax.bar(
        x,
        tab["delta_mesma_familia_menos_outros_julgadores"],
        color=colors,
        alpha=0.85,
    )

    for bar, row in zip(bars, tab.itertuples()):
        y = row.delta_mesma_familia_menos_outros_julgadores

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + (0.03 if y >= 0 else -0.08),
            f"{y:.2f}",
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tab["familia_modelo_avaliado"].str.upper())
    ax.set_ylabel("Diferença média de nota")
    ax.set_title(
        "Viés pareado: juiz da família menos média dos outros juízes",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    salvar_fig(fig, "06_grafico_vies_pareado")


# ============================================================
# RELATÓRIO TEXTUAL
# ============================================================

def gerar_relatorio(df, t1, t2, t3):
    lines = []

    lines.append("# Análise de circularidade juiz–respondente\n")
    lines.append(f"Total de respostas carregadas: **{len(df)}**.\n")
    lines.append("\n## Modelos encontrados por família\n")

    fam_counts = (
        df.groupby(["familia_modelo", "modelo_avaliado"])
        .size()
        .reset_index(name="n")
        .sort_values(["familia_modelo", "modelo_avaliado"])
    )

    for fam, sub in fam_counts.groupby("familia_modelo"):
        lines.append(f"\n### Família: {fam}\n")
        for row in sub.itertuples():
            lines.append(f"- {row.modelo_avaliado}: {row.n} respostas\n")

    lines.append("\n## 1. Juiz avaliando própria família vs demais modelos\n\n")
    lines.append(
        "Esta análise verifica se cada juiz atribui notas médias maiores aos modelos "
        "da sua própria família do que aos demais modelos.\n\n"
    )

    for row in t1.itertuples():
        lines.append(
            f"- **{row.juiz.upper()}**: própria família = "
            f"{row.media_nota_familia_propria:.3f} "
            f"(n={row.n_respostas_familia_propria}), demais modelos = "
            f"{row.media_nota_demais_modelos:.3f} "
            f"(n={row.n_respostas_demais_modelos}), "
            f"Δ = {row.delta_self_menos_outros:.3f}.\n"
        )

    lines.append("\n## 2. Comparação pareada no mesmo conjunto de respostas\n\n")
    lines.append(
        "Esta é a análise mais forte. Para cada família de modelo avaliado, compara-se "
        "a nota do juiz da mesma família com a média dos outros dois juízes nas mesmas respostas.\n\n"
    )

    for row in t2.itertuples():
        if row.n_respostas == 0:
            lines.append(
                f"- **{row.familia_modelo_avaliado.upper()}**: sem modelo dessa família nos dados.\n"
            )
            continue

        interpretacao = "possível favorecimento" if row.delta_mesma_familia_menos_outros_julgadores > 0 else "sem favorecimento médio"

        lines.append(
            f"- **{row.familia_modelo_avaliado.upper()}**: juiz da família = "
            f"{row.media_juiz_mesma_familia:.3f}, outros juízes = "
            f"{row.media_outros_julgadores:.3f}, Δ = "
            f"{row.delta_mesma_familia_menos_outros_julgadores:.3f} "
            f"(IC95 bootstrap: {row.ic95_delta_low:.3f} a {row.ic95_delta_high:.3f}); "
            f"{interpretacao}.\n"
        )

    lines.append("\n## Como reportar no artigo\n\n")
    lines.append(
        "Sugestão de texto: "
        "\"Para verificar possível circularidade juiz–respondente, comparamos a nota média atribuída "
        "por cada julgador aos modelos de sua própria família com as notas atribuídas aos demais modelos. "
        "Além disso, realizamos uma comparação pareada, contrastando, para as mesmas respostas, a nota do "
        "julgador da família do modelo com a média dos outros dois julgadores. Essa análise permite verificar "
        "se há evidência de favorecimento sistemático associado à família do modelo avaliado.\"\n"
    )

    (OUT / "04_resumo_textual.md").write_text("".join(lines), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    warnings.filterwarnings("ignore")

    if len(sys.argv) >= 2:
        input_path = Path(sys.argv[1])
    else:
        input_path = Path(".")

    print(f"Entrada: {input_path}")

    df = carregar_dados(input_path)
    df = preparar_dados(df)

    # Remove linhas sem nenhum score válido
    score_cols = list(JUDGES.values())
    df = df.dropna(subset=score_cols, how="all").copy()

    print("\nModelos avaliados encontrados:")
    print(df["modelo_avaliado"].value_counts().to_string())

    print("\nFamílias detectadas:")
    print(df["familia_modelo"].value_counts().to_string())

    t1 = analisar_juiz_self_vs_outros_modelos(df)
    t2 = analisar_mesmo_modelo_julgador_familia_vs_outros_julgadores(df)
    t3 = resumo_por_modelo_e_julgador(df)

    # Salvar saídas
    t1.to_csv(OUT / "01_self_family_vs_outros_modelos.csv", index=False, encoding="utf-8-sig")
    t2.to_csv(OUT / "02_mesmo_modelo_julgador_familia_vs_outros_julgadores.csv", index=False, encoding="utf-8-sig")
    t3.to_csv(OUT / "03_media_score_por_modelo_e_julgador.csv", index=False, encoding="utf-8-sig")

    gerar_relatorio(df, t1, t2, t3)
    grafico_self_vs_outros(t1)
    grafico_vies_pareado(t2)

    print("\nResumo 1 — juiz própria família vs demais modelos:")
    print(t1.to_string(index=False))

    print("\nResumo 2 — comparação pareada no mesmo modelo:")
    print(t2.to_string(index=False))

    print(f"\nArquivos salvos em: {OUT.resolve()}")


if __name__ == "__main__":
    main()
