# -*- coding: utf-8 -*-
"""
Gera as versões reformuladas das Figuras 2 e 4 do artigo.

Figura 2 (calibracao_julgadores.pdf): dot plot com as médias de GPT, Gemini
e Claude por modelo respondente + linha do comitê. Lê os CSVs de julgadores.

Figura 4 (desalinhamento_ranking.pdf): slope chart ligando o ranking
jurídico (Tabela 2) ao ranking médio das métricas (Tabela 5). Dados embutidos.

Uso (a partir da raiz do repositório):
    pip install pandas numpy matplotlib
    python src/05_graficos/graficos_calibracao_e_ranking.py [pasta_dos_csvs]
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PASTA_PADRAO = "results/scores_julgadores"
PASTA_SAIDA = Path("results/figuras")
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

# Rótulos curtos, na ordem do ranking jurídico (Tabela 2)
ORDEM = [
    ("gpt-5.1", "GPT-5.1"),
    ("claude-opus-4-6", "Claude Opus 4.6"),
    ("glm-5.1", "GLM-5.1"),
    ("deepseek-v4-flash", "DeepSeek V4 Flash"),
    ("gemma-4-31b-it", "Gemma 4 31B"),
    ("qwen3.5-122b-a10b", "Qwen3.5 122B"),
    ("gemini-3-pro-preview", "Gemini 3 Pro"),
    ("mixtral-8x22b-instruct-v0.1", "Mixtral 8x22B"),
    ("openai-gpt-oss-120b", "GPT-OSS 120B"),
    ("qwen2-72B-Instruct", "Qwen2 72B"),
    ("llama-3.3-70b-instruct", "Llama 3.3 70B"),
]

# Tabela 5 do artigo: ROUGE-L, BLEU, BERTScore F1, Connectedness
METRICAS_T5 = {
    "gpt-5.1":                     (0.3625, 0.1531, 0.7735, 0.8248),
    "claude-opus-4-6":             (0.3016, 0.0956, 0.7327, 0.7837),
    "glm-5.1":                     (0.4295, 0.2993, 0.7980, 0.8225),
    "deepseek-v4-flash":           (0.4267, 0.2529, 0.7980, 0.8111),
    "gemma-4-31b-it":              (0.4404, 0.2813, 0.8041, 0.8159),
    "qwen3.5-122b-a10b":           (0.4258, 0.2584, 0.7967, 0.8251),
    "gemini-3-pro-preview":        (0.4111, 0.2765, 0.7973, 0.8165),
    "mixtral-8x22b-instruct-v0.1": (0.4490, 0.3090, 0.7999, 0.8305),
    "openai-gpt-oss-120b":         (0.3393, 0.1203, 0.7668, 0.8204),
    "qwen2-72B-Instruct":          (0.4280, 0.2946, 0.7998, 0.8324),
    "llama-3.3-70b-instruct":      (0.4254, 0.3131, 0.7931, 0.8012),
}

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})


def chave(modelo_csv: str) -> str:
    """Normaliza o nome vindo do CSV para a chave curta usada em ORDEM."""
    m = modelo_csv.lower()
    for k, _ in ORDEM:
        nucleo = k.lower().replace("-", "").replace(".", "").replace("_", "")
        alvo = m.replace("-", "").replace(".", "").replace("_", "")
        if nucleo in alvo or alvo.endswith(nucleo):
            return k
    return modelo_csv


def figura2(pasta: str):
    arquivos = [a for a in sorted(glob.glob(os.path.join(pasta, "*.csv")))
                if "all_results" not in os.path.basename(a).lower()]
    medias = {}
    for a in arquivos:
        df = pd.read_csv(a)
        k = chave(str(df["modelo_avaliado"].iloc[0]))
        medias[k] = (df["gpt_judge_score_0_5"].mean(),
                     df["gemini_judge_score_0_5"].mean(),
                     df["claude_judge_score_0_5"].mean(),
                     df["score_oficial_0_5"].mean())

    labels = [r for _, r in ORDEM]
    x = np.arange(len(ORDEM))
    gpt = [medias[k][0] for k, _ in ORDEM]
    gem = [medias[k][1] for k, _ in ORDEM]
    cla = [medias[k][2] for k, _ in ORDEM]
    com = [medias[k][3] for k, _ in ORDEM]

    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    for xi in x:
        ax.plot([xi, xi], [min(gpt[xi], gem[xi], cla[xi]),
                           max(gpt[xi], gem[xi], cla[xi])],
                color="0.85", lw=3, zorder=1)
    ax.scatter(x, gem, marker="^", s=28, label="Gemini", color="#e08214", zorder=3)
    ax.scatter(x, gpt, marker="o", s=28, label="GPT", color="#2166ac", zorder=3)
    ax.scatter(x, cla, marker="s", s=28, label="Claude", color="#1a9850", zorder=3)
    ax.plot(x, com, color="#b2182b", lw=1.2, ls="--", marker=".", ms=6,
            label="Comitê (média)", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("Pontuação média (0–5)")
    ax.set_ylim(3.4, 4.7)
    ax.legend(ncol=4, fontsize=7.5, frameon=False, loc="upper right")
    ax.grid(axis="y", color="0.9", lw=0.6)
    fig.tight_layout()
    saida = PASTA_SAIDA / "calibracao_julgadores.pdf"
    fig.savefig(saida, bbox_inches="tight")
    print(f"Gerado: {saida}")


def figura4():
    labels = [r for _, r in ORDEM]
    n = len(ORDEM)
    rank_jur = {k: i + 1 for i, (k, _) in enumerate(ORDEM)}

    vals = np.array([METRICAS_T5[k] for k, _ in ORDEM])
    ranks_por_metrica = np.zeros_like(vals)
    for j in range(vals.shape[1]):
        ordem = np.argsort(-vals[:, j])           # maior valor = rank 1
        for pos, idx in enumerate(ordem):
            ranks_por_metrica[idx, j] = pos + 1
    rank_met_medio = ranks_por_metrica.mean(axis=1)
    # converte média de ranks em ordenação 1..11
    rank_met = {ORDEM[i][0]: r + 1
                for r, i in enumerate(np.argsort(rank_met_medio))}

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for i, (k, rotulo) in enumerate(ORDEM):
        y0, y1 = rank_jur[k], rank_met[k]
        piora = y1 - y0
        cor = "#b2182b" if piora < -2 else ("#2166ac" if piora > 2 else "0.55")
        ax.plot([0, 1], [y0, y1], color=cor, lw=1.4, zorder=2)
        ax.scatter([0, 1], [y0, y1], color=cor, s=18, zorder=3)
        ax.text(-0.05, y0, f"{rotulo}  {y0}", ha="right", va="center", fontsize=7.5)
        ax.text(1.05, y1, f"{y1}", ha="left", va="center", fontsize=7.5)

    ax.set_xlim(-0.75, 1.25)
    ax.set_ylim(n + 0.6, 0.4)                     # rank 1 no topo
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Ranking jurídico\n(Multi-LLM Judge)",
                        "Ranking médio\n(métricas automáticas)"], fontsize=8)
    ax.set_yticks([])
    for lado in ("left", "bottom"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    saida = PASTA_SAIDA / "desalinhamento_ranking.pdf"
    fig.savefig(saida, bbox_inches="tight")
    print(f"Gerado: {saida}")


if __name__ == "__main__":
    pasta = sys.argv[1] if len(sys.argv) > 1 else PASTA_PADRAO
    figura2(pasta)
    figura4()
