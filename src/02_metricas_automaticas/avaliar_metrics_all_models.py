import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_FILE = "results/legacy/QAG/aberto/metrics_all_models.csv"
OUTPUT_DIR = Path("results/metricas")
OUTPUT_DIR.mkdir(exist_ok=True)

METRIC_COLS = ["rouge-l", "bleu", "bert_f1", "connectedness"]

WEIGHTS = {
    "rouge-l": 1.0,
    "bleu": 1.0,
    "bert_f1": 1.0,
    "connectedness": 1.0,
}


df = pd.read_csv(INPUT_FILE)

missing_cols = [col for col in ["modelo"] + METRIC_COLS if col not in df.columns]

if missing_cols:
    raise ValueError(f"Colunas ausentes no CSV: {missing_cols}")

df_eval = df[["modelo"] + METRIC_COLS].copy()


for metric in METRIC_COLS:
    df_eval[f"rank_{metric}"] = df_eval[metric].rank(
        ascending=False,
        method="min"
    ).astype(int)

total_weight = sum(WEIGHTS.values())

df_eval["score_geral"] = sum(
    df_eval[metric] * WEIGHTS[metric]
    for metric in METRIC_COLS
) / total_weight

df_eval["score_geral_100"] = df_eval["score_geral"] * 100

df_eval["rank_geral"] = df_eval["score_geral"].rank(
    ascending=False,
    method="min"
).astype(int)

df_eval = df_eval.sort_values(
    by="score_geral",
    ascending=False
).reset_index(drop=True)


# =========================
# Salvando resultados
# =========================

ranking_file = OUTPUT_DIR / "ranking_geral_metricas.csv"
df_eval.to_csv(ranking_file, index=False)

print("\n=== Ranking geral dos modelos ===\n")

cols_to_show = [
    "rank_geral",
    "modelo",
    "rouge-l",
    "bleu",
    "bert_f1",
    "connectedness",
    "score_geral_100",
]

print(
    df_eval[cols_to_show]
    .to_string(index=False)
)

print(f"\nArquivo salvo em: {ranking_file}")


# =========================
# Melhor modelo por métrica
# =========================

print("\n=== Melhor modelo por métrica ===\n")

best_rows = []

for metric in METRIC_COLS:
    best = df_eval.sort_values(by=metric, ascending=False).iloc[0]
    best_rows.append({
        "metrica": metric,
        "melhor_modelo": best["modelo"],
        "valor": best[metric]
    })

best_df = pd.DataFrame(best_rows)

best_file = OUTPUT_DIR / "melhor_modelo_por_metrica.csv"
best_df.to_csv(best_file, index=False)

print(best_df.to_string(index=False))
print(f"\nArquivo salvo em: {best_file}")


# =========================
# Gráfico 1: Ranking geral
# =========================

plt.figure(figsize=(12, 7))

plt.barh(
    df_eval["modelo"],
    df_eval["score_geral_100"]
)

plt.xlabel("Score geral médio (0 a 100)")
plt.ylabel("Modelo")
plt.title("Ranking geral dos modelos")
plt.gca().invert_yaxis()
plt.grid(axis="x", linestyle="--", alpha=0.6)

plt.tight_layout()

ranking_chart = OUTPUT_DIR / "ranking_geral.png"
plt.savefig(ranking_chart, dpi=300)
plt.show()

print(f"Gráfico salvo em: {ranking_chart}")


# =========================
# Gráfico 2: Comparação por métrica
# =========================

df_plot = df_eval.set_index("modelo")[METRIC_COLS]

plt.figure(figsize=(14, 7))

df_plot.plot(kind="bar", figsize=(14, 7))

plt.title("Comparação dos modelos por métrica")
plt.xlabel("Modelo")
plt.ylabel("Valor da métrica")
plt.xticks(rotation=75, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()

comparison_chart = OUTPUT_DIR / "comparacao_metricas.png"
plt.savefig(comparison_chart, dpi=300)
plt.show()

print(f"Gráfico salvo em: {comparison_chart}")


# =========================
# Tabela resumida para artigo/TCC
# =========================

summary = df_eval[
    [
        "modelo",
        "rouge-l",
        "bleu",
        "bert_f1",
        "connectedness",
        "score_geral_100",
        "rank_geral"
    ]
].copy()

summary = summary.round({
    "rouge-l": 4,
    "bleu": 4,
    "bert_f1": 4,
    "connectedness": 4,
    "score_geral_100": 2
})

summary_file = OUTPUT_DIR / "tabela_resumo_artigo.csv"
summary.to_csv(summary_file, index=False)

print(f"\nTabela resumida salva em: {summary_file}")