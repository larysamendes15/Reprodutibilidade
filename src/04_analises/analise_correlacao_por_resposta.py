# -*- coding: utf-8 -*-
"""
Correlação entre métricas automáticas e pontuação do comitê Multi-LLM Judge
no NÍVEL DA RESPOSTA (n = 1.111), para substituir/complementar a Tabela 6.

Recalcula BLEU, ROUGE-L, BERTScore F1 e connectedness por resposta a partir
dos CSVs de julgadores (colunas Q, A, A_model, score_oficial_0_5) e
correlaciona com a nota consolidada do comitê.

Uso:
    pip install pandas numpy scipy rouge-score sacrebleu bert-score sentence-transformers
    python analise_correlacao_por_resposta.py [pasta_dos_csvs]

Padrão: results/judges_scores

ATENÇÃO: BERTScore e Sentence-BERT baixam modelos do HuggingFace na primeira
execução e levam vários minutos em CPU para 1.111 respostas. O script salva
um cache (metricas_por_resposta.csv) para não recalcular em execuções futuras.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

PASTA_PADRAO = "results/judges_scores"
CACHE = "results/metricas/metricas_por_resposta.csv"
SEED = 42

SCORE_COL = "score_oficial_0_5"


def carregar_respostas(pasta: str) -> pd.DataFrame:
    arquivos = sorted(glob.glob(os.path.join(pasta, "*.csv")))
    arquivos = [a for a in arquivos
                if "all_results" not in os.path.basename(a).lower()]
    if not arquivos:
        sys.exit(f"Nenhum CSV encontrado em '{pasta}'.")
    frames = []
    for a in arquivos:
        df = pd.read_csv(a)
        need = ["Q", "A", "A_model", "modelo_avaliado", SCORE_COL]
        faltando = [c for c in need if c not in df.columns]
        if faltando:
            sys.exit(f"{os.path.basename(a)}: colunas ausentes {faltando}")
        frames.append(df[need])
    out = pd.concat(frames, ignore_index=True)
    out["A"] = out["A"].fillna("").astype(str)
    out["A_model"] = out["A_model"].fillna("").astype(str)
    print(f"Carregadas {len(out)} respostas de {len(arquivos)} modelos.")
    return out


def calcular_metricas(df: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(SEED)

    # ---------------- ROUGE-L ----------------
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    print("Calculando ROUGE-L...")
    df["rouge_l"] = [
        rs.score(ref, hyp)["rougeL"].fmeasure
        for ref, hyp in zip(df["A"], df["A_model"])
    ]

    # ---------------- BLEU (sentença, com suavização) ----------------
    import sacrebleu
    print("Calculando BLEU...")
    df["bleu"] = [
        sacrebleu.sentence_bleu(hyp, [ref], smooth_method="exp").score / 100.0
        for ref, hyp in zip(df["A"], df["A_model"])
    ]

    # ---------------- BERTScore F1 ----------------
    from bert_score import score as bertscore
    print("Calculando BERTScore F1 (pode demorar)...")
    _, _, f1 = bertscore(
        cands=df["A_model"].tolist(),
        refs=df["A"].tolist(),
        lang="pt",
        verbose=True,
        batch_size=16,
    )
    df["bert_f1"] = f1.numpy()

    # ---------------- Connectedness (Sentence-BERT, cosseno) ----------------
    from sentence_transformers import SentenceTransformer
    print("Calculando connectedness (pode demorar)...")
    st = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    emb_ref = st.encode(df["A"].tolist(), batch_size=32,
                        normalize_embeddings=True, show_progress_bar=True)
    emb_hyp = st.encode(df["A_model"].tolist(), batch_size=32,
                        normalize_embeddings=True, show_progress_bar=True)
    df["connectedness"] = (emb_ref * emb_hyp).sum(axis=1)

    return df


def main():
    pasta = sys.argv[1] if len(sys.argv) > 1 else PASTA_PADRAO

    if os.path.exists(CACHE):
        print(f"Usando cache {CACHE} (apague-o para recalcular).")
        df = pd.read_csv(CACHE)
    else:
        df = carregar_respostas(pasta)
        df = calcular_metricas(df)
        df.to_csv(CACHE, index=False)
        print(f"Cache salvo em {CACHE}.")

    metricas = ["bleu", "rouge_l", "bert_f1", "connectedness"]
    nomes = {"bleu": "BLEU", "rouge_l": "ROUGE-L",
             "bert_f1": "BERTScore F1", "connectedness": "Connectedness"}

    # ---------- Sanidade: médias por modelo vs Tabela 5 ----------
    print("\nMédias por modelo (compare com a Tabela 5 do artigo):")
    print(df.groupby("modelo_avaliado")[metricas].mean().round(4).to_string())

    # ---------- Correlação no nível da resposta (n = 1.111) ----------
    y = df[SCORE_COL].astype(float)
    print("\n========== Correlação por RESPOSTA (n = %d) ==========" % len(df))
    print(f"{'Métrica':15s} {'Pearson':>9s} {'p-valor':>12s} {'Spearman':>9s} {'p-valor':>12s}")
    for m in metricas:
        x = df[m].astype(float)
        rp, pp = stats.pearsonr(x, y)
        rsq, ps = stats.spearmanr(x, y)
        print(f"{nomes[m]:15s} {rp:9.3f} {pp:12.2e} {rsq:9.3f} {ps:12.2e}")

    # ---------- Correlação por modelo (n = 11), para comparação ----------
    ag = df.groupby("modelo_avaliado").agg(
        {**{m: "mean" for m in metricas}, SCORE_COL: "mean"})
    print("\n========== Correlação por MODELO (n = %d, como na Tabela 6) ==========" % len(ag))
    print(f"{'Métrica':15s} {'Pearson':>9s} {'p-valor':>9s} {'Spearman':>9s} {'p-valor':>9s}")
    for m in metricas:
        rp, pp = stats.pearsonr(ag[m], ag[SCORE_COL])
        rsq, ps = stats.spearmanr(ag[m], ag[SCORE_COL])
        print(f"{nomes[m]:15s} {rp:9.3f} {pp:9.3f} {rsq:9.3f} {ps:9.3f}")


if __name__ == "__main__":
    main()
