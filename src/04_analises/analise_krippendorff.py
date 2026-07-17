# -*- coding: utf-8 -*-
"""
Krippendorff's alpha (ordinal) por modelo respondente e geral.
Substitui a coluna "Fleiss' K" da Tabela 3 do artigo.

Uso:
    pip install krippendorff pandas numpy
    python analise_krippendorff.py [csv_consolidado_ou_pasta]

Sem argumento, usa a pasta results/scores_julgadores
(concatena os csvs individuais por modelo, ignorando all_results*).

Definições alinhadas ao artigo:
- Unanimidade = os 3 julgadores atribuem a MESMA CATEGORIA
  (CORRETO / PARCIALMENTE_CORRETO / INCORRETO), como na Tabela 3.
- Desvio médio = média do desvio-padrão amostral (ddof=1) das 3 notas.
- Alpha ordinal = Krippendorff com value_domain 0..5.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import krippendorff

CAMINHO_PADRAO = "results/scores_julgadores"

SCORE_COLS = ["gpt_judge_score_0_5", "gemini_judge_score_0_5",
              "claude_judge_score_0_5"]
CAT_COLS = ["gpt_judge_categoria", "gemini_judge_categoria",
            "claude_judge_categoria"]
COL_MODELO = "modelo_avaliado"


def carregar(caminho: str) -> pd.DataFrame:
    if os.path.isdir(caminho):
        arquivos = sorted(glob.glob(os.path.join(caminho, "*.csv")))
        arquivos_modelo = [a for a in arquivos
                           if "all_results" not in os.path.basename(a).lower()]
        frames = [pd.read_csv(a) for a in (arquivos_modelo or arquivos)]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(caminho)

    faltando = [c for c in SCORE_COLS + CAT_COLS + [COL_MODELO]
                if c not in df.columns]
    if faltando:
        sys.exit(f"Colunas ausentes: {faltando}\nDisponíveis: {list(df.columns)}")
    return df


def alpha_ordinal(df: pd.DataFrame) -> float:
    notas = df[SCORE_COLS].to_numpy(dtype=float).T  # (juizes, itens)
    return krippendorff.alpha(
        reliability_data=notas,
        level_of_measurement="ordinal",
        value_domain=[0, 1, 2, 3, 4, 5],
    )


def metricas(df: pd.DataFrame):
    unan_cat = (df[CAT_COLS].nunique(axis=1) == 1).mean() * 100
    notas = df[SCORE_COLS].to_numpy(dtype=float)
    desvio = np.nanstd(notas, axis=1, ddof=1).mean()
    return alpha_ordinal(df), unan_cat, desvio, len(df)


def main():
    caminho = sys.argv[1] if len(sys.argv) > 1 else CAMINHO_PADRAO
    df = carregar(caminho)

    print(f"Total de respostas: {len(df)}  |  "
          f"Modelos: {df[COL_MODELO].nunique()}\n")

    linhas = []
    for modelo, grupo in df.groupby(COL_MODELO):
        a, u, d, n = metricas(grupo)
        linhas.append((modelo, n, u, d, a))

    print("============= TABELA 3 (versão Krippendorff ordinal) =============")
    print(f"{'Modelo':32s} {'n':>4s} {'Unanim.':>9s} {'Desv.méd':>9s} {'Alpha ord.':>11s}")
    for modelo, n, u, d, a in sorted(linhas, key=lambda x: -x[2]):
        print(f"{modelo:32s} {n:4d} {u:8.2f}% {d:9.3f} {a:11.3f}")

    a, u, d, n = metricas(df)
    print("-" * 72)
    print(f"{'GERAL':32s} {n:4d} {u:8.2f}% {d:9.3f} {a:11.3f}")

    print("\nInterpretação (Krippendorff): alpha >= 0,80 confiável; "
          "0,667-0,80 aceitável para conclusões provisórias.")
    print("Validação: para qwen2-72B-Instruct devem sair 70,30% e 0,490, "
          "batendo com a Tabela 3 atual do artigo.")


if __name__ == "__main__":
    main()