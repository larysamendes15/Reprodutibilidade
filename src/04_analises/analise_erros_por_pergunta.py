#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analisa_erros_por_pergunta.py

Analisa os CSVs detalhados de julgamento por modelo e identifica:
- perguntas mais erradas;
- perguntas sem nenhuma resposta correta;
- concentração dos erros;
- resumo por modelo;
- perguntas incorretas/parciais por modelo;
- matriz modelo x pergunta.

IMPORTANTE:
Este script espera 11 modelos. Se algum CSV detalhado estiver faltando,
ele interrompe a execução para evitar gerar gráficos errados com apenas 10 modelos.

Entrada esperada:
arquivos do tipo:
    *_QAG*_julgadores_score*.csv

Exemplos:
    gpt-5.1_QAG_101_3_julgadores_score_batch.csv
    claude-opus-4-6_QAG_101_3_julgadores_score_batch.csv
    qwen2-72B-Instruct_old_QAG_101_3_julgadores_score_batch.csv
    deepseek-ai_deepseek-v4-flash_QAG_julgadores_score.csv

Como rodar:
    python analisa_erros_por_pergunta.py \
      --input-dir results/legacy/result_judges_score \
      --out-dir results/erros

Depois gere os gráficos novamente:
    python graficos_explicativos_erros.py \
      --input-dir results/erros \
      --out-dir results/figuras \
      --top-n 14
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configurações
# ============================================================

GLOB_PATTERN = "*_QAG*_julgadores_score*.csv"

COL_PERGUNTA = "Q"
COL_RESPOSTA_OFICIAL = "A"
COL_RESPOSTA_MODELO = "A_model"

COL_MODELO = "modelo_avaliado"
COL_RESULTADO = "resultado_oficial"
COL_SCORE = "score_oficial_0_5"
COL_DESVIO = "desvio_score_entre_julgadores"

EXPECTED_MODELS = [
    "gpt-5.1",
    "claude-opus-4-6",
    "z-ai_glm-5.1",
    "deepseek-ai_deepseek-v4-flash",
    "google_gemma-4-31b-it",
    "qwen_qwen3.5-122b-a10b",
    "gemini-3-pro-preview",
    "mistralai_mixtral-8x22b-instruct-v0.1",
    "qwen2-72B-Instruct",
    "openai_gpt-oss-120b",
    "meta_llama-3.3-70b-instruct",
]


# ============================================================
# Utilitários
# ============================================================

def inferir_modelo_pelo_nome(path: Path) -> str:
    """
    Exemplo:
    gpt-5.1_QAG_101_3_julgadores_score_batch.csv -> gpt-5.1
    deepseek-ai_deepseek-v4-flash_QAG_julgadores_score.csv -> deepseek-ai_deepseek-v4-flash
    qwen2-72B-Instruct_old_QAG_101_3_julgadores_score_batch.csv -> qwen2-72B-Instruct_old
    """
    nome = path.stem
    nome = re.sub(r"_QAG.*$", "", nome)
    return nome


def normalizar_nome_modelo(nome: str) -> str:
    """
    Normaliza pequenas variações de nome para evitar duplicidade.
    """
    nome = str(nome).strip()

    aliases = {
        "deepseek-v4-flash": "deepseek-ai_deepseek-v4-flash",
        "google/gemma-4-31b-it": "google_gemma-4-31b-it",
        "gemma-4-31b-it": "google_gemma-4-31b-it",
        "meta/llama-3.3-70b-instruct": "meta_llama-3.3-70b-instruct",
        "llama-3.3-70b-instruct": "meta_llama-3.3-70b-instruct",
        "mistralai/mixtral-8x22b-instruct-v0.1": "mistralai_mixtral-8x22b-instruct-v0.1",
        "mixtral-8x22b-instruct-v0.1": "mistralai_mixtral-8x22b-instruct-v0.1",
        "openai/gpt-oss-120b": "openai_gpt-oss-120b",
        "gpt-oss-120b": "openai_gpt-oss-120b",
        "qwen/qwen3.5-122b-a10b": "qwen_qwen3.5-122b-a10b",
        "qwen3.5-122b-a10b": "qwen_qwen3.5-122b-a10b",
        "z-ai/glm-5.1": "z-ai_glm-5.1",
        "glm-5.1": "z-ai_glm-5.1",
        "qwen2-72b-instruct-old": "qwen2-72B-Instruct_old",
        "qwen2-72b-instruct_old": "qwen2-72B-Instruct_old",
        "qwen2-72B-Instruct-old": "qwen2-72B-Instruct_old",
    }

    return aliases.get(nome, aliases.get(nome.lower(), nome))


def normalizar_pergunta(q: str) -> str:
    """
    Normaliza pergunta para agrupar pequenas variações textuais.
    """
    q = str(q)
    q = re.sub(r"\s+", " ", q).strip()

    q = re.sub(
        r"elaboração da escrituração contábil citada na questão anterior,?",
        "elaboração da escrituração contábil,",
        q,
        flags=re.IGNORECASE,
    )

    q = re.sub(r"\s+", " ", q).strip().lower()

    return q


def encurtar_texto(txt: str, n: int = 120) -> str:
    txt = re.sub(r"\s+", " ", str(txt)).strip()

    if len(txt) <= n:
        return txt

    return txt[: n - 3] + "..."


def classificar_por_score(score: float) -> str:
    """
    Regra usada no seu experimento:
    - score >= 4: correto
    - score == 3: parcialmente correto
    - score < 3: incorreto
    """
    try:
        score = float(score)
    except Exception:
        return "ERRO"

    if score >= 4:
        return "CORRETO"

    if score == 3:
        return "PARCIALMENTE_CORRETO"

    return "INCORRETO"


def garantir_colunas(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    obrigatorias = [
        COL_PERGUNTA,
        COL_RESPOSTA_OFICIAL,
        COL_SCORE,
    ]

    faltando = [c for c in obrigatorias if c not in df.columns]

    if faltando:
        raise ValueError(
            f"Arquivo {path.name} sem colunas obrigatórias: {faltando}\n"
            f"Colunas encontradas: {list(df.columns)}"
        )

    modelo_inferido = inferir_modelo_pelo_nome(path)

    if COL_MODELO not in df.columns:
        df[COL_MODELO] = modelo_inferido

    df[COL_MODELO] = df[COL_MODELO].fillna(modelo_inferido)
    df[COL_MODELO] = df[COL_MODELO].apply(normalizar_nome_modelo)

    if COL_RESULTADO not in df.columns:
        df[COL_RESULTADO] = df[COL_SCORE].apply(classificar_por_score)

    if COL_DESVIO not in df.columns:
        df[COL_DESVIO] = np.nan

    if COL_RESPOSTA_MODELO not in df.columns:
        df[COL_RESPOSTA_MODELO] = ""

    return df


def carregar_arquivos(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob(GLOB_PATTERN))

    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo encontrado em {input_dir.resolve()} "
            f"com padrão {GLOB_PATTERN}"
        )

    print(f"\nArquivos encontrados: {len(files)}")
    for f in files:
        print(" -", f.name)

    dfs = []

    for path in files:
        df = pd.read_csv(path)
        df = garantir_colunas(df, path)

        df["arquivo_origem"] = path.name
        df["pergunta_norm"] = df[COL_PERGUNTA].apply(normalizar_pergunta)

        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    return all_df


def validar_modelos(df: pd.DataFrame, expected_models: list[str]) -> None:
    modelos_carregados = sorted(df[COL_MODELO].dropna().unique().tolist())
    esperados = sorted(expected_models)

    faltando = sorted(set(esperados) - set(modelos_carregados))
    extras = sorted(set(modelos_carregados) - set(esperados))

    print("\n=== MODELOS CARREGADOS ===")
    for m in modelos_carregados:
        qtd = int((df[COL_MODELO] == m).sum())
        print(f" - {m}: {qtd} linhas")

    print(f"\nTotal de modelos carregados: {len(modelos_carregados)}")

    if extras:
        print("\nModelos extras encontrados:")
        for m in extras:
            print(f" - {m}")

    if faltando:
        print("\nModelos esperados que estão faltando:")
        for m in faltando:
            print(f" - {m}")

        raise RuntimeError(
            f"Foram carregados {len(modelos_carregados)} modelos, "
            f"mas eram esperados {len(expected_models)}.\n"
            "Corrija a pasta de entrada antes de gerar a análise. "
            "Provavelmente falta algum CSV detalhado, como o qwen2-72B-Instruct_old."
        )

    if len(modelos_carregados) != len(expected_models):
        raise RuntimeError(
            f"Foram carregados {len(modelos_carregados)} modelos, "
            f"mas eram esperados {len(expected_models)}."
        )

    print("\nValidação OK: todos os 11 modelos foram carregados.")


def validar_101_perguntas_por_modelo(df: pd.DataFrame) -> None:
    problemas = []

    for modelo, g in df.groupby(COL_MODELO):
        n_linhas = len(g)
        n_perguntas = g["pergunta_norm"].nunique()

        if n_linhas != 101 or n_perguntas != 101:
            problemas.append(
                {
                    "modelo": modelo,
                    "n_linhas": n_linhas,
                    "n_perguntas_unicas": n_perguntas,
                }
            )

    if problemas:
        print("\nProblemas de quantidade de perguntas por modelo:")
        for p in problemas:
            print(
                f" - {p['modelo']}: "
                f"{p['n_linhas']} linhas, "
                f"{p['n_perguntas_unicas']} perguntas únicas"
            )

        raise RuntimeError(
            "Algum modelo não possui exatamente 101 perguntas. "
            "Verifique se os CSVs detalhados estão completos."
        )

    print("Validação OK: cada modelo possui 101 perguntas.")


# ============================================================
# Análises
# ============================================================

def resumo_por_modelo(df: pd.DataFrame) -> pd.DataFrame:
    resumo = (
        df.groupby(COL_MODELO)
        .agg(
            n_total=(COL_PERGUNTA, "count"),
            n_correto=(COL_RESULTADO, lambda s: (s == "CORRETO").sum()),
            n_parcialmente_correto=(COL_RESULTADO, lambda s: (s == "PARCIALMENTE_CORRETO").sum()),
            n_incorreto=(COL_RESULTADO, lambda s: (s == "INCORRETO").sum()),
            score_medio_0_5=(COL_SCORE, "mean"),
            desvio_medio_entre_julgadores=(COL_DESVIO, "mean"),
        )
        .reset_index()
    )

    resumo["score_normalizado"] = resumo["score_medio_0_5"] / 5
    resumo["taxa_erro"] = resumo["n_incorreto"] / resumo["n_total"]
    resumo["taxa_nao_correto"] = (
        resumo["n_incorreto"] + resumo["n_parcialmente_correto"]
    ) / resumo["n_total"]

    resumo = resumo.sort_values(
        ["score_medio_0_5", "n_correto", "n_incorreto"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    resumo.insert(0, "posicao", np.arange(1, len(resumo) + 1))

    return resumo


def ranking_perguntas(df: pd.DataFrame) -> pd.DataFrame:
    base = (
        df.groupby("pergunta_norm")
        .agg(
            pergunta=(COL_PERGUNTA, "first"),
            resposta_oficial=(COL_RESPOSTA_OFICIAL, "first"),
            n_modelos=(COL_MODELO, "nunique"),
            n_correto=(COL_RESULTADO, lambda s: (s == "CORRETO").sum()),
            n_parcialmente_correto=(COL_RESULTADO, lambda s: (s == "PARCIALMENTE_CORRETO").sum()),
            n_incorreto=(COL_RESULTADO, lambda s: (s == "INCORRETO").sum()),
            score_medio_0_5=(COL_SCORE, "mean"),
            score_min=(COL_SCORE, "min"),
            score_max=(COL_SCORE, "max"),
            desvio_medio_entre_julgadores=(COL_DESVIO, "mean"),
        )
        .reset_index()
    )

    modelos_incorretos = (
        df[df[COL_RESULTADO] == "INCORRETO"]
        .groupby("pergunta_norm")[COL_MODELO]
        .apply(lambda s: ", ".join(sorted(set(s))))
        .reset_index(name="modelos_incorretos")
    )

    modelos_parciais = (
        df[df[COL_RESULTADO] == "PARCIALMENTE_CORRETO"]
        .groupby("pergunta_norm")[COL_MODELO]
        .apply(lambda s: ", ".join(sorted(set(s))))
        .reset_index(name="modelos_parciais")
    )

    modelos_corretos = (
        df[df[COL_RESULTADO] == "CORRETO"]
        .groupby("pergunta_norm")[COL_MODELO]
        .apply(lambda s: ", ".join(sorted(set(s))))
        .reset_index(name="modelos_corretos")
    )

    base = base.merge(modelos_incorretos, on="pergunta_norm", how="left")
    base = base.merge(modelos_parciais, on="pergunta_norm", how="left")
    base = base.merge(modelos_corretos, on="pergunta_norm", how="left")

    for col in ["modelos_incorretos", "modelos_parciais", "modelos_corretos"]:
        base[col] = base[col].fillna("")

    base["taxa_erro"] = base["n_incorreto"] / base["n_modelos"]
    base["taxa_nao_correto"] = (
        base["n_incorreto"] + base["n_parcialmente_correto"]
    ) / base["n_modelos"]

    base = base.sort_values(
        ["n_incorreto", "n_parcialmente_correto", "score_medio_0_5"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    base.insert(0, "rank_dificuldade", np.arange(1, len(base) + 1))

    return base


def top_perguntas_por_modelo(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    partes = []

    ordem_categoria = {
        "INCORRETO": 0,
        "PARCIALMENTE_CORRETO": 1,
        "CORRETO": 2,
    }

    df2 = df.copy()
    df2["ordem_categoria"] = df2[COL_RESULTADO].map(ordem_categoria).fillna(9)

    colunas_preferidas = [
        COL_MODELO,
        COL_PERGUNTA,
        COL_RESPOSTA_OFICIAL,
        COL_RESPOSTA_MODELO,
        COL_RESULTADO,
        COL_SCORE,
        COL_DESVIO,
        "gpt_judge_score_0_5",
        "gemini_judge_score_0_5",
        "claude_judge_score_0_5",
        "gpt_judge_raciocinio",
        "gemini_judge_raciocinio",
        "claude_judge_raciocinio",
    ]

    colunas_existentes = [c for c in colunas_preferidas if c in df2.columns]

    for modelo, g in df2.groupby(COL_MODELO):
        problemas = g[
            g[COL_RESULTADO].isin(["INCORRETO", "PARCIALMENTE_CORRETO"])
        ].copy()

        problemas = problemas.sort_values(
            ["ordem_categoria", COL_SCORE, COL_DESVIO],
            ascending=[True, True, False],
        ).head(top_n)

        partes.append(problemas[colunas_existentes].copy())

    if not partes:
        return pd.DataFrame()

    return pd.concat(partes, ignore_index=True)


def matriz_modelo_pergunta(df: pd.DataFrame) -> pd.DataFrame:
    mapa = {
        "CORRETO": "C",
        "PARCIALMENTE_CORRETO": "P",
        "INCORRETO": "I",
    }

    temp = df.copy()
    temp["categoria_curta"] = temp[COL_RESULTADO].map(mapa).fillna("?")

    matriz = temp.pivot_table(
        index="pergunta_norm",
        columns=COL_MODELO,
        values="categoria_curta",
        aggfunc="first",
    ).reset_index()

    perguntas = (
        df.groupby("pergunta_norm")[COL_PERGUNTA]
        .first()
        .reset_index(name="pergunta")
    )

    matriz = perguntas.merge(matriz, on="pergunta_norm", how="left")

    return matriz


def resumo_concentracao(ranking: pd.DataFrame, total_erros: int) -> pd.DataFrame:
    linhas = []

    for n in [5, 10, 14, 20]:
        n_real = min(n, len(ranking))
        erros = int(ranking.head(n_real)["n_incorreto"].sum())
        perc = erros / total_erros if total_erros else 0

        linhas.append(
            {
                "top_n_perguntas": n_real,
                "erros_acumulados": erros,
                "percentual_dos_erros": perc,
            }
        )

    return pd.DataFrame(linhas)


# ============================================================
# Impressão
# ============================================================

def imprimir_resumo(
    df: pd.DataFrame,
    resumo_modelos: pd.DataFrame,
    ranking: pd.DataFrame,
    concentracao: pd.DataFrame,
) -> None:
    total = len(df)

    n_c = int((df[COL_RESULTADO] == "CORRETO").sum())
    n_p = int((df[COL_RESULTADO] == "PARCIALMENTE_CORRETO").sum())
    n_i = int((df[COL_RESULTADO] == "INCORRETO").sum())

    print("\n=== RESUMO GERAL ===")
    print(f"Total de avaliações: {total}")
    print(f"Corretas: {n_c}")
    print(f"Parcialmente corretas: {n_p}")
    print(f"Incorretas: {n_i}")
    print(f"Perguntas únicas: {ranking.shape[0]}")
    print(f"Modelos únicos: {df[COL_MODELO].nunique()}")

    print("\n=== CONCENTRAÇÃO DOS ERROS ===")
    print(concentracao.to_string(index=False))

    print("\n=== RESUMO POR MODELO ===")
    cols_modelo = [
        "posicao",
        COL_MODELO,
        "score_medio_0_5",
        "n_correto",
        "n_parcialmente_correto",
        "n_incorreto",
        "desvio_medio_entre_julgadores",
    ]

    print(resumo_modelos[cols_modelo].to_string(index=False))

    print("\n=== TOP 15 PERGUNTAS MAIS ERRADAS ===")
    cols_top = [
        "rank_dificuldade",
        "pergunta",
        "n_modelos",
        "n_correto",
        "n_parcialmente_correto",
        "n_incorreto",
        "score_medio_0_5",
    ]

    tmp = ranking[cols_top].head(15).copy()
    tmp["pergunta"] = tmp["pergunta"].apply(lambda x: encurtar_texto(x, 100))

    print(tmp.to_string(index=False))

    sem_erro = int((ranking["n_incorreto"] == 0).sum())
    todos_erraram = int((ranking["n_incorreto"] == ranking["n_modelos"]).sum())
    nenhuma_correta = int((ranking["n_correto"] == 0).sum())

    print("\n=== INDICADORES POR PERGUNTA ===")
    print(f"Perguntas sem nenhum erro: {sem_erro}")
    print(f"Perguntas em que todos os modelos erraram: {todos_erraram}")
    print(f"Perguntas sem nenhuma resposta correta: {nenhuma_correta}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=str,
        default=".",
        help="Pasta onde estão os CSVs detalhados por modelo.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/erros",
        help="Pasta onde os arquivos de saída serão salvos.",
    )

    parser.add_argument(
        "--top-model",
        type=int,
        default=10,
        help="Quantidade de perguntas problemáticas por modelo.",
    )

    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Ignora a validação dos 11 modelos esperados.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    df = carregar_arquivos(input_dir)

    if not args.skip_model_check:
        validar_modelos(df, EXPECTED_MODELS)
        validar_101_perguntas_por_modelo(df)

    resumo_modelos = resumo_por_modelo(df)
    ranking = ranking_perguntas(df)
    top_modelos = top_perguntas_por_modelo(df, top_n=args.top_model)
    matriz = matriz_modelo_pergunta(df)

    total_erros = int((df[COL_RESULTADO] == "INCORRETO").sum())
    concentracao = resumo_concentracao(ranking, total_erros)

    resumo_modelos.to_csv(
        out_dir / "01_resumo_por_modelo.csv",
        index=False,
        encoding="utf-8",
    )

    ranking.to_csv(
        out_dir / "02_ranking_perguntas_mais_erradas.csv",
        index=False,
        encoding="utf-8",
    )

    top_modelos.to_csv(
        out_dir / "03_top_perguntas_problematicas_por_modelo.csv",
        index=False,
        encoding="utf-8",
    )

    matriz.to_csv(
        out_dir / "04_matriz_modelo_pergunta_categoria.csv",
        index=False,
        encoding="utf-8",
    )

    concentracao.to_csv(
        out_dir / "05_concentracao_erros.csv",
        index=False,
        encoding="utf-8",
    )

    imprimir_resumo(df, resumo_modelos, ranking, concentracao)

    print("\nArquivos salvos em:", out_dir.resolve())
    print(" - 01_resumo_por_modelo.csv")
    print(" - 02_ranking_perguntas_mais_erradas.csv")
    print(" - 03_top_perguntas_problematicas_por_modelo.csv")
    print(" - 04_matriz_modelo_pergunta_categoria.csv")
    print(" - 05_concentracao_erros.csv")


if __name__ == "__main__":
    main()