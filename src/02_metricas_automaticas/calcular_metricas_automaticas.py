#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Avalia métricas (ROUGE, BLEU, BERTScore, Connectedness) para TODOS os CSVs *_QAG.csv em results/legacy/QAG.

Cada arquivo *_QAG.csv deve conter colunas:
  Q        -> pergunta
  A        -> resposta ouro (gold)
  A_model  -> resposta do modelo

Saídas:
  results/legacy/QAG/metrics_all_models.csv   (uma linha por modelo)
  results/legacy/QAG/<modelo>_metrics.csv     (opcional, um por modelo)

Instalar:
  pip install pandas rouge sacrebleu bert-score sentence-transformers torch tqdm transformers
"""

from __future__ import annotations

from pathlib import Path
import re
import random
import numpy as np
import pandas as pd
from rouge import Rouge
from sacrebleu.metrics import BLEU
from bert_score import score
from sentence_transformers import SentenceTransformer
import torch

# =========================
# REPRODUCIBILIDADE
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================
# DETECÇÃO DE DEVICE
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# CONFIGURAÇÕES
# =========================
INPUT_DIR = Path("results/qag")
GLOB_PATTERN = "*_QAG.csv"

# Saída agregada
CSV_OUTPUT_ALL = INPUT_DIR / "metrics_all_models_new.csv"

# Se True, salva também um arquivo por modelo
SAVE_PER_MODEL = True


# SentenceTransformer usado no connectedness
ST_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"


def infer_model_name_from_filename(path: Path) -> str:
    # Ex.: llama-3.3-70b-instruct_QAG.csv -> llama-3.3-70b-instruct
    name = path.stem
    name = re.sub(r"_QAG$", "", name)
    return name


def compute_metrics_for_file(
    csv_path: Path,
    rouge: Rouge,
    bleu: BLEU,
    st_model: SentenceTransformer,
) -> dict:
    df = pd.read_csv(csv_path)

    for col in ["Q", "A", "A_model"]:
        if col not in df.columns:
            raise ValueError(
                f"[{csv_path.name}] Coluna '{col}' não encontrada. Colunas: {list(df.columns)}"
            )

    refs = df["A"].astype(str).tolist()
    hyps = df["A_model"].astype(str).tolist()

    # ROUGE
    rouge_scores = rouge.get_scores(hyps, refs, avg=True)

    # BLEU
    bleu_score = bleu.corpus_score(hyps, [refs])

    # =========================
    # BERTScore - EXATAMENTE COMO NO NOTEBOOK
    # =========================
    P, R, F1 = score(hyps, refs, lang="pt", device="cpu", verbose=True)

    # Connectedness
    emb_refs = st_model.encode(refs, convert_to_tensor=True, show_progress_bar=False)
    emb_hyps = st_model.encode(hyps, convert_to_tensor=True, show_progress_bar=False)

    emb_refs = torch.nn.functional.normalize(emb_refs, p=2, dim=1)
    emb_hyps = torch.nn.functional.normalize(emb_hyps, p=2, dim=1)

    conn_scores = (emb_refs * emb_hyps).sum(dim=1)
    connectedness = conn_scores.mean().item()

    model_name = infer_model_name_from_filename(csv_path)

    return {
        "modelo": model_name,
        "arquivo_qag": csv_path.name,
        "rouge-1": float(rouge_scores["rouge-1"]["f"]),
        "rouge-2": float(rouge_scores["rouge-2"]["f"]),
        "rouge-l": float(rouge_scores["rouge-l"]["f"]),
        "bleu": float(bleu_score.score / 100.0),
        "bert_precision": float(P.mean().item()),
        "bert_recall": float(R.mean().item()),
        "bert_f1": float(F1.mean().item()),
        "connectedness": float(connectedness),
        "num_exemplos": int(len(df)),
        "device": DEVICE,
    }


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {INPUT_DIR.resolve()}")

    files = sorted(INPUT_DIR.glob(GLOB_PATTERN))
    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo encontrado com padrão {GLOB_PATTERN} em {INPUT_DIR.resolve()}"
        )

    print(f"Encontrados {len(files)} arquivos para avaliar em: {INPUT_DIR.resolve()}")
    for f in files:
        print(" -", f.name)

    rouge = Rouge()
    bleu = BLEU()

    print(f"\nCarregando SentenceTransformer ({ST_MODEL_NAME}) em {DEVICE}...")
    st_model = SentenceTransformer(ST_MODEL_NAME, device=DEVICE)

    all_metrics = []

    for csv_path in files:
        print(f"\nAvaliando: {csv_path.name}")
        metrics = compute_metrics_for_file(
            csv_path,
            rouge=rouge,
            bleu=bleu,
            st_model=st_model,
        )
        all_metrics.append(metrics)

        print(
            f"  modelo={metrics['modelo']} | N={metrics['num_exemplos']} | "
            f"ROUGE-L={metrics['rouge-l']:.4f} | BLEU={metrics['bleu']:.4f} | "
            f"BERT-F1={metrics['bert_f1']:.4f} | Conn={metrics['connectedness']:.4f} | "
            f"DEVICE={metrics['device']}"
        )

        if SAVE_PER_MODEL:
            out_path = INPUT_DIR / f"{metrics['modelo']}_metrics.csv"
            pd.DataFrame([metrics]).to_csv(out_path, index=False, encoding="utf-8")
            print(f"  -> salvo: {out_path.name}")

    out_df = pd.DataFrame(all_metrics)
    out_df.sort_values(by=["bert_f1", "rouge-l", "bleu"], ascending=False, inplace=True)
    out_df.to_csv(CSV_OUTPUT_ALL, index=False, encoding="utf-8")

    print(f"\n✅ Arquivo agregado salvo em: {CSV_OUTPUT_ALL.resolve()}")
    print("Fim.")


if __name__ == "__main__":
    main()
