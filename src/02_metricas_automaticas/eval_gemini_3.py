"""
Script de avaliação das respostas do GEMINI no experimento de QA.

Supõe que o qa_gemini.py gerou um CSV no formato:

    saidas/gemini-3-pro-preview_QAG.csv

com colunas:
    Q        -> pergunta
    A        -> resposta ouro (gold)
    A_model  -> resposta do Gemini

Requisitos:

    pip install pandas rouge sacrebleu bert-score torch matplotlib sentence-transformers
"""

from pathlib import Path

import pandas as pd
from rouge import Rouge
from sacrebleu.metrics import BLEU
from bert_score import score
from sentence_transformers import SentenceTransformer
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_INPUT = Path("results/legacy/gemini-3-pro-preview_QAG.csv")

CSV_OUTPUT = Path("results/legacy/gemini-3-pro-preview_metrics.csv")


def main():
    if not CSV_INPUT.exists():
        raise FileNotFoundError(f"CSV de entrada não encontrado: {CSV_INPUT.resolve()}")

    print(f"Lendo respostas do Gemini em: {CSV_INPUT}")
    df = pd.read_csv(CSV_INPUT)

    for col in ["Q", "A", "A_model"]:
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não encontrada no CSV.")

    refs = df["A"].astype(str).tolist()
    hyps = df["A_model"].astype(str).tolist()

    print(f"Número de exemplos: {len(df)}")

    rouge = Rouge()
    bleu = BLEU()

    print("\nCalculando ROUGE...")
    rouge_scores = rouge.get_scores(hyps, refs, avg=True)

    print("Calculando BLEU...")
    bleu_score = bleu.corpus_score(hyps, [refs])

    print(f"Calculando BERTScore (device={DEVICE})...")
    P, R, F1 = score(
        hyps,
        refs,
        lang="pt",    
        device=DEVICE,
        verbose=True,
    )


    print(f"Calculando Connectedness (embeddings, device={DEVICE})...")
    st_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device=DEVICE)

    emb_refs = st_model.encode(refs, convert_to_tensor=True, show_progress_bar=True)
    emb_hyps = st_model.encode(hyps, convert_to_tensor=True, show_progress_bar=True)

    emb_refs = torch.nn.functional.normalize(emb_refs, p=2, dim=1)
    emb_hyps = torch.nn.functional.normalize(emb_hyps, p=2, dim=1)

    conn_scores = (emb_refs * emb_hyps).sum(dim=1)  
    connectedness = conn_scores.mean().item()

    metrics = {
        "modelo": "gemini-3-pro-preview",
        "rouge-1": rouge_scores["rouge-1"]["f"],
        "rouge-2": rouge_scores["rouge-2"]["f"],
        "rouge-l": rouge_scores["rouge-l"]["f"],
        "bleu": bleu_score.score / 100.0,
        "bert_precision": P.mean().item(),
        "bert_recall": R.mean().item(),
        "bert_f1": F1.mean().item(),
        "connectedness": connectedness,
        "num_exemplos": len(df),
    }

    print("\n===== MÉTRICAS GEMINI =====")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:15s}: {v:.4f}")
        else:
            print(f"{k:15s}: {v}")

    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame([metrics])
    out_df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8")

    print(f"\nMétricas salvas em: {CSV_OUTPUT.resolve()}")
    print("Fim da avaliação do Gemini.")


if __name__ == "__main__":
    main()
