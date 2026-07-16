"""Script de QA usando Gemini.

Estrutura esperada de pastas (mesma pasta do script):

    qa_experiment/
        qa_gemini.py
        data/tax_law_brazil_cosit/   <- você coloca aqui o dataset salvo com save_to_disk
        saidas/                 <- aqui serão salvos os CSVs gerados

Para rodar:
    pip install google-generativeai datasets pandas tqdm
    python qa_gemini.py
"""

from pathlib import Path

import os
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm
import google.generativeai as genai
import time 

API_KEY = os.getenv("GEMINI_API_KEY", "")

# Caminho do dataset salvo com load_from_disk 
DATASET_DIR = "data/tax_law_brazil_cosit"

# Pasta onde o CSV de saída será salvo
OUTPUT_DIR = "results/legacy"

# Nome do split dentro do DatasetDict
SPLIT_NAME = "tax_law"

# Modelo Gemini a ser usado
MODEL_NAME = "gemini-3.1-pro-preview"

# Número máximo de linhas para processar
MAX_ROWS = None 


RETRIEVER_TEMPLATE = """Use the following pieces of legal information from laws to answer the user's question.
If the answer is not clear in context, try to figure out by interpreting the information.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Do not quote the "contextual information" provided in the answer, do not say "according to the information" or anything like that, use the information only to answer the question.
Only return the helpful answer below and nothing else.
REMEMBER: answer the question in portuguese.
Helpful answer:"""


def configure_gemini(api_key: str, model_name: str):
    if not api_key or api_key == "SUA_CHAVE_GEMINI_AQUI":
        raise RuntimeError(
            "API_KEY não definida ou ainda está com o valor padrão.\n"
            "Edite o arquivo qa_gemini.py e troque API_KEY pela sua chave real."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def gemini_infer(model, context: str, question: str) -> str:
    """Chama o Gemini com o prompt formatado."""
    prompt_text = RETRIEVER_TEMPLATE.format(context=context, question=question)
    response = model.generate_content(prompt_text)
    return getattr(response, "text", "").strip()


def run_experiment(
    api_key: str,
    dataset_dir: str,
    split_name: str,
    output_dir: str,
    model_name: str,
    max_rows: int | None = None,
):
    start_time = time.time()
     
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Carregando dataset de: {dataset_dir}")
    dataset_dict = load_from_disk(dataset_dir)

    if split_name not in dataset_dict:
        raise ValueError(
            f"Split '{split_name}' não encontrado no dataset. "
            f"Splits disponíveis: {list(dataset_dict.keys())}"
        )

    ds = dataset_dict[split_name]

    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))

    print(f"Número de exemplos no split '{split_name}': {len(ds)}")

    model = configure_gemini(api_key, model_name=model_name)

    data = []
    for row in tqdm(ds, desc=f"Processing with {model_name}"):
        question = row["question"]
        gold_passage = row["gold_passage"]
        answer = row["answer"]

        try:
            model_answer = gemini_infer(model, gold_passage, question)
        except Exception as e:
            model_answer = f"<<ERRO AO GERAR RESPOSTA: {e}>>"

        data.append(
            {
                "Q": question,
                "A": answer,
                "A_model": model_answer,
            }
        )

    df = pd.DataFrame(data, columns=["Q", "A", "A_model"])
    out_file = output_path / f"{model_name.replace('/', '_')}_QAG.csv"
    df.to_csv(out_file, index=False, encoding="utf-8")

    print(f"Arquivo salvo em: {out_file}")
    print("Finish")
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Tempo total do experimento: {elapsed:.2f} segundos (~{elapsed/60:.2f} minutos)")

def main():
    run_experiment(
        api_key=API_KEY,
        dataset_dir=DATASET_DIR,
        split_name=SPLIT_NAME,
        output_dir=OUTPUT_DIR,
        model_name=MODEL_NAME,
        max_rows=MAX_ROWS,
    )


if __name__ == "__main__":
    main()
