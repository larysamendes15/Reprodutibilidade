import re
import unicodedata
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

# CSV com o ranking de erros por pergunta, gerado na etapa 4 por
# analise_erros_por_pergunta.py.
ARQUIVO_RANKING_ERROS = Path("results/erros/02_ranking_perguntas_mais_erradas.csv")
# Pasta onde estão os arquivos QAG dos modelos.
# Cada arquivo deve ter colunas parecidas com:
# Q, A, A_model
PASTA_QAG_MODELOS = Path("results/respostas_geradas")

# Pasta de saída
PASTA_SAIDA = Path("results/ranking_erros_perguntas")
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

ARQUIVO_SAIDA_XLSX = PASTA_SAIDA / "ranking_perguntas_erros_com_respostas.xlsx"
ARQUIVO_SAIDA_CSV = PASTA_SAIDA / "ranking_perguntas_erros_com_respostas.csv"
ARQUIVO_PRISM = PASTA_SAIDA / "dados_prism_ranking_erros.csv"


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def ler_csv(caminho: Path) -> pd.DataFrame:
    """
    Lê CSV tentando detectar separador automaticamente.
    """
    try:
        return pd.read_csv(caminho, sep=None, engine="python")
    except Exception:
        try:
            return pd.read_csv(caminho, sep=";")
        except Exception:
            return pd.read_csv(caminho, sep=",")


def normalizar_texto(texto: str) -> str:
    """
    Normaliza texto para comparar perguntas vindas de arquivos diferentes.
    """
    if pd.isna(texto):
        return ""

    texto = str(texto).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(r"[^\w\s]", "", texto)
    return texto.strip()


def encontrar_coluna(df: pd.DataFrame, candidatos: list[str]) -> str:
    """
    Encontra uma coluna no DataFrame a partir de possíveis nomes.
    """
    colunas_norm = {normalizar_texto(c): c for c in df.columns}

    for candidato in candidatos:
        candidato_norm = normalizar_texto(candidato)
        if candidato_norm in colunas_norm:
            return colunas_norm[candidato_norm]

    raise ValueError(
        f"Nenhuma das colunas {candidatos} foi encontrada. "
        f"Colunas disponíveis: {list(df.columns)}"
    )


def nome_modelo_por_arquivo(caminho: Path) -> str:
    """
    Extrai o nome do modelo a partir do nome do arquivo.
    """
    nome = caminho.stem

    sufixos = [
        "_QAG",
        "-QAG",
        "_qag",
        "-qag",
        "_respostas",
        "_responses",
    ]

    for sufixo in sufixos:
        nome = nome.replace(sufixo, "")

    return nome


def tokenizar(texto: str) -> list[str]:
    """
    Tokenização simples para métricas lexicais.
    """
    texto = str(texto).lower()
    return re.findall(r"\w+", texto, flags=re.UNICODE)


def lcs_len(a: list[str], b: list[str]) -> int:
    """
    Longest Common Subsequence para ROUGE-L.
    Usa programação dinâmica com memória reduzida.
    """
    if not a or not b:
        return 0

    anterior = [0] * (len(b) + 1)

    for token_a in a:
        atual = [0] * (len(b) + 1)

        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                atual[j] = anterior[j - 1] + 1
            else:
                atual[j] = max(anterior[j], atual[j - 1])

        anterior = atual

    return anterior[-1]


def rouge_l_f1(referencia: str, resposta: str) -> float:
    """
    Calcula ROUGE-L F1 simples entre resposta esperada e resposta do modelo.
    """
    ref_tokens = tokenizar(referencia)
    resp_tokens = tokenizar(resposta)

    if not ref_tokens or not resp_tokens:
        return 0.0

    lcs = lcs_len(ref_tokens, resp_tokens)

    precisao = lcs / len(resp_tokens)
    revocacao = lcs / len(ref_tokens)

    if precisao + revocacao == 0:
        return 0.0

    return 2 * precisao * revocacao / (precisao + revocacao)


def jaccard(referencia: str, resposta: str) -> float:
    """
    Similaridade Jaccard entre conjuntos de palavras.
    """
    ref = set(tokenizar(referencia))
    resp = set(tokenizar(resposta))

    if not ref or not resp:
        return 0.0

    return len(ref & resp) / len(ref | resp)


def cobertura_referencia(referencia: str, resposta: str) -> float:
    """
    Mede quanto da resposta esperada aparece lexicalmente na resposta do modelo.
    """
    ref = set(tokenizar(referencia))
    resp = set(tokenizar(resposta))

    if not ref:
        return 0.0

    return len(ref & resp) / len(ref)


def media_segura(valores: list[float]) -> float:
    valores = [v for v in valores if pd.notna(v)]
    if not valores:
        return 0.0
    return sum(valores) / len(valores)


# ============================================================
# 1. CARREGAR RANKING DE ERROS
# ============================================================

ranking = ler_csv(ARQUIVO_RANKING_ERROS)

col_pergunta = encontrar_coluna(
    ranking,
    ["pergunta", "Pergunta", "pergunta_completa", "Pergunta completa", "Q"]
)

try:
    col_id = encontrar_coluna(
        ranking,
        ["id_pergunta", "pergunta_curta", "ID", "id", "Pergunta ID"]
    )
except ValueError:
    col_id = None

col_incorreto = encontrar_coluna(
    ranking,
    ["n_incorreto", "Incorretas", "incorretas", "n_erros"]
)

try:
    col_parcial = encontrar_coluna(
        ranking,
        ["n_parcialmente_correto", "Parcialmente corretas", "parciais"]
    )
except ValueError:
    col_parcial = None

try:
    col_correto = encontrar_coluna(
        ranking,
        ["n_correto", "Corretas", "corretas"]
    )
except ValueError:
    col_correto = None

try:
    col_score = encontrar_coluna(
        ranking,
        ["score_medio_0_5", "Score médio (0-5)", "score_medio", "score"]
    )
except ValueError:
    col_score = None

ranking["pergunta_normalizada"] = ranking[col_pergunta].apply(normalizar_texto)


# ============================================================
# 2. CARREGAR RESPOSTAS DOS MODELOS
# ============================================================

arquivos_qag = sorted(PASTA_QAG_MODELOS.glob("*_QAG.csv"))

if not arquivos_qag:
    raise FileNotFoundError(
        f"Nenhum CSV encontrado em {PASTA_QAG_MODELOS}. "
        "Confira se a pasta está correta."
    )

resposta_esperada_por_pergunta = {}
respostas_modelos_por_pergunta = defaultdict(dict)

modelos = []

for arquivo in arquivos_qag:
    modelo = nome_modelo_por_arquivo(arquivo)
    modelos.append(modelo)

    df = ler_csv(arquivo)

    col_q = encontrar_coluna(
        df,
        ["Q", "pergunta", "Pergunta", "question"]
    )

    col_a = encontrar_coluna(
        df,
        ["A", "resposta_esperada", "resposta oficial", "resposta_oficial", "answer"]
    )

    col_a_model = encontrar_coluna(
        df,
        ["A_model", "resposta_modelo", "resposta gerada", "resposta_gerada", "model_answer"]
    )

    for _, linha in df.iterrows():
        pergunta = linha[col_q]
        chave = normalizar_texto(pergunta)

        resposta_esperada = linha[col_a]
        resposta_modelo = linha[col_a_model]

        if chave not in resposta_esperada_por_pergunta:
            resposta_esperada_por_pergunta[chave] = resposta_esperada

        respostas_modelos_por_pergunta[chave][modelo] = resposta_modelo

modelos = sorted(set(modelos))


# ============================================================
# 3. MONTAR RANKING FINAL
# ============================================================

linhas_saida = []

for idx, linha in ranking.iterrows():
    chave = linha["pergunta_normalizada"]

    pergunta = linha[col_pergunta]
    resposta_esperada = resposta_esperada_por_pergunta.get(chave, "")

    n_incorretas = int(linha[col_incorreto]) if pd.notna(linha[col_incorreto]) else 0

    if col_parcial:
        n_parciais = int(linha[col_parcial]) if pd.notna(linha[col_parcial]) else 0
    else:
        n_parciais = 0

    if col_correto:
        n_corretas = int(linha[col_correto]) if pd.notna(linha[col_correto]) else 0
    else:
        total_modelos = len(modelos)
        n_corretas = max(total_modelos - n_incorretas - n_parciais, 0)

    total_avaliado = n_corretas + n_parciais + n_incorretas

    if total_avaliado == 0:
        taxa_erro = 0.0
    else:
        taxa_erro = 100 * n_incorretas / total_avaliado

    if col_score:
        score_medio = float(linha[col_score]) if pd.notna(linha[col_score]) else None
    else:
        score_medio = None

    indice_dificuldade = n_incorretas + (0.5 * n_parciais)

    respostas_da_pergunta = respostas_modelos_por_pergunta.get(chave, {})

    rouges = []
    jaccards = []
    coberturas = []
    tamanhos = []

    for modelo in modelos:
        resposta_modelo = respostas_da_pergunta.get(modelo, "")

        if resposta_esperada and resposta_modelo:
            rouges.append(rouge_l_f1(resposta_esperada, resposta_modelo))
            jaccards.append(jaccard(resposta_esperada, resposta_modelo))
            coberturas.append(cobertura_referencia(resposta_esperada, resposta_modelo))
            tamanhos.append(len(tokenizar(resposta_modelo)))

    id_pergunta = linha[col_id] if col_id else f"Q{idx + 1}"

    registro = {
        "ranking_erro": None,
        "id_pergunta": id_pergunta,
        "pergunta": pergunta,
        "resposta_esperada": resposta_esperada,
        "n_corretas": n_corretas,
        "n_parcialmente_corretas": n_parciais,
        "n_incorretas": n_incorretas,
        "total_modelos_avaliados": total_avaliado,
        "taxa_erro_%": round(taxa_erro, 2),
        "score_medio_0_5": round(score_medio, 4) if score_medio is not None else "",
        "indice_dificuldade": round(indice_dificuldade, 2),
        "rouge_l_medio": round(media_segura(rouges), 4),
        "jaccard_medio": round(media_segura(jaccards), 4),
        "cobertura_resposta_esperada_media": round(media_segura(coberturas), 4),
        "tamanho_medio_resposta_palavras": round(media_segura(tamanhos), 2),
        "modelos_com_resposta": len(respostas_da_pergunta),
        "modelos_sem_resposta": len(modelos) - len(respostas_da_pergunta),
    }

    for modelo in modelos:
        registro[f"resposta_{modelo}"] = respostas_da_pergunta.get(modelo, "")

    linhas_saida.append(registro)

saida = pd.DataFrame(linhas_saida)


# ============================================================
# 4. ORDENAR DA PERGUNTA MAIS ERRADA PARA A MENOS ERRADA
# ============================================================

saida = saida.sort_values(
    by=[
        "n_incorretas",
        "n_parcialmente_corretas",
        "indice_dificuldade",
        "score_medio_0_5",
    ],
    ascending=[False, False, False, True],
    na_position="last"
).reset_index(drop=True)

saida["ranking_erro"] = range(1, len(saida) + 1)


# ============================================================
# 5. GERAR ARQUIVO PARA O PRISM
# ============================================================

prism = saida[
    [
        "id_pergunta",
        "n_corretas",
        "n_parcialmente_corretas",
        "n_incorretas",
        "taxa_erro_%",
        "score_medio_0_5",
    ]
].copy()

prism = prism.rename(
    columns={
        "id_pergunta": "Pergunta",
        "n_corretas": "Corretas",
        "n_parcialmente_corretas": "Parcialmente corretas",
        "n_incorretas": "Incorretas",
        "taxa_erro_%": "Taxa de erro (%)",
        "score_medio_0_5": "Score médio (0-5)",
    }
)


# ============================================================
# 6. SALVAR RESULTADOS
# ============================================================

saida.to_csv(ARQUIVO_SAIDA_CSV, index=False, encoding="utf-8-sig")
prism.to_csv(ARQUIVO_PRISM, index=False, encoding="utf-8-sig")

try:
    with pd.ExcelWriter(ARQUIVO_SAIDA_XLSX, engine="xlsxwriter") as writer:
        saida.to_excel(writer, sheet_name="Ranking completo", index=False)
        prism.to_excel(writer, sheet_name="Prism", index=False)

        workbook = writer.book

        formato_header = workbook.add_format(
            {
                "bold": True,
                "bg_color": "#D9EAF7",
                "border": 1,
                "text_wrap": True,
                "valign": "top",
            }
        )

        formato_texto = workbook.add_format(
            {
                "text_wrap": True,
                "valign": "top",
            }
        )

        formato_numero = workbook.add_format(
            {
                "num_format": "0.00",
                "valign": "top",
            }
        )

        for sheet_name in ["Ranking completo", "Prism"]:
            worksheet = writer.sheets[sheet_name]
            df_sheet = saida if sheet_name == "Ranking completo" else prism

            for col_num, col_name in enumerate(df_sheet.columns):
                worksheet.write(0, col_num, col_name, formato_header)

                if col_name in ["pergunta", "resposta_esperada"] or col_name.startswith("resposta_"):
                    worksheet.set_column(col_num, col_num, 45, formato_texto)
                elif "taxa" in col_name.lower() or "score" in col_name.lower() or "medio" in col_name.lower():
                    worksheet.set_column(col_num, col_num, 15, formato_numero)
                else:
                    worksheet.set_column(col_num, col_num, 18, formato_texto)

            worksheet.freeze_panes(1, 0)
            worksheet.autofilter(0, 0, len(df_sheet), len(df_sheet.columns) - 1)

except ModuleNotFoundError:
    print("Aviso: xlsxwriter não está instalado. O Excel não foi gerado.")
    print("Instale com: pip install xlsxwriter")


print("\nArquivos gerados:")
print(f"- {ARQUIVO_SAIDA_CSV}")
print(f"- {ARQUIVO_PRISM}")
print(f"- {ARQUIVO_SAIDA_XLSX}")