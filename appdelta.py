# -*- coding: utf-8 -*-
"""
Verificador de Duplicidade — Versão Refatorada e Consolidada
============================================================

Este aplicativo implementa todas as funcionalidades descritas no "Guia de Implementação Técnica",
combinando a lógica do app original com as melhorias propostas.

Funcionalidades Principais:
- Carregamento de dados de atividades do MySQL.
- Algoritmo de similaridade avançado com RapidFuzz, penalidade de tamanho e bônus por campos.
- Pré-índice (bucketing) por pasta, CNJ e, opcionalmente, pela pasta real do DataJuri.
- Agrupamento transitivo (BFS) para formar grupos de duplicatas.
- Interface rica com modo de exibição estrito, seleção do "melhor principal", e comparação visual (diff).
- Integração com API de cancelamento, incluindo resiliência (tentativas, rate-limit) e modo de teste (dry-run).
- Painel de calibração para ajustar os limiares de similaridade por pasta.
"""
from __future__ import annotations

import os
import re
import html
import logging
import time
import math
from datetime import datetime, timedelta, date
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from zoneinfo import ZoneInfo
from unidecode import unidecode
from rapidfuzz import fuzz, process
from difflib import SequenceMatcher

# Importa os clientes de API. Certifique-se de que os arquivos
# api_functions_retry.py e datajuri_client.py estão na mesma pasta.
try:
    from api_functions_retry import HttpClientRetry
except ImportError:
    st.error("Erro: O arquivo 'api_functions_retry.py' não foi encontrado. Ele é necessário para a comunicação com a API de cancelamento.")
    HttpClientRetry = None

try:
    from datajuri_client import DataJuriClient
except ImportError:
    st.warning("Aviso: O arquivo 'datajuri_client.py' não foi encontrado. A funcionalidade de busca por pasta no DataJuri será desativada.")
    DataJuriClient = None

# Opcional para o gráfico de calibração
try:
    import altair as alt
except ImportError:
    alt = None

# =============================================================================
# CONFIGURAÇÃO GERAL E CONSTANTES
# =============================================================================
APP_TITLE = "Verificador de Duplicidade Avançado"
TZ_SP = ZoneInfo("America/Sao_Paulo")
TZ_UTC = ZoneInfo("UTC")

# Chaves para o session_state do Streamlit, para evitar colisões
SUFFIX = "_v2_refatorado"
class SK:
    USERNAME = f"username_{SUFFIX}"
    SIMILARITY_CACHE = f"simcache_{SUFFIX}"
    PAGE_NUMBER = f"page_{SUFFIX}"
    GROUP_STATES = f"group_states_{SUFFIX}"
    CFG = f"cfg_{SUFFIX}"

# Valores padrão caso não sejam definidos nos secrets
DEFAULTS = {
    "itens_por_pagina": 10,
    "dias_filtro_inicio": 7,
    "dias_filtro_fim": 14,
    "min_sim_global": 90,
    "min_containment": 55,
    "pre_cutoff_delta": 10,
    "diff_hard_limit": 12000,
}

# Configuração da página e estilos CSS
st.set_page_config(layout="wide", page_title=APP_TITLE)
st.markdown("""
<style>
    /* Estilos para o visualizador de diferenças (diff) */
    pre.highlighted-text {
        white-space: pre-wrap;
        word-wrap: break-word;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
        font-size: .9em;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        background-color: #f9f9f9;
        height: 360px;
        overflow-y: auto;
    }
    .diff-del { background-color: #ffcdd2 !important; text-decoration: none !important; }
    .diff-ins { background-color: #c8e6c9 !important; text-decoration: none !important; }

    /* Estilos para os cards de atividade */
    .card { border-left: 5px solid #ccc; padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: #fff; }
    .card-cancelado { background-color: #f5f5f5; border-left: 5px solid #e0e0e0; }
    .card-principal { border-left: 5px solid #4CAF50; }

    /* Estilos para os badges de similaridade */
    .similarity-badge { padding: 3px 6px; border-radius: 5px; color: black; font-weight: 600; display: inline-block; margin-bottom: 6px; }
    .badge-green { background:#C8E6C9; }
    .badge-yellow { background:#FFF9C4; }
    .badge-red { background:#FFCDD2; }

    /* Outros estilos */
    .meta-chip { background:#E0F7FA; padding:2px 6px; margin-right:6px; border-radius:8px; display:inline-block; font-size:0.85em; }
    .small-muted { color:#777; font-size:0.85em; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# INICIALIZAÇÃO DE SERVIÇOS (BANCO, APIS)
# =============================================================================

@st.cache_resource
def db_engine_mysql() -> Optional[Engine]:
    """Cria e armazena em cache a engine de conexão com o banco de dados MySQL."""
    cfg = st.secrets.get("database", {})
    db_params = {k: cfg.get(k) for k in ["host", "user", "password", "name"]}
    if not all(db_params.values()):
        st.error("Credenciais do banco de dados (MySQL) ausentes em st.secrets['database'].")
        st.stop()
    try:
        engine = create_engine(
            f"mysql+mysqlconnector://{db_params['user']}:{db_params['password']}@{db_params['host']}/{db_params['name']}",
            pool_pre_ping=True,
            pool_recycle=3600
        )
        # Testa a conexão
        with engine.connect():
            pass
        return engine
    except exc.SQLAlchemyError as e:
        logging.exception(e)
        st.error(f"Erro ao conectar no banco de dados (MySQL): {e}")
        st.stop()

@st.cache_resource
def api_client() -> Optional[HttpClientRetry]:
    """Cria e armazena em cache o cliente para a API de cancelamento."""
    if HttpClientRetry is None: return None
    
    api_cfg = st.secrets.get("api", {})
    client_cfg = st.secrets.get("api_client", {})
    
    api_params = {k: api_cfg.get(k) for k in ["url_api", "entity_id", "token"]}
    if not all(api_params.values()):
        st.warning("Configuração da API de cancelamento ausente ou incompleta em st.secrets['api']. A funcionalidade de cancelamento será desativada.")
        return None
        
    return HttpClientRetry(
        base_url=api_params["url_api"],
        entity_id=api_params["entity_id"],
        token=api_params["token"],
        calls_per_second=float(client_cfg.get("calls_per_second", 3.0)),
        max_attempts=int(client_cfg.get("max_attempts", 3)),
        timeout=int(client_cfg.get("timeout", 15)),
        dry_run=bool(client_cfg.get("dry_run", False))
    )

@st.cache_resource
def datajuri_client_instance() -> Optional[DataJuriClient]:
    """Cria e armazena em cache o cliente para a API do DataJuri."""
    if DataJuriClient is None: return None
    
    dj_cfg = st.secrets.get("datajuri", {})
    # O cliente aceita vários nomes de chave, então apenas passamos o dict
    if not dj_cfg or not all(k in dj_cfg for k in ["DATAJURI_BASE_URL", "DATAJURI_CLIENT_ID", "DATAJURI_SECRET_ID", "DATAJURI_USERNAME", "DATAJURI_PASSWORD"]):
        st.info("Credenciais do DataJuri não encontradas em st.secrets['datajuri']. A funcionalidade será desativada.")
        return None
        
    try:
        # Renomeia as chaves para corresponder aos argumentos do construtor
        client_args = {
            "base_url": dj_cfg["DATAJURI_BASE_URL"],
            "client_id": dj_cfg["DATAJURI_CLIENT_ID"],
            "secret_id": dj_cfg["DATAJURI_SECRET_ID"],
            "username": dj_cfg["DATAJURI_USERNAME"],
            "password": dj_cfg["DATAJURI_PASSWORD"]
        }
        client = DataJuriClient(**client_args)
        client.ensure_token() # Testa a autenticação na inicialização
        st.sidebar.success("Cliente DataJuri conectado. ✅")
        return client
    except Exception as e:
        st.sidebar.warning(f"Falha ao conectar no DataJuri: {e}. A funcionalidade será desativada.")
        return None

# =============================================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO DE DADOS
# =============================================================================

@st.cache_data(ttl=1800, hash_funcs={Engine: lambda _: None})
def carregar_dados_mysql(_eng: Engine, dias_historico: int) -> pd.DataFrame:
    """Carrega atividades do banco de dados, incluindo abertas e fechadas recentes."""
    limite = date.today() - timedelta(days=dias_historico)
    query = text("""
        SELECT activity_id, activity_folder, user_profile_name, activity_date, activity_status, Texto
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type='Verificar'
          AND (activity_status='Aberta' OR DATE(activity_date) >= :limite)
    """)
    try:
        with _eng.connect() as conn:
            df = pd.read_sql(query, conn, params={"limite": limite})
        if df.empty:
            return pd.DataFrame()
        
        # Limpeza e formatação inicial
        df["activity_id"] = df["activity_id"].astype(str)
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
        df["Texto"] = df["Texto"].fillna("").astype(str)
        
        # Lógica para manter apenas a atividade mais relevante (Aberta > Fechada)
        df["status_ord"] = df["activity_status"].map({"Aberta": 0}).fillna(1)
        df = df.sort_values(["activity_id", "status_ord"]).drop_duplicates("activity_id", keep="first").drop(columns="status_ord")
        
        df = df.sort_values(["activity_folder", "activity_date"], ascending=[True, False])
        return df
    except exc.SQLAlchemyError as e:
        logging.exception(e)
        st.error(f"Erro ao carregar dados do banco: {e}")
        return pd.DataFrame()

# =============================================================================
# LÓGICA DE SIMILARIDADE E NORMALIZAÇÃO
# =============================================================================

# Expressões Regulares para normalização
CNJ_RE = re.compile(r"(?:\b|^)(\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})(?:\b|$)")
URL_RE = re.compile(r"https?://\S+")
DATENUM_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
NUM_RE = re.compile(r"\b\d+\b")

# Stopwords básicas para o idioma português e contexto jurídico
STOPWORDS_BASE = set("""
    de da do das dos e em a o os as na no para por com que ao aos às à um uma umas uns
    tipo titulo inteiro teor publicado publicacao disponibilizacao orgao vara tribunal
    processo recurso intimacao notificacao justica nacional diario djen poder judiciario trabalho
""".split())

def extract_meta(text: str) -> Dict[str, str]:
    """Extrai metadados estruturados (CNJ, órgão, etc.) do texto da atividade."""
    t = text or ""
    meta = {}
    
    # Processo/CNJ
    cnj_match = CNJ_RE.search(t)
    cnj = cnj_match.group(1) if cnj_match else None
    if not cnj:
        proc_match = re.search(r"PROCESSO:\s*([0-9\-.]+)", t, re.IGNORECASE)
        if proc_match: cnj = proc_match.group(1)
    meta["processo"] = cnj or ""

    # Outros campos
    patterns = {
        "orgao": r"\bORGAO:\s*([^-\n\r]+)",
        "vara": r"\bVARA\s+DO\s+TRABALHO\s+[^-\n\r]+",
        "tipo_doc": r"\bTIPO\s+DE\s+DOCUMENTO:\s*([^-]+)",
        "tipo_com": r"\bTIPO\s+DE\s+COMUNICACAO:\s*([^-]+)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, t, re.IGNORECASE)
        if match:
            meta[key] = match.group(1).strip() if key != "vara" else match.group(0).strip()
            
    return meta

def normalize_for_match(text: str, stopwords_extra: List[str]) -> str:
    """Aplica uma série de normalizações ao texto para melhorar a comparação."""
    if not isinstance(text, str): return ""
    t = text
    t = URL_RE.sub(" url ", t)
    t = CNJ_RE.sub(" numproc ", t)
    t = DATENUM_RE.sub(" data ", t)
    t = NUM_RE.sub(" # ", t)
    t = unidecode(t.lower())
    t = re.sub(r"[^\w\s]", " ", t)
    
    all_stopwords = STOPWORDS_BASE.union(stopwords_extra)
    tokens = [w for w in t.split() if w not in all_stopwords]
    return " ".join(tokens)

def token_containment(a_tokens: List[str], b_tokens: List[str]) -> float:
    """Calcula a porcentagem de tokens do texto menor que estão contidos no texto maior."""
    if not a_tokens or not b_tokens: return 0.0
    
    # Garante que 'small' seja sempre a menor lista de tokens
    small, big = (a_tokens, set(b_tokens)) if len(a_tokens) <= len(b_tokens) else (b_tokens, set(a_tokens))
    
    intersection_count = sum(1 for token in small if token in big)
    return 100.0 * (intersection_count / len(small))

def length_penalty(len_a: int, len_b: int) -> float:
    """Aplica uma penalidade se os textos tiverem tamanhos muito diferentes."""
    if len_a == 0 or len_b == 0: return 0.9 # Penalidade alta para texto vazio
    diff_ratio = abs(len_a - len_b) / max(len_a, len_b)
    # A penalidade é de até 10% (1.0 - diff_ratio * 0.1) e no mínimo 0.9
    return max(0.9, 1.0 - diff_ratio * 0.1)

def fields_bonus(meta_a: Dict[str,str], meta_b: Dict[str,str]) -> int:
    """Concede um bônus de pontuação se certos metadados forem idênticos."""
    bonus = 0
    if meta_a.get("processo") and meta_a.get("processo") == meta_b.get("processo"):
        bonus += 6
    if meta_a.get("orgao") and meta_a.get("orgao") == meta_b.get("orgao"):
        bonus += 3
    if meta_a.get("tipo_doc") and meta_a.get("tipo_doc") == meta_b.get("tipo_doc"):
        bonus += 3
    if meta_a.get("tipo_com") and meta_a.get("tipo_com") == meta_b.get("tipo_com"):
        bonus += 2
    return bonus

def combined_score(a_norm: str, b_norm: str, meta_a: Dict[str,str], meta_b: Dict[str,str]) -> Tuple[float, Dict[str,float]]:
    """Calcula o score final de similaridade combinando várias métricas."""
    # Métricas base
    set_ratio = fuzz.token_set_ratio(a_norm, b_norm)
    sort_ratio = fuzz.token_sort_ratio(a_norm, b_norm)
    contain = token_containment(a_norm.split(), b_norm.split())
    
    # Ponderação e modificadores
    lp = length_penalty(len(a_norm), len(b_norm))
    bonus = fields_bonus(meta_a, meta_b)
    
    # Fórmula final
    base_score = 0.6 * set_ratio + 0.2 * sort_ratio + 0.2 * contain
    final_score = max(0.0, min(100.0, base_score * lp + bonus))
    
    details = {"set": set_ratio, "sort": sort_ratio, "contain": contain, "len_pen": lp, "bonus": bonus, "base": base_score}
    return final_score, details

# =============================================================================
# LÓGICA DE AGRUPAMENTO (BUCKETING E BFS)
# =============================================================================

def build_buckets(df: pd.DataFrame, use_cnj: bool, use_datajuri: bool, dj_client: Optional[DataJuriClient]) -> Dict[str, List[int]]:
    """Agrupa atividades em 'baldes' (buckets) para otimizar a comparação."""
    buckets = defaultdict(list)
    for i, row in df.iterrows():
        folder = str(row.get("activity_folder") or "SEM_PASTA")
        cnj = row.get("_meta", {}).get("processo", "")
        
        # A chave base é sempre a pasta da atividade
        key = f"folder::{folder}"
        
        # Se DataJuri estiver ativo e um CNJ for encontrado, tenta resolver a pasta real
        if use_datajuri and dj_client and cnj:
            pasta_real = dj_client.get_pasta_by_cnj_cached(cnj)
            if pasta_real:
                key = f"pasta_dj::{pasta_real}" # Usa a pasta do DataJuri como chave
        # Senão, se a restrição por CNJ estiver ativa, adiciona o CNJ à chave
        elif use_cnj:
            key = f"{key}::cnj::{cnj or 'SEM_CNJ'}"
            
        buckets[key].append(i)
    return buckets

def criar_grupos_de_duplicatas(df: pd.DataFrame, params: Dict) -> List[List[Dict]]:
    """Função principal que orquestra a identificação e o agrupamento de duplicatas."""
    if df.empty: return []

    # Cria uma "assinatura" dos parâmetros para invalidar o cache se algo mudar
    cutoffs_tuple = tuple(sorted(st.secrets.get("similarity", {}).get("cutoffs_por_pasta", {}).items()))
    sig = (
        tuple(sorted(df["activity_id"])),
        params['min_sim'], params['min_containment'], params['pre_delta'],
        params['use_cnj'], params['use_datajuri'], cutoffs_tuple
    )
    
    cached = st.session_state.get(SK.SIMILARITY_CACHE)
    if cached and cached.get("sig") == sig:
        return cached["groups"]

    work_df = df.copy()
    
    # Pré-calcula metadados e textos normalizados para performance
    stopwords_extra = st.secrets.get("similarity", {}).get("stopwords_extra", [])
    work_df["_meta"] = work_df["Texto"].apply(extract_meta)
    work_df["_norm"] = work_df["Texto"].apply(lambda t: normalize_for_match(t, stopwords_extra))

    buckets = build_buckets(work_df, params['use_cnj'], params['use_datajuri'], params['dj_client'])
    cutoffs_map = st.secrets.get("similarity", {}).get("cutoffs_por_pasta", {})

    groups = []
    progress_bar = st.sidebar.progress(0, text="Agrupando duplicatas...")
    total_processed = 0
    total_items = len(work_df)

    for bkey, idxs in buckets.items():
        if len(idxs) < 2:
            total_processed += len(idxs)
            continue

        bucket_df = work_df.loc[idxs].reset_index().rename(columns={"index": "orig_idx"})
        texts = bucket_df["_norm"].tolist()

        # Define o limiar de similaridade para este bucket (pasta específica ou global)
        folder_name = bkey.split("::")[1] if bkey.startswith("folder::") else None
        min_sim_bucket = float(cutoffs_map.get(folder_name, params['min_sim']))
        
        pre_cutoff = max(0, int(min_sim_bucket * 100) - params['pre_delta'])

        # 1. Pré-corte: usa uma métrica rápida para eliminar pares obviamente diferentes
        prelim_matrix = process.cdist(texts, texts, scorer=fuzz.token_set_ratio, score_cutoff=pre_cutoff)
        
        # 2. Construção do Grafo e Agrupamento (BFS)
        n = len(bucket_df)
        visited = set()
        memo_score: Dict[Tuple[int, int], Tuple[float, Dict]] = {}

        def are_connected(i, j) -> bool:
            """Verifica se dois itens são duplicados usando o score completo."""
            key = tuple(sorted((i, j)))
            if key in memo_score:
                score, details = memo_score[key]
            else:
                score, details = combined_score(
                    bucket_df.loc[i, "_norm"], bucket_df.loc[j, "_norm"],
                    bucket_df.loc[i, "_meta"], bucket_df.loc[j, "_meta"]
                )
                memo_score[key] = (score, details)
            
            return details["contain"] >= params['min_containment'] and score >= (min_sim_bucket * 100.0)

        for i in range(n):
            if i in visited: continue
            
            component = {i}
            queue = deque([i])
            visited.add(i)
            
            while queue:
                current_node = queue.popleft()
                for neighbor in range(n):
                    if neighbor not in visited and prelim_matrix[current_node][neighbor] >= pre_cutoff and are_connected(current_node, neighbor):
                        visited.add(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
            
            if len(component) > 1:
                # Ordena os membros do grupo por data, do mais recente para o mais antigo
                sorted_idxs = sorted(list(component), key=lambda ix: bucket_df.loc[ix, "activity_date"], reverse=True)
                group_data = [work_df.loc[bucket_df.loc[ix, "orig_idx"]].to_dict() for ix in sorted_idxs]
                groups.append(group_data)

        total_processed += len(idxs)
        progress_bar.progress(min(1.0, total_processed / total_items), text=f"Agrupando (bucket {bkey})...")
    
    progress_bar.empty()
    st.session_state[SK.SIMILARITY_CACHE] = {"sig": sig, "groups": groups}
    return groups

# =============================================================================
# COMPONENTES DE UI E RENDERIZAÇÃO
# =============================================================================

def highlight_diffs_safe(text1: str, text2: str, hard_limit: int) -> Tuple[str,str]:
    """Gera um diff visual entre dois textos, com um fallback para textos muito grandes."""
    t1, t2 = (text1 or ""), (text2 or "")
    if (len(t1) + len(t2)) > hard_limit:
        # Fallback: compara apenas as primeiras N sentenças para evitar travamentos
        s1 = " ".join(re.split(r'([.!?\n]+)', t1)[:100])
        s2 = " ".join(re.split(r'([.!?\n]+)', t2)[:100])
        h1, h2 = highlight_diffs(s1, s2)
        note = "<div class='small-muted'>⚠️ Diff parcial por tamanho. Comparando apenas o início dos textos.</div>"
        return (note + h1, note + h2)
    return highlight_diffs(t1, t2)

def highlight_diffs(a: str, b: str) -> Tuple[str,str]:
    """Usa difflib para criar o HTML do diff."""
    tokens1 = [tok for tok in re.split(r'(\W+)', a or "") if tok]
    tokens2 = [tok for tok in re.split(r'(\W+)', b or "") if tok]
    sm = SequenceMatcher(None, tokens1, tokens2, autojunk=False)
    out1, out2 = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        s1, s2 = html.escape("".join(tokens1[i1:i2])), html.escape("".join(tokens2[j1:j2]))
        if tag == 'equal':
            out1.append(s1); out2.append(s2)
        elif tag == 'replace':
            out1.append(f"<span class='diff-del'>{s1}</span>"); out2.append(f"<span class='diff-ins'>{s2}</span>")
        elif tag == 'delete':
            out1.append(f"<span class='diff-del'>{s1}</span>")
        elif tag == 'insert':
            out2.append(f"<span class='diff-ins'>{s2}</span>")
    return (f"<pre class='highlighted-text'>{''.join(out1)}</pre>", f"<pre class='highlighted-text'>{''.join(out2)}</pre>")

def sidebar_controls(df_full: pd.DataFrame) -> Dict:
    """Renderiza todos os controles da barra lateral e retorna os parâmetros selecionados."""
    st.sidebar.header("👤 Sessão")
    username = st.session_state.get(SK.USERNAME, "Não logado")
    st.sidebar.success(f"Logado como: **{username}**")
    if st.sidebar.button("🔄 Forçar Atualização dos Dados"):
        st.session_state.pop(SK.SIMILARITY_CACHE, None)
        carregar_dados_mysql.clear()
        st.rerun()

    st.sidebar.header("⚙️ Parâmetros de Similaridade")
    sim_cfg = st.secrets.get("similarity", {})
    
    # Controles principais
    min_sim = st.sidebar.slider("Similaridade Mínima Global (%)", 0, 100, int(sim_cfg.get("min_sim_global", DEFAULTS["min_sim_global"])), 1) / 100.0
    min_containment = st.sidebar.slider("Containment Mínimo (%)", 0, 100, int(sim_cfg.get("min_containment", DEFAULTS["min_containment"])), 1)
    pre_delta = st.sidebar.slider("Delta do Pré-corte", 0, 30, int(sim_cfg.get("pre_cutoff_delta", DEFAULTS["pre_cutoff_delta"])), 1, help="Aumentar torna o pré-corte mais agressivo, acelerando o processo mas podendo perder alguns matches.")
    diff_limit = st.sidebar.number_input("Limite de Caracteres do Diff", min_value=5000, value=int(sim_cfg.get("diff_hard_limit", DEFAULTS["diff_hard_limit"])), step=1000)

    st.sidebar.header("👁️ Filtros de Exibição")
    # Filtros de data e escopo
    dias_hist = st.sidebar.number_input("Dias de Histórico para Análise", min_value=7, max_value=365, value=90, step=1)
    data_inicio = st.sidebar.date_input("Data Início", date.today() - timedelta(days=DEFAULTS["dias_filtro_inicio"]))
    data_fim = st.sidebar.date_input("Data Fim", date.today() + timedelta(days=DEFAULTS["dias_filtro_fim"]))
    
    # Filtros de conteúdo
    pastas_opts = sorted(df_full["activity_folder"].dropna().unique()) if not df_full.empty else []
    status_opts = sorted(df_full["activity_status"].dropna().unique()) if not df_full.empty else []
    pastas_sel = st.sidebar.multiselect("Filtrar por Pastas", pastas_opts)
    status_sel = st.sidebar.multiselect("Filtrar por Status", status_opts)
    
    # Modo de exibição estrito
    strict_only = st.sidebar.toggle("Modo Estrito", value=True, help="Exibe apenas itens com similaridade e containment acima do limiar em relação ao item principal do grupo.")

    st.sidebar.header("🚀 Otimizações (Pré-índice)")
    use_cnj = st.sidebar.toggle("Restringir por Nº do Processo (CNJ)", value=True, help="Compara apenas atividades que compartilham o mesmo número de processo.")
    use_datajuri = st.sidebar.toggle("Usar Pasta Real (DataJuri)", value=True, disabled=(datajuri_client_instance() is None), help="Usa a API do DataJuri para agrupar pela pasta real do processo, mais preciso que o CNJ.")
    
    st.sidebar.header("📡 API de Cancelamento")
    dry_run = st.sidebar.toggle("Modo Teste (Dry-run)", value=bool(st.secrets.get("api_client", {}).get("dry_run", False)), help="Se ativo, simula as chamadas de API sem cancelar as atividades de fato.")
    st.session_state[SK.CFG] = {"dry_run": dry_run}
    
    # Exibe regras por pasta para informação
    with st.sidebar.expander("Regras de Similaridade por Pasta"):
        st.json(st.secrets.get("similarity", {}).get("cutoffs_por_pasta", {}))

    return dict(
        min_sim=min_sim, min_containment=min_containment, pre_delta=pre_delta,
        diff_limit=diff_limit, dias_hist=dias_hist, data_inicio=data_inicio, data_fim=data_fim,
        pastas=pastas_sel, status=status_sel, use_cnj=use_cnj, use_datajuri=use_datajuri,
        strict_only=strict_only, dj_client=datajuri_client_instance()
    )

def get_best_principal_id(group_rows: List[Dict], min_sim_pct: float, min_containment_pct: float) -> str:
    """Calcula qual item do grupo é o 'melhor principal' (medoid)."""
    best_id, max_avg_score = None, -1.0
    
    # Cache para evitar recalcular normalizações e metadados
    cache = {r['activity_id']: (normalize_for_match(r.get('Texto', ''), []), extract_meta(r.get('Texto', ''))) for r in group_rows}

    for candidate in group_rows:
        candidate_id = candidate['activity_id']
        c_norm, c_meta = cache[candidate_id]
        scores = []
        for other in group_rows:
            if other['activity_id'] == candidate_id: continue
            o_norm, o_meta = cache[other['activity_id']]
            
            score, details = combined_score(c_norm, o_norm, c_meta, o_meta)
            if score >= min_sim_pct and details['contain'] >= min_containment_pct:
                scores.append(score)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        if avg_score > max_avg_score:
            max_avg_score, best_id = avg_score, candidate_id
            
    return best_id or group_rows[0]['activity_id'] # Fallback para o primeiro item

def render_group(group_rows: List[Dict], params: Dict):
    """Renderiza um único grupo de atividades duplicadas."""
    group_id = group_rows[0]["activity_id"]
    
    # Inicializa ou recupera o estado deste grupo
    state = st.session_state[SK.GROUP_STATES].setdefault(group_id, {
        "principal_id": group_rows[0]["activity_id"],
        "open_compare": None,
        "cancelados": set()
    })

    # Garante que o principal_id ainda existe no grupo (pode mudar com filtros)
    if not any(r["activity_id"] == state["principal_id"] for r in group_rows):
        state["principal_id"] = group_rows[0]["activity_id"]

    principal = next(r for r in group_rows if r["activity_id"] == state["principal_id"])
    p_norm = normalize_for_match(principal.get("Texto", ""), [])
    p_meta = extract_meta(principal.get("Texto", ""))

    # Filtra os itens a serem exibidos com base no modo estrito
    if params['strict_only']:
        visible_rows = [principal]
        for row in group_rows:
            if row["activity_id"] == principal["activity_id"]: continue
            r_norm = normalize_for_match(row.get("Texto", ""), [])
            r_meta = extract_meta(row.get("Texto", ""))
            score, details = combined_score(p_norm, r_norm, p_meta, r_meta)
            if score >= (params['min_sim'] * 100) and details['contain'] >= params['min_containment']:
                visible_rows.append(row)
    else:
        visible_rows = group_rows

    expander_title = f"Grupo com {len(group_rows)} itens (exibindo {len(visible_rows)}) — Pasta: {group_rows[0].get('activity_folder', '')}"
    
    with st.expander(expander_title):
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            st.info("Dica: para marcar um item para cancelar, primeiro clique em '⚖️ Comparar com Principal'.")
        with col2:
            if st.button("⭐ Definir Melhor Principal", key=f"auto_princ_{group_id}", use_container_width=True):
                best_id = get_best_principal_id(group_rows, params['min_sim'] * 100, params['min_containment'])
                state["principal_id"] = best_id
                state["open_compare"] = None # Fecha comparações abertas
                st.rerun()

        st.markdown("---")

        for row in visible_rows:
            rid = row["activity_id"]
            is_principal = (rid == state["principal_id"])
            is_comparing = (rid == state["open_compare"])
            is_marked_for_cancel = (rid in state["cancelados"])

            card_class = "card"
            if is_principal: card_class += " card-principal"
            if is_marked_for_cancel: card_class += " card-cancelado"
            
            with st.container():
                st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)
                c1, c2 = st.columns([0.7, 0.3])
                with c1:
                    # Informações básicas da atividade
                    dt = pd.to_datetime(row.get("activity_date")).tz_localize(TZ_UTC).tz_convert(TZ_SP) if pd.notna(row.get("activity_date")) else None
                    st.markdown(f"**ID:** `{rid}` {'⭐ **Principal**' if is_principal else ''} {'🗑️ **Marcado p/ Cancelar**' if is_marked_for_cancel else ''}")
                    st.caption(f"**Data:** {dt.strftime('%d/%m/%Y %H:%M') if dt else 'N/A'} | **Status:** {row.get('activity_status','')} | **Usuário:** {row.get('user_profile_name','')}")

                    # Badge de similaridade
                    if not is_principal:
                        r_norm = normalize_for_match(row.get("Texto", ""), [])
                        r_meta = extract_meta(row.get("Texto", ""))
                        score, details = combined_score(p_norm, r_norm, p_meta, r_meta)
                        
                        score_pct = params['min_sim'] * 100
                        badge_color = "badge-green" if score >= score_pct + 5 else "badge-yellow" if score >= score_pct else "badge-red"
                        tooltip = f"Set: {details['set']:.0f}% | Sort: {details['sort']:.0f}% | Contain: {details['contain']:.0f}% | Bônus: {details['bonus']}"
                        st.markdown(f"<span class='similarity-badge {badge_color}' title='{tooltip}'>Similaridade: {score:.0f}%</span>", unsafe_allow_html=True)

                    # Texto resumido
                    st.text_area("Texto", row.get("Texto", ""), height=100, disabled=True, key=f"txt_{rid}")

                with c2:
                    # Botões de ação
                    if not is_principal:
                        if st.button("⭐ Tornar Principal", key=f"mkp_{rid}", use_container_width=True):
                            state["principal_id"] = rid
                            state["open_compare"] = None
                            st.rerun()
                        
                        if st.button("⚖️ Comparar com Principal", key=f"cmp_{rid}", use_container_width=True):
                            state["open_compare"] = rid if not is_comparing else None # Toggle
                            st.rerun()

                    # Checkbox de cancelamento (só aparece após comparar)
                    if not is_principal and is_comparing:
                        st.markdown("---")
                        cancel_checked = st.checkbox("🗑️ Marcar para Cancelar", value=is_marked_for_cancel, key=f"cancel_{rid}")
                        if cancel_checked != is_marked_for_cancel:
                            if cancel_checked:
                                state["cancelados"].add(rid)
                            else:
                                state["cancelados"].discard(rid)
                            st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

        # Se uma comparação estiver aberta, renderiza o diff
        if state["open_compare"]:
            comparado_row = next((r for r in group_rows if r["activity_id"] == state["open_compare"]), None)
            if comparado_row:
                st.markdown("---")
                st.subheader("Comparação Detalhada (Diff)")
                c1, c2 = st.columns(2)
                c1.markdown(f"**Principal: ID `{principal['activity_id']}`**")
                c2.markdown(f"**Comparado: ID `{comparado_row['activity_id']}`**")
                
                hA, hB = highlight_diffs_safe(principal.get("Texto", ""), comparado_row.get("Texto", ""), params['diff_limit'])
                c1.markdown(hA, unsafe_allow_html=True)
                c2.markdown(hB, unsafe_allow_html=True)

# =============================================================================
# AÇÕES (EXPORTAR, PROCESSAR) E CALIBRAÇÃO
# =============================================================================

def export_groups_csv(groups: List[List[Dict]]) -> bytes:
    """Gera um arquivo CSV a partir dos grupos de duplicatas."""
    rows = []
    for i, g in enumerate(groups):
        for r in g:
            rows.append({
                "group_index": i + 1,
                "group_size": len(g),
                "activity_id": r.get("activity_id"),
                "activity_folder": r.get("activity_folder"),
                "activity_date": r.get("activity_date"),
                "activity_status": r.get("activity_status"),
                "user_profile_name": r.get("user_profile_name"),
                "Texto": r.get("Texto","")
            })
    if not rows: return b""
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

def process_cancellations(groups: List[List[Dict]], user: str):
    """Processa todos os cancelamentos marcados na interface."""
    client = api_client()
    if not client:
        st.error("Cliente de API não configurado. Não é possível processar cancelamentos.")
        return

    # Atualiza o modo dry_run do cliente com base na seleção da UI
    client.dry_run = st.session_state[SK.CFG].get("dry_run", True)
    
    to_cancel = []
    for g in groups:
        gid = g[0]["activity_id"]
        state = st.session_state[SK.GROUP_STATES].get(gid, {})
        to_cancel.extend(list(state.get("cancelados", set())))

    if not to_cancel:
        st.info("Nenhuma atividade foi marcada para cancelamento.")
        return

    st.info(f"Iniciando o cancelamento de {len(to_cancel)} atividades...")
    progress = st.progress(0)
    results = {"ok": 0, "err": 0}
    
    for i, act_id in enumerate(to_cancel):
        try:
            response = client.activity_canceled(activity_id=act_id, user_name=user)
            if response and (response.get("ok") or response.get("success")):
                results["ok"] += 1
                logging.info(f"Sucesso ao cancelar {act_id}.")
            else:
                results["err"] += 1
                st.warning(f"Falha ao cancelar {act_id}. Resposta: {response}")
                logging.error(f"Falha ao cancelar {act_id}. Resposta: {response}")
        except Exception as e:
            results["err"] += 1
            st.error(f"Erro de exceção ao cancelar {act_id}: {e}")
            logging.exception(f"Erro de exceção ao cancelar {act_id}")
        
        progress.progress((i + 1) / len(to_cancel))

    st.success(f"Processamento concluído! Sucessos: {results['ok']}, Falhas: {results['err']}.")
    if client.dry_run:
        st.warning("Atenção: O modo Teste (Dry-run) está ativo. Nenhuma atividade foi realmente cancelada.")
    
    # Limpa os estados de cancelamento após o processamento
    for g in groups:
        gid = g[0]["activity_id"]
        if gid in st.session_state[SK.GROUP_STATES]:
            st.session_state[SK.GROUP_STATES][gid]["cancelados"].clear()
    
    # Força a atualização dos dados para refletir as mudanças
    carregar_dados_mysql.clear()
    st.session_state.pop(SK.SIMILARITY_CACHE, None)
    st.rerun()

def render_calibration_tab(df: pd.DataFrame):
    """Renderiza a aba de calibração para análise de similaridade."""
    st.subheader("📊 Calibração de Similaridade por Pasta")
    st.info("Esta ferramenta ajuda a encontrar o limiar de similaridade ideal para cada pasta, analisando pares aleatórios de atividades.")

    if df.empty:
        st.warning("Não há dados para calibrar.")
        return

    pasta_opts = sorted(df["activity_folder"].dropna().unique())
    pasta = st.selectbox("Selecione uma pasta para amostragem:", pasta_opts)
    
    col1, col2 = st.columns(2)
    num_samples = col1.slider("Nº de Pares Aleatórios", 50, 2000, 500, 50)
    min_containment_filter = col2.slider("Filtro de Containment Mínimo (%)", 0, 100, 0, 1)

    if st.button("Analisar Pasta"):
        sample_df = df[df["activity_folder"] == pasta].copy()
        if len(sample_df) < 2:
            st.warning("A pasta selecionada tem menos de 2 atividades para comparar."); return

        # Prepara dados para análise
        stopwords_extra = st.secrets.get("similarity", {}).get("stopwords_extra", [])
        sample_df["_meta"] = sample_df["Texto"].apply(extract_meta)
        sample_df["_norm"] = sample_df["Texto"].apply(lambda t: normalize_for_match(t, stopwords_extra))
        sample_df = sample_df.reset_index()

        # Gera pares aleatórios
        n = len(sample_df)
        indices = np.arange(n)
        pairs = set()
        rng = np.random.default_rng(seed=42)
        while len(pairs) < min(num_samples, (n * (n - 1)) // 2):
            i, j = rng.choice(indices, size=2, replace=False)
            pairs.add(tuple(sorted((i, j))))

        scores = []
        progress = st.progress(0, text="Calculando scores...")
        for i, (idx1, idx2) in enumerate(pairs):
            row1, row2 = sample_df.iloc[idx1], sample_df.iloc[idx2]
            score, details = combined_score(row1["_norm"], row2["_norm"], row1["_meta"], row2["_meta"])
            if details["contain"] >= min_containment_filter:
                scores.append({"score": score, "containment": details["contain"]})
            progress.progress((i + 1) / len(pairs))
        progress.empty()
        
        if not scores:
            st.info("Nenhum par encontrado após aplicar o filtro de containment."); return
            
        df_scores = pd.DataFrame(scores)
        
        st.write("Estatísticas Descritivas dos Scores:")
        st.dataframe(df_scores["score"].describe(percentiles=[.25, .5, .75, .9, .95, .99]))

        if alt:
            chart = alt.Chart(df_scores).mark_bar().encode(
                x=alt.X("score:Q", bin=alt.Bin(maxbins=50), title="Score de Similaridade"),
                y=alt.Y("count()", title="Contagem de Pares")
            ).properties(
                title=f"Distribuição de Similaridade para a Pasta: {pasta}",
                height=300
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("Biblioteca 'altair' não instalada. Exibindo gráfico simples.")
            st.line_chart(df_scores["score"])

# =============================================================================
# FLUXO PRINCIPAL DO APLICATIVO
# =============================================================================

def main():
    """Função principal que executa o aplicativo Streamlit."""
    st.title(APP_TITLE)
    
    # Inicializa o session_state
    for key in [SK.USERNAME, SK.SIMILARITY_CACHE, SK.PAGE_NUMBER, SK.GROUP_STATES, SK.CFG]:
        if key not in st.session_state:
            st.session_state[key] = 0 if key == SK.PAGE_NUMBER else {}

    # Login simples para registrar o usuário nas ações
    if not st.session_state.get(SK.USERNAME):
        with st.sidebar.form("login_form"):
            username = st.text_input("Nome de Usuário")
            if st.form_submit_button("Entrar"):
                if username:
                    st.session_state[SK.USERNAME] = username
                    st.rerun()
                else:
                    st.sidebar.error("Por favor, insira um nome de usuário.")
        st.info("👋 Bem-vindo! Por favor, faça o login na barra lateral para começar.")
        st.stop()

    engine = db_engine_mysql()
    df_full = carregar_dados_mysql(engine, 365) # Carrega um ano para os filtros
    
    params = sidebar_controls(df_full)
    
    # Carrega os dados para o período de análise selecionado
    df_analysis = carregar_dados_mysql(engine, params["dias_hist"])
    if df_analysis.empty:
        st.warning("Nenhuma atividade encontrada para o período de análise selecionado.")
        st.stop()

    # Aplica filtros de visualização (data, pasta, status)
    mask = (
        (df_analysis["activity_date"].dt.date >= params["data_inicio"]) &
        (df_analysis["activity_date"].dt.date <= params["data_fim"])
    )
    if params["pastas"]: mask &= df_analysis["activity_folder"].isin(params["pastas"])
    if params["status"]: mask &= df_analysis["activity_status"].isin(params["status"])
    df_view = df_analysis[mask].copy()

    # Abas principais da aplicação
    tab1, tab2 = st.tabs(["🔎 Análise de Duplicidades", "📊 Calibração"])

    with tab1:
        groups = criar_grupos_de_duplicatas(df_view, params)
        
        st.metric("Grupos de Duplicatas Encontrados", len(groups))

        # Paginação dos grupos
        page_size = st.number_input("Grupos por página", min_value=5, value=DEFAULTS["itens_por_pagina"], step=5)
        total_pages = max(1, math.ceil(len(groups) / page_size))
        page_num = st.number_input("Página", min_value=1, max_value=total_pages, value=1, step=1)
        
        start_idx = (page_num - 1) * page_size
        end_idx = start_idx + page_size
        
        st.caption(f"Exibindo grupos {start_idx + 1}–{min(end_idx, len(groups))} de {len(groups)}")

        for group in groups[start_idx:end_idx]:
            render_group(group, params)

        st.markdown("---")
        st.header("⚡ Ações em Massa")
        col_a, col_b = st.columns(2)
        with col_a:
            csv_data = export_groups_csv(groups)
            st.download_button(
                "⬇️ Exportar Grupos para CSV",
                data=csv_data,
                file_name="relatorio_duplicatas.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_b:
            if st.button("🚀 Processar Cancelamentos Marcados", type="primary", use_container_width=True):
                process_cancellations(groups, st.session_state.get(SK.USERNAME, "usuário_desconhecido"))

    with tab2:
        render_calibration_tab(df_full)

if __name__ == "__main__":
    main()
