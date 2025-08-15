# -*- coding: utf-8 -*-
"""
Verificador de Duplicidade — Upgrades Extras
- Pré‑índice por CNJ e/ou Pasta (DataJuri opcional)
- Cutoffs por pasta via st.secrets
- Histograma de similaridade para calibração
- Retry/Rate‑limit/Dry‑run no client HTTP
- Campos destacáveis (Processo/CNJ, Órgão, Tipo de Doc/Comunicação)
- Badge com % de similaridade e tooltip com componentes
- Diff adaptativo (não trava com textos muito grandes)
- Checkbox “Marcar para cancelar” só aparece após abrir a comparação
"""
from __future__ import annotations

import os, re, json, html, logging, time, math
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

# ====== opcionais ======
try:
    import altair as alt
except Exception:
    alt = None

# Clientes locais
try:
    from api_functions_retry import HttpClientRetry
except Exception:
    HttpClientRetry = None

try:
    from datajuri_client import DataJuriClient
except Exception:
    DataJuriClient = None

# =============================================================================
# Configuração básica / Constantes
# =============================================================================
APP_TITLE = "Verificador de Duplicidade — Upgrades"
TZ_SP  = ZoneInfo("America/Sao_Paulo")
TZ_UTC = ZoneInfo("UTC")

SUFFIX = "_v24_upgrades"
class SK:
    LOGGED_IN = "logged_in"
    USERNAME = "username"
    LAST_UPDATE = f"last_update_{SUFFIX}"
    SIMILARITY_CACHE = f"simcache_{SUFFIX}"
    PAGE_NUMBER = f"page_{SUFFIX}"
    SAVED_FILTERS = f"saved_filters_{SUFFIX}"
    GROUP_STATES = f"group_states_{SUFFIX}"
    CFG = f"cfg_{SUFFIX}"

DEFAULTS = {
    "itens_por_pagina": 10,
    "dias_filtro_inicio": 7,
    "dias_filtro_fim": 14,
    "min_sim_global": 90,
    "min_containment": 55,
    "pre_cutoff_delta": 10,
    "diff_hard_limit": 12000,
}

st.set_page_config(layout="wide", page_title=APP_TITLE)
st.markdown("""
<style>
pre.highlighted-text {
  white-space: pre-wrap; word-wrap: break-word; font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
  font-size: .9em; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; height: 360px; overflow-y: auto;
}
.similarity-badge { padding: 3px 6px; border-radius: 5px; color: black; font-weight: 600; display: inline-block; margin-bottom: 6px; }
.diff-del { background-color: #ffcdd2 !important; text-decoration: none !important; }
.diff-ins { background-color: #c8e6c9 !important; text-decoration: none !important; }
.card-cancelado { background-color: #f5f5f5; border-left: 5px solid #e0e0e0; padding: 10px; margin-bottom: 5px; border-radius: 5px;}
.card-principal { border-left: 5px solid #4CAF50; }
.badge-green { background:#C8E6C9; }
.badge-yellow{ background:#FFF9C4; }
.badge-red   { background:#FFCDD2; }
.meta-chip { background:#E0F7FA; padding:2px 6px; margin-right:6px; border-radius:8px; display:inline-block; font-size:0.85em; }
.small-muted { color:#777; font-size:0.85em; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Inicialização de serviços
# =============================================================================

def _init_session_state():
    if SK.CFG not in st.session_state:
        st.session_state[SK.CFG] = {}
    for k in [SK.LOGGED_IN, SK.USERNAME, SK.LAST_UPDATE, SK.SIMILARITY_CACHE, SK.PAGE_NUMBER, SK.SAVED_FILTERS, SK.GROUP_STATES]:
        st.session_state.setdefault(k, None if k in [SK.LAST_UPDATE, SK.SIMILARITY_CACHE] else {} if k==SK.SAVED_FILTERS else False if k==SK.LOGGED_IN else 0 if k==SK.PAGE_NUMBER else {} if k==SK.GROUP_STATES else "")

@st.cache_resource
def db_engine_mysql() -> Engine:
    cfg = st.secrets.get("database", {})
    host, user, pw, db = [cfg.get(k) for k in ["host", "user", "password", "name"]]
    if not all([host, user, pw, db]):
        st.error("Credenciais do banco (MySQL) ausentes em st.secrets['database']."); st.stop()
    try:
        engine = create_engine(f"mysql+mysqlconnector://{user}:{pw}@{host}/{db}", pool_pre_ping=True, pool_recycle=3600)
        with engine.connect(): pass
        return engine
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error("Erro ao conectar no banco (MySQL)."); st.stop()

@st.cache_resource
def api_client() -> Optional[HttpClientRetry]:
    cfg = st.secrets.get("api", {})
    url_api, entity_id, token = [cfg.get(k) for k in ["url_api", "entity_id", "token"]]
    client_cfg = st.secrets.get("api_client", {})
    if not all([url_api, entity_id, token]) or HttpClientRetry is None:
        return None
    return HttpClientRetry(
        base_url=url_api, entity_id=entity_id, token=token,
        calls_per_second=float(client_cfg.get("calls_per_second", 3.0)),
        max_attempts=int(client_cfg.get("max_attempts", 3)),
        timeout=int(client_cfg.get("timeout", 15)),
        dry_run=bool(client_cfg.get("dry_run", False))
    )

@st.cache_resource
def datajuri_client() -> Optional[DataJuriClient]:
    dj = st.secrets.get("datajuri", {})
    if not dj or DataJuriClient is None:
        return None
    try:
        c = DataJuriClient(**dj)
        c.ensure_token()
        return c
    except Exception as e:
        st.warning(f"DataJuri não inicializado: {e}")
        return None

# =============================================================================
# Leitura de dados
# =============================================================================

@st.cache_data(ttl=3600, hash_funcs={Engine: lambda _: None})
def carregar_dados_mysql(eng: Engine, dias_historico: int) -> pd.DataFrame:
    limite = date.today() - timedelta(days=dias_historico)
    q = text("""
        SELECT activity_id, activity_folder, user_profile_name, activity_date, activity_status, Texto
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type='Verificar'
          AND (activity_status='Aberta' OR DATE(activity_date) >= :limite)
    """)
    try:
        with eng.connect() as conn:
            df = pd.read_sql(q, conn, params={"limite": limite})
        if df.empty: return pd.DataFrame()
        df["activity_id"] = df["activity_id"].astype(str)
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
        df["Texto"] = df["Texto"].fillna("").astype(str)
        df["status_ord"] = df["activity_status"].map({"Aberta": 0}).fillna(1)
        df = df.sort_values(["activity_id", "status_ord"]).drop_duplicates("activity_id", keep="first").drop(columns="status_ord")
        df = df.sort_values(["activity_folder", "activity_date"], ascending=[True, False])
        return df
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error("Erro ao carregar dados do banco principal."); return pd.DataFrame()

# =============================================================================
# Normalização, extração de metacampos e similaridade
# =============================================================================

CNJ_RE = re.compile(r"(?:\b|^)(\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})(?:\b|$)")
URL_RE = re.compile(r"https?://\S+")
DATENUM_RE = re.compile(r"\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})\b")
NUM_RE = re.compile(r"\d+")

STOPWORDS_BASE = set("""de da do das dos e em a o os as na no para por com que ao aos às à um uma umas uns tipo titulo inteiro teor publicado publicacao disponibilizacao orgao vara tribunal processo recurso intimacao notificacao justica nacional diario djen poder judiciario trabalho""".split())

def extract_cnj(text: str) -> Optional[str]:
    m = CNJ_RE.search(text or "")
    return m.group(1) if m else None

def extract_meta(text: str) -> Dict[str, str]:
    t = text or ""
    meta = {}
    # Processo/CNJ
    cnj = extract_cnj(t)
    if not cnj:
        m = re.search(r"PROCESSO:\s*([0-9\-.]+)", t, re.IGNORECASE)
        if m: cnj = m.group(1)
    meta["processo"] = cnj or ""

    # Órgão/Vara/Tipo
    m = re.search(r"\bORGAO:\s*([^-\n\r]+)", t, re.IGNORECASE)
    if m: meta["orgao"] = m.group(1).strip()
    m = re.search(r"\bVARA\s+DO\s+TRABALHO\s+[^-\n\r]+", t, re.IGNORECASE)
    if m: meta["vara"] = m.group(0).strip()
    m = re.search(r"\bTIPO\s+DE\s+DOCUMENTO:\s*([^-]+)", t, re.IGNORECASE)
    if m: meta["tipo_doc"] = m.group(1).strip()
    m = re.search(r"\bTIPO\s+DE\s+COMUNICACAO:\s*([^-]+)", t, re.IGNORECASE)
    if m: meta["tipo_com"] = m.group(1).strip()
    return meta

def normalize_for_match(text: str, stopwords_extra: List[str] | None = None) -> str:
    if not isinstance(text, str): return ""
    t = text
    t = URL_RE.sub(" url ", t)
    t = CNJ_RE.sub(" numproc ", t)
    t = DATENUM_RE.sub(" data ", t)
    t = NUM_RE.sub(" # ", t)
    t = unidecode(t.lower())
    t = re.sub(r"[^\w\s]", " ", t)
    tokens = [w for w in t.split() if w not in STOPWORDS_BASE and (not stopwords_extra or w not in stopwords_extra)]
    return " ".join(tokens)

def token_containment(a_tokens: List[str], b_tokens: List[str]) -> float:
    if not a_tokens or not b_tokens: return 0.0
    if len(a_tokens) <= len(b_tokens):
        small, big = a_tokens, set(b_tokens)
    else:
        small, big = b_tokens, set(a_tokens)
    hits = sum(1 for w in small if w in big)
    return 100.0 * (hits / max(1, len(small)))

def fields_bonus(meta_a: Dict[str,str], meta_b: Dict[str,str]) -> int:
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

def length_penalty(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 or lb == 0: return 0.8
    diff = abs(la - lb) / max(la, lb)
    # penaliza até 10%
    return max(0.9, 1.0 - diff * 0.5)

def combined_score(a_norm: str, b_norm: str, meta_a: Dict[str,str], meta_b: Dict[str,str]) -> Tuple[float, Dict[str,float]]:
    set_ratio = fuzz.token_set_ratio(a_norm, b_norm)
    sort_ratio = fuzz.token_sort_ratio(a_norm, b_norm)
    a_toks = a_norm.split(); b_toks = b_norm.split()
    cont = token_containment(a_toks, b_toks)
    lp = length_penalty(a_norm, b_norm)
    bonus = fields_bonus(meta_a, meta_b)
    base = 0.6*set_ratio + 0.2*sort_ratio + 0.2*cont
    score = max(0.0, min(100.0, base * lp + bonus))
    details = {"set": set_ratio, "sort": sort_ratio, "contain": cont, "len_pen": lp, "bonus": bonus, "base": base}
    return score, details

# =============================================================================
# Agrupamento
# =============================================================================

def build_buckets(df: pd.DataFrame, use_cnj: bool, use_datajuri: bool, dj_client: Optional[DataJuriClient]) -> Dict[str, List[int]]:
    """
    Retorna dict bucket_key -> list(row_index)
    - Sempre agrupa por activity_folder
    - Se use_cnj: sub-bucket por CNJ extraído
    - Se use_datajuri: tenta Pasta real via DataJuri
    """
    buckets = defaultdict(list)
    for i, row in df.iterrows():
        folder = str(row.get("activity_folder") or "")
        text = row.get("Texto") or ""
        meta = row.get("_meta", {})
        cnj = meta.get("processo") or extract_cnj(text) or ""
        key = f"folder::{folder}"
        # DataJuri: substitui bucket pelo nome da Pasta, se disponível
        if use_datajuri and dj_client and cnj:
            try:
                pasta = dj_client.get_pasta_by_cnj_cached(cnj)
                if pasta:
                    key = f"pasta::{pasta}"
            except Exception:
                pass
        elif use_cnj:
            key = f"{key}::cnj::{cnj or 'SEM_CNJ'}"
        buckets[key].append(i)
    return buckets

def criar_grupos_de_duplicatas(df: pd.DataFrame,
                               min_sim: float,
                               min_containment: float,
                               pre_cutoff_delta: int,
                               use_cnj: bool,
                               use_datajuri: bool,
                               dj_client: Optional[DataJuriClient],
                               stopwords_extra: Optional[List[str]] = None) -> List[List[Dict]]:
    """
    Retorna lista de grupos (cada grupo é lista de dicts de linhas do df).
    Respeita cutoffs por pasta definidos em st.secrets['similarity']['cutoffs_por_pasta'].
    """
    if df.empty: return []
    # Cache control
    sig = (tuple(sorted(df["activity_id"])), round(min_sim,3), round(min_containment,3), pre_cutoff_delta, use_cnj, use_datajuri, "cutoffs_v1")
    cached = st.session_state.get(SK.SIMILARITY_CACHE)
    if cached and cached.get("sig") == sig:
        return cached["groups"]

    work = df.copy()
    # precompute meta & norms
    stop_extras = [s.strip() for s in st.secrets.get("similarity", {}).get("stopwords_extra", [])] if stopwords_extra is None else stopwords_extra
    work["_meta"] = work["Texto"].apply(extract_meta)
    work["_norm"] = work["Texto"].apply(lambda t: normalize_for_match(t, stop_extras))

    # bucketing
    buckets = build_buckets(work, use_cnj=use_cnj, use_datajuri=use_datajuri, dj_client=dj_client)

    # cutoffs por pasta
    cutoffs_map = st.secrets.get("similarity", {}).get("cutoffs_por_pasta", {}) or {}

    groups = []
    # Barra de progresso
    ph = st.sidebar.empty(); pb = ph.progress(0, text="Agrupando duplicatas...")
    processed = 0; total = len(work)

    def eff_threshold_for_bucket(bkey: str) -> float:
        # bkey: "folder::<folder>::cnj::<n>" ou "pasta::<pasta>"
        if bkey.startswith("folder::"):
            folder = bkey.split("::")[1]
            return float(cutoffs_map.get(folder, min_sim))
        elif bkey.startswith("pasta::"):
            pasta = bkey.split("::", 1)[1]
            return float(cutoffs_map.get(pasta, min_sim))
        return min_sim

    for bkey, idxs in buckets.items():
        if len(idxs) < 2:
            processed += len(idxs); pb.progress(min(1.0, processed/total)); continue

        dff = work.loc[idxs].reset_index(drop=False).rename(columns={"index":"orig_idx"})
        texts = dff["_norm"].tolist()

        # thresholds para este bucket
        eff_min = eff_threshold_for_bucket(bkey)
        pre_cut = max(0, int(eff_min*100) - pre_cutoff_delta)

        # Pré filtro com cdist no token_set_ratio
        prelim = process.cdist(texts, texts, scorer=fuzz.token_set_ratio, score_cutoff=pre_cut)
        n = len(dff)
        visited = set()
        memo_score: Dict[Tuple[int,int], Tuple[float,Dict[str,float]]] = {}

        def edge_ok(i,j) -> bool:
            key = (i,j) if i<=j else (j,i)
            if key in memo_score: s, det = memo_score[key]
            else:
                s, det = combined_score(dff.loc[i, "_norm"], dff.loc[j, "_norm"], dff.loc[i, "_meta"], dff.loc[j, "_meta"])
                memo_score[key] = (s, det)
            if det["contain"] < min_containment: return False
            return s >= (eff_min*100.0)

        for i in range(n):
            if i in visited: continue
            comp = {i}; dq = deque([i]); visited.add(i)
            while dq:
                k = dq.popleft()
                for j in range(n):
                    if j in visited: continue
                    if prelim[k][j] >= pre_cut and edge_ok(k,j):
                        visited.add(j); comp.add(j); dq.append(j)
            if len(comp) > 1:
                comp_idxs = sorted(list(comp), key=lambda ix: dff.loc[ix, "activity_date"] if pd.notna(dff.loc[ix, "activity_date"]) else pd.Timestamp(0), reverse=True)
                groups.append([work.iloc[dff.loc[ix, "orig_idx"]].to_dict() for ix in comp_idxs])

        processed += len(idxs)
        pb.progress(min(1.0, processed/total), text=f"Agrupando (bucket {bkey})...")
    ph.empty()

    st.session_state[SK.SIMILARITY_CACHE] = {"sig": sig, "groups": groups}
    return groups

# =============================================================================
# Diff seguro (evita travar)
# =============================================================================

def highlight_diffs_safe(text1: str, text2: str, hard_limit: int = 12000) -> Tuple[str,str]:
    t1, t2 = text1 or "", text2 or ""
    if (len(t1) + len(t2)) <= hard_limit:
        return highlight_diffs(t1, t2)
    # fallback por sentença / blocos
    def split_sentences(t: str, max_parts=400):
        parts = re.split(r"([.!?\\n]+)", t)
        # recombina mantendo pontuação
        seq = []
        for i in range(0, len(parts), 2):
            seg = parts[i]
            sep = parts[i+1] if i+1 < len(parts) else ""
            seq.append((seg+sep).strip())
            if len(seq) >= max_parts: break
        return seq
    s1 = " ".join(split_sentences(t1))
    s2 = " ".join(split_sentences(t2))
    h1, h2 = highlight_diffs(s1, s2)
    note = "<div class='small-muted'>⚠️ Diff parcial por tamanho — comparado por sentenças.</div>"
    return (note + h1, note + h2)

def highlight_diffs(a: str, b: str) -> Tuple[str,str]:
    tokens1 = [tok for tok in re.split(r'(\W+)', a or "") if tok]
    tokens2 = [tok for tok in re.split(r'(\W+)', b or "") if tok]
    sm = SequenceMatcher(None, tokens1, tokens2, autojunk=False)
    out1, out2 = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        s1, s2 = html.escape("".join(tokens1[i1:i2])), html.escape("".join(tokens2[j1:j2]))
        if tag == 'equal': out1.append(s1); out2.append(s2)
        elif tag == 'replace': out1.append(f"<span class='diff-del'>{s1}</span>"); out2.append(f"<span class='diff-ins'>{s2}</span>")
        elif tag == 'delete': out1.append(f"<span class='diff-del'>{s1}</span>")
        elif tag == 'insert': out2.append(f"<span class='diff-ins'>{s2}</span>")
    return (f"<pre class='highlighted-text'>{''.join(out1)}</pre>", f"<pre class='highlighted-text'>{''.join(out2)}</pre>")

def as_sp(ts: any) -> Optional[datetime]:
    if pd.isna(ts): return None
    if isinstance(ts, str): ts = pd.to_datetime(ts)
    if isinstance(ts, (int, float)): ts = pd.to_datetime(ts, unit='s')
    if ts.tzinfo is None: ts = ts.tz_localize(TZ_UTC)
    return ts.tz_convert(TZ_SP)

# =============================================================================
# Sidebar / Filtros
# =============================================================================

def sidebar_controls(df_full: pd.DataFrame) -> Dict:
    st.sidebar.header("Sessão")
    if st.session_state.get(SK.USERNAME):
        st.sidebar.success(f"Logado como **{st.session_state[SK.USERNAME]}**")
    if st.sidebar.button("🔄 Atualizar dados"):
        st.session_state.pop(SK.SIMILARITY_CACHE, None)
        carregar_dados_mysql.clear()

    st.sidebar.header("Parâmetros de Similaridade")
    sim_cfg = st.secrets.get("similarity", {})
    min_sim = st.sidebar.slider("Similaridade mínima (%)", 0, 100, int(sim_cfg.get("min_sim_global", DEFAULTS["min_sim_global"])), 1) / 100.0
    min_containment = st.sidebar.slider("Containment mínimo (%)", 0, 100, int(sim_cfg.get("min_containment", DEFAULTS["min_containment"])), 1)
    pre_delta = st.sidebar.slider("Pré‑corte (delta)", 0, 30, int(sim_cfg.get("pre_cutoff_delta", DEFAULTS["pre_cutoff_delta"])), 1)
    diff_limit = st.sidebar.number_input("Limite duro do Diff (caracteres)", min_value=3000, value=int(sim_cfg.get("diff_hard_limit", DEFAULTS["diff_hard_limit"])), step=1000)

    st.sidebar.subheader("Escopo da comparação")
    dias_hist = st.sidebar.number_input("Dias de histórico", min_value=7, value=90, step=1)
    data_inicio = st.sidebar.date_input("Data Início", date.today()-timedelta(days=DEFAULTS["dias_filtro_inicio"]))
    data_fim = st.sidebar.date_input("Data Fim", date.today()+timedelta(days=DEFAULTS["dias_filtro_fim"]))
    pastas_opts = sorted(df_full["activity_folder"].dropna().unique()) if not df_full.empty else []
    status_opts = sorted(df_full["activity_status"].dropna().unique()) if not df_full.empty else []
    pastas_sel = st.sidebar.multiselect("Pastas", pastas_opts)
    status_sel = st.sidebar.multiselect("Status", status_opts)

    st.sidebar.subheader("Pré‑índice")
    use_cnj = st.sidebar.toggle("Restringir por nº do processo (CNJ)", value=True)
    use_datajuri = st.sidebar.toggle("Usar DataJuri p/ Pasta", value=False if datajuri_client() is None else True)
    st.sidebar.caption("Se ligado, só compara documentos com o mesmo CNJ ou mesma Pasta DataJuri.")

    st.sidebar.subheader("API de Cancelamento")
    dry_run = st.sidebar.toggle("Dry‑run (não chama a API real)", value=bool(st.secrets.get("api_client", {}).get("dry_run", False)))
    st.session_state[SK.CFG]["dry_run"] = dry_run

    # Regras por pasta (informativas)
    st.sidebar.subheader("Regras por Pasta (secrets)")
    cutoffs_por_pasta = st.secrets.get("similarity", {}).get("cutoffs_por_pasta", {})
    if cutoffs_por_pasta:
        st.sidebar.json(cutoffs_por_pasta, expanded=False)

    return dict(
        min_sim=min_sim, min_containment=min_containment, pre_delta=pre_delta,
        diff_limit=diff_limit, dias_hist=dias_hist, data_inicio=data_inicio, data_fim=data_fim,
        pastas=pastas_sel, status=status_sel, use_cnj=use_cnj, use_datajuri=use_datajuri
    )

# =============================================================================
# UI de grupos
# =============================================================================

def color_for_badge(score: float, min_sim_pct: float) -> str:
    if score >= (min_sim_pct + 5): return "badge-green"
    if score >= (min_sim_pct): return "badge-yellow"
    return "badge-red"

def render_group(group_rows: List[Dict], min_sim_pct: float, diff_limit: int):
    group_id = group_rows[0]["activity_id"]
    state = st.session_state[SK.GROUP_STATES].setdefault(group_id, {
        "principal_id": group_rows[0]["activity_id"],
        "open_compare": None,
        "cancelados": set()
    })

    with st.expander(f"Grupo com {len(group_rows)} itens — Pasta: {group_rows[0].get('activity_folder','')}", expanded=False):
        # Mapa de metas e norms para o principal
        principal = next(r for r in group_rows if r["activity_id"] == state["principal_id"])
        p_meta = extract_meta(principal.get("Texto","")); p_norm = normalize_for_match(principal.get("Texto",""))
        min_sim_abs = min_sim_pct

        for row in group_rows:
            rid = row["activity_id"]
            is_principal = (rid == state["principal_id"])
            is_open = (rid == state["open_compare"])
            is_cancel = (rid in state["cancelados"])

            card_class = "card-principal" if is_principal else ("card-cancelado" if is_cancel else "")
            st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)

            c1, c2 = st.columns([0.72, 0.28])
            with c1:
                dt = as_sp(row.get("activity_date"))
                st.markdown(f"**ID:** `{rid}` {'⭐ **Principal**' if is_principal else ''} {'🗑️ **Selecionado p/ cancelar**' if is_cancel else ''}")
                st.caption(f"**Data:** {dt.strftime('%d/%m/%Y %H:%M') if dt else 'N/A'} | **Status:** {row.get('activity_status','')} | **Usuário:** {row.get('user_profile_name','')}")
                # Badge de similaridade (somente se não for principal)
                if not is_principal:
                    r_norm = normalize_for_match(row.get("Texto",""))
                    r_meta = extract_meta(row.get("Texto",""))
                    s, det = combined_score(p_norm, r_norm, p_meta, r_meta)
                    badge = color_for_badge(s, min_sim_abs)
                    st.markdown(f"<span class='similarity-badge {badge}'>Similaridade: {int(round(s))}%</span>", unsafe_allow_html=True)
                    st.caption(f"set={int(det['set'])} | sort={int(det['sort'])} | contain={int(det['contain'])} | len_pen={det['len_pen']:.2f} | bonus={det['bonus']}")
                # Chips de metacampos
                r_meta_show = extract_meta(row.get("Texto",""))
                chips = []
                if r_meta_show.get("processo"): chips.append(f"<span class='meta-chip'>Processo: {r_meta_show['processo']}</span>")
                if r_meta_show.get("orgao"): chips.append(f"<span class='meta-chip'>Órgão: {r_meta_show['orgao']}</span>")
                if r_meta_show.get("tipo_doc"): chips.append(f"<span class='meta-chip'>Tipo Doc: {r_meta_show['tipo_doc']}</span>")
                if r_meta_show.get("tipo_com"): chips.append(f"<span class='meta-chip'>Tipo Com: {r_meta_show['tipo_com']}</span>")
                if chips: st.markdown(" ".join(chips), unsafe_allow_html=True)

                st.text_area("Texto", row.get("Texto",""), height=100, disabled=True, key=f"txt_{rid}")

            with c2:
                if not is_principal and st.button("⭐ Tornar Principal", key=f"mkp_{rid}"):
                    state["principal_id"] = rid; state["open_compare"] = None; st.rerun()

                if not is_principal and st.button("⚖️ Comparar com Principal", key=f"cmp_{rid}"):
                    state["open_compare"] = rid; st.rerun()

                # ⚠️ Checkbox aparece somente após abrir a comparação deste item
                if not is_principal and is_open:
                    ck = st.checkbox("🗑️ Marcar para Cancelar", value=is_cancel, key=f"cancel_{rid}")
                    if ck: state["cancelados"].add(rid)
                    else:  state["cancelados"].discard(rid)

            st.markdown("</div>", unsafe_allow_html=True)

        if state["open_compare"]:
            principal_row = next(r for r in group_rows if r["activity_id"] == state["principal_id"])
            comparado_row = next(r for r in group_rows if r["activity_id"] == state["open_compare"])
            st.markdown("---")
            c1, c2 = st.columns(2)
            c1.markdown(f"**Principal: ID `{principal_row['activity_id']}`**")
            c2.markdown(f"**Comparado: ID `{comparado_row['activity_id']}`**")
            hA, hB = highlight_diffs_safe(principal_row.get("Texto",""), comparado_row.get("Texto",""), diff_limit)
            c1.markdown(hA, unsafe_allow_html=True)
            c2.markdown(hB, unsafe_allow_html=True)
            if st.button("❌ Fechar comparação", key=f"close_{group_id}"):
                state["open_compare"] = None; st.rerun()

# =============================================================================
# Export e processamento
# =============================================================================

def export_groups_csv(groups: List[List[Dict]]) -> bytes:
    rows = []
    for g in groups:
        for r in g:
            rows.append({
                "group_size": len(g),
                "activity_id": r.get("activity_id"),
                "activity_folder": r.get("activity_folder"),
                "activity_date": r.get("activity_date"),
                "activity_status": r.get("activity_status"),
                "user_profile_name": r.get("user_profile_name"),
                "Texto": r.get("Texto","")
            })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

def process_actions(groups: List[List[Dict]], user: str):
    cli = api_client()
    if cli is None:
        st.error("Cliente de API não configurado (api_functions_retry.HttpClientRetry indisponível ou secrets incompletos).")
        return
    dry_run = st.session_state[SK.CFG].get("dry_run", False)
    total = 0; ok = 0; errs = 0
    for g in groups:
        gid = None
        # encontra estado do grupo pela primeira activity_id
        gid = g[0]["activity_id"]
        state = st.session_state[SK.GROUP_STATES].get(gid, {})
        canc = state.get("cancelados", set())
        if not canc: continue
        for rid in list(canc):
            total += 1
            resp = cli.activity_canceled(activity_id=rid, user_name=user)
            if resp and isinstance(resp, dict) and (resp.get("ok") or resp.get("success") or resp.get("status") in (200,201)):
                ok += 1
            elif dry_run:
                ok += 1
            else:
                errs += 1
    if ok and not errs:
        st.success(f"Ações processadas com sucesso: {ok}/{total}.")
    elif ok:
        st.warning(f"Ações parcialmente concluídas: {ok}/{total} com {errs} erros.")
    else:
        st.error("Falha ao processar as ações.")

# =============================================================================
# Calibração (histograma)
# =============================================================================

def render_calibration(df: pd.DataFrame, use_cnj: bool, use_datajuri: bool, dj_client: Optional[DataJuriClient]):
    st.subheader("📊 Calibração de Similaridade")
    pasta = st.selectbox("Selecione uma pasta (activity_folder) para amostragem:", sorted(df["activity_folder"].dropna().unique()))
    amostras = st.slider("Amostras (pares aleatórios)", 50, 1000, 200, 50)
    min_cont = st.slider("Containment mínimo para considerar no gráfico (%)", 0, 100, 0, 1)

    sample_df = df[df["activity_folder"]==pasta].copy()
    if len(sample_df) < 2:
        st.info("Pasta com menos de 2 itens.")
        return

    # prepara norms/meta
    stop_extras = [s.strip() for s in st.secrets.get("similarity", {}).get("stopwords_extra", [])]
    sample_df["_meta"] = sample_df["Texto"].apply(extract_meta)
    sample_df["_norm"] = sample_df["Texto"].apply(lambda t: normalize_for_match(t, stop_extras))

    # pega pares aleatórios
    idxs = sample_df.index.tolist()
    rng = np.random.default_rng(42)
    pairs = set()
    while len(pairs) < min(amostras, (len(idxs)*(len(idxs)-1))//2):
        i, j = tuple(rng.choice(idxs, size=2, replace=False))
        a = min(i,j); b = max(i,j)
        if a!=b: pairs.add((a,b))

    scores = []
    for a,b in pairs:
        s, det = combined_score(sample_df.at[a,"_norm"], sample_df.at[b,"_norm"], sample_df.at[a,"_meta"], sample_df.at[b,"_meta"])
        if det["contain"] >= min_cont:
            scores.append({"score": s, "set": det["set"], "sort": det["sort"], "contain": det["contain"]})

    if not scores:
        st.info("Nenhum par após o filtro de containment.")
        return
    df_scores = pd.DataFrame(scores)

    st.write("Estatísticas:", df_scores.describe()[["score","contain"]])

    if alt is not None:
        chart = alt.Chart(df_scores).mark_bar().encode(
            x=alt.X("score:Q", bin=alt.Bin(maxbins=30)),
            y="count()"
        ).properties(height=240)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.line_chart(df_scores["score"])

# =============================================================================
# App principal
# =============================================================================

def main():
    _init_session_state()
    st.title(APP_TITLE)

    # (Opcional) login muito simples para registrar usuário executor
    with st.sidebar.expander("Login (opcional)"):
        u = st.text_input("Usuário")
        if st.button("Entrar"):
            st.session_state[SK.USERNAME] = u or "usuário"
            st.success(f"Logado como {st.session_state[SK.USERNAME]}")

    eng = db_engine_mysql()
    df_full = carregar_dados_mysql(eng, 90)

    cfg = sidebar_controls(df_full)

    df = carregar_dados_mysql(eng, cfg["dias_hist"])
    if df.empty:
        st.warning("Nenhuma atividade encontrada para o período.")
        st.stop()

    # Aplicar filtros de visualização
    mask = pd.Series(True, index=df.index)
    if cfg["pastas"]:
        mask &= df["activity_folder"].isin(cfg["pastas"])
    if cfg["status"]:
        mask &= df["activity_status"].isin(cfg["status"])
    if cfg["data_inicio"]:
        mask &= (df["activity_date"].dt.date >= cfg["data_inicio"])
    if cfg["data_fim"]:
        mask &= (df["activity_date"].dt.date <= cfg["data_fim"])
    df_view = df[mask].copy()

    # Tabs
    t1, t2 = st.tabs(["🔎 Análise de Duplicidades", "📊 Calibração"])

    with t1:
        st.info("Dica: para marcar cancelamento, **abra a comparação** do item primeiro.")
        groups = criar_grupos_de_duplicatas(df, cfg["min_sim"], cfg["min_containment"], cfg["pre_delta"], cfg["use_cnj"], cfg["use_datajuri"], datajuri_client())

        # Mostra só grupos que contenham alguma atividade dentro do período/filters
        def group_visible(g):
            ids = {r["activity_id"] for r in g}
            sub = df_view[df_view["activity_id"].isin(ids)]
            return not sub.empty
        visible_groups = [g for g in groups if group_visible(g)]

        # Paginação
        page_size = st.number_input("Itens por página (grupos)", min_value=1, value=DEFAULTS["itens_por_pagina"], step=1)
        total_pages = max(1, math.ceil(len(visible_groups)/page_size))
        page = st.number_input("Página", min_value=1, max_value=total_pages, value=1, step=1)
        start = (page-1)*page_size; end = start + page_size
        st.caption(f"Exibindo grupos {start+1}–{min(end, len(visible_groups))} de {len(visible_groups)}")

        for g in visible_groups[start:end]:
            render_group(g, cfg["min_sim"]*100.0, cfg["diff_limit"])

        col_a, col_b = st.columns([0.5,0.5])
        with col_a:
            csv_bytes = export_groups_csv(visible_groups)
            st.download_button("⬇️ Exportar CSV (grupos visíveis)", data=csv_bytes, file_name="duplicatas.csv", mime="text/csv", use_container_width=True)
        with col_b:
            user = st.session_state.get(SK.USERNAME) or "usuario"
            if st.button("🚀 Processar Ações (cancelamentos marcados)", type="primary", use_container_width=True):
                process_actions(visible_groups, user)

    with t2:
        render_calibration(df, cfg["use_cnj"], cfg["use_datajuri"], datajuri_client())

if __name__ == "__main__":
    main()
