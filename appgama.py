
import streamlit as st
import pandas as pd
import re, html, os, logging, json, hashlib
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from unidecode import unidecode
from rapidfuzz import fuzz, process
from api_functions import HttpClient
from difflib import SequenceMatcher
from collections import deque
import firebase_admin
from firebase_admin import credentials, firestore

# ==============================================================================
# 1. CONFIGURAÇÕES E CONSTANTES GLOBAIS
# ==============================================================================
SUFFIX = "_v25_revisado"

class SK:
    LOGGED_IN = "logged_in"
    USERNAME = "username"
    LAST_UPDATE = f"last_update_{SUFFIX}"
    SIMILARITY_CACHE = f"simcache_{SUFFIX}"
    PAGE_NUMBER = f"page_{SUFFIX}"
    SAVED_FILTERS = f"saved_filters_{SUFFIX}"
    GROUP_STATES = f"group_states_{SUFFIX}"

ITENS_POR_PAGINA = 10
DIAS_FILTRO_PADRAO_INICIO = 7
DIAS_FILTRO_PADRAO_FIM = 14
TZ_SP  = ZoneInfo("America/Sao_Paulo")
TZ_UTC = ZoneInfo("UTC")

# Limiares para badge (ajustados ao novo score combinado)
SIM_HIGH = 85
SIM_MED  = 70

# Parâmetros do agrupamento (afináveis via st.secrets["similarity"])
SIM_MIN_CONTAINMENT = int(st.secrets.get("similarity", {}).get("min_containment", 55))  # %
SIM_PRE_CUTOFF_DELTA = int(st.secrets.get("similarity", {}).get("pre_cutoff_delta", 10)) # pontos a menos que cutoff final
DIFF_HARD_LIMIT = int(st.secrets.get("similarity", {}).get("diff_hard_limit", 12000))    # nº máx de chars para diff palavra-a-palavra

st.set_page_config(layout="wide", page_title="Verificador de Duplicidade")
st.markdown("""
<style>
    pre.highlighted-text {
        white-space: pre-wrap; word-wrap: break-word; font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
        font-size: .9em; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; height: 300px; overflow-y: auto;
    }
    .similarity-badge { padding: 3px 6px; border-radius: 5px; color: black; font-weight: 600; display: inline-block; margin-left: 8px; }
    .diff-del { background-color: #ffcdd2 !important; text-decoration: none !important; }
    .diff-ins { background-color: #c8e6c9 !important; text-decoration: none !important; }
    .vertical-align-bottom { display: flex; align-items: flex-end; height: 100%;}
    .card-cancelado { background-color: #f5f5f5; border-left: 5px solid #e0e0e0; padding: 10px; margin-bottom: 5px; border-radius: 5px;}
    .card-principal { border-left: 5px solid #4CAF50; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. INICIALIZAÇÃO DE SERVIÇOS E ESTADO
# ==============================================================================

@st.cache_resource
def inicializar_api_client() -> HttpClient:
    cfg = st.secrets.get("api", {})
    url_api, entity_id, token = [cfg.get(k) for k in ["url_api", "entity_id", "token"]]
    if not all([url_api, entity_id, token]):
        st.error("Credenciais da API não configuradas.")
        st.stop()
    try:
        entity_id = int(entity_id)  # garante numérico para o cliente atual
    except Exception:
        st.error("`entity_id` inválido nos secrets. Deve ser numérico.")
        st.stop()
    return HttpClient(base_url=url_api, entity_id=entity_id, token=token)

def inicializar_session_state():
    defaults = {
        SK.LOGGED_IN: False, SK.USERNAME: "", SK.LAST_UPDATE: None,
        SK.SIMILARITY_CACHE: None, SK.PAGE_NUMBER: 1,
        SK.SAVED_FILTERS: {}, SK.GROUP_STATES: {}
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

@st.cache_resource
def initialize_firebase():
    """Inicializa a conexão com o Firebase usando os secrets do Streamlit."""
    try:
        if "firebase_credentials" not in st.secrets:
            st.error("Credenciais do Firebase não encontradas nos secrets do Streamlit.")
            st.stop()
        
        creds_dict = dict(st.secrets["firebase_credentials"])
        creds_dict["private_key"] = creds_dict["private_key"].replace('\\n', '\n')
        cred = credentials.Certificate(creds_dict)
        
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        
        return firestore.client()
    except Exception as e:
        st.error(f"Falha ao inicializar o Firebase: {e}")
        st.stop()

# ==============================================================================
# 3. NORMALIZAÇÃO E SIMILARIDADE
# ==============================================================================

_re_url = re.compile(r"https?://\S+")
_re_proc = re.compile(r"\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b")
_re_idtoken = re.compile(r"#id[:\s]*[a-z0-9]+", re.IGNORECASE)
_re_num = re.compile(r"\b\d{2,}\b")               # sequências numéricas longas
_re_date = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_re_time = re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\b")

# Stopwords / boilerplate (após unidecode + lower)
BASE_STOPWORDS = set("""
de do da dos das e a o os as para por com sem no na nos nas ao aos às que em um uma uns umas 
poder judiciario tribunal regional do trabalho justica do trabalho vara do trabalho diario de justica eletronico nacional 
dejtn djej djen publicacao numeroarquivo numeropublicacao titulo cabecalho textopublicacao inteiro teor
tipo de comunicacao tipo de documento orgao processo autos intimacao citacao intimado citado intimados citados
advogado advogados oab juiz juiza substituta secretaria pericia perito laudo diligencia despacho ata
cidade estado brasil sao jose dos campos sp
""".split())

EXTRA_STOPWORDS = set(map(str.lower, st.secrets.get("similarity", {}).get("stopwords_extra", [])))

def norm(text: str | None) -> str:
    if not isinstance(text, str): return ""
    text = unidecode(text.lower())
    text = _re_url.sub(" url ", text)
    text = _re_proc.sub(" numproc ", text)
    text = _re_idtoken.sub(" idtoken ", text)
    text = _re_date.sub(" data ", text)
    text = _re_time.sub(" hora ", text)
    text = _re_num.sub(" num ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def norm_drop_boilerplate(text: str) -> str:
    """Remove tokens hiperfrequentes/boilerplate para reduzir falsos positivos."""
    tokens = [t for t in text.split() if t not in BASE_STOPWORDS and t not in EXTRA_STOPWORDS]
    return " ".join(tokens)

def prep_for_similarity(text: str | None) -> str:
    return norm_drop_boilerplate(norm(text))

@st.cache_data(ttl=3600, show_spinner=False)
def _combined_similarity(a: str, b: str) -> tuple[int, dict]:
    """Score combinado que penaliza anexos longos e boilerplate.
       Retorna (score_final_0a100, breakdown_dict)."""
    A = prep_for_similarity(a); B = prep_for_similarity(b)
    if not A or not B:
        return (0, {"set": 0, "tok": 0, "containment": 0, "len_pen": 0, "jaccard": 0})
    s_set = int(fuzz.token_set_ratio(A, B))
    s_tok = int(fuzz.token_ratio(A, B))
    sa, sb = set(A.split()), set(B.split())
    if sa and sb:
        inter = len(sa & sb); uni = len(sa | sb)
        containment = int(100 * inter / max(len(sa), len(sb)))
        jaccard = int(100 * inter / max(1, uni))
    else:
        containment = jaccard = 0
    len_pen = int(100 * min(len(A), len(B)) / max(len(A), len(B)))
    score = round(0.40 * s_set + 0.30 * s_tok + 0.20 * containment + 0.10 * len_pen)
    return score, {"set": s_set, "tok": s_tok, "containment": containment, "len_pen": len_pen, "jaccard": jaccard}

def compute_similarity_pct(a: str, b: str) -> int:
    score, _ = _combined_similarity(a, b)
    return int(score)

def similarity_badge(pct: int, label_prefix: str = "Similaridade", breakdown: dict | None = None) -> str:
    if pct >= SIM_HIGH:
        bg = "#C8E6C9"
    elif pct >= SIM_MED:
        bg = "#FFE082"
    else:
        bg = "#FFCDD2"
    title = ""
    if breakdown:
        title = f"title='set:{breakdown['set']} tok:{breakdown['tok']} cont:{breakdown['containment']} len:{breakdown['len_pen']}'"
    return f"<span class='similarity-badge' {title} style='background-color:{bg}'>{label_prefix}: {pct}%</span>"

# ==============================================================================
# 4. DIFF OTIMIZADO (rápido para textos longos)
# ==============================================================================

def _highlight_diffs_tokens(tokens1, tokens2):
    sm = SequenceMatcher(None, tokens1, tokens2)  # autojunk=True (default) acelera textos grandes
    out1, out2 = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        slice1 = html.escape("".join(tokens1[i1:i2] if isinstance(tokens1[0], str) else tokens1[i1:i2]))
        slice2 = html.escape("".join(tokens2[j1:j2] if isinstance(tokens2[0], str) else tokens2[j1:j2]))
        if tag == 'equal':
            out1.append(slice1); out2.append(slice2)
        elif tag == 'replace':
            out1.append(f"<span class='diff-del'>{slice1}</span>"); out2.append(f"<span class='diff-ins'>{slice2}</span>")
        elif tag == 'delete':
            out1.append(f"<span class='diff-del'>{slice1}</span>")
        elif tag == 'insert':
            out2.append(f"<span class='diff-ins'>{slice2}</span>")
    return (f"<pre class='highlighted-text'>{''.join(out1)}</pre>", f"<pre class='highlighted-text'>{''.join(out2)}</pre>")

@st.cache_data(ttl=3600, show_spinner=False)
def highlight_diffs(text1: str, text2: str) -> tuple[str, str]:
    # Estratégia adaptativa: palavra-a-palavra até DIFF_HARD_LIMIT; senão, por sentenças/linhas
    if (len(text1) + len(text2)) <= DIFF_HARD_LIMIT:
        tokens1 = [t for t in re.split(r'(\W+)', text1 or "") if t]
        tokens2 = [t for t in re.split(r'(\W+)', text2 or "") if t]
        return _highlight_diffs_tokens(tokens1, tokens2)
    # fallback por sentenças/linhas (muito mais rápido)
    sent_split = r'(?<=[\.\!\?\:;])\s+|\n+'
    t1 = [s + "\n" for s in re.split(sent_split, text1 or "") if s]
    t2 = [s + "\n" for s in re.split(sent_split, text2 or "") if s]
    return _highlight_diffs_tokens(t1, t2)

def df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

def _to_int(val):
    try:
        return int(str(val).strip())
    except Exception:
        return None

# ==============================================================================
# 5. LÓGICA DE DADOS (MYSQL PARA DADOS, FIREBASE PARA LOG)
# ==============================================================================

@st.cache_resource
def db_engine_mysql() -> Engine:
    cfg = st.secrets.get("database", {}); host, user, pw, db = [cfg.get(k) for k in ["host", "user", "password", "name"]]
    if not all([host, user, pw, db]):
        st.error("Credenciais de banco ausentes.")
        st.stop()
    try:
        engine = create_engine(f"mysql+mysqlconnector://{user}:{pw}@{host}/{db}", pool_pre_ping=True, pool_recycle=3600)
        with engine.connect(): pass
        return engine
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error("Erro ao conectar ao banco de dados principal."); st.stop()

@st.cache_data(ttl=3600, hash_funcs={Engine: lambda _: None})
def carregar_dados_mysql(eng: Engine, dias_historico: int) -> pd.DataFrame:
    """Carrega atividades do tipo 'Verificar' com janela histórica (Aberta sempre entra; Fechada só dentro do limite)."""
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
        if df.empty:
            return pd.DataFrame()
        df["activity_id"] = df["activity_id"].astype(str)
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
        df["Texto"] = df["Texto"].fillna("").astype(str)
        # Remover possíveis duplicatas do SELECT (mantém 'Aberta' se houver conflito)
        df["status_ord"] = df["activity_status"].map({"Aberta": 0}).fillna(1)
        df = (
            df.sort_values(["activity_id", "status_ord"])
              .drop_duplicates("activity_id", keep="first")
              .drop(columns="status_ord")
        )
        # Pré-processamentos p/ similaridade
        df["prep_text"] = df["Texto"].apply(prep_for_similarity)
        df["prep_len"] = df["prep_text"].str.len().fillna(0).astype(int)
        df["prep_set"] = df["prep_text"].apply(lambda s: set(s.split()) if s else set())
        return df.sort_values(["activity_folder", "activity_date"], ascending=[True, False])
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error("Erro ao carregar dados do banco principal."); return pd.DataFrame()

@st.cache_data(ttl=60)
def carregar_log_firestore(_db_client, search_term: str, start_date, end_date) -> pd.DataFrame:
    """Carrega o log de auditoria do Firestore."""
    db = initialize_firebase() # Garante que temos o cliente
    collection_ref = db.collection('log_auditoria_duplicatas')
    query = collection_ref.order_by('timestamp', direction=firestore.Query.DESCENDING)

    docs = query.limit(500).stream()
    log_data = [doc.to_dict() for doc in docs]

    if not log_data: return pd.DataFrame()

    df = pd.DataFrame(log_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if start_date: df = df[df['timestamp'].dt.date >= start_date]
    if end_date: df = df[df['timestamp'].dt.date <= end_date]
    if search_term:
        term = search_term.lower()
        df = df[df.apply(lambda row: term in str(row['usuario']).lower() or term in str(row['acao']).lower() or term in str(row['detalhes']).lower(), axis=1)]
        
    return df.head(200)

def log_action_firestore(db, user: str, action: str, details: dict):
    """Insere uma nova entrada na coleção de log do Firestore."""
    try:
        log_entry = {
            "timestamp": datetime.now(TZ_UTC),
            "usuario": user,
            "acao": action,
            "detalhes": json.dumps(details)
        }
        db.collection('log_auditoria_duplicatas').add(log_entry)
        carregar_log_firestore.clear()
    except Exception as e:
        logging.error(f"Falha ao registrar log no Firestore: {e}")

# ==============================================================================
# 6. AGRUPAMENTO COM PENALIZAÇÃO POR CONTEÚDO EXTRA
# ==============================================================================

def criar_grupos_de_duplicatas(df: pd.DataFrame, min_sim: float) -> list:
    """Agrupa duplicatas por pasta com duas etapas:
       1) Candidatos rápidos via token_set (RapidFuzz, C-level)
       2) Validação com score combinado + containment mínimo"""
    if df.empty:
        return []

    df = df.copy()
    cutoff = int(min_sim * 100)
    pre_cutoff = max(60, cutoff - SIM_PRE_CUTOFF_DELTA)

    # Assinatura de cache inclui id, pasta e fingerprint do texto (len) + limiar e parâmetros de sim
    sig = (
        tuple(sorted((row["activity_id"], row.get("activity_folder") or "", int(row.get("prep_len",0)))
                     for _, row in df.iterrows())),
        cutoff, pre_cutoff, SIM_MIN_CONTAINMENT
    )
    cached = st.session_state.get(SK.SIMILARITY_CACHE)
    if cached and cached.get("sig") == sig:
        return cached["groups"]

    groups = []
    ph = st.sidebar.empty()
    pb = ph.progress(0, text="Agrupando duplicatas...")

    total_processado = 0
    for folder, dff in df.groupby("activity_folder", dropna=False):
        n = len(dff)
        if n < 2:
            total_processado += n
            continue

        texts = dff["prep_text"].tolist()
        sets_list = dff["prep_set"].tolist()

        # Matriz rápida para cortar pares óbvios
        sim_fast = process.cdist(texts, texts, scorer=fuzz.token_set_ratio, score_cutoff=pre_cutoff)

        visitados = set()
        for i in range(n):
            if i in visitados: continue
            comp = {i}
            fila = deque([i])
            visitados.add(i)
            while fila:
                k = fila.popleft()
                for j in range(n):
                    if j in visitados: 
                        continue
                    if sim_fast[k][j] < pre_cutoff:
                        continue
                    # Validação mais rigorosa
                    score, breakdown = _combined_similarity(dff.iloc[k]["Texto"], dff.iloc[j]["Texto"])
                    # containment direto via sets (mais barato e consistente)
                    sa, sb = sets_list[k], sets_list[j]
                    contain = int(100 * (len(sa & sb) / max(1, max(len(sa), len(sb)))))
                    if score >= cutoff and contain >= SIM_MIN_CONTAINMENT:
                        visitados.add(j); comp.add(j); fila.append(j)

            if len(comp) > 1:
                comp_idxs = sorted(list(comp), key=lambda idx: dff.iloc[idx]["activity_date"], reverse=True)
                groups.append([dff.iloc[idx].to_dict() for idx in comp_idxs])

        total_processado += n
        pb.progress(min(1.0, total_processado / len(df)), text="Analisando pastas...")
    ph.empty()
    st.session_state[SK.SIMILARITY_CACHE] = {"sig": sig, "groups": groups}
    return groups

# ==============================================================================
# 7. COMPONENTES DE UI
# ==============================================================================

def renderizar_sidebar(df_completo: pd.DataFrame) -> dict:
    st.sidebar.success(f"Logado como **{st.session_state[SK.USERNAME]}**")
    if st.sidebar.button("Logout"):
        st.session_state.clear(); st.rerun()

    if st.sidebar.button("🔄 Atualizar dados"):
        st.session_state.pop(SK.SIMILARITY_CACHE, None)
        st.session_state.pop(SK.GROUP_STATES, None)
        st.session_state[SK.PAGE_NUMBER] = 1
        carregar_dados_mysql.clear()

    st.sidebar.header("Filtros")

    # Widgets com chaves fixas para permitir reidratação
    dias_historico = st.sidebar.number_input(
        "Dias para Comparação", min_value=7, value=st.session_state.get("k_dias_historico", 90), step=1, key="k_dias_historico"
    )
    data_inicio = st.sidebar.date_input(
        "Data Início", st.session_state.get("k_data_inicio", date.today() - timedelta(days=DIAS_FILTRO_PADRAO_INICIO)), key="k_data_inicio"
    )
    data_fim = st.sidebar.date_input(
        "Data Fim", st.session_state.get("k_data_fim", date.today() + timedelta(days=DIAS_FILTRO_PADRAO_FIM)), key="k_data_fim"
    )

    pastas_opts = sorted(df_completo["activity_folder"].dropna().unique()) if not df_completo.empty else []
    status_opts = sorted(df_completo["activity_status"].dropna().unique()) if not df_completo.empty else []

    pastas = st.sidebar.multiselect("Pastas", pastas_opts, default=st.session_state.get("k_pastas", []), key="k_pastas")
    status = st.sidebar.multiselect("Status", status_opts, default=st.session_state.get("k_status", []), key="k_status")
    min_sim = st.sidebar.slider("Similaridade Mínima (%)", 0, 100, st.session_state.get("k_min_sim", 90), 1, key="k_min_sim") / 100

    # Salvar/Carregar
    st.sidebar.subheader("Salvar/Carregar Filtros")
    saved_filters = st.session_state[SK.SAVED_FILTERS]
    filter_name = st.sidebar.text_input("Nome para salvar filtro")
    if st.sidebar.button("Salvar Filtros Atuais") and filter_name:
        saved_filters[filter_name] = {
            "k_dias_historico": dias_historico,
            "k_data_inicio": data_inicio,
            "k_data_fim": data_fim,
            "k_pastas": pastas,
            "k_status": status,
            "k_min_sim": int(min_sim * 100),
        }
        st.sidebar.success(f"Filtro '{filter_name}' salvo!")

    if saved_filters:
        selected_filter = st.sidebar.selectbox("Carregar Filtro Salvo", [""] + list(saved_filters.keys()))
        if selected_filter:
            f = saved_filters[selected_filter]
            st.session_state["k_dias_historico"] = f.get("k_dias_historico", 90)
            st.session_state["k_data_inicio"] = f.get("k_data_inicio")
            st.session_state["k_data_fim"] = f.get("k_data_fim")
            st.session_state["k_pastas"] = f.get("k_pastas", [])
            st.session_state["k_status"] = f.get("k_status", [])
            st.session_state["k_min_sim"] = f.get("k_min_sim", 90)
            st.sidebar.success("Filtros aplicados.")
            st.session_state[SK.PAGE_NUMBER] = 1
            st.rerun()

    return {
        "dias_historico": st.session_state["k_dias_historico"],
        "data_inicio": st.session_state["k_data_inicio"],
        "data_fim": st.session_state["k_data_fim"],
        "pastas": st.session_state.get("k_pastas", []),
        "status": st.session_state.get("k_status", []),
        "min_sim": st.session_state.get("k_min_sim", 90) / 100,
    }

def renderizar_grupo_duplicatas(group_data: list, expand: bool = False):
    # chave do estado do grupo baseada no primeiro id (como antes)
    group_id = group_data[0]['activity_id']
    group_state = st.session_state[SK.GROUP_STATES].setdefault(group_id, {
        "principal_id": group_data[0]['activity_id'], "cancelados": set(), "comparacao_aberta": None
    })

    with st.expander(f"Grupo de Duplicatas ({len(group_data)} atividades) - Pasta: {group_data[0]['activity_folder']}", expanded=expand):
        # Localiza dados da principal atual
        principal_data = next(item for item in group_data if item['activity_id'] == group_state["principal_id"])

        for item_data in group_data:
            item_id = item_data['activity_id']
            is_principal = (item_id == group_state["principal_id"])
            is_cancelado = (item_id in group_state["cancelados"])

            # Similaridade vs principal (100% para a própria principal)
            if is_principal:
                sim_pct, breakdown = 100, {"set":100,"tok":100,"containment":100,"len_pen":100,"jaccard":100}
            else:
                sim_pct, breakdown = _combined_similarity(principal_data['Texto'], item_data['Texto'])
            badge = similarity_badge(int(sim_pct), "Similaridade c/ principal", breakdown)

            card_class = "card-principal" if is_principal else "card-cancelado" if is_cancelado else ""
            st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)

            c1, c2 = st.columns([0.7, 0.3])
            with c1:
                dt = as_sp(item_data["activity_date"])
                st.markdown(
                    f"**ID:** `{item_id}` "
                    f"{'⭐ **Principal**' if is_principal else ''} "
                    f"{'🗑️ **Cancelado**' if is_cancelado else ''} "
                    f"{badge}",
                    unsafe_allow_html=True
                )
                st.caption(f"**Data:** {dt.strftime('%d/%m/%Y %H:%M') if dt else 'N/A'} | "
                           f"**Status:** {item_data['activity_status']} | "
                           f"**Usuário:** {item_data['user_profile_name']}")
                st.text_area("Texto", item_data['Texto'], height=80, disabled=True, key=f"text_{item_id}")

            with c2:
                if not is_principal and st.button("⭐ Tornar Principal", key=f"principal_{item_id}"):
                    group_state["principal_id"] = item_id; st.rerun()

                # [GATING] Só mostra o checkbox DEPOIS de abrir a comparação
                show_cancel = (group_state["comparacao_aberta"] == item_id)
                if show_cancel and st.checkbox("Marcar para Cancelar", value=is_cancelado, key=f"cancel_{item_id}"):
                    if not is_cancelado: group_state["cancelados"].add(item_id)
                elif show_cancel and is_cancelado:
                    # permitir desmarcar
                    if not st.session_state.get(f"cancel_{item_id}"):
                        group_state["cancelados"].discard(item_id)

                if not is_principal and st.button("⚖ Comparar com Principal", key=f"compare_{item_id}"):
                    group_state["comparacao_aberta"] = item_id; st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        if group_state["comparacao_aberta"]:
            principal_data = next(item for item in group_data if item['activity_id'] == group_state["principal_id"])
            comparado_data = next(item for item in group_data if item['activity_id'] == group_state["comparacao_aberta"])
            st.markdown("---")
            # Badge de similaridade na área de comparação (com breakdown)
            sim_pct, breakdown = _combined_similarity(principal_data['Texto'], comparado_data['Texto'])
            badge = similarity_badge(int(sim_pct), "Similaridade (diff)", breakdown)
            ctitle1, ctitle2 = st.columns(2)
            ctitle1.markdown(f"**Principal: ID `{principal_data['activity_id']}`** {badge}", unsafe_allow_html=True)
            ctitle2.markdown(f"**Comparado: ID `{comparado_data['activity_id']}`**", unsafe_allow_html=True)

            with st.spinner("Calculando diferenças..."):
                hA, hB = highlight_diffs(principal_data['Texto'], comparado_data['Texto'])
            c1, c2 = st.columns(2)
            c1.markdown(hA, unsafe_allow_html=True)
            c2.markdown(hB, unsafe_allow_html=True)
            if st.button("❌ Fechar Comparação", key=f"close_compare_{group_id}"):
                group_state["comparacao_aberta"] = None; st.rerun()

# ==============================================================================
# 8. APLICAÇÃO PRINCIPAL
# ==============================================================================

def app():
    api_client = inicializar_api_client()
    mysql_engine = db_engine_mysql()
    firestore_db = initialize_firebase()

    # Carrega base histórica ampla (para popular filtros)
    df_completo = carregar_dados_mysql(mysql_engine, st.session_state.get("k_dias_historico", 90))

    filtros = renderizar_sidebar(df_completo)

    # Recarrega conforme dias_historico atual
    df_completo = carregar_dados_mysql(mysql_engine, filtros['dias_historico'])
    if df_completo.empty:
        st.warning("Nenhuma atividade encontrada para o período.")
        st.stop()

    tab_principal, tab_log = st.tabs(["🔎 Análise de Duplicidades", "📜 Histórico de Ações"])

    with tab_principal:
        # --------- Escopo: Comparação Histórica Ampla ---------
        # Filtros de pasta/status aplicam-se à base para comparação
        df_base = df_completo.copy()
        if filtros['pastas']:
            df_base = df_base[df_base['activity_folder'].isin(filtros['pastas'])]
        if filtros['status']:
            df_base = df_base[df_base['activity_status'].isin(filtros['status'])]

        # Define ids "recentes" apenas para o que será exibido
        mask_recente = df_base['activity_date'].dt.date.between(filtros['data_inicio'], filtros['data_fim'])
        ids_recente = set(df_base.loc[mask_recente, 'activity_id'])

        # Agrupamento considera RECENTE + HISTÓRICO (com validação rigorosa)
        grupos_todos = criar_grupos_de_duplicatas(df_base, filtros["min_sim"])

        # Exibir apenas grupos que toquem o período recente
        grupos_duplicados = [g for g in grupos_todos if any(item['activity_id'] in ids_recente for item in g)]

        # Sincroniza estado se necessário
        if st.session_state.get(SK.GROUP_STATES) and not grupos_duplicados:
            st.session_state[SK.GROUP_STATES] = {}

        header_c1, header_c2, header_c3 = st.columns([4, 1, 1])
        header_c1.markdown(f"### {len(grupos_duplicados)} Grupos de Duplicatas Encontrados")

        # Processar ações: chama API e registra no Firestore
        if header_c2.button("Processar Ações"):
            resultados = {"principais": [], "cancelados_ok": [], "cancelados_fail": []}
            usuario = st.session_state[SK.USERNAME]

            for group_id, state in st.session_state[SK.GROUP_STATES].items():
                # Cancelamentos via API + log no Firestore
                for act_id in state['cancelados']:
                    aid = _to_int(act_id)
                    try:
                        if aid is None:
                            raise ValueError(f"ID inválido: {act_id}")
                        resp = api_client.activity_canceled(aid, usuario)
                        if isinstance(resp, dict) and resp.get("error"):
                            resultados["cancelados_fail"].append({"id": act_id, "erro": resp["error"]})
                            log_action_firestore(firestore_db, usuario, "Cancelamento(API-erro)",
                                                 {"activity_id": act_id, "grupo": group_id, "erro": resp["error"]})
                        elif resp:
                            resultados["cancelados_ok"].append({"id": act_id, "resposta": resp})
                            log_action_firestore(firestore_db, usuario, "Cancelamento(API)",
                                                 {"activity_id": act_id, "grupo": group_id, "api_response": resp})
                        else:
                            resultados["cancelados_fail"].append({"id": act_id, "erro": "Sem resposta"})
                            log_action_firestore(firestore_db, usuario, "Cancelamento(API-erro)",
                                                 {"activity_id": act_id, "grupo": group_id, "erro": "Sem resposta"})
                    except Exception as e:
                        resultados["cancelados_fail"].append({"id": act_id, "erro": str(e)})
                        log_action_firestore(firestore_db, usuario, "Cancelamento(API-erro)",
                                             {"activity_id": act_id, "grupo": group_id, "erro": str(e)})

                # Marcação de principal (apenas log permanente)
                resultados["principais"].append(state['principal_id'])
                log_action_firestore(firestore_db, usuario, "Marcar Principal",
                                     {"activity_id": state['principal_id'], "grupo": group_id})

            # Limpar e feedback
            st.session_state[SK.GROUP_STATES] = {}
            st.success(f"Principais: {len(resultados['principais'])} | "
                       f"Canceladas OK: {len(resultados['cancelados_ok'])} | "
                       f"Falhas: {len(resultados['cancelados_fail'])}")
            with st.expander("Detalhes da execução"):
                st.json(resultados)
            st.rerun()

        # Exportação
        if grupos_duplicados:
            export_data = [{"grupo_id": i + 1, **item} for i, group in enumerate(grupos_duplicados) for item in group]
            df_export = pd.DataFrame(export_data)
            header_c3.download_button("Exportar para CSV", df_to_csv(df_export), "relatorio_duplicatas.csv", "text/csv")

        # --------- Paginação (sempre visível) ---------
        if not grupos_duplicados:
            st.info("Nenhum grupo de duplicatas encontrado para os filtros selecionados.")
        else:
            total = len(grupos_duplicados)
            itens_por_pagina = st.session_state.get("k_itens_pagina", ITENS_POR_PAGINA)
            col_a, col_b, col_c, col_d, col_e = st.columns([2,1,2,1,2])
            with col_a:
                itens_por_pagina = st.number_input("Itens/página", min_value=5, max_value=50,
                                                   value=itens_por_pagina, step=5, key="k_itens_pagina")
            page_count = max(1, (total + itens_por_pagina - 1) // itens_por_pagina)
            current_page = st.session_state.get(SK.PAGE_NUMBER, 1)
            current_page = min(max(1, current_page), page_count)
            with col_b:
                st.write("")
                if st.button("◀️ Anterior", disabled=(current_page <= 1)):
                    st.session_state[SK.PAGE_NUMBER] = max(1, current_page - 1)
                    st.rerun()
            with col_c:
                current_page = st.number_input("Página", min_value=1, max_value=page_count,
                                               value=current_page, step=1, key="k_page_num")
                st.session_state[SK.PAGE_NUMBER] = current_page
            with col_d:
                st.write("")
                if st.button("Próxima ▶️", disabled=(current_page >= page_count)):
                    st.session_state[SK.PAGE_NUMBER] = min(page_count, current_page + 1)
                    st.rerun()
            with col_e:
                expand_all = st.toggle("Expandir todos", value=st.session_state.get("k_expand_all", False), key="k_expand_all")

            ini = (st.session_state[SK.PAGE_NUMBER] - 1) * itens_por_pagina
            fim = ini + itens_por_pagina
            st.caption(f"Mostrando grupos {ini+1}–{min(fim, total)} de {total}")
            st.divider()

            grupos_page = grupos_duplicados[ini:fim]
            for i, grupo in enumerate(grupos_page):
                renderizar_grupo_duplicatas(grupo, expand=expand_all)

    with tab_log:
        st.header("Histórico de Ações")
        c1, c2, c3 = st.columns([2,1,1])
        search_term = c1.text_input("Pesquisar por usuário, ação ou detalhes")
        start_date = c2.date_input("De", None)
        end_date = c3.date_input("Até", None)
        
        df_log = carregar_log_firestore(firestore_db, search_term, start_date, end_date)
        if df_log.empty:
            st.info("Nenhuma ação registrada para os filtros selecionados.")
        else:
            for entry in df_log.itertuples():
                ts = entry.timestamp if isinstance(entry.timestamp, pd.Timestamp) else pd.to_datetime(entry.timestamp)
                ts_fmt = (ts.tz_localize(TZ_UTC).tz_convert(TZ_SP).strftime('%d/%m/%Y %H:%M:%S')
                          if ts.tzinfo is None else ts.tz_convert(TZ_SP).strftime('%d/%m/%Y %H:%M:%S'))
                st.markdown(f"**Ação:** `{entry.acao}` | **Usuário:** `{entry.usuario}` | **Data:** `{ts_fmt}`")
                try:
                    st.json(json.loads(entry.detalhes) if isinstance(entry.detalhes, str) else entry.detalhes)
                except Exception:
                    st.write(entry.detalhes)
                st.divider()

# ==============================================================================
# 9. LÓGICA DE LOGIN
# ==============================================================================

def cred_ok(u, p):
    users = st.secrets.get("credentials", {}).get("usernames", {})
    return u in users and str(users[u]) == p

def login():
    st.header("Login")
    with st.form("login_form_main"):
        username, password = st.text_input("Usuário"), st.text_input("Senha", type="password")
        if st.form_submit_button("Entrar"):
            if cred_ok(username, password):
                st.session_state[SK.LOGGED_IN] = True; st.session_state[SK.USERNAME] = username
                st.rerun()
            else: st.error("Credenciais inválidas.")

# ==============================================================================
# PONTO DE ENTRADA PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    inicializar_session_state()
    (app() if st.session_state[SK.LOGGED_IN] else login())
