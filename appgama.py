
import streamlit as st
import pandas as pd
import re, html, os, logging, json
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
SUFFIX = "_v24_firebase_sim"

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

# Limiares de similaridade para badge
SIM_HIGH = 90
SIM_MED  = 75

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
        SK.SIMILARITY_CACHE: None, SK.PAGE_NUMBER: 0,
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
# 3. FUNÇÕES AUXILIARES E DE LÓGICA
# ==============================================================================

def as_sp(ts: any) -> datetime | None:
    if pd.isna(ts): return None
    if isinstance(ts, str): ts = pd.to_datetime(ts)
    if isinstance(ts, (int, float)): ts = pd.to_datetime(ts, unit='s')
    if ts.tzinfo is None: ts = ts.tz_localize(TZ_UTC)
    return ts.tz_convert(TZ_SP)

def norm(text: str | None) -> str:
    if not isinstance(text, str): return ""
    text = unidecode(text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def compute_similarity_pct(a: str, b: str) -> int:
    """Similaridade 0..100 usando token_set_ratio após normalização."""
    return int(fuzz.token_set_ratio(norm(a), norm(b)))

def similarity_badge(pct: int, label_prefix: str = "Similaridade") -> str:
    """Gera um badge colorido conforme thresholds."""
    if pct >= SIM_HIGH:
        bg = "#C8E6C9"   # verde claro
    elif pct >= SIM_MED:
        bg = "#FFE082"   # âmbar
    else:
        bg = "#FFCDD2"   # vermelho claro
    return f"<span class='similarity-badge' style='background-color:{bg}'>{label_prefix}: {pct}%</span>"

def highlight_diffs(text1: str, text2: str) -> tuple[str, str]:
    tokens1 = [token for token in re.split(r'(\W+)', text1 or "") if token]
    tokens2 = [token for token in re.split(r'(\W+)', text2 or "") if token]
    sm = SequenceMatcher(None, tokens1, tokens2, autojunk=False)
    out1, out2 = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        slice1, slice2 = html.escape("".join(tokens1[i1:i2])), html.escape("".join(tokens2[j1:j2]))
        if tag == 'equal':
            out1.append(slice1); out2.append(slice2)
        elif tag == 'replace':
            out1.append(f"<span class='diff-del'>{slice1}</span>"); out2.append(f"<span class='diff-ins'>{slice2}</span>")
        elif tag == 'delete':
            out1.append(f"<span class='diff-del'>{slice1}</span>")
        elif tag == 'insert':
            out2.append(f"<span class='diff-ins'>{slice2}</span>")
    return (f"<pre class='highlighted-text'>{''.join(out1)}</pre>", f"<pre class='highlighted-text'>{''.join(out2)}</pre>")

def df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

def _to_int(val):
    try:
        return int(str(val).strip())
    except Exception:
        return None

# ==============================================================================
# 4. LÓGICA DE DADOS (MYSQL PARA DADOS, FIREBASE PARA LOG)
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
        return df.sort_values(["activity_folder", "activity_date"], ascending=[True, False])
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error("Erro ao carregar dados do banco principal."); return pd.DataFrame()

def criar_grupos_de_duplicatas(df: pd.DataFrame, min_sim: float) -> list:
    """Agrupa duplicatas por pasta usando similaridade token_set_ratio e componentes conexos (BFS)."""
    if df.empty:
        return []

    df = df.copy()
    df['norm_texto'] = df['Texto'].apply(norm)

    # Assinatura de cache inclui id, pasta e fingerprint do texto (len) + limiar
    sig = (
        tuple(sorted((row["activity_id"], row.get("activity_folder") or "", len(row["norm_texto"]))
                     for _, row in df.iterrows())),
        round(min_sim, 3)
    )
    cached = st.session_state.get(SK.SIMILARITY_CACHE)
    if cached and cached.get("sig") == sig:
        return cached["groups"]

    groups = []
    cutoff = int(min_sim * 100)
    ph = st.sidebar.empty()
    pb = ph.progress(0, text="Agrupando duplicatas...")

    total_processado = 0
    for folder, dff in df.groupby("activity_folder", dropna=False):
        n = len(dff)
        if n < 2:
            total_processado += n
            continue
        texts = dff["norm_texto"].tolist()
        sim = process.cdist(texts, texts, scorer=fuzz.token_set_ratio, score_cutoff=cutoff)
        visitados = set()
        for i in range(n):
            if i in visitados: continue
            comp = {i}
            fila = deque([i])
            visitados.add(i)
            while fila:
                k = fila.popleft()
                for j in range(n):
                    if j not in visitados and sim[k][j] >= cutoff:
                        visitados.add(j); comp.add(j); fila.append(j)
            if len(comp) > 1:
                comp_idxs = sorted(list(comp), key=lambda idx: dff.iloc[idx]["activity_date"], reverse=True)
                groups.append([dff.iloc[idx].to_dict() for idx in comp_idxs])
        total_processado += n
        pb.progress(min(1.0, total_processado / len(df)), text="Analisando pastas...")
    ph.empty()
    st.session_state[SK.SIMILARITY_CACHE] = {"sig": sig, "groups": groups}
    return groups

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
# 5. COMPONENTES DE UI
# ==============================================================================

def renderizar_sidebar(df_completo: pd.DataFrame) -> dict:
    st.sidebar.success(f"Logado como **{st.session_state[SK.USERNAME]}**")
    if st.sidebar.button("Logout"):
        st.session_state.clear(); st.rerun()

    if st.sidebar.button("🔄 Atualizar dados"):
        st.session_state.pop(SK.SIMILARITY_CACHE, None)
        st.session_state.pop(SK.GROUP_STATES, None)
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
            st.rerun()

    return {
        "dias_historico": st.session_state["k_dias_historico"],
        "data_inicio": st.session_state["k_data_inicio"],
        "data_fim": st.session_state["k_data_fim"],
        "pastas": st.session_state.get("k_pastas", []),
        "status": st.session_state.get("k_status", []),
        "min_sim": st.session_state.get("k_min_sim", 90) / 100,
    }

def renderizar_grupo_duplicatas(group_data: list):
    # chave do estado do grupo baseada no primeiro id (como antes)
    group_id = group_data[0]['activity_id']
    group_state = st.session_state[SK.GROUP_STATES].setdefault(group_id, {
        "principal_id": group_data[0]['activity_id'], "cancelados": set(), "comparacao_aberta": None
    })

    with st.expander(f"Grupo de Duplicatas ({len(group_data)} atividades) - Pasta: {group_data[0]['activity_folder']}", expanded=True):
        # Localiza dados da principal atual
        principal_data = next(item for item in group_data if item['activity_id'] == group_state["principal_id"])

        for item_data in group_data:
            item_id = item_data['activity_id']
            is_principal = (item_id == group_state["principal_id"])
            is_cancelado = (item_id in group_state["cancelados"])

            # Similaridade vs principal (100% para a própria principal)
            if is_principal:
                sim_pct = 100
            else:
                sim_pct = compute_similarity_pct(principal_data['Texto'], item_data['Texto'])
            badge = similarity_badge(sim_pct, "Similaridade c/ principal")

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

                if st.checkbox("Marcar para Cancelar", value=is_cancelado, key=f"cancel_{item_id}"):
                    if not is_cancelado: group_state["cancelados"].add(item_id)
                elif is_cancelado:
                    group_state["cancelados"].discard(item_id)

                if not is_principal and st.button("⚖ Comparar com Principal", key=f"compare_{item_id}"):
                    group_state["comparacao_aberta"] = item_id; st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        if group_state["comparacao_aberta"]:
            principal_data = next(item for item in group_data if item['activity_id'] == group_state["principal_id"])
            comparado_data = next(item for item in group_data if item['activity_id'] == group_state["comparacao_aberta"])
            st.markdown("---")
            # Badge de similaridade na área de comparação
            sim_pct = compute_similarity_pct(principal_data['Texto'], comparado_data['Texto'])
            badge = similarity_badge(sim_pct, "Similaridade (diff)")
            ctitle1, ctitle2 = st.columns(2)
            ctitle1.markdown(f"**Principal: ID `{principal_data['activity_id']}`** {badge}", unsafe_allow_html=True)
            ctitle2.markdown(f"**Comparado: ID `{comparado_data['activity_id']}`**", unsafe_allow_html=True)

            hA, hB = highlight_diffs(principal_data['Texto'], comparado_data['Texto'])
            c1, c2 = st.columns(2)
            c1.markdown(hA, unsafe_allow_html=True)
            c2.markdown(hB, unsafe_allow_html=True)
            if st.button("❌ Fechar Comparação", key=f"close_compare_{group_id}"):
                group_state["comparacao_aberta"] = None; st.rerun()

# ==============================================================================
# 6. APLICAÇÃO PRINCIPAL
# ==============================================================================

def app():
    api_client = inicializar_api_client()
    mysql_engine = db_engine_mysql()
    firestore_db = initialize_firebase()

    # Carrega base histórica ampla
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

        # Agrupamento considera RECENTE + HISTÓRICO
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
                        # Cliente atual retorna None em falhas; tratamos isso
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

        # Render
        if not grupos_duplicados:
            st.info("Nenhum grupo de duplicatas encontrado para os filtros selecionados.")
        else:
            # Paginação simples (opcional)
            total = len(grupos_duplicados)
            page_count = (total + ITENS_POR_PAGINA - 1) // ITENS_POR_PAGINA
            if page_count > 1:
                page = st.number_input("Página", min_value=1, max_value=page_count, value=1, step=1)
                ini = (page - 1) * ITENS_POR_PAGINA
                fim = ini + ITENS_POR_PAGINA
                grupos_page = grupos_duplicados[ini:fim]
            else:
                grupos_page = grupos_duplicados

            for i, grupo in enumerate(grupos_page):
                renderizar_grupo_duplicatas(grupo)

    with tab_log:
        st.header("Histórico de Ações")
        c1, c2, c3 = st.columns([2,1,1])
        search_term = c1.text_input("Pesquisar por usuário, ação ou detalhes")
        # Alguns ambientes do Streamlit não aceitam None; inicializa sem filtro e trate abaixo se necessário
        start_date = c2.date_input("De", None)
        end_date = c3.date_input("Até", None)
        
        df_log = carregar_log_firestore(firestore_db, search_term, start_date, end_date)
        if df_log.empty:
            st.info("Nenhuma ação registrada para os filtros selecionados.")
        else:
            for entry in df_log.itertuples():
                ts_fmt = as_sp(entry.timestamp).strftime('%d/%m/%Y %H:%M:%S') if as_sp(entry.timestamp) else str(entry.timestamp)
                st.markdown(f"**Ação:** `{entry.acao}` | **Usuário:** `{entry.usuario}` | **Data:** `{ts_fmt}`")
                try:
                    st.json(json.loads(entry.detalhes) if isinstance(entry.detalhes, str) else entry.detalhes)
                except Exception:
                    st.write(entry.detalhes)
                st.divider()

# ==============================================================================
# 7. LÓGICA DE LOGIN
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
