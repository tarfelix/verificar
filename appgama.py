import streamlit as st
import pandas as pd
import re, html, os, logging
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from unidecode import unidecode
from rapidfuzz import fuzz, process
from api_functions import HttpClient
from difflib import SequenceMatcher
from collections import deque # <--- 1. Importação para BFS otimizado

# ==============================================================================
# 1. CONFIGURAÇÕES E CONSTANTES GLOBAIS
# ==============================================================================
SUFFIX = "_v17_polished"

class SK:
    LOGGED_IN = "logged_in"
    USERNAME = "username"
    LAST_UPDATE = f"last_update_{SUFFIX}"
    SIMILARITY_CACHE = f"simcache_{SUFFIX}"
    PAGE_NUMBER = f"page_{SUFFIX}"
    SAVED_FILTERS = f"saved_filters_{SUFFIX}"
    AUDIT_LOG = f"audit_log_{SUFFIX}"
    GROUP_STATES = f"group_states_{SUFFIX}"

ITENS_POR_PAGINA = 10
DIAS_FILTRO_PADRAO_INICIO = 7
DIAS_FILTRO_PADRAO_FIM = 14
TZ_SP  = ZoneInfo("America/Sao_Paulo")
TZ_UTC = ZoneInfo("UTC")

st.set_page_config(layout="wide", page_title="Verificador de Duplicidade")
st.markdown("""
<style>
    pre.highlighted-text {
        white-space: pre-wrap; word-wrap: break-word; font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
        font-size: .9em; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; height: 300px; overflow-y: auto;
    }
    .similarity-badge { padding: 3px 6px; border-radius: 5px; color: black; font-weight: 500; display: inline-block; margin-bottom: 4px; }
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
    if not all([url_api, entity_id, token]): st.error("Credenciais da API não configuradas."); st.stop()
    return HttpClient(base_url=url_api, entity_id=entity_id, token=token)

def inicializar_session_state():
    defaults = {
        SK.LOGGED_IN: False, SK.USERNAME: "", SK.LAST_UPDATE: None,
        SK.SIMILARITY_CACHE: None, SK.PAGE_NUMBER: 0,
        SK.SAVED_FILTERS: {}, SK.AUDIT_LOG: [], SK.GROUP_STATES: {}
    }
    for key, value in defaults.items(): st.session_state.setdefault(key, value)

# ==============================================================================
# 3. FUNÇÕES AUXILIARES E DE LÓGICA
# ==============================================================================

def as_sp(ts: pd.Timestamp | None) -> datetime | None:
    if pd.isna(ts): return None
    if ts.tzinfo is None: ts = ts.tz_localize(TZ_UTC)
    return ts.tz_convert(TZ_SP)

def norm(text: str | None) -> str:
    if not isinstance(text, str): return ""
    text = unidecode(text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def highlight_diffs(text1: str, text2: str) -> tuple[str, str]:
    tokens1 = [token for token in re.split(r'(\W+)', text1 or "") if token]
    tokens2 = [token for token in re.split(r'(\W+)', text2 or "") if token]
    sm = SequenceMatcher(None, tokens1, tokens2, autojunk=False)
    out1, out2 = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        slice1, slice2 = html.escape("".join(tokens1[i1:i2])), html.escape("".join(tokens2[j1:j2]))
        if tag == 'equal': out1.append(slice1); out2.append(slice2)
        elif tag == 'replace': out1.append(f"<span class='diff-del'>{slice1}</span>"); out2.append(f"<span class='diff-ins'>{slice2}</span>")
        elif tag == 'delete': out1.append(f"<span class='diff-del'>{slice1}</span>")
        elif tag == 'insert': out2.append(f"<span class='diff-ins'>{slice2}</span>")
    return (f"<pre class='highlighted-text'>{''.join(out1)}</pre>", f"<pre class='highlighted-text'>{''.join(out2)}</pre>")

def log_action(user: str, action: str, details: dict):
    """Adiciona uma entrada ao log de auditoria na sessão."""
    log_entry = {"timestamp": datetime.now(TZ_SP), "user": user, "action": action, "details": details}
    st.session_state[SK.AUDIT_LOG].insert(0, log_entry)

def df_to_csv(df: pd.DataFrame) -> bytes:
    """Converte um DataFrame para bytes CSV para download.""" # <--- 4. Docstring corrigida
    return df.to_csv(index=False).encode('utf-8')

# ==============================================================================
# 4. LÓGICA DE DADOS (BANCO DE DADOS E CACHE)
# ==============================================================================

@st.cache_resource
def db_engine() -> Engine:
    cfg = st.secrets.get("database", {}); host, user, pw, db = [cfg.get(k) for k in ["host", "user", "password", "name"]]
    if not all([host, user, pw, db]): st.error("Credenciais de banco ausentes."); st.stop()
    try:
        engine = create_engine(f"mysql+mysqlconnector://{user}:{pw}@{host}/{db}", pool_pre_ping=True, pool_recycle=3600)
        with engine.connect(): pass
        return engine
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error("Erro ao conectar ao banco de dados."); st.stop()

@st.cache_data(ttl=3600, hash_funcs={Engine: lambda _: None})
def carregar_dados(eng: Engine, dias_historico: int) -> pd.DataFrame:
    limite = date.today() - timedelta(days=dias_historico)
    q = text("SELECT activity_id, activity_folder, user_profile_name, activity_date, activity_status, Texto FROM ViewGrdAtividadesTarcisio WHERE activity_type='Verificar' AND (activity_status='Aberta' OR DATE(activity_date) >= :limite)")
    try:
        with eng.connect() as conn: df = pd.read_sql(q, conn, params={"limite": limite})
        if df.empty: return pd.DataFrame()
        df["activity_id"] = df["activity_id"].astype(str)
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
        df["Texto"] = df["Texto"].fillna("").astype(str)
        df["status_ord"] = df["activity_status"].map({"Aberta": 0}).fillna(1)
        df = df.sort_values(["activity_id", "status_ord"]).drop_duplicates("activity_id", keep="first").drop(columns="status_ord")
        return df.sort_values(["activity_folder", "activity_date"], ascending=[True, False])
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error("Erro ao carregar dados."); return pd.DataFrame()

def criar_grupos_de_duplicatas(df: pd.DataFrame, min_sim: float) -> list:
    """Agrupa duplicatas por componentes conexos (transitivo) dentro de cada pasta."""
    sig = (tuple(sorted(df["activity_id"])), min_sim)
    cached = st.session_state.get(SK.SIMILARITY_CACHE)
    if cached and cached.get("sig") == sig:
        return cached["groups"]

    if df.empty:
        st.session_state[SK.SIMILARITY_CACHE] = {"sig": sig, "groups": []}
        return []

    df = df.copy()
    df["norm_texto"] = df["Texto"].apply(norm)

    groups = []
    cutoff = int(min_sim * 100)

    ph = st.sidebar.empty()
    pb = ph.progress(0, text="Agrupando duplicatas...")

    total_por_pasta = df.groupby("activity_folder", dropna=False).size().sum()
    avan = 0
    for folder, dff in df.groupby("activity_folder", dropna=False):
        ids = dff["activity_id"].tolist()
        texts = dff["norm_texto"].tolist()
        n = len(ids)
        if n < 2:
            avan += n
            pb.progress(min(100, int(100 * avan / max(1, total_por_pasta))), text=f"Agrupando… {avan}/{total_por_pasta}")
            continue

        sim = process.cdist(texts, texts, scorer=fuzz.token_set_ratio, score_cutoff=cutoff)

        visitados = set()
        for i in range(n):
            if i in visitados:
                avan += 1
                pb.progress(min(100, int(100 * avan / max(1, total_por_pasta))), text=f"Agrupando… {avan}/{total_por_pasta}")
                continue

            comp = {i}
            fila = deque([i]) # <--- 1. Usando deque
            visitados.add(i)
            while fila:
                k = fila.popleft() # <--- 1. Usando popleft()
                for j in range(n):
                    if j not in visitados and sim[k][j] >= cutoff:
                        visitados.add(j)
                        comp.add(j)
                        fila.append(j)
            
            avan += 1
            pb.progress(min(100, int(100 * avan / max(1, total_por_pasta))), text=f"Agrupando… {avan}/{total_por_pasta}")

            if len(comp) > 1:
                comp_idxs = sorted(list(comp), key=lambda idx: dff.iloc[idx]["activity_date"], reverse=True)
                group = [dff.iloc[idx].to_dict() for idx in comp_idxs]
                groups.append(group)

    ph.empty()

    st.session_state[SK.SIMILARITY_CACHE] = {"sig": sig, "groups": groups}
    return groups

# ==============================================================================
# 5. COMPONENTES DE UI
# ==============================================================================

def renderizar_sidebar_primaria() -> dict:
    """Renderiza os controles da sidebar que não dependem dos dados carregados."""
    st.sidebar.success(f"Logado como **{st.session_state[SK.USERNAME]}**")
    if st.sidebar.button("Logout"): 
        st.session_state.clear()
        st.rerun()

    if st.sidebar.button("🔄 Atualizar dados"):
        st.session_state[SK.LAST_UPDATE] = datetime.now(TZ_SP)
        carregar_dados.clear()
        st.session_state.pop(SK.SIMILARITY_CACHE, None)
        st.session_state.pop(SK.GROUP_STATES, None) # <--- 2. Limpa estados de grupo
    
    up = st.session_state.get(SK.LAST_UPDATE) or datetime.now(TZ_SP)
    st.sidebar.caption(f"Dados de: {up:%d/%m/%Y %H:%M}")
    
    st.sidebar.header("Filtros")
    
    return {
        "dias_historico": st.sidebar.number_input("Dias para Comparação", min_value=7, value=90, step=1),
        "data_inicio": st.sidebar.date_input("Data Início", date.today() - timedelta(days=DIAS_FILTRO_PADRAO_INICIO)),
        "data_fim": st.sidebar.date_input("Data Fim", date.today() + timedelta(days=DIAS_FILTRO_PADRAO_FIM)),
    }

def renderizar_sidebar_secundaria(df_completo: pd.DataFrame, filtros_primarios: dict) -> dict:
    """Renderiza os controles da sidebar que dependem dos dados carregados."""
    pastas_options = sorted(df_completo["activity_folder"].dropna().unique())
    status_options = sorted(df_completo["activity_status"].dropna().unique())

    filtros_secundarios = {
        "pastas": st.sidebar.multiselect("Pastas", pastas_options),
        "status": st.sidebar.multiselect("Status", status_options),
        "min_sim": st.sidebar.slider("Similaridade Mínima (%)", 0, 100, 90, 1) / 100
    }

    st.sidebar.subheader("Salvar/Carregar Filtros")
    saved_filters = st.session_state[SK.SAVED_FILTERS]
    filter_name = st.sidebar.text_input("Nome para salvar filtro")
    if st.sidebar.button("Salvar Filtros Atuais") and filter_name:
        filtros_para_salvar = {**filtros_primarios, **filtros_secundarios}
        saved_filters[filter_name] = filtros_para_salvar
        st.sidebar.success(f"Filtro '{filter_name}' salvo!")
    
    if saved_filters:
        selected_filter = st.sidebar.selectbox("Carregar Filtro Salvo", [""] + list(saved_filters.keys()))
        if selected_filter:
            st.sidebar.info("Filtros carregados. Clique em 'Atualizar dados' para aplicar.")
    
    return filtros_secundarios

def renderizar_grupo_duplicatas(group_data: list, group_index: int):
    """Renderiza um 'super card' para um grupo de atividades duplicadas."""
    group_id = group_data[0]['activity_id']
    
    group_state = st.session_state[SK.GROUP_STATES].setdefault(group_id, {
        "principal_id": group_data[0]['activity_id'],
        "cancelados": set(),
        "comparacao_aberta": None
    })

    with st.expander(f"Grupo de Duplicatas ({len(group_data)} atividades) - Pasta: {group_data[0]['activity_folder']}", expanded=True):
        for item_data in group_data:
            item_id = item_data['activity_id']
            is_principal = (item_id == group_state["principal_id"])
            is_cancelado = (item_id in group_state["cancelados"])
            
            card_class = "card-principal" if is_principal else "card-cancelado" if is_cancelado else ""
            st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)
            
            c1, c2 = st.columns([0.7, 0.3])
            with c1:
                dt = as_sp(item_data["activity_date"])
                st.markdown(f"**ID:** `{item_id}` {'⭐ **Principal**' if is_principal else ''} {'🗑️ **Cancelado**' if is_cancelado else ''}")
                st.caption(f"**Data:** {dt.strftime('%d/%m/%Y %H:%M') if dt else 'N/A'} | **Status:** {item_data['activity_status']} | **Usuário:** {item_data['user_profile_name']}")
                st.text_area("Texto", item_data['Texto'], height=80, disabled=True, key=f"text_{item_id}")

            with c2:
                if not is_principal:
                    if st.button("⭐ Tornar Principal", key=f"principal_{item_id}"):
                        group_state["principal_id"] = item_id
                        st.rerun()
                
                if st.checkbox("Marcar para Cancelar", value=is_cancelado, key=f"cancel_{item_id}"):
                    if not is_cancelado: group_state["cancelados"].add(item_id)
                elif is_cancelado:
                    group_state["cancelados"].discard(item_id)

                if not is_principal:
                    if st.button("⚖ Comparar com Principal", key=f"compare_{item_id}"):
                        group_state["comparacao_aberta"] = item_id
                        st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)

        if group_state["comparacao_aberta"]:
            principal_data = next(item for item in group_data if item['activity_id'] == group_state["principal_id"])
            comparado_data = next(item for item in group_data if item['activity_id'] == group_state["comparacao_aberta"])
            
            st.markdown("---")
            hA, hB = highlight_diffs(principal_data['Texto'], comparado_data['Texto'])
            c1, c2 = st.columns(2)
            c1.markdown(f"**Principal: ID `{principal_data['activity_id']}`**"); c1.markdown(hA, unsafe_allow_html=True)
            c2.markdown(f"**Comparado: ID `{comparado_data['activity_id']}`**"); c2.markdown(hB, unsafe_allow_html=True)
            if st.button("❌ Fechar Comparação", key=f"close_compare_{group_id}"):
                group_state["comparacao_aberta"] = None
                st.rerun()

# ==============================================================================
# 6. APLICAÇÃO PRINCIPAL
# ==============================================================================

def app():
    api_client = inicializar_api_client()
    eng = db_engine()
    
    filtros_primarios = renderizar_sidebar_primaria()
    df_completo = carregar_dados(eng, filtros_primarios['dias_historico'])
    if df_completo.empty: st.warning("Nenhuma atividade encontrada."); st.stop()
    
    filtros_secundarios = renderizar_sidebar_secundaria(df_completo, filtros_primarios)
    filtros = {**filtros_primarios, **filtros_secundarios}

    tab_principal, tab_log = st.tabs(["🔎 Análise de Duplicidades", "📜 Histórico de Ações"])

    with tab_principal:
        df_filtrado = df_completo.copy()
        if filtros['pastas']: df_filtrado = df_filtrado[df_filtrado['activity_folder'].isin(filtros['pastas'])]
        if filtros['status']: df_filtrado = df_filtrado[df_filtrado['activity_status'].isin(filtros['status'])]
        df_filtrado = df_filtrado[df_filtrado["activity_date"].dt.date.between(filtros["data_inicio"], filtros["data_fim"])]

        grupos_duplicados = criar_grupos_de_duplicatas(df_filtrado, filtros["min_sim"])
        
        # <--- 3. (Opcional) Limpa estados se não houver mais grupos
        if st.session_state.get(SK.GROUP_STATES) and not grupos_duplicados:
            st.session_state[SK.GROUP_STATES] = {}

        header_c1, header_c2, header_c3 = st.columns([4, 1, 1])
        header_c1.markdown(f"### {len(grupos_duplicados)} Grupos de Duplicatas Encontrados")
        
        if header_c2.button("Processar Cancelamentos"):
            total = ok = fail = 0
            erros = []
            for group_id, state in st.session_state[SK.GROUP_STATES].items():
                for act_id in list(state["cancelados"]):
                    total += 1
                    try:
                        resp = api_client.activity_canceled(activity_id=int(act_id),
                                                            user_name=st.session_state[SK.USERNAME])
                        if resp is not None:
                            ok += 1
                            log_action(st.session_state[SK.USERNAME], "Cancelamento", {"activity_id": act_id, "grupo": group_id, "api_resp": resp})
                        else:
                            fail += 1
                            erros.append(str(act_id))
                    except Exception as e:
                        fail += 1
                        erros.append(f"{act_id} ({e})")

            if ok:
                st.success(f"{ok}/{total} cancelamento(s) enviados com sucesso.")
            if fail:
                st.error(f"{fail}/{total} falharam: {', '.join(erros[:10])}{'...' if len(erros) > 10 else ''}")
            st.rerun()

        if grupos_duplicados:
            export_data = []
            for i, group in enumerate(grupos_duplicados):
                for item in group:
                    export_data.append({"grupo_id": i + 1, **item})
            df_export = pd.DataFrame(export_data)
            header_c3.download_button("Exportar para CSV", df_to_csv(df_export), "relatorio_duplicatas.csv", "text/csv")

        if not grupos_duplicados:
            st.info("Nenhum grupo de duplicatas encontrado para os filtros selecionados.")
        else:
            for i, grupo in enumerate(grupos_duplicados):
                renderizar_grupo_duplicatas(grupo, i)

    with tab_log:
        st.header("Histórico de Ações Recentes")
        log = st.session_state[SK.AUDIT_LOG]
        if not log:
            st.info("Nenhuma ação registrada ainda.")
        else:
            for entry in log:
                st.markdown(f"**Ação:** `{entry['action']}` | **Usuário:** `{entry['user']}` | **Data:** `{entry['timestamp'].strftime('%d/%m/%Y %H:%M:%S')}`")
                st.json(entry['details'])
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
