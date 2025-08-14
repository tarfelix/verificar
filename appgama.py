import streamlit as st
import pandas as pd
import re, html, os, logging
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from unidecode import unidecode
from rapidfuzz import fuzz
from api_functions import HttpClient
from difflib import SequenceMatcher

# ==============================================================================
# 1. CONFIGURAÇÕES E CONSTANTES GLOBAIS
# ==============================================================================

# --- Constantes da Aplicação ---
ITENS_POR_PAGINA = 20
DIAS_HISTORICO_COMPARACAO = 90
DIAS_FILTRO_PADRAO_INICIO = 7
DIAS_FILTRO_PADRAO_FIM = 14
SUFFIX = "_v10_performance" # Sufixo para evitar conflitos de cache/sessão com versões antigas

# --- Zonas de Tempo ---
TZ_SP  = ZoneInfo("America/Sao_Paulo")
TZ_UTC = ZoneInfo("UTC")

# --- Chaves da Sessão (Session Keys) ---
class SK:
    LOGGED_IN = "logged_in"
    USERNAME = "username"
    LAST_UPDATE = f"last_update_{SUFFIX}"
    SIMILARITY_CACHE = f"simcache_cruzado_{SUFFIX}"
    OPEN_COMPARISONS = f"open_cmps_{SUFFIX}"
    PAGE_NUMBER = f"page_{SUFFIX}"
    FULL_TEXT_DATA = f"full_act_{SUFFIX}"
    SHOW_FULL_TEXT_DIALOG = f"show_text_{SUFFIX}"
    STEP_CANCEL = "step_cancel"
    PROCESS_CANCEL = "process_cancel"
    STEP_CANCEL_PROCESSADO = "step_cancel_processado"


# --- Configuração da Página do Streamlit ---
st.set_page_config(layout="wide", page_title="Verificador de Duplicidade")

# --- Estilos CSS ---
st.markdown("""
<style>
    pre.highlighted-text {
        white-space: pre-wrap; word-wrap: break-word; font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
        font-size: .9em; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;
        height: 400px; overflow-y: auto;
    }
    .similarity-badge {
        padding: 3px 6px; border-radius: 5px; color: black; font-weight: 500;
        display: inline-block; margin-bottom: 4px;
    }
    .diff-del { background-color: #ffcdd2 !important; text-decoration: none !important; }
    .diff-ins { background-color: #c8e6c9 !important; text-decoration: none !important; }
    .vertical-align-bottom { display: flex; align-items: flex-end; height: 100%;}
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
        st.error("Credenciais da API não configuradas nos secrets.")
        st.stop()
    return HttpClient(base_url=url_api, entity_id=entity_id, token=token)

def inicializar_session_state():
    defaults = {
        SK.LOGGED_IN: False, SK.USERNAME: "", SK.LAST_UPDATE: None,
        SK.SIMILARITY_CACHE: None, SK.OPEN_COMPARISONS: set(), SK.PAGE_NUMBER: 0,
        SK.FULL_TEXT_DATA: None, SK.SHOW_FULL_TEXT_DIALOG: False, SK.STEP_CANCEL: None,
        SK.PROCESS_CANCEL: False, SK.STEP_CANCEL_PROCESSADO: False
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

# ==============================================================================
# 3. FUNÇÕES AUXILIARES E DE LÓGICA
# ==============================================================================

def as_sp(ts: pd.Timestamp | None) -> datetime | None:
    if pd.isna(ts): return None
    if ts.tzinfo is None: ts = ts.tz_localize(TZ_UTC)
    return ts.tz_convert(TZ_SP)

def norm(text: str | None) -> str:
    if not isinstance(text, str): return ""
    text_sem_acento = unidecode(text.lower())
    text_limpo = re.sub(r"[^\w\s]", " ", text_sem_acento)
    return re.sub(r"\s+", " ", text_limpo).strip()

def calc_sim_on_norm(norm_a: str, norm_b: str) -> float:
    """Calcula a similaridade em textos JÁ normalizados para otimização."""
    if not norm_a or not norm_b or abs(len(norm_a) - len(norm_b)) > 0.3 * max(len(norm_a), len(norm_b)):
        return 0.0
    return fuzz.token_set_ratio(norm_a, norm_b) / 100

def cor_sim(ratio: float) -> str:
    if ratio >= 0.9: return "#FF5252"
    if ratio >= 0.7: return "#FFB74D"
    return "#FFD54F"

def highlight_diffs(text1: str, text2: str) -> tuple[str, str]:
    tokens1 = [token for token in re.split(r'(\W+)', text1 or "") if token]
    tokens2 = [token for token in re.split(r'(\W+)', text2 or "") if token]
    sm = SequenceMatcher(None, tokens1, tokens2, autojunk=False)
    out1, out2 = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        slice1 = html.escape("".join(tokens1[i1:i2]))
        slice2 = html.escape("".join(tokens2[j1:j2]))
        if tag == 'equal':
            out1.append(slice1); out2.append(slice2)
        elif tag == 'replace':
            out1.append(f"<span class='diff-del'>{slice1}</span>"); out2.append(f"<span class='diff-ins'>{slice2}</span>")
        elif tag == 'delete':
            out1.append(f"<span class='diff-del'>{slice1}</span>")
        elif tag == 'insert':
            out2.append(f"<span class='diff-ins'>{slice2}</span>")
    return (f"<pre class='highlighted-text'>{''.join(out1)}</pre>", f"<pre class='highlighted-text'>{''.join(out2)}</pre>")

# ==============================================================================
# 4. LÓGICA DE DADOS (BANCO DE DADOS E CACHE)
# ==============================================================================

@st.cache_resource
def db_engine() -> Engine:
    cfg = st.secrets.get("database", {})
    host, user, pw, db = [cfg.get(k) or os.getenv(f"DB_{k.upper()}") for k in ["host", "user", "password", "name"]]
    if not all([host, user, pw, db]):
        st.error("Credenciais de banco de dados ausentes."); st.stop()
    try:
        engine = create_engine(f"mysql+mysqlconnector://{user}:{pw}@{host}/{db}", pool_pre_ping=True, pool_recycle=3600)
        with engine.connect(): pass
        return engine
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error("Erro ao conectar ao banco de dados."); st.stop()

@st.cache_data(ttl=3600, hash_funcs={Engine: lambda _: None})
def carregar_dados(eng: Engine) -> pd.DataFrame:
    limite_historico = date.today() - timedelta(days=DIAS_HISTORICO_COMPARACAO)
    query_abertas = text("SELECT activity_id, activity_folder, user_profile_name, activity_date, activity_status, Texto FROM ViewGrdAtividadesTarcisio WHERE activity_type='Verificar' AND activity_status='Aberta'")
    query_historico = text("SELECT activity_id, activity_folder, user_profile_name, activity_date, activity_status, Texto FROM ViewGrdAtividadesTarcisio WHERE activity_type='Verificar' AND DATE(activity_date) >= :limite")
    try:
        with eng.connect() as conn:
            df = pd.concat([pd.read_sql(query_abertas, conn), pd.read_sql(query_historico, conn, params={"limite": limite_historico})], ignore_index=True)
        if df.empty: return pd.DataFrame()
        df["activity_id"] = df["activity_id"].astype(str)
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
        df["Texto"] = df["Texto"].astype(str).fillna("")
        df["status_ord"] = df["activity_status"].map({"Aberta": 0}).fillna(1)
        df = df.sort_values(["activity_id", "status_ord"]).drop_duplicates("activity_id", keep="first").drop(columns="status_ord")
        return df.sort_values(["activity_folder", "activity_date", "activity_id"], ascending=[True, False, False])
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error("Erro ao carregar dados do banco."); return pd.DataFrame()

def gerar_mapa_similaridade(df_exibir: pd.DataFrame, df_comparar: pd.DataFrame, min_sim: float) -> tuple[dict, set]:
    """
    [OTIMIZADO] Compara cada item em df_exibir com todos os itens em df_comparar,
    agrupando por 'activity_folder' para máxima performance.
    """
    if df_exibir.empty:
        return {}, set()
        
    sig = (tuple(sorted(df_exibir["activity_id"])), tuple(sorted(df_comparar["activity_id"])), min_sim)
    cached = st.session_state.get(SK.SIMILARITY_CACHE)
    if cached and cached.get("sig") == sig:
        return cached["map"], cached["dup"]

    mapa_similaridade = {}
    ids_duplicados = set()
    
    # Otimização 2: Pré-calcular texto normalizado para evitar recálculos
    df_exibir = df_exibir.copy()
    df_comparar = df_comparar.copy()
    df_exibir['norm_texto'] = df_exibir['Texto'].apply(norm)
    df_comparar['norm_texto'] = df_comparar['Texto'].apply(norm)
    
    # Otimização 1: Agrupar por pasta para reduzir drasticamente as comparações
    pastas_para_analise = df_exibir["activity_folder"].dropna().unique()
    
    bar = st.sidebar.progress(0, text="Calculando similaridades por pasta…")
    total_pastas = len(pastas_para_analise)

    for i, pasta in enumerate(pastas_para_analise):
        bar.progress((i + 1) / total_pastas, text=f"Analisando pasta {i+1}/{total_pastas}")

        sub_df_exibir = df_exibir[df_exibir["activity_folder"] == pasta]
        sub_df_comparar = df_comparar[df_comparar["activity_folder"] == pasta]

        atividades_para_exibir = sub_df_exibir.to_dict("records")
        atividades_para_comparar = sub_df_comparar.to_dict("records")

        for atividade_principal in atividades_para_exibir:
            id_principal = atividade_principal["activity_id"]
            mapa_similaridade.setdefault(id_principal, [])
            
            for atividade_historica in atividades_para_comparar:
                id_historico = atividade_historica["activity_id"]
                if id_principal == id_historico: continue

                ratio = calc_sim_on_norm(atividade_principal["norm_texto"], atividade_historica["norm_texto"])
                if ratio >= min_sim:
                    ids_duplicados.add(id_principal)
                    mapa_similaridade[id_principal].append({"id": id_historico, "ratio": ratio, "cor": cor_sim(ratio)})
    
    bar.empty()
    
    for k in mapa_similaridade:
        mapa_similaridade[k].sort(key=lambda z: z["ratio"], reverse=True)
        
    st.session_state[SK.SIMILARITY_CACHE] = {"sig": sig, "map": mapa_similaridade, "dup": ids_duplicados}
    return mapa_similaridade, ids_duplicados

# ==============================================================================
# 5. COMPONENTES DE UI E LÓGICA DE CANCELAMENTO
# ==============================================================================

def renderizar_sidebar(df_completo: pd.DataFrame) -> dict:
    st.sidebar.success(f"Logado como **{st.session_state[SK.USERNAME]}**")
    if st.sidebar.button("Logout", key="logout_button"):
        st.session_state.clear(); st.rerun()

    if st.sidebar.button("🔄 Atualizar dados", key="update_data_button"):
        st.session_state[SK.LAST_UPDATE] = datetime.now(TZ_SP)
        carregar_dados.clear()
        st.session_state.pop(SK.SIMILARITY_CACHE, None)

    up = st.session_state.get(SK.LAST_UPDATE) or datetime.now(TZ_SP)
    st.sidebar.caption(f"Dados atualizados em: {up:%d/%m/%Y %H:%M:%S}")
    
    st.sidebar.header("Filtros de Visualização")
    hoje = date.today()
    d_ini = st.sidebar.date_input("Data Início", hoje - timedelta(days=DIAS_FILTRO_PADRAO_INICIO))
    d_fim = st.sidebar.date_input("Data Fim", hoje + timedelta(days=DIAS_FILTRO_PADRAO_FIM), min_value=d_ini)
    if d_ini > d_fim:
        st.sidebar.error("Data de início não pode ser maior que a data de fim."); st.stop()

    return {
        "data_inicio": d_ini, "data_fim": d_fim,
        "pastas": st.sidebar.multiselect("Pastas p/ Análise", sorted(df_completo["activity_folder"].dropna().unique())),
        "status": st.sidebar.multiselect("Status p/ Exibição", sorted(df_completo["activity_status"].dropna().unique())),
        "min_sim": st.sidebar.slider("Similaridade mínima (%)", 0, 100, 90, 5) / 100,
        "only_dup": st.sidebar.checkbox("Mostrar somente duplicatas", True),
        "only_multi": st.sidebar.checkbox("Apenas pastas com múltiplos responsáveis"),
        "usuarios": st.sidebar.multiselect("Usuários", sorted(df_completo["user_profile_name"].dropna().unique())),
    }

def filtrar_dados(df_completo: pd.DataFrame, filtros: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_base_comparacao = df_completo if not filtros["pastas"] else df_completo[df_completo["activity_folder"].isin(filtros["pastas"])]
    df_para_exibir = df_completo[df_completo["activity_date"].dt.date.between(filtros["data_inicio"], filtros["data_fim"])]
    if filtros["pastas"]:
        df_para_exibir = df_para_exibir[df_para_exibir["activity_folder"].isin(filtros["pastas"])]
    if filtros["status"]:
        df_para_exibir = df_para_exibir[df_para_exibir["activity_status"].isin(filtros["status"])]
    if filtros["only_multi"]:
        pastas_multi_users = {p for p, g in df_base_comparacao.groupby("activity_folder") if g["user_profile_name"].nunique() > 1}
        df_para_exibir = df_para_exibir[df_para_exibir["activity_folder"].isin(pastas_multi_users)]
    if filtros["usuarios"]:
        df_para_exibir = df_para_exibir[df_para_exibir["user_profile_name"].isin(filtros["usuarios"])]
    return df_para_exibir, df_base_comparacao

@st.dialog("Texto completo")
def exibir_dialogo_texto_completo():
    data = st.session_state[SK.FULL_TEXT_DATA]
    if data is None: return
    dt = as_sp(data["activity_date"])
    st.markdown(f"### ID {data['activity_id']} – {dt.strftime('%d/%m/%Y %H:%M') if dt else 'N/A'}")
    st.markdown(f"<pre style='max-height:400px;overflow:auto'>{html.escape(data['Texto'])}</pre>", unsafe_allow_html=True)
    st.button("Fechar", key="dialog_close_full_text", on_click=lambda: st.session_state.update({SK.SHOW_FULL_TEXT_DIALOG: False}))

def confirmar_cancelamento(api_client: HttpClient):
    def get_cancelados():
        return sorted([k.split('_')[-1] for k, v in st.session_state.items() if k.startswith("cancel_") and v])

    cancelados = get_cancelados()
    if not cancelados:
        st.write("Nenhuma atividade foi marcada para cancelamento.")
        if st.button("❌ Fechar"): st.session_state.pop(SK.STEP_CANCEL, None); st.rerun()
        return

    st.write("As seguintes atividades serão canceladas:"); st.code("\n".join(cancelados))

    if not st.session_state.get(SK.PROCESS_CANCEL) and not st.session_state.get(SK.STEP_CANCEL_PROCESSADO):
        c1, c2 = st.columns(2)
        c1.button("✅ Confirmar", key=SK.PROCESS_CANCEL)
        if c2.button("❌ Cancelar"): st.session_state.pop(SK.STEP_CANCEL, None); st.rerun()

    elif not st.session_state.get(SK.STEP_CANCEL_PROCESSADO):
        progress = st.progress(0, "Iniciando...")
        for i, act_id in enumerate(cancelados, 1):
            with st.status(f"Cancelando {act_id}..."):
                try:
                    response = api_client.activity_canceled(act_id, st.session_state[SK.USERNAME])
                    if response and response.get("code") == '200': st.success(f"✅ {act_id} cancelada.")
                    else: st.error(f"❌ Erro ao cancelar {act_id}: {response.get('message', 'Sem detalhes')}")
                except Exception as e: st.error(f"❌ Falha na requisição para {act_id}: {e}")
            progress.progress(i / len(cancelados), f"{i}/{len(cancelados)} concluídos")
        st.success("Processo finalizado."); st.session_state[SK.STEP_CANCEL_PROCESSADO] = True

    if st.session_state.get(SK.STEP_CANCEL_PROCESSADO):
        if st.button("✅ Concluir"):
            def clean_cancelados():
                for k in [key for key in st.session_state if key.startswith("cancel_")]: del st.session_state[k]
            
            clean_cancelados()
            st.session_state.pop(SK.STEP_CANCEL, None)
            st.session_state.pop(SK.STEP_CANCEL_PROCESSADO, None)
            st.session_state.pop(SK.PROCESS_CANCEL, None)
            st.session_state[SK.LAST_UPDATE] = datetime.now(TZ_SP)
            carregar_dados.clear()
            st.session_state.pop(SK.SIMILARITY_CACHE, None)
            st.rerun()

# ==============================================================================
# 6. APLICAÇÃO PRINCIPAL
# ==============================================================================

def app():
    api_client = inicializar_api_client()
    eng = db_engine()
    df_completo = carregar_dados(eng)
    if df_completo.empty:
        st.warning("Nenhuma atividade encontrada para análise."); st.stop()

    filtros = renderizar_sidebar(df_completo)
    df_para_exibir, df_base_comparacao = filtrar_dados(df_completo, filtros)
    sim_map, dup_ids = gerar_mapa_similaridade(df_para_exibir, df_base_comparacao, filtros["min_sim"])
    
    df_view = df_para_exibir.copy()
    if filtros["only_dup"]:
        df_view = df_view[df_view["activity_id"].isin(dup_ids)]

    idx_map_completo = df_base_comparacao.set_index("activity_id").to_dict("index")

    c1, c2 = st.columns([6, 2])
    with c1:
        st.markdown(f"<div class='vertical-align-bottom'><h3>🔎 Análise de Duplicidades ({len(df_view)} atividades exibidas)</h3></div>", unsafe_allow_html=True)
    with c2:
        if st.button("Cancelar Selecionado(s)", key="process_cancel_button"):
            st.session_state[SK.STEP_CANCEL] = "confirmar"; st.session_state[SK.STEP_CANCEL_PROCESSADO] = False
            confirmar_cancelamento(api_client)

    pastas_ord = sorted(df_view["activity_folder"].dropna().unique())
    if not pastas_ord:
        st.info("Nenhum resultado para os filtros selecionados."); st.stop()

    total_paginas = max(1, (len(pastas_ord) + ITENS_POR_PAGINA - 1) // ITENS_POR_PAGINA)
    page_num = st.session_state.get(SK.PAGE_NUMBER, 0)
    page_num = max(0, min(page_num, total_paginas - 1))
    st.session_state[SK.PAGE_NUMBER] = page_num

    if total_paginas > 1:
        l, mid, r = st.columns([1, 2, 1])
        if l.button("⬅", disabled=page_num == 0): st.session_state[SK.PAGE_NUMBER] -= 1; st.rerun()
        mid.markdown(f"<p style='text-align:center'>Página {page_num + 1}/{total_paginas}</p>", unsafe_allow_html=True)
        if r.button("➡", disabled=page_num == total_paginas - 1): st.session_state[SK.PAGE_NUMBER] += 1; st.rerun()

    pastas_na_pagina = pastas_ord[page_num * ITENS_POR_PAGINA : (page_num + 1) * ITENS_POR_PAGINA]
    for pasta in pastas_na_pagina:
        df_pasta_atual = df_view[df_view["activity_folder"] == pasta]
        total_na_pasta_base = len(df_base_comparacao[df_base_comparacao["activity_folder"] == pasta])
        max_selecoes = max(0, total_na_pasta_base - 1)
        def check_total(p): return sum(1 for k, v in st.session_state.items() if k.startswith(f"cancel_{p}_") and v)
        num_selecionados_atual = check_total(pasta)
        
        exp_title = f"📁 {pasta} ({len(df_pasta_atual)} de {total_na_pasta_base}) - Selecionados: {num_selecionados_atual}/{max_selecoes}"
        with st.expander(exp_title, expanded=True):
            for row in df_pasta_atual.itertuples():
                renderizar_cartao_atividade(row, pasta, sim_map, idx_map_completo, max_selecoes, num_selecionados_atual)
            st.divider()

    if st.session_state.get(SK.STEP_CANCEL) == "confirmar":
        confirmar_cancelamento(api_client)
    if st.session_state.get(SK.SHOW_FULL_TEXT_DIALOG):
        exibir_dialogo_texto_completo()

def renderizar_cartao_atividade(row, pasta, sim_map, idx_map, max_selecoes, num_selecionados):
    act_id = row.activity_id
    c1, c2 = st.columns([.6, .4], gap="small")
    with c1:
        dt = as_sp(row.activity_date)
        st.markdown(f"**ID** `{act_id}` • {dt.strftime('%d/%m/%Y %H:%M') if dt else 'N/A'} • `{row.activity_status}`")
        st.markdown(f"**Usuário:** {row.user_profile_name}")
        st.text_area("Texto", row.Texto, height=100, disabled=True, key=f"txt_{pasta}_{act_id}")
        b1, b2, b3 = st.columns(3)
        b1.button("👁 Completo", key=f"full_{act_id}", on_click=lambda r=row: st.session_state.update({SK.FULL_TEXT_DATA: r._asdict(), SK.SHOW_FULL_TEXT_DIALOG: True}))
        links = link_z(act_id); b2.link_button("ZFlow v1", links["antigo"]); b3.link_button("ZFlow v2", links["novo"])
    with c2:
        similares = sim_map.get(act_id, [])
        if similares:
            st.markdown(f"**Duplicatas Encontradas:** {len(similares)}")
            for s in similares:
                info = idx_map.get(s["id"])
                if not info: continue
                info_id = s["id"]; d = as_sp(info["activity_date"]); d_fmt = d.strftime("%d/%m/%y %H:%M") if d else "N/A"
                st.markdown(f"<div class='similarity-badge' style='background:{s['cor']};'><b>ID {info_id}</b> • {s['ratio']:.0%}<br>{d_fmt} • {info['activity_status']}<br>{info['user_profile_name']}</div>", unsafe_allow_html=True)
                def add_comp(id1, id2): st.session_state[SK.OPEN_COMPARISONS].add(tuple(sorted((id1, id2))))
                st.button("⚖ Comparar", key=f"cmp_{act_id}_{info_id}", on_click=add_comp, args=(act_id, info_id))

    open_comps = st.session_state.get(SK.OPEN_COMPARISONS, set())
    comps_to_show = [c for c in open_comps if act_id in c]
    if comps_to_show: st.markdown("---")
    for comp_pair in comps_to_show:
        comp_id = next(c_id for c_id in comp_pair if c_id != act_id)
        renderizar_visualizacao_comparacao(row, idx_map.get(comp_id), pasta, max_selecoes, num_selecionados, sim_map)

def renderizar_visualizacao_comparacao(base_data_row, comp_data_dict, pasta, max_sel, num_sel, sim_map):
    if not comp_data_dict: return
    base_id = base_data_row.activity_id; comp_id = comp_data_dict["activity_id"]
    with st.container(border=True):
        st.markdown("""<div style="font-size: 0.85em; margin-bottom: 10px; padding: 5px; background-color: #f0f2f6; border-radius: 5px;">
            <b>Legenda:</b> <span style="padding: 0 3px; background-color: #ffcdd2;">Texto removido</span> | <span style="padding: 0 3px; background-color: #c8e6c9;">Texto adicionado</span>
        </div>""", unsafe_allow_html=True)
        hA, hB = highlight_diffs(base_data_row.Texto, comp_data_dict["Texto"])
        colA, colB = st.columns(2)
        with colA:
            st.markdown(f"**Original: ID `{base_id}`**"); st.markdown(hA, unsafe_allow_html=True)
        with colB:
            sim_info = next((s for s in sim_map.get(base_id, []) if s['id'] == comp_id), {})
            ratio_str = f"{sim_info.get('ratio', 0):.0%}"
            st.markdown(f"**Comparado: ID `{comp_id}` ({ratio_str})**"); st.markdown(hB, unsafe_allow_html=True)

        st.markdown("##### ❎ Marcar para cancelamento")
        limite_atingido = num_sel >= max_sel
        col_chk1, col_chk2 = st.columns(2)
        def toggle_cancel_state(p, a_id): st.session_state[f"cancel_{p}_{a_id}"] = not st.session_state.get(f"cancel_{p}_{a_id}", False)
        with col_chk1:
            is_checked1 = st.session_state.get(f"cancel_{pasta}_{base_id}", False)
            st.checkbox(f"Cancelar ID {base_id}", value=is_checked1, key=f"chk_{base_id}_vs_{comp_id}", on_change=toggle_cancel_state, args=(pasta, base_id), disabled=limite_atingido and not is_checked1)
        with col_chk2:
            is_checked2 = st.session_state.get(f"cancel_{pasta}_{comp_id}", False)
            st.checkbox(f"Cancelar ID {comp_id}", value=is_checked2, key=f"chk_{comp_id}_vs_{base_id}", on_change=toggle_cancel_state, args=(pasta, comp_id), disabled=limite_atingido and not is_checked2)

        def remove_comp(b_id, c_id): st.session_state[SK.OPEN_COMPARISONS].discard(tuple(sorted((b_id, c_id))))
        st.button("❌ Fechar comparação", key=f"cls_{base_id}_{comp_id}", on_click=remove_comp, args=(base_id, comp_id))

# ==============================================================================
# 7. LÓGICA DE LOGIN
# ==============================================================================

def cred_ok(u, p):
    users = st.secrets.get("credentials", {}).get("usernames", {})
    return u in users and str(users[u]) == p

def login():
    st.header("Login")
    with st.form("login_form_main"):
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
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
    if st.session_state[SK.LOGGED_IN]:
        app()
    else:
        login()
