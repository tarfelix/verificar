import streamlit as st
import pandas as pd
import re, io, html, os, logging
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from unidecode import unidecode
from rapidfuzz import fuzz

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================
SUFFIX_STATE = "_final_v7_refinado" # Sufixo atualizado para esta versão
ITENS_POR_PAGINA = 20
HIGHLIGHT_COLOR = "#a8d1ff"
TZ_SP = ZoneInfo("America/Sao_Paulo") # Fuso Horário de São Paulo

st.set_page_config(layout="wide", page_title="Verificador de Duplicidade (Refinado)")

# CSS para destacar semelhanças e formatar texto
st.markdown(f"""
<style>
mark.common {{background-color:{HIGHLIGHT_COLOR}; padding:0 2px; font-weight:bold;}}
pre.highlighted-text {{
    white-space: pre-wrap; word-wrap: break-word;
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
    font-size: 0.9em; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;
}}
.similarity-badge {{
    padding: 3px 6px; border-radius: 5px; color: black; 
    font-weight: 500; display: inline-block; margin-bottom: 4px;
}}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================
def as_sp(timestamp: pd.Timestamp | None) -> datetime | None:
    """Converte um timestamp para o fuso horário de São Paulo, tratando NaT."""
    if pd.isna(timestamp): return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC") # Assume UTC se não houver fuso
    return timestamp.tz_convert(TZ_SP)

def normalizar_texto(txt: str | None) -> str:
    """Remove acentos, pontuação e espaços múltiplos, retornando lower-case."""
    if not isinstance(txt, str): return ""
    txt = unidecode(txt.lower())
    txt = re.sub(r"[^\w\s]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()

def calcular_similaridade(texto_a: str, texto_b: str) -> float:
    """Calcula a similaridade usando rapidfuzz após normalizar os textos."""
    a, b = normalizar_texto(texto_a), normalizar_texto(texto_b)
    if not a or not b: return 0.0
    # Heurística para pular comparações entre textos de tamanhos muito diferentes
    if abs(len(a) - len(b)) > 0.3 * max(len(a), len(b)): return 0.0
    return fuzz.token_set_ratio(a, b) / 100

def cor_sim(ratio: float) -> str:
    """Retorna a cor correspondente com base no nível de similaridade."""
    if ratio >= .9: return "#FF5252" # Vermelho
    if ratio >= .7: return "#FFB74D" # Laranja
    return "#FFD54F" # Amarelo

def highlight_common(t1: str, t2: str, min_len: int = 3) -> tuple[str, str]:
    """Gera HTML para dois textos com palavras em comum destacadas."""
    tok1 = re.findall(r"\w+", normalizar_texto(t1))
    tok2 = re.findall(r"\w+", normalizar_texto(t2))
    comuns = {w for w in tok1 if w in tok2 and len(w) >= min_len}

    def wrap(txt):
        out = []
        # Divide o texto original mantendo os delimitadores (espaços, pontuação)
        for part in re.split(r"(\W+)", txt):
            if not part: continue
            # Destaca a palavra se sua versão normalizada estiver no conjunto de palavras comuns
            if re.match(r"\w+", part) and normalizar_texto(part) in comuns:
                out.append(f"<mark class='common'>{html.escape(part)}</mark>")
            else:
                out.append(html.escape(part))
        return "<pre class='highlighted-text'>" + "".join(out) + "</pre>"
    return wrap(t1), wrap(t2)

# ==============================================================================
# BANCO DE DADOS (MySQL)
# ==============================================================================
@st.cache_resource
def db_engine() -> Engine | None:
    """Cria e retorna uma engine de conexão com o banco."""
    cfg = st.secrets.get("database", {})
    host = cfg.get("host") or os.getenv("DB_HOST")
    user = cfg.get("user") or os.getenv("DB_USER")
    pw = cfg.get("password") or os.getenv("DB_PASS")
    db = cfg.get("name") or os.getenv("DB_NAME")
    if not all([host, user, pw, db]):
        st.error("Credenciais do banco ausentes em `st.secrets` ou variáveis de ambiente."); return None
    try:
        eng = create_engine(f"mysql+mysqlconnector://{user}:{pw}@{host}/{db}", pool_pre_ping=True, pool_recycle=3600)
        with eng.connect(): pass
        return eng
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error("Erro ao conectar no banco."); return None

@st.cache_data(ttl=3600, hash_funcs={Engine: lambda _: None})
def carregar_dados(eng: Engine) -> pd.DataFrame:
    """Lê atividades 'Verificar' Abertas (sem limite futuro) + últimos 7 dias do histórico."""
    limite = date.today() - timedelta(days=7)
    q_abertas = text("SELECT activity_id, activity_folder, user_profile_name, activity_date, activity_status, Texto FROM ViewGrdAtividadesTarcisio WHERE activity_type='Verificar' AND activity_status='Aberta'")
    q_hist = text("SELECT activity_id, activity_folder, user_profile_name, activity_date, activity_status, Texto FROM ViewGrdAtividadesTarcisio WHERE activity_type='Verificar' AND DATE(activity_date) >= :limite")
    try:
        with eng.connect() as c:
            df_abertas = pd.read_sql(q_abertas, c)
            df_historico = pd.read_sql(q_hist, c, params={"limite": limite})
        df = pd.concat([df_abertas, df_historico], ignore_index=True)
        if df.empty: return df
        
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
        df["Texto"] = df["Texto"].astype(str).fillna("")
        
        # Cria uma coluna temporária para priorizar o status 'Aberta'
        df["status_ord"] = df["activity_status"].map({"Aberta": 0}).fillna(1)
        df_final = (df.sort_values(["activity_id", "status_ord"])
                      .drop_duplicates("activity_id", keep="first")
                      .drop(columns="status_ord")
                      .sort_values(["activity_folder", "activity_date", "activity_id"], ascending=[True, False, False])
                      .reset_index(drop=True))
        return df_final
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error("Erro ao executar consulta SQL."); return pd.DataFrame()

# ==============================================================================
# CACHE DE SIMILARIDADE
# ==============================================================================
def get_similarity_map_cached(df: pd.DataFrame, min_sim: float):
    """Calcula ou recupera do cache o mapa de similaridades."""
    if df.empty: return {}, set()
    
    ids_tuple = tuple(sorted(df["activity_id"]))
    signature = (ids_tuple, min_sim)
    cache_key = "simcache" + SUFFIX_STATE
    
    cached_data = st.session_state.get(cache_key)
    if cached_data and cached_data.get("sig") == signature:
        return cached_data["map"], cached_data["dup"]

    mapa_similaridade, ids_com_duplicatas = {}, set()
    prog_bar = st.sidebar.progress(0, text="Calculando similaridades...")
    grupos_pasta = list(df.groupby("activity_folder"))
    
    for idx, (nome_pasta, grupo) in enumerate(grupos_pasta, 1):
        prog_bar.progress(idx / len(grupos_pasta), text=f"Analisando pasta: {nome_pasta}")
        atividades = grupo.to_dict("records")
        if len(atividades) < 2: continue
        for i, atividade_base in enumerate(atividades):
            mapa_similaridade.setdefault(atividade_base["activity_id"], [])
            for atividade_comparar in atividades[i+1:]:
                ratio = calcular_similaridade(atividade_base["Texto"], atividade_comparar["Texto"])
                if ratio >= min_sim:
                    cor = cor_sim(ratio)
                    ids_com_duplicatas.update([atividade_base["activity_id"], atividade_comparar["activity_id"]])
                    mapa_similaridade[atividade_base["activity_id"]].append(dict(id_similar=atividade_comparar["activity_id"], ratio=ratio, cor=cor))
                    mapa_similaridade.setdefault(atividade_comparar["activity_id"], []).append(dict(id_similar=atividade_base["activity_id"], ratio=ratio, cor=cor))
    
    prog_bar.empty()
    for k in mapa_similaridade: mapa_similaridade[k].sort(key=lambda x: x["ratio"], reverse=True)
    
    st.session_state[cache_key] = {"sig": signature, "map": mapa_similaridade, "dup": ids_com_duplicatas}
    return mapa_similaridade, ids_com_duplicatas

# ==============================================================================
# ESTADO E UI
# ==============================================================================
link_z = lambda i: {"antigo": f"https://zflow.zionbyonset.com.br/activity/3/details/{i}", "novo": f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={i}#/fixcol1"}

for key_base in ["show_text_dialog", "full_act", "cmp", "pagina_atual", "last_update"]:
    st.session_state.setdefault(f"{key_base}{SUFFIX_STATE}", False if "show" in key_base else (0 if "pagina" in key_base else None))

@st.dialog("Texto completo")
def dlg_full_text():
    d = st.session_state[f"full_act{SUFFIX_STATE}"]
    if d is None: return
    data_fmt = as_sp(d["activity_date"]).strftime("%d/%m/%Y %H:%M") if pd.notna(d["activity_date"]) else "N/A"
    st.markdown(f"### ID {d['activity_id']} – {data_fmt}")
    st.markdown(f"<pre style='max-height:400px;overflow:auto'>{html.escape(d['Texto'])}</pre>", unsafe_allow_html=True)
    st.button("Fechar", on_click=lambda: st.session_state.update({f"show_text{SUFFIX_STATE}": False}))

def app():
    st.sidebar.success(f"Logado como **{st.session_state['username']}**")
    if st.sidebar.button("Logout"):
        st.session_state.clear(); st.rerun()

    engine = db_engine()
    if not engine: st.stop()

    if st.sidebar.button("🔄 Atualizar dados"):
        carregar_dados.clear(); st.session_state.pop("simcache"+SUFFIX_STATE, None)
        st.session_state[f"cmp{SUFFIX_STATE}"] = None
        st.session_state[f"last_update{SUFFIX_STATE}"] = datetime.now(TZ_SP)

    df_raw = carregar_dados(engine)
    if isinstance(df_raw, tuple) and df_raw[1]: st.exception(df_raw[1]); st.stop()
    if df_raw.empty: st.warning("Sem atividades base carregadas."); st.stop()

    if not st.session_state[f"last_update{SUFFIX_STATE}"]:
        st.session_state[f"last_update{SUFFIX_STATE}"] = datetime.now(TZ_SP)
    st.sidebar.caption(f"Dados do banco atualizados em: {st.session_state[f'last_update{SUFFIX_STATE}']: %d/%m/%Y %H:%M:%S}")

    st.sidebar.header("Período de Exibição")
    hoje = date.today()
    data_inicio_padrao = hoje - timedelta(days=1)
    df_futuras = df_raw[(df_raw["activity_status"] == "Aberta") & (df_raw["activity_date"].notna()) & (df_raw["activity_date"].dt.date > hoje)]
    data_fim_padrao = df_futuras["activity_date"].dt.date.max() if not df_futuras.empty else hoje + timedelta(days=14)
    if data_inicio_padrao > data_fim_padrao: data_inicio_padrao = data_fim_padrao - timedelta(days=1)
    
    d_ini = st.sidebar.date_input("Início", data_inicio_padrao, key=f"di_{SUFFIX_STATE}")
    d_fim = st.sidebar.date_input("Fim", d_fim_padrao, min_value=d_ini, key=f"df_{SUFFIX_STATE}")
    
    df_periodo = df_raw[df_raw["activity_date"].notna() & df_raw["activity_date"].dt.date.between(d_ini, d_fim)]
    st.title(f"🔎 Verificador de Duplicidade ({len(df_periodo)} atividades no período)")

    st.sidebar.header("Filtros")
    pastas_sel = st.sidebar.multiselect("Pastas para Análise:", sorted(df_periodo["activity_folder"].dropna().unique()))
    
    # DataFrame para análise de similaridade ignora filtros de status/usuário
    df_para_analise = df_periodo[df_periodo["activity_folder"].isin(pastas_sel)] if pastas_sel else df_periodo

    status_sel = st.sidebar.multiselect("Status para Exibição:", sorted(df_para_analise["activity_status"].dropna().unique()))
    
    min_sim = st.sidebar.slider("Similaridade mínima (%)", 0, 100, 70, 5, key=f"sim_slider_{SUFFIX_STATE}") / 100
    only_dup = st.sidebar.checkbox("Exibir Somente com Duplicatas", True)
    
    pastas_multi = {p for p, g in df_para_analise.groupby("activity_folder") if g["user_profile_name"].nunique() > 1}
    only_multi = st.sidebar.checkbox("Pastas com múltiplos responsáveis")
    
    users_sel = st.sidebar.multiselect("Usuários para Exibição:", sorted(df_para_analise["user_profile_name"].dropna().unique()))

    sim_map, ids_dup = get_similarity_map_cached(df_para_analise, min_sim)

    # DataFrame final para exibição
    df_view = df_para_analise.copy()
    if status_sel: df_view = df_view[df_view["activity_status"].isin(status_sel)]
    if only_dup: df_view = df_view[df_view["activity_id"].isin(ids_dup)]
    if only_multi: df_view = df_view[df_view["activity_folder"].isin(pastas_multi)]
    if users_sel: df_view = df_view[df_view["user_profile_name"].isin(users_sel)]
    
    # Exportação XLSX
    st.sidebar.markdown("---")
    if st.sidebar.button("📥 Exportar para XLSX"):
        #... (lógica de exportação mantida como na sua versão)

    # Comparação Lado-a-Lado
    cmp_state = st.session_state[f"cmp{SUFFIX_STATE}"]
    if cmp_state:
        base_rows = df_raw[df_raw.activity_id == cmp_state["base_id"]]
        comp_rows = df_raw[df_raw.activity_id == cmp_state["comp_id"]]
        if not base_rows.empty and not comp_rows.empty:
            a_base, b_comp = base_rows.iloc[0], comp_rows.iloc[0]
            with st.container(border=True):
                st.subheader(f"🔎 Comparação: ID `{a_base['activity_id']}` vs `{b_comp['activity_id']}`")
                col1_cmp, col2_cmp = st.columns(2)
                html_a, html_b = highlight_common(a_base["Texto"], b_comp["Texto"])
                with col1_cmp: st.markdown(f"**ID {a_base['activity_id']} (Base)**<br>{html_a}", unsafe_allow_html=True)
                with col2_cmp: st.markdown(f"**ID {b_comp['activity_id']} (Similar)**<br>{html_b}", unsafe_allow_html=True)
                if st.button("Ocultar Comparação", key=f"fechar_comp{SUFFIX_STATE}"):
                    st.session_state[f"cmp{SUFFIX_STATE}"] = None; st.rerun()
            st.markdown("---")
    
    # Listagem e Paginação
    st.header("Análise Detalhada por Pasta")
    if df_view.empty: st.info("Nenhuma atividade para os filtros de exibição selecionados.")

    pastas_ord = sorted(df_view["activity_folder"].dropna().unique())
    pagina_atual = st.session_state[f"page{SUFFIX_STATE}"]
    total_paginas = max(1, (len(pastas_ord) + ITENS_POR_PAGINA - 1) // ITENS_POR_PAGINA)
    pagina_atual = max(0, min(pagina_atual, total_paginas - 1)); st.session_state[f"page{SUFFIX_STATE}"] = pagina_atual

    if total_paginas > 1:
        c1,c2,c3 = st.columns([1,2,1])
        if c1.button("⬅️", disabled=pagina_atual==0): st.session_state[f"page{SUFFIX_STATE}"]-=1; st.rerun()
        c2.markdown(f"<p style='text-align:center'>Página {pagina_atual+1}/{total_paginas}</p>", unsafe_allow_html=True)
        if c3.button("➡️", disabled=pagina_atual>=total_paginas-1): st.session_state[f"page{SUFFIX_STATE}"]+=1; st.rerun()

    start, end = pagina_atual * ITENS_POR_PAGINA, (pagina_atual + 1) * ITENS_POR_PAGINA
    for pasta in pastas_ord[start:end]:
        df_p = df_view[df_view["activity_folder"] == pasta]
        analisadas = len(df_para_analise[df_para_analise["activity_folder"] == pasta])
        with st.expander(f"📁 {pasta} ({len(df_p)} exibidas / {analisadas} analisadas)", expanded=False):
            for _, r in df_p.iterrows():
                act_id = int(r["activity_id"])
                c1, c2 = st.columns([.6, .4], gap="small")
                with c1:
                    data = r["activity_date"].strftime("%d/%m/%Y %H:%M") if pd.notna(r["activity_date"]) else "N/A"
                    st.markdown(f"**ID** `{act_id}` • {data} • `{r['activity_status']}`")
                    st.markdown(f"**Usuário:** {r['user_profile_name']}")
                    st.text_area("Texto", r["Texto"], height=100, disabled=True, key=f"txt_{pasta}_{act_id}_{pagina_atual}")
                    b1,b2,b3 = st.columns(3)
                    b1.button("👁 Completo", key=f"full_{pasta}_{act_id}_{pagina_atual}", on_click=lambda act=r: st.session_state.update({f"full_act{SUFFIX_STATE}": act.to_dict(), f"show_text{SUFFIX_STATE}": True}))
                    lnk = link_z(act_id)
                    b2.link_button("ZFlow v1", lnk["antigo"])
                    b3.link_button("ZFlow v2", lnk["novo"])
                with c2:
                    sims = sim_map.get(act_id, [])
                    if sims:
                        st.markdown(f"**Duplicatas:** {len(sims)}")
                        for s in sims:
                            # A busca da info da duplicata deve ser feita em df_para_analise, que tem todos os status
                            info_rows = df_para_analise[df_para_analise["activity_id"] == s["id_similar"]]
                            if not info_rows.empty:
                                info = info_rows.iloc[0]
                                d = info["activity_date"].strftime("%d/%m/%y %H:%M") if pd.notna(info["activity_date"]) else "N/A"
                                badge = (f"<div class='similarity-badge' style='background:{s['cor']};'>"
                                         f"<b>{info['activity_id']}</b> • {s['ratio']:.0%}<br>"
                                         f"{d} • {info['activity_status']}<br>{info['user_profile_name']}</div>")
                                st.markdown(badge, unsafe_allow_html=True)
                                st.button("⚖ Comparar", key=f"cmp_{act_id}_{info['activity_id']}_{pagina_atual}",
                                          on_click=lambda a=r, b=info: st.session_state.update({f"cmp{SUFFIX_STATE}": {"base_id": a.activity_id, "comp_id": b.activity_id}}))
                    elif not only_dup:
                        st.markdown("<span style='color:green;'>Sem duplicatas</span>", unsafe_allow_html=True)
    if st.session_state[f"show_text{SUFFIX_STATE}"]: dlg_full()

# ==============================================================================
# LOGIN
# ==============================================================================
def cred_ok(u,p):
    creds = st.secrets.get("credentials", {}).get("usernames", {})
    return u in creds and str(creds[u]) == p

def login_form(): # Renomeado de login
    st.header("Login")
    with st.form("login_form"): # Renomeado
        u = st.text_input("Usuário")
        p = st.text_input("Senha", type="password")
        if st.form_submit_button("Entrar"):
            if cred_ok(u,p):
                st.session_state.update({"logged_in":True,"username":u}); st.rerun()
            else: st.error("Credenciais inválidas.")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    if not st.session_state.get("logged_in"): st.session_state["logged_in"] = False
    if st.session_state["logged_in"]: app()
    else: login_form()
