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
# CONFIGURAÇÕES
# ==============================================================================
SUFFIX_STATE    = "_final_v8_corrigido" # Sufixo atualizado para esta versão
ITENS_POR_PAGINA = 20
HIGHLIGHT_COLOR  = "#a8d1ff"
TZ_SP            = ZoneInfo("America/Sao_Paulo")

st.set_page_config(layout="wide", page_title="Verificador de Duplicidade (Refinado)")

st.markdown(
    f"""
    <style>
    mark.common {{ background-color:{HIGHLIGHT_COLOR}; padding:0 2px; font-weight: bold;}}
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
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================
def as_sp(timestamp: pd.Timestamp | None) -> datetime | None:
    if pd.isna(timestamp): return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert(TZ_SP)

def normalizar_texto(txt: str | None) -> str:
    if not isinstance(txt, str): return ""
    txt = unidecode(txt.lower())
    txt = re.sub(r"[^\w\s]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()

def calcular_similaridade(a: str, b: str) -> float:
    a, b = normalizar_texto(a), normalizar_texto(b)
    if not a or not b: return 0.0
    if abs(len(a) - len(b)) > 0.3 * max(len(a), len(b)): return 0.0
    return fuzz.token_set_ratio(a, b) / 100

def cor_sim(ratio: float) -> str:
    return "#FF5252" if ratio >= .9 else "#FFB74D" if ratio >= .7 else "#FFD54F"

def highlight_common(t1: str, t2: str, min_len: int = 3):
    tok1 = re.findall(r"\w+", normalizar_texto(t1))
    tok2 = re.findall(r"\w+", normalizar_texto(t2))
    comuns = {w for w in tok1 if w in tok2 and len(w) >= min_len}

    def wrap(txt):
        out = []
        for part in re.split(r"(\W+)", txt):
            if not part: continue
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
    cfg = st.secrets.get("database", {})
    host = cfg.get("host") or os.getenv("DB_HOST")
    user = cfg.get("user") or os.getenv("DB_USER")
    pw   = cfg.get("password") or os.getenv("DB_PASS")
    db   = cfg.get("name") or os.getenv("DB_NAME")
    if not all([host, user, pw, db]):
        st.error("Credenciais do banco ausentes."); return None
    try:
        eng = create_engine(f"mysql+mysqlconnector://{user}:{pw}@{host}/{db}", pool_pre_ping=True, pool_recycle=3600)
        with eng.connect(): pass
        return eng
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error("Erro de conexão."); return None

@st.cache_data(ttl=3600, hash_funcs={Engine: lambda _: None})
def carregar_dados(eng: Engine) -> pd.DataFrame:
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
        df["status_ord"] = df["activity_status"].map({"Aberta": 0}).fillna(1)
        df_final = (df.sort_values(["activity_id", "status_ord"])
                      .drop_duplicates("activity_id", keep="first")
                      .drop(columns="status_ord")
                      .sort_values(["activity_folder", "activity_date", "activity_id"], ascending=[True, False, False])
                      .reset_index(drop=True))
        return df_final
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error("Erro SQL"); return pd.DataFrame()

# ==============================================================================
# CACHE DE SIMILARIDADE
# ==============================================================================
def sim_map_cached(df: pd.DataFrame, min_sim: float):
    if df.empty: return {}, set()
    sig = (tuple(sorted(df["activity_id"])), min_sim)
    key = "simcache" + SUFFIX_STATE
    cache = st.session_state.get(key)
    if cache and cache.get("sig") == sig: return cache["map"], cache["dup"]

    mapa, dup_ids = {}, set()
    bar = st.sidebar.progress(0, text="Calculando similaridades...")
    grupos = list(df.groupby("activity_folder"))
    for i, (nome_pasta, g) in enumerate(grupos, 1):
        bar.progress(i / len(grupos), text=f"Analisando: {nome_pasta}")
        acts = g.to_dict("records")
        for idx, a in enumerate(acts):
            mapa.setdefault(a["activity_id"], [])
            for b in acts[idx + 1:]:
                r = calcular_similaridade(a["Texto"], b["Texto"])
                if r >= min_sim:
                    c = cor_sim(r)
                    dup_ids.update([a["activity_id"], b["activity_id"]])
                    mapa[a["activity_id"]].append(dict(id=b["activity_id"], ratio=r, cor=c))
                    mapa.setdefault(b["activity_id"], []).append(dict(id=a["activity_id"], ratio=r, cor=c))
    bar.empty()
    for k in mapa: mapa[k].sort(key=lambda x: x["ratio"], reverse=True)
    st.session_state[key] = {"sig": sig, "map": mapa, "dup": dup_ids}
    return mapa, dup_ids

# ==============================================================================
# ESTADO E UI
# ==============================================================================
link_z = lambda i: {"antigo": f"https://zflow.zionbyonset.com.br/activity/3/details/{i}", "novo": f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={i}#/fixcol1"}

for key_base in ["show_text_dialog", "full_act", "cmp", "pagina_atual", "last_update"]:
    st.session_state.setdefault(f"{key_base}{SUFFIX_STATE}", False if "show" in key_base else (0 if "pagina" in key_base else None))

@st.dialog("Texto completo")
def dlg_full():
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

    eng = db_engine()
    if not eng: st.stop()

    if st.sidebar.button("🔄 Atualizar dados"):
        carregar_dados.clear(); st.session_state.pop("simcache"+SUFFIX_STATE, None)
        st.session_state[f"cmp{SUFFIX_STATE}"] = None
        st.session_state[f"last_update{SUFFIX_STATE}"] = datetime.now(TZ_SP)

    df_raw = carregar_dados(eng)
    if isinstance(df_raw, tuple): st.exception(df_raw[1]); st.stop()
    if df_raw.empty: st.warning("Sem atividades base carregadas."); st.stop()

    up = st.session_state[f"last_update{SUFFIX_STATE}"] or datetime.now(TZ_SP)
    st.sidebar.caption(f"Dados do banco atualizados em: {up:%d/%m/%Y %H:%M:%S}")

    st.sidebar.header("Período de Exibição")
    hoje = date.today()
    d_ini_def = hoje - timedelta(days=1)
    df_futuras = df_raw[(df_raw["activity_status"] == "Aberta") & (df_raw["activity_date"].notna()) & (df_raw["activity_date"].dt.date > hoje)]
    d_fim_def = df_futuras["activity_date"].dt.date.max() if not df_futuras.empty else hoje + timedelta(days=14)
    if d_ini_def > d_fim_def: d_ini_def = d_fim_def - timedelta(days=1)
    
    d_ini = st.sidebar.date_input("Início", d_ini_def)
    d_fim = st.sidebar.date_input("Fim", d_fim_def, min_value=d_ini)
    
    df_periodo = df_raw[df_raw["activity_date"].notna() & df_raw["activity_date"].dt.date.between(d_ini, d_fim)]
    st.title(f"🔎 Verificador de Duplicidade ({len(df_periodo)} atividades no período)")

    st.sidebar.header("Filtros")
    pastas_sel = st.sidebar.multiselect("Pastas para Análise:", sorted(df_periodo["activity_folder"].dropna().unique()))
    
    # --- AJUSTE NA LÓGICA DE FILTROS ---
    # DataFrame para análise de similaridade ignora filtros de status/usuário da UI
    df_para_analise = df_periodo[df_periodo["activity_folder"].isin(pastas_sel)] if pastas_sel else df_periodo

    status_disp = sorted(df_para_analise["activity_status"].dropna().unique())
    status_sel_exibicao = st.sidebar.multiselect("Status para Exibição:", status_disp)
    
    min_sim = st.sidebar.slider("Similaridade mínima (%)", 0, 100, 70, 5) / 100
    only_dup = st.sidebar.checkbox("Exibir Somente com Duplicatas", True)
    
    pastas_multi = {p for p, g in df_para_analise.groupby("activity_folder") if g["user_profile_name"].nunique() > 1}
    only_multi = st.sidebar.checkbox("Pastas com múltiplos responsáveis")
    
    users_sel = st.sidebar.multiselect("Usuários para Exibição:", sorted(df_para_analise["user_profile_name"].dropna().unique()))

    sim_map, ids_dup = sim_map_cached(df_para_analise, min_sim)

    # DataFrame final para exibição
    df_view = df_para_analise.copy()
    if status_sel_exibicao: df_view = df_view[df_view["activity_status"].isin(status_sel_exibicao)]
    if only_dup: df_view = df_view[df_view["activity_id"].isin(ids_dup)]
    if only_multi: df_view = df_view[df_view["activity_folder"].isin(pastas_multi)]
    if users_sel: df_view = df_view[df_view["user_profile_name"].isin(users_sel)]
    
    # ... Lógica de exportação ...
    
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
    idx_map = df_para_analise.set_index("activity_id").to_dict("index")

    for pasta in pastas_ord[start:end]:
        df_p = df_view[df_view["activity_folder"] == pasta]
        analisadas = len(df_para_analise[df_para_analise["activity_folder"] == pasta])
        with st.expander(f"📁 {pasta} ({len(df_p)} exibidas / {analisadas} analisadas)", expanded=False):
            for _, r in df_p.iterrows():
                act_id = int(r["activity_id"])
                c1, c2 = st.columns([.6, .4], gap="small")
                with c1:
                    data_fmt = as_sp(r["activity_date"]).strftime("%d/%m/%Y %H:%M") if pd.notna(r["activity_date"]) else "N/A"
                    st.markdown(f"**ID** `{act_id}` • {data_fmt} • `{r['activity_status']}`")
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
                            info = idx_map.get(s["id"])
                            if not info: continue
                            d = as_sp(info["activity_date"]).strftime("%d/%m/%y %H:%M") if pd.notna(info["activity_date"]) else "N/A"
                            badge = (f"<div class='similarity-badge' style='background:{s['cor']};'>"
                                     f"<b>{info['activity_id']}</b> • {s['ratio']:.0%}<br>"
                                     f"{d} • {info['activity_status']}<br>{info['user_profile_name']}</div>")
                            st.markdown(badge, unsafe_allow_html=True)
                            st.button("⚖ Comparar", key=f"cmp_{act_id}_{info['activity_id']}_{pagina_atual}",
                                      on_click=lambda a_id=act_id, b_id=info['activity_id']: st.session_state.update({f"cmp{SUFFIX_STATE}": {"base_id": a_id, "comp_id": b_id}}))
                    elif not only_dup:
                        st.markdown("<span style='color:green;'>Sem duplicatas</span>", unsafe_allow_html=True)

    if st.session_state[f"show_text_dialog{SUFFIX_STATE}"]: dlg_full()

# ==============================================================================
# LOGIN
# ==============================================================================
def cred_ok(u,p):
    creds = st.secrets.get("credentials", {}).get("usernames", {})
    return u in creds and str(creds[u]) == p

def login_form():
    st.header("Login")
    with st.form("login_main_form"):
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
