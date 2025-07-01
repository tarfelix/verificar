# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import re, html, os, logging
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from unidecode import unidecode
from rapidfuzz import fuzz
from difflib import SequenceMatcher
from dataclasses import dataclass

# ═════════════════ CONFIGURAÇÕES GLOBAIS ═════════════════

ITENS_POR_PAGINA = 20
TZ_SP = ZoneInfo("America/Sao_Paulo")
TZ_UTC = ZoneInfo("UTC")

class SK:
    LOGGED_IN = "logged_in"
    USERNAME = "username"
    LAST_UPDATE = "last_update"
    SIM_CACHE = "sim_cache"
    PAGE = "page"
    SHOW_FULL_TEXT_DIALOG = "show_text_dialog"
    FULL_TEXT_DATA = "full_text_data"
    COMPARE_STATE = "compare_state"

@dataclass
class CompareState:
    base_id: int
    comp_id: int

st.set_page_config(layout="wide", page_title="Verificador de Duplicidade")
st.markdown("""
<style>
    pre.highlighted-text {
        white-space: pre-wrap;
        word-wrap: break-word;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
        font-size: .9em;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        background-color: #f9f9f9;
        height: 400px; 
        overflow-y: auto;
    }
    .similarity-badge {
        padding: 3px 6px; border-radius: 5px; color: black; font-weight: 500;
        display: inline-block; margin-bottom: 4px;
    }
    .highlighted-text del {
        background-color: #ffcdd2; /* Vermelho claro */
        text-decoration: none;
    }
    .highlighted-text ins {
        background-color: #c8e6c9; /* Verde claro */
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)


# ═════════════ FUNÇÕES AUXILIARES ═════════════

def as_sp(ts: pd.Timestamp | None):
    if pd.isna(ts): return None
    if ts.tzinfo is None: ts = ts.tz_localize(TZ_UTC)
    return ts.tz_convert(TZ_SP)

def norm(t: str | None) -> str:
    if not isinstance(t, str): return ""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", unidecode(t.lower()))).strip()

def calc_sim(a: str, b: str) -> float:
    a_norm, b_norm = norm(a), norm(b)
    if not a_norm or not b_norm or abs(len(a_norm) - len(b_norm)) > 0.3 * max(len(a_norm), len(b_norm)):
        return 0.0
    return fuzz.token_set_ratio(a_norm, b_norm) / 100

def cor_sim(r: float) -> str:
    return "#FF5252" if r >= 0.9 else "#FFB74D" if r >= 0.7 else "#FFD54F"

# (VERSÃO FINAL E DEFINITIVA DA FUNÇÃO)
def highlight_diffs(t1: str, t2: str) -> tuple[str, str]:
    """
    Compara dois textos e destaca as diferenças (adições/remoções)
    usando uma tokenização granular que preserva palavras e pontuações.
    """
    t1, t2 = (t1 or ""), (t2 or "")
    
    # Tokeniza em qualquer caractere não-alfanumérico (\W+), mantendo os delimitadores.
    # Isso separa palavras, espaços e pontuações de forma eficaz.
    t1_tokens = [token for token in re.split(r'(\W+)', t1) if token]
    t2_tokens = [token for token in re.split(r'(\W+)', t2) if token]

    sm = SequenceMatcher(None, t1_tokens, t2_tokens, autojunk=False)

    out1, out2 = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        slice1 = html.escape("".join(t1_tokens[i1:i2]))
        slice2 = html.escape("".join(t2_tokens[j1:j2]))

        if tag == 'equal':
            out1.append(slice1)
            out2.append(slice2)
        elif tag == 'replace':
            out1.append(f"<del>{slice1}</del>")
            out2.append(f"<ins>{slice2}</ins>")
        elif tag == 'delete':
            out1.append(f"<del>{slice1}</del>")
        elif tag == 'insert':
            out2.append(f"<ins>{slice2}</ins>")
            
    h1 = f"<pre class='highlighted-text'>{''.join(out1)}</pre>"
    h2 = f"<pre class='highlighted-text'>{''.join(out2)}</pre>"
    return h1, h2

# ═════════════ BANCO DE DADOS (Inalterado) ═════════════
@st.cache_resource
def db_engine() -> Engine | None:
    cfg=st.secrets.get("database",{})
    host,user,pw,db=[cfg.get(k) or os.getenv(f"DB_{k.upper()}") for k in["host","user","password","name"]]
    if not all([host,user,pw,db]): st.error("Credenciais de banco ausentes."); return None
    try:
        eng=create_engine(f"mysql+mysqlconnector://{user}:{pw}@{host}/{db}",pool_pre_ping=True,pool_recycle=3600)
        with eng.connect(): pass
        return eng
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error("Erro ao conectar."); return None

@st.cache_data(ttl=3600, hash_funcs={Engine: lambda _: None})
def carregar(eng: Engine) -> pd.DataFrame:
    # A data do WHERE foi ajustada para um período maior para garantir que os dados de exemplo sejam carregados
    lim=date.today()-timedelta(days=90) 
    q_open=text("""SELECT activity_id,activity_folder,user_profile_name,
                   activity_date,activity_status,Texto
                   FROM ViewGrdAtividadesTarcisio
                   WHERE activity_type='Verificar' AND activity_status='Aberta'""")
    q_hist=text("""SELECT activity_id,activity_folder,user_profile_name,
                   activity_date,activity_status,Texto
                   FROM ViewGrdAtividadesTarcisio
                   WHERE activity_type='Verificar' AND DATE(activity_date)>=:lim""")
    try:
        with eng.connect() as c:
            df=pd.concat([pd.read_sql(q_open,c),
                          pd.read_sql(q_hist,c,params={"lim":lim})],ignore_index=True)
        if df.empty: return df
        df["activity_date"]=pd.to_datetime(df["activity_date"],errors="coerce")
        df["Texto"]=df["Texto"].astype(str).fillna("")
        df["status_ord"]=df["activity_status"].map({"Aberta":0}).fillna(1)
        df=(df.sort_values(["activity_id","status_ord"])
                .drop(columns="status_ord")
                .drop_duplicates("activity_id"))
        return df.sort_values(["activity_folder","activity_date","activity_id"],
                              ascending=[True,False,False])
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error("Erro SQL"); return pd.DataFrame()


# ═════════════ LÓGICA DO APP (Inalterado na maior parte) ═════════════
def sim_cache(df: pd.DataFrame, min_sim: float):
    sig = (tuple(sorted(df["activity_id"])), min_sim)
    cached = st.session_state.get(SK.SIM_CACHE)
    if cached and cached.get("sig") == sig:
        return cached["map"], cached["dup"]
    mapa, dup = {}, set()
    bar = st.sidebar.progress(0, text="Calculando similaridades…")
    groups = list(df.groupby("activity_folder"))
    for i, (_, g) in enumerate(groups, 1):
        bar.progress(i / len(groups), text=f"Calculando... {i}/{len(groups)} pastas")
        acts = g.to_dict("records")
        for idx, a in enumerate(acts):
            mapa.setdefault(a["activity_id"], [])
            for b in acts[idx + 1:]:
                r = calc_sim(a["Texto"], b["Texto"])
                if r >= min_sim:
                    dup.update([a["activity_id"], b["activity_id"]])
                    mapa[a["activity_id"]].append({"id": b["activity_id"], "ratio": r, "cor": cor_sim(r)})
                    mapa.setdefault(b["activity_id"], []).append({"id": a["activity_id"], "ratio": r, "cor": cor_sim(r)})
    bar.empty()
    for k in mapa:
        mapa[k].sort(key=lambda z: z["ratio"], reverse=True)
    st.session_state[SK.SIM_CACHE] = {"sig": sig, "map": mapa, "dup": dup}
    return mapa, dup

def init_session_state():
    defaults = {
        SK.LOGGED_IN: False, SK.USERNAME: "", SK.LAST_UPDATE: None,
        SK.SIM_CACHE: None, SK.PAGE: 0, SK.SHOW_FULL_TEXT_DIALOG: False,
        SK.FULL_TEXT_DATA: None, SK.COMPARE_STATE: None
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

@st.dialog("Texto completo")
def show_full_text_dialog():
    d = st.session_state.get(SK.FULL_TEXT_DATA)
    if not d: return
    dt = as_sp(d["activity_date"])
    st.markdown(f"### ID {d['activity_id']} – {dt.strftime('%d/%m/%Y %H:%M') if dt else 'N/A'}")
    st.text_area("Texto Completo", d['Texto'], height=400, disabled=True)
    if st.button("Fechar"):
        st.session_state[SK.SHOW_FULL_TEXT_DIALOG] = False
        st.rerun()

def app():
    st.sidebar.success(f"Logado como **{st.session_state[SK.USERNAME]}**")
    if st.sidebar.button("Logout"):
        st.session_state.clear(); st.rerun()

    eng = db_engine()
    if not eng: st.stop()

    if st.sidebar.button("🔄 Atualizar dados"):
        carregar.clear()
        st.session_state[SK.SIM_CACHE] = None
        st.session_state[SK.LAST_UPDATE] = datetime.now(TZ_SP)

    df = carregar(eng)
    if df.empty:
        st.warning("Nenhuma atividade encontrada para análise."); st.stop()
    
    up = st.session_state.get(SK.LAST_UPDATE) or datetime.now(TZ_SP)
    st.sidebar.caption(f"Dados de {up:%d/%m/%Y %H:%M:%S}")

    st.sidebar.header("Filtros")
    hoje = date.today()
    # Intervalo de datas padrão ajustado para garantir que os exemplos apareçam
    d_ini = st.sidebar.date_input("Início", hoje - timedelta(days=40))
    d_fim = st.sidebar.date_input("Fim", hoje + timedelta(days=14), min_value=d_ini)
    if d_ini > d_fim:
        st.sidebar.error("Data de início não pode ser maior que a data de fim."); st.stop()

    df_per = df[df["activity_date"].notna() & df["activity_date"].dt.date.between(d_ini, d_fim)]
    
    pastas_sel = st.sidebar.multiselect("Pastas p/ Análise", sorted(df_per["activity_folder"].dropna().unique()))
    df_ana = df_per if not pastas_sel else df_per[df_per["activity_folder"].isin(pastas_sel)]

    status_sel = st.sidebar.multiselect("Status", sorted(df_ana["activity_status"].dropna().unique()))
    min_sim = st.sidebar.slider("Similaridade mínima (%)", 0, 100, 90, 5) / 100
    only_dup = st.sidebar.checkbox("Mostrar somente duplicatas", True)
    only_multi = st.sidebar.checkbox("Apenas pastas com múltiplos responsáveis")
    users_sel = st.sidebar.multiselect("Usuários", sorted(df_ana["user_profile_name"].dropna().unique()))
    
    sim_map, dup_ids = sim_cache(df_ana, min_sim)

    df_view = df_ana.copy()
    if status_sel: df_view = df_view[df_view["activity_status"].isin(status_sel)]
    if only_dup: df_view = df_view[df_view["activity_id"].isin(dup_ids)]
    if only_multi:
        mult = {p for p, g in df_ana.groupby("activity_folder") if g["user_profile_name"].nunique() > 1}
        df_view = df_view[df_view["activity_folder"].isin(mult)]
    if users_sel: df_view = df_view[df_view["user_profile_name"].isin(users_sel)]
    
    st.title(f"🔎 Análise de Duplicidades ({len(df_view)} atividades exibidas)")
    
    idx_map = df_ana.set_index("activity_id").to_dict("index")
    pastas_ord = sorted(df_view["activity_folder"].dropna().unique())
    
    if not pastas_ord:
        st.info("Nenhum resultado para os filtros selecionados."); st.stop()

    total_paginas = max(1, (len(pastas_ord) + ITENS_POR_PAGINA - 1) // ITENS_POR_PAGINA)
    page_num = st.session_state.get(SK.PAGE, 0)
    page_num = max(0, min(page_num, total_paginas - 1))
    st.session_state[SK.PAGE] = page_num
    
    if total_paginas > 1:
        l, mid, r = st.columns([1, 2, 1])
        if l.button("⬅", disabled=page_num == 0): st.session_state[SK.PAGE] -= 1; st.rerun()
        mid.markdown(f"<p style='text-align:center'>Página {page_num + 1}/{total_paginas}</p>", unsafe_allow_html=True)
        if r.button("➡", disabled=page_num == total_paginas - 1): st.session_state[SK.PAGE] += 1; st.rerun()

    cmp_state = st.session_state.get(SK.COMPARE_STATE)
    pastas_na_pagina = pastas_ord[page_num * ITENS_POR_PAGINA : (page_num + 1) * ITENS_POR_PAGINA]

    for pasta in pastas_na_pagina:
        df_p = df_view[df_view["activity_folder"] == pasta]
        total_na_pasta_analisada = len(df_ana[df_ana["activity_folder"] == pasta])
        exp_title = f"📁 {pasta} ({len(df_p)} de {total_na_pasta_analisada} na análise)"
        
        with st.expander(exp_title, expanded=True):
            for row in df_p.itertuples(index=False):
                act_id = row.activity_id
                c1, c2 = st.columns([.6, .4], gap="large")

                with c1:
                    dt = as_sp(row.activity_date)
                    st.markdown(f"**ID:** {act_id}  •  **Status:** {row.activity_status}  •  **Data:** {dt.strftime('%d/%m/%y %H:%M') if dt else 'N/A'}")
                    st.markdown(f"**Usuário:** {row.user_profile_name}")
                    
                    b1, b2, b3 = st.columns(3)
                    if b1.button("👁 Texto Completo", key=f"full_{act_id}"):
                        st.session_state[SK.FULL_TEXT_DATA] = row._asdict()
                        st.session_state[SK.SHOW_FULL_TEXT_DIALOG] = True
                    link_zflow_v1 = f"https://zflow.zionbyonset.com.br/activity/3/details/{act_id}"
                    link_zflow_v2 = f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={act_id}#/fixcol1"
                    b2.link_button("ZFlow v1", link_zflow_v1); b3.link_button("ZFlow v2", link_zflow_v2)

                with c2:
                    sims = sim_map.get(act_id, [])
                    if sims:
                        st.markdown(f"**Duplicatas Encontradas:** {len(sims)}")
                        for s in sims[:5]:
                            info = idx_map.get(s["id"])
                            if not info: continue
                            d = as_sp(info["activity_date"])
                            d_fmt = d.strftime("%d/%m/%y %H:%M") if d else "N/A"
                            badge_html = (
                                f"<div class='similarity-badge' style='background:{s['cor']};'>"
                                f"<b>ID {s['id']}</b> • {s['ratio']:.0%}<br>"
                                f"{d_fmt} • {info['user_profile_name']}"
                                "</div>")
                            st.markdown(badge_html, unsafe_allow_html=True)
                            if st.button("⚖️ Comparar", key=f"cmp_{act_id}_{s['id']}"):
                                st.session_state[SK.COMPARE_STATE] = CompareState(base_id=act_id, comp_id=s['id'])
                                st.rerun()

                if cmp_state and cmp_state.base_id == act_id:
                    comp_data = idx_map.get(cmp_state.comp_id)
                    if comp_data:
                        sim_info = next((s for s in sim_map.get(act_id, []) if s['id'] == cmp_state.comp_id), None)
                        ratio_str = f"{sim_info['ratio']:.0%}" if sim_info else "N/A"
                        st.markdown("---")
                        
                        st.markdown("""
                        <div style="font-size: 0.85em; margin-bottom: 10px; padding: 5px; background-color: #f0f2f6; border-radius: 5px;">
                            <b>Legenda:</b>
                            <span style="background-color: #ffcdd2; padding: 0 3px; margin: 0 5px;">Texto removido do original</span> |
                            <span style="background-color: #c8e6c9; padding: 0 3px; margin: 0 5px;">Texto adicionado no comparado</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        hA, hB = highlight_diffs(row.Texto, comp_data["Texto"])
                        
                        colA, colB = st.columns(2)
                        with colA:
                            st.markdown(f"**Original: ID {act_id}**"); st.markdown(hA, unsafe_allow_html=True)
                        with colB:
                            st.markdown(f"**Comparado: ID {cmp_state.comp_id} ({ratio_str})**"); st.markdown(hB, unsafe_allow_html=True)
                        
                        if st.button("❌ Fechar Comparação", key=f"cls_{act_id}_{cmp_state.comp_id}"):
                            st.session_state[SK.COMPARE_STATE] = None; st.rerun()
                st.divider()

    if st.session_state[SK.SHOW_FULL_TEXT_DIALOG]: show_full_text_dialog()

def login():
    st.header("Login")
    with st.form("login_form_main"):
        u = st.text_input("Usuário"); p = st.text_input("Senha", type="password")
        if st.form_submit_button("Entrar"):
            creds = st.secrets.get("credentials", {}); users = creds.get("usernames", {})
            if u in users and str(users[u]) == p:
                st.session_state[SK.LOGGED_IN] = True; st.session_state[SK.USERNAME] = u; st.rerun()
            else: st.error("Credenciais inválidas.")

if __name__ == "__main__":
    init_session_state()
    (app() if st.session_state.get(SK.LOGGED_IN) else login())
