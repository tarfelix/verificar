import streamlit as st
import pandas as pd
import re, io, html, os
from datetime import datetime, timedelta, date
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from unidecode import unidecode
from rapidfuzz import fuzz

# ═════════ CONFIG ════════════════════════════════════════════════════════════
SUFFIX_STATE      = "_final_fix"
ITENS_POR_PAGINA  = 20
HIGHLIGHT_COLOR   = "#a8d1ff"

st.set_page_config(layout="wide", page_title="Verificador de Duplicidade")

st.markdown(f"""
<style>
 mark.common {{background:{HIGHLIGHT_COLOR};padding:0 2px;font-weight:bold}}
 pre.highlighted-text {{
   white-space:pre-wrap;word-wrap:break-word;
   font-family:'SFMono-Regular',Consolas,'Liberation Mono',Menlo,Courier,monospace;
   font-size:.9em;padding:10px;border:1px solid #ddd;border-radius:5px;background:#f9f9f9;
 }}
 .similarity-badge {{padding:3px 6px;border-radius:5px;color:black;font-weight:500;display:inline-block}}
</style>""", unsafe_allow_html=True)

# ═════════ FUNÇÕES TEXTO / SIMILARIDADE ═══════
def normalizar_texto(t): ...
def calcular_similaridade(a,b): ...
def cor_sim(r): ...
def highlight_common(t1,t2,min_len: int=3): ...

# (omiti o corpo para economizar espaço – igual à versão anterior)

# ═════════ BANCO  ═════════════════════════════
@st.cache_resource
def db_engine(): ...
@st.cache_data(ttl=3600, hash_funcs={Engine:lambda _:None})
def carregar_dados(eng): ...

# ═════════ CACHE DE SIMILARIDADE ══════════════
def sim_map_cached(df: pd.DataFrame, min_sim: float):
    sig = (tuple(sorted(df["activity_id"])), min_sim)
    key = "simcache"+SUFFIX_STATE
    if key in st.session_state and st.session_state[key]["sig"] == sig:
        c = st.session_state[key]; return c["map"], c["dup"]

    mapa, ids = {}, set()
    barra = st.sidebar.progress(0)
    for idx, (_, g) in enumerate(df.groupby("activity_folder")):
        barra.progress((idx+1)/df.groupby("activity_folder").ngroups)
        acts = g.to_dict("records")
        for i, a in enumerate(acts):
            mapa.setdefault(a["activity_id"], [])
            for b in acts[i+1:]:
                r = calcular_similaridade(a["Texto"], b["Texto"])
                if r >= min_sim:
                    c = cor_sim(r)
                    ids.update([a["activity_id"], b["activity_id"]])
                    mapa[a["activity_id"]].append(dict(id=b["activity_id"],ratio=r,cor=c))
                    mapa.setdefault(b["activity_id"], []).append(dict(id=a["activity_id"],ratio=r,cor=c))
    barra.empty()
    for k in mapa: mapa[k].sort(key=lambda x:x["ratio"], reverse=True)
    st.session_state[key] = {"sig":sig,"map":mapa,"dup":ids}
    return mapa, ids

# ═════════ LINKS ══════════════════════════════
link_z = lambda i: {"antigo":f"https://zflow.zionbyonset.com.br/activity/3/details/{i}",
                    "novo":  f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={i}#/fixcol1"}

# ═════════ ESTADO INICIAL ═════════════════════
for k,v in {
    f"show_text{SUFFIX_STATE}":False,
    f"full_act{SUFFIX_STATE}":None,
    f"cmp{SUFFIX_STATE}":None,
    f"page{SUFFIX_STATE}":0,
    f"last_update{SUFFIX_STATE}":None,
}.items(): st.session_state.setdefault(k,v)

@st.dialog("Texto completo")
def dlg_full():  # igual à versão anterior
    ...

# ═════════ APP PRINCIPAL ══════════════════════
def app():
    st.sidebar.success(f"Logado como **{st.session_state['username']}**")
    if st.sidebar.button("Logout"): st.session_state.clear(); st.rerun()

    eng = db_engine();  # conexão
    if not eng: st.stop()

    # ------ Botão Atualizar ------
    if st.sidebar.button("🔄 Atualizar dados"):
        carregar_dados.clear()                       # limpa cache de dados
        st.session_state.pop("simcache"+SUFFIX_STATE, None)   # limpa cache de similaridade
        st.session_state[f"last_update{SUFFIX_STATE}"] = datetime.now()

    # ------ Carrega dados ------
    df, err = carregar_dados(eng)
    if err: st.exception(err); st.stop()
    if df.empty: st.warning("Sem atividades."); st.stop()

    # Info da última atualização
    up = st.session_state.get(f"last_update{SUFFIX_STATE}")
    if up: st.sidebar.caption(f"Dados atualizados em: {up:%d/%m/%Y %H:%M:%S}")
    else:  st.sidebar.caption("Dados provenientes do cache.")

    # (Restante do app permanece igual ao arquivo anterior)

    # ...  filtros, paginação, comparação embutida etc ...

# ═════════ LOGIN & MAIN ═══════════════════════
def cred_ok(u,p): ...
def login(): ...
if __name__ == "__main__":
    if not st.session_state.get("logged_in"): st.session_state["logged_in"]=False
    if st.session_state["logged_in"]: app()
    else: login()
