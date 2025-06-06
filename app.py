import streamlit as st
import pandas as pd
import re, io, html, os, logging
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo          # ← fuso horário oficial
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from unidecode import unidecode
from rapidfuzz import fuzz

# ═════════ CONFIGURAÇÃO ═══════════════════════════════════════════════════════
SUFFIX_STATE      = "_final_v6"
ITENS_POR_PAGINA  = 20
HIGHLIGHT_COLOR   = "#a8d1ff"
TZ_SP             = ZoneInfo("America/Sao_Paulo")     # Fuso São Paulo

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
</style>
""", unsafe_allow_html=True)

# ═════════ FUNÇÕES TEXTO / SIMILARIDADE ══════════════════════════════════════
def normalizar_texto(txt: str | None) -> str:
    if not txt or not isinstance(txt, str): return ""
    txt = unidecode(txt.lower())
    txt = re.sub(r"[^\w\s]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()

def calcular_similaridade(a: str, b: str) -> float:
    a, b = normalizar_texto(a), normalizar_texto(b)
    if not a or not b: return 0.0
    if abs(len(a) - len(b)) > .3 * max(len(a), len(b)): return 0.0
    return fuzz.token_set_ratio(a, b) / 100

cor_sim = lambda r: "#FF5252" if r >= .9 else "#FFB74D" if r >= .7 else "#FFD54F"

def highlight_common(t1: str, t2: str, min_len: int = 3):
    tokens1 = re.findall(r"\w+", normalizar_texto(t1))
    tokens2 = re.findall(r"\w+", normalizar_texto(t2))
    comunes  = {w for w in tokens1 if w in tokens2 and len(w) >= min_len}

    def wrap(txt):
        out = []
        for part in re.split(r"(\W+)", txt):
            if not part: continue
            if re.match(r"\w+", part) and normalizar_texto(part) in comunes:
                out.append(f"<mark class='common'>{html.escape(part)}</mark>")
            else:
                out.append(html.escape(part))
        return "<pre class='highlighted-text'>" + "".join(out) + "</pre>"
    return wrap(t1), wrap(t2)

# ═════════ BANCO  ════════════════════════════════════════════════════════════
@st.cache_resource
def db_engine() -> Engine | None:
    cfg = st.secrets.get("database", {})
    host = cfg.get("host") or os.getenv("DB_HOST")
    user = cfg.get("user") or os.getenv("DB_USER")
    pw   = cfg.get("password") or os.getenv("DB_PASS")
    db   = cfg.get("name") or os.getenv("DB_NAME")
    if not all([host, user, pw, db]):
        st.error("Credenciais do banco ausentes em st.secrets ou variáveis DB_*"); return None
    try:
        eng = create_engine(f"mysql+mysqlconnector://{user}:{pw}@{host}/{db}",
                            pool_pre_ping=True, pool_recycle=3600)
        with eng.connect(): pass
        return eng
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error("Erro ao conectar no banco."); return None

@st.cache_data(ttl=3600, hash_funcs={Engine:lambda _:None})
def carregar_dados(eng: Engine) -> pd.DataFrame:
    hoje, lim = date.today(), date.today() - timedelta(days=7)
    q_open = text("""SELECT activity_id, activity_folder, user_profile_name,
                            activity_date, activity_status, Texto
                     FROM ViewGrdAtividadesTarcisio
                     WHERE activity_type='Verificar' AND activity_status='Aberta'""")
    q_hist = text("""SELECT activity_id, activity_folder, user_profile_name,
                            activity_date, activity_status, Texto
                     FROM ViewGrdAtividadesTarcisio
                     WHERE activity_type='Verificar' AND DATE(activity_date)>=:lim""")
    try:
        with eng.connect() as c:
            df = pd.concat([pd.read_sql(q_open,c),
                            pd.read_sql(q_hist,c,params={"lim":lim})], ignore_index=True)
        if df.empty: return df
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
        df["Texto"] = df["Texto"].astype(str).fillna("")
        return (df.sort_values(["activity_folder","activity_date","activity_id"],
                               ascending=[True,False,False])
                  .drop_duplicates("activity_id"))
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error("Erro SQL"); return pd.DataFrame()

# ═════════ SIMILARIDADE CACHE ════════════════════════════════════════════════
def sim_map_cached(df: pd.DataFrame, min_sim: float):
    sig = (tuple(sorted(df["activity_id"])), min_sim)
    key = "simcache"+SUFFIX_STATE
    cache = st.session_state.get(key)
    if cache and cache["sig"] == sig: return cache["map"], cache["dup"]

    mapa, dup_ids = {}, set()
    bar = st.sidebar.progress(0, text="Calculando similaridades...")
    groups = list(df.groupby("activity_folder"))
    for i, (_, g) in enumerate(groups, 1):
        bar.progress(i/len(groups))
        acts = g.to_dict("records")
        for idx, a in enumerate(acts):
            mapa.setdefault(a["activity_id"], [])
            for b in acts[idx+1:]:
                r = calcular_similaridade(a["Texto"], b["Texto"])
                if r >= min_sim:
                    c = cor_sim(r)
                    dup_ids.update([a["activity_id"], b["activity_id"]])
                    mapa[a["activity_id"]].append(dict(id=b["activity_id"], ratio=r, cor=c))
                    mapa.setdefault(b["activity_id"], []).append(dict(id=a["activity_id"], ratio=r, cor=c))
    bar.empty()
    for k in mapa: mapa[k].sort(key=lambda x:x["ratio"], reverse=True)
    st.session_state[key] = {"sig":sig,"map":mapa,"dup":dup_ids}
    return mapa, dup_ids

# ═════════ LINKS ZFLOW ═══════════════════════════════════════════════════════
link_z = lambda i: {
    "antigo":f"https://zflow.zionbyonset.com.br/activity/3/details/{i}",
    "novo"  :f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={i}#/fixcol1"
}

# ═════════ ESTADO PADRÃO ═════════════════════════════════════════════════════
for k,v in {
    f"show_text{SUFFIX_STATE}":False,
    f"full_act{SUFFIX_STATE}":None,
    f"cmp{SUFFIX_STATE}":None,
    f"page{SUFFIX_STATE}":0,
    f"last_update{SUFFIX_STATE}":None,
}.items(): st.session_state.setdefault(k,v)

# ═════════ DIALOG TEXTO COMPLETO ═════════════════════════════════════════════
@st.dialog("Texto completo")
def dlg_full():
    d = st.session_state[f"full_act{SUFFIX_STATE}"];  # dict
    if d is None: return
    data = d["activity_date"].astimezone(TZ_SP).strftime("%d/%m/%Y %H:%M") if pd.notna(d["activity_date"]) else "N/A"
    st.markdown(f"### ID {d['activity_id']} – {data}")
    html_pre = f"<pre style='max-height:400px;overflow:auto'>{html.escape(d['Texto'])}</pre>"
    st.markdown(html_pre, unsafe_allow_html=True)
    st.button("Fechar", on_click=lambda: st.session_state.update({f"show_text{SUFFIX_STATE}":False}))

# ═════════ APP PRINCIPAL ═════════════════════════════════════════════════════
def app():
    st.sidebar.success(f"Logado como **{st.session_state['username']}**")
    if st.sidebar.button("Logout"):
        st.session_state.clear(); st.rerun()

    eng = db_engine()
    if not eng: st.stop()

    # Atualizar dados
    if st.sidebar.button("🔄 Atualizar dados"):
        carregar_dados.clear()
        st.session_state.pop("simcache"+SUFFIX_STATE, None)
        st.session_state[f"last_update{SUFFIX_STATE}"] = datetime.now(TZ_SP)

    df = carregar_dados(eng)
    if df.empty: st.warning("Sem atividades."); st.stop()

    up = st.session_state[f"last_update{SUFFIX_STATE}"] or datetime.now(TZ_SP)
    st.sidebar.caption(f"Dados atualizados em: {up:%d/%m/%Y %H:%M:%S}")

    # Período
    hoje = date.today()
    d_ini = st.sidebar.date_input("Início", hoje - timedelta(days=1))
    d_fim = st.sidebar.date_input("Fim",    hoje + timedelta(days=14), min_value=d_ini)
    if d_ini > d_fim:
        st.sidebar.error("Início > fim."); st.stop()

    df_per = df[(df["activity_date"].notna()) & df["activity_date"].dt.date.between(d_ini,d_fim)]
    st.title(f"🔎 Duplicidades ({len(df_per)} atividades)")

    # Filtros
    pastas = sorted(df_per["activity_folder"].dropna().unique())
    pastas_sel_ana = st.sidebar.multiselect("Pastas para Análise", pastas)
    df_ana = df_per if not pastas_sel_ana else df_per[df_per["activity_folder"].isin(pastas_sel_ana)]

    status_disp = sorted(df_ana["activity_status"].dropna().unique())
    status_sel  = st.sidebar.multiselect("Status para Exibição", status_disp)
    min_sim     = st.sidebar.slider("Similaridade mínima (%)",0,100,90,5)/100
    only_dup    = st.sidebar.checkbox("Somente duplicatas", True)
    pastas_multi= {p for p,g in df_ana.groupby("activity_folder") if g["user_profile_name"].nunique()>1}
    only_multi  = st.sidebar.checkbox("Pastas com múltiplos responsáveis")
    users_disp  = sorted(df_ana["user_profile_name"].dropna().unique())
    users_sel   = st.sidebar.multiselect("Usuários", users_disp)

    # Similaridade
    sim_map, ids_dup = sim_map_cached(df_ana, min_sim)

    # Exibição final
    df_view = df_ana.copy()
    if status_sel: df_view = df_view[df_view["activity_status"].isin(status_sel)]
    if only_dup:   df_view = df_view[df_view["activity_id"].isin(ids_dup)]
    if only_multi: df_view = df_view[df_view["activity_folder"].isin(pastas_multi)]
    if users_sel:  df_view = df_view[df_view["user_profile_name"].isin(users_sel)]

    # Índice O(1)
    idx_map = {row.activity_id: row for row in df_ana.itertuples()}

    # Paginação
    pastas_ord = sorted(df_view["activity_folder"].dropna().unique())
    page = st.session_state[f"page{SUFFIX_STATE}"]
    total_pages = max(1,(len(pastas_ord)+ITENS_POR_PAGINA-1)//ITENS_POR_PAGINA)
    page = max(0,min(page,total_pages-1)); st.session_state[f"page{SUFFIX_STATE}"]=page

    if total_pages>1:
        a,b,c = st.columns([1,2,1])
        if a.button("⬅",disabled=page==0): st.session_state[f"page{SUFFIX_STATE}"]-=1; st.rerun()
        b.markdown(f"<p style='text-align:center'>Página {page+1}/{total_pages}</p>",unsafe_allow_html=True)
        if c.button("➡",disabled=page==total_pages-1): st.session_state[f"page{SUFFIX_STATE}"]+=1; st.rerun()

    # Comparação ativa
    cmp_state = st.session_state[f"cmp{SUFFIX_STATE}"]

    # Loop pastas
    start,end = page*ITENS_POR_PAGINA,(page+1)*ITENS_POR_PAGINA
    for pasta in pastas_ord[start:end]:
        df_p = df_view[df_view["activity_folder"] == pasta]
        tot_ana = len(df_ana[df_ana["activity_folder"] == pasta])
        with st.expander(f"📁 {pasta} ({len(df_p)}/{tot_ana})"):
            for row in df_p.itertuples():
                act_id = row.activity_id
                c1,c2 = st.columns([.6,.4], gap="small")
                # ---------- Info ----------
                with c1:
                    data_fmt = row.activity_date.astimezone(TZ_SP).strftime("%d/%m/%Y %H:%M") if pd.notna(row.activity_date) else "N/A"
                    st.markdown(f"**ID** `{act_id}` • {data_fmt} • `{row.activity_status}`")
                    st.markdown(f"**Usuário:** {row.user_profile_name}")
                    st.text_area("Texto", row.Texto, height=100, disabled=True,
                                 key=f"txt_{pasta}_{act_id}_{page}")
                    b1,b2,b3 = st.columns(3)
                    b1.button("👁 Completo",
                              key=f"full_{pasta}_{act_id}_{page}",
                              on_click=lambda r=row: st.session_state.update({
                                  f"full_act{SUFFIX_STATE}": r._asdict(),
                                  f"show_text{SUFFIX_STATE}": True}))
                    links = link_z(act_id)
                    b2.link_button("ZFlow v1", links["antigo"])
                    b3.link_button("ZFlow v2", links["novo"])
                # ---------- Dupes ----------
                with c2:
                    sims = sim_map.get(act_id, [])
                    if sims:
                        st.markdown(f"**Duplicatas:** {len(sims)}")
                        for s in sims:
                            info = idx_map.get(s["id"])
                            if not info: continue
                            badge = (f"<div class='similarity-badge' style='background:{s['cor']};'>"
                                     f"<b>{info.activity_id}</b> • {s['ratio']:.0%}<br>"
                                     f"{info.activity_date.astimezone(TZ_SP).strftime('%d/%m/%y %H:%M') if pd.notna(info.activity_date) else 'N/A'}"
                                     f" • {info.activity_status}<br>{info.user_profile_name}</div>")
                            st.markdown(badge, unsafe_allow_html=True)
                            st.button("⚖ Comparar",
                                      key=f"cmp_{page}_{pasta}_{act_id}_{info.activity_id}",
                                      on_click=lambda a=act_id,b=info.activity_id:
                                          st.session_state.update({f"cmp{SUFFIX_STATE}":{"base_id":a,"comp_id":b}}))
                    elif not only_dup:
                        st.markdown("<span style='color:green;'>Sem duplicatas</span>",unsafe_allow_html=True)

                # ---------- Comparação embutida ----------
                if cmp_state and cmp_state["base_id"] == act_id:
                    comp = idx_map.get(cmp_state["comp_id"])
                    if comp:
                        html_a, html_b = highlight_common(row.Texto, comp.Texto)
                        with st.container():
                            st.markdown("---")
                            st.markdown(f"### Comparação {act_id} × {comp.activity_id}")
                            ca,cb = st.columns(2)
                            ca.markdown(html_a,unsafe_allow_html=True)
                            cb.markdown(html_b,unsafe_allow_html=True)
                            if st.button("❌ Fechar comparação", key=f"cls_{act_id}_{page}"):
                                st.session_state[f"cmp{SUFFIX_STATE}"]=None; st.rerun()

    # Dialog texto completo
    if st.session_state[f"show_text{SUFFIX_STATE}"]: dlg_full()

# ═════════ LOGIN / MAIN ══════════════════════════════════════════════════════
def cred_ok(u,p):
    creds = st.secrets.get("credentials", {}).get("usernames", {})
    return u in creds and str(creds[u]) == p

def login():
    st.header("Login")
    with st.form("login"):
        u = st.text_input("Usuário")
        p = st.text_input("Senha", type="password")
        if st.form_submit_button("Entrar"):
            if cred_ok(u,p):
                st.session_state.update({"logged_in":True,"username":u}); st.rerun()
            else:
                st.error("Credenciais inválidas.")

if __name__ == "__main__":
    if not st.session_state.get("logged_in"): st.session_state["logged_in"]=False
    if st.session_state["logged_in"]:
        app()
    else:
        login()
