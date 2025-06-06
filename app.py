import streamlit as st
import pandas as pd
import re, io, html, os
from datetime import datetime, timedelta, date
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from unidecode import unidecode
from rapidfuzz import fuzz

# ═════════ CONFIGURAÇÃO ═══════════════════════════════════════════════════════
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

# ═════════ FUNÇÕES DE TEXTOS / SIMILARIDADE ═══════════════════════════════════
def normalizar_texto(t: str | None) -> str:
    if not t or not isinstance(t, str): return ""
    t = unidecode(t.lower())
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def calcular_similaridade(a: str, b: str) -> float:
    a, b = normalizar_texto(a), normalizar_texto(b)
    if not a or not b: return 0.0
    if abs(len(a) - len(b)) > 0.5 * max(len(a), len(b)): return 0.0
    return fuzz.token_set_ratio(a, b) / 100

def cor_sim(r: float) -> str:
    return "#FF5252" if r >= .9 else "#FFB74D" if r >= .7 else "#FFD54F"

def highlight_common(t1: str, t2: str, min_len: int = 3):
    tok1 = re.findall(r"\w+", normalizar_texto(t1))
    tok2 = re.findall(r"\w+", normalizar_texto(t2))
    comuns = {w for w in tok1 if w in tok2 and len(w) >= min_len}

    def wrap(txt):
        parts = re.split(r"(\W+)", txt)
        out   = []
        for p in parts:
            if not p: continue
            if re.match(r"\w+", p) and normalizar_texto(p) in comuns:
                out.append(f"<mark class='common'>{html.escape(p)}</mark>")
            else:
                out.append(html.escape(p))
        return "<pre class='highlighted-text'>" + "".join(out) + "</pre>"

    return wrap(t1), wrap(t2)

# ═════════ BANCO DE DADOS ════════════════════════════════════════════════════
@st.cache_resource
def db_engine() -> Engine | None:
    h = st.secrets.get("database", {}).get("host") or os.getenv("DB_HOST")
    u = st.secrets.get("database", {}).get("user") or os.getenv("DB_USER")
    p = st.secrets.get("database", {}).get("password") or os.getenv("DB_PASS")
    n = st.secrets.get("database", {}).get("name") or os.getenv("DB_NAME")
    if not all([h, u, p, n]):
        st.error("Credenciais do banco ausentes.")
        return None
    try:
        eng = create_engine(f"mysql+mysqlconnector://{u}:{p}@{h}/{n}",
                            pool_pre_ping=True, pool_recycle=3600)
        with eng.connect(): pass
        return eng
    except exc.SQLAlchemyError as e:
        st.exception(e); return None

@st.cache_data(ttl=3600, hash_funcs={Engine: lambda _: None})
def carregar_dados(eng: Engine):
    hoje, lim = date.today(), date.today() - timedelta(days=7)
    q1 = text("""
        SELECT activity_id,activity_folder,user_profile_name,activity_date,
               activity_status,Texto
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type='Verificar' AND activity_status='Aberta'""")
    q2 = text("""
        SELECT activity_id,activity_folder,user_profile_name,activity_date,
               activity_status,Texto
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type='Verificar' AND DATE(activity_date)>=:lim""")
    try:
        with eng.connect() as c:
            df1 = pd.read_sql(q1, c)
            df2 = pd.read_sql(q2, c, params={"lim": lim})
        df = pd.concat([df1, df2], ignore_index=True)
        if df.empty:
            cols = ["activity_id","activity_folder","user_profile_name",
                    "activity_date","activity_status","Texto"]
            return pd.DataFrame(columns=cols), None
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
        df["Texto"] = df["Texto"].astype(str).fillna("")
        df = (df.sort_values(["activity_folder","activity_date","activity_id"],
                             ascending=[True,False,False])
                .drop_duplicates("activity_id"))
        return df, None
    except exc.SQLAlchemyError as e:
        return None, e

# ═════════ SIMILARIDADE (CACHE) ══════════════════════════════════════════════
def sim_map_cached(df: pd.DataFrame, min_sim: float):
    sig = (tuple(sorted(df["activity_id"])), min_sim)
    key = "simcache" + SUFFIX_STATE
    if key in st.session_state and st.session_state[key]["sig"] == sig:
        c = st.session_state[key]; return c["map"], c["dup"]

    mapa, ids_dup = {}, set()
    barra = st.sidebar.progress(0)
    grupos = df.groupby("activity_folder")
    for idx, (_, g) in enumerate(grupos):
        barra.progress((idx+1)/grupos.ngroups)
        acts = g.to_dict("records")
        for i, a in enumerate(acts):
            mapa.setdefault(a["activity_id"], [])
            for b in acts[i+1:]:
                r = calcular_similaridade(a["Texto"], b["Texto"])
                if r >= min_sim:
                    c = cor_sim(r)
                    ids_dup.update([a["activity_id"], b["activity_id"]])
                    mapa[a["activity_id"]].append(dict(id=b["activity_id"], ratio=r, cor=c))
                    mapa.setdefault(b["activity_id"], []).append(dict(id=a["activity_id"], ratio=r, cor=c))
    barra.empty()
    for k in mapa: mapa[k].sort(key=lambda x: x["ratio"], reverse=True)
    st.session_state[key] = {"sig": sig, "map": mapa, "dup": ids_dup}
    return mapa, ids_dup

# ═════════ LINKS ZFLOW ═══════════════════════════════════════════════════════
link_z = lambda i: dict(
    antigo=f"https://zflow.zionbyonset.com.br/activity/3/details/{i}",
    novo  =f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={i}#/fixcol1"
)

# ═════════ ESTADO INICIAL ════════════════════════════════════════════════════
for k, v in {
    f"show_text{SUFFIX_STATE}": False,
    f"full_act{SUFFIX_STATE}":  None,
    f"cmp{SUFFIX_STATE}":       None,  # {"base_id": .., "comp_id": ..}
    f"page{SUFFIX_STATE}":      0,
}.items():
    st.session_state.setdefault(k, v)

# ═════════ DIALOG TEXTO COMPLETO ═════════════════════════════════════════════
@st.dialog("Texto completo")
def dlg_full():
    d = st.session_state[f"full_act{SUFFIX_STATE}"]
    if d is None: return
    data = d["activity_date"].strftime("%d/%m/%Y %H:%M") if pd.notna(d["activity_date"]) else "N/A"
    st.markdown(f"### ID {d['activity_id']} – {data}")
    st.text_area("Texto", d["Texto"], height=400, disabled=True,
                 key=f"dlg_txt_{d['activity_id']}{SUFFIX_STATE}")
    if st.button("Fechar"): st.session_state[f"show_text{SUFFIX_STATE}"] = False

# ═════════ APP PRINCIPAL ═════════════════════════════════════════════════════
def app():
    st.sidebar.success(f"Logado como **{st.session_state['username']}**")
    if st.sidebar.button("Logout"):
        st.session_state.clear(); st.rerun()

    eng = db_engine()
    if not eng: st.stop()

    if st.sidebar.button("🔄 Atualizar dados"):
        carregar_dados.clear(); sim_map_cached.clear()

    df, err = carregar_dados(eng)
    if err: st.exception(err); st.stop()
    if df.empty: st.warning("Sem atividades."); st.stop()

    # -------- Período --------
    hoje = date.today()
    d_ini = st.sidebar.date_input("Início", hoje - timedelta(days=1))
    d_fim = st.sidebar.date_input("Fim",   hoje + timedelta(days=14), min_value=d_ini)
    if d_ini > d_fim: st.sidebar.error("Início > fim."); st.stop()

    df_per = df[(df["activity_date"].notna()) &
                df["activity_date"].dt.date.between(d_ini, d_fim)]
    st.title(f"🔎 Duplicidades ({len(df_per)} atividades)")

    # -------- Filtros --------
    pastas = sorted(df_per["activity_folder"].dropna().unique().tolist())
    ps = st.sidebar.multiselect("Pastas", pastas)
    status = sorted(df_per["activity_status"].dropna().unique().tolist())
    ss = st.sidebar.multiselect("Status", status)

    df_a = df_per.copy()
    if ps: df_a = df_a[df_a["activity_folder"].isin(ps)]
    if ss: df_a = df_a[df_a["activity_status"].isin(ss)]

    min_sim = st.sidebar.slider("Similaridade mínima (%)", 0, 100, 70, 5) / 100
    only_dup = st.sidebar.checkbox("Somente duplicatas", True)
    pastas_multi = {p for p, g in df_a.groupby("activity_folder")
                    if g["user_profile_name"].nunique() > 1}
    only_multi = st.sidebar.checkbox("Pastas com múltiplos responsáveis")

    users = sorted(df_a["user_profile_name"].dropna().unique())
    us = st.sidebar.multiselect("Usuários", users)

    # -------- Similaridade --------
    sim_map, ids_dup = sim_map_cached(df_a, min_sim)

    df_v = df_a.copy()
    if only_dup:   df_v = df_v[df_v["activity_id"].isin(ids_dup)]
    if only_multi: df_v = df_v[df_v["activity_folder"].isin(pastas_multi)]
    if us:         df_v = df_v[df_v["user_profile_name"].isin(us)]

    # -------- Paginação --------
    pastas_ord = sorted(df_v["activity_folder"].dropna().unique().tolist())
    page = st.session_state[f"page{SUFFIX_STATE}"]
    tot  = max(1, (len(pastas_ord) + ITENS_POR_PAGINA - 1) // ITENS_POR_PAGINA)
    page = max(0, min(page, tot-1)); st.session_state[f"page{SUFFIX_STATE}"] = page

    if tot > 1:
        b1, _, b2 = st.columns([1,2,1])
        if b1.button("⬅", disabled=page==0):
            st.session_state[f"page{SUFFIX_STATE}"] -= 1; st.rerun()
        st.markdown(f"<p style='text-align:center;'>Página {page+1}/{tot}</p>", unsafe_allow_html=True)
        if b2.button("➡", disabled=page==tot-1):
            st.session_state[f"page{SUFFIX_STATE}"] += 1; st.rerun()

    # -------- Loop por pasta --------
    ini, fim = page*ITENS_POR_PAGINA, (page+1)*ITENS_POR_PAGINA
    cmp_state = st.session_state[f"cmp{SUFFIX_STATE}"]

    for pasta in pastas_ord[ini:fim]:
        df_p = df_v[df_v["activity_folder"] == pasta]
        analisadas = len(df_a[df_a["activity_folder"] == pasta])
        with st.expander(f"📁 {pasta} ({len(df_p)}/{analisadas})", expanded=False):
            for _, r in df_p.iterrows():
                act_id = int(r["activity_id"])
                c1, c2 = st.columns([.6, .4])
                # --- Info ---
                with c1:
                    d = r["activity_date"].strftime("%d/%m/%Y %H:%M") if pd.notna(r["activity_date"]) else "N/A"
                    st.markdown(f"**ID** `{act_id}` • {d} • `{r['activity_status']}`")
                    st.markdown(f"**Usuário:** {r['user_profile_name']}")
                    st.text_area("Texto", r["Texto"], height=100, disabled=True,
                                 key=f"txt_{pasta}_{act_id}")
                    st.button("👁 Completo",
                              key=f"full_{act_id}",
                              on_click=lambda act=r: st.session_state.update({
                                  f"full_act{SUFFIX_STATE}": act,
                                  f"show_text{SUFFIX_STATE}": True}))
                    l = link_z(act_id)
                    st.link_button("ZFlow v1", l["antigo"])
                    st.link_button("ZFlow v2", l["novo"])
                # --- Duplicatas ---
                with c2:
                    sims = sim_map.get(act_id, [])
                    if sims:
                        st.markdown(f"**Duplicatas:** {len(sims)}")
                        for s in sims:
                            info = df[df["activity_id"] == s["id"]].iloc[0]
                            badge = (f"<div class='similarity-badge' style='background:{s['cor']};'>"
                                     f"<b>{info['activity_id']}</b> • {s['ratio']:.0%}<br>"
                                     f"{info['activity_date'].strftime('%d/%m/%y %H:%M') if pd.notna(info['activity_date']) else 'N/A'}"
                                     f" • {info['activity_status']}<br>{info['user_profile_name']}</div>")
                            st.markdown(badge, unsafe_allow_html=True)
                            st.button("⚖ Comparar",
                                      key=f"cmp_{act_id}_{info['activity_id']}",
                                      on_click=lambda a=act_id, b=int(info["activity_id"]):
                                          st.session_state.update({f"cmp{SUFFIX_STATE}": {"base_id": a, "comp_id": b}}))
                    elif not only_dup:
                        st.markdown("<span style='color:green;'>Sem duplicatas</span>", unsafe_allow_html=True)

                # --- Comparação embutida ---
                if cmp_state and cmp_state["base_id"] == act_id:
                    comp_id = cmp_state["comp_id"]
                    texto_comp = df[df["activity_id"] == comp_id].iloc[0]["Texto"]
                    html_a, html_b = highlight_common(r["Texto"], texto_comp)
                    with st.container():
                        st.markdown("---")
                        st.markdown(f"### Comparação {act_id} × {comp_id}")
                        col_a, col_b = st.columns(2)
                        col_a.markdown(html_a, unsafe_allow_html=True)
                        col_b.markdown(html_b, unsafe_allow_html=True)
                        if st.button("❌ Fechar comparação", key=f"close_cmp_{act_id}"):
                            st.session_state[f"cmp{SUFFIX_STATE}"] = None
                            st.rerun()

    if st.session_state[f"show_text{SUFFIX_STATE}"]:
        dlg_full()

# ═════════ LOGIN ═════════════════════════════════════════════════════════════
def cred_ok(u, p):
    d = st.secrets.get("credentials", {}).get("usernames", {})
    return u in d and str(d[u]) == p

def login():
    st.header("Login")
    with st.form("log"):
        u = st.text_input("Usuário")
        p = st.text_input("Senha", type="password")
        if st.form_submit_button("Entrar"):
            if cred_ok(u, p):
                st.session_state.update({"logged_in": True, "username": u}); st.rerun()
            else:
                st.error("Credenciais inválidas.")

# ═════════ MAIN ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if not st.session_state.get("logged_in"):
        st.session_state["logged_in"] = False
    if st.session_state["logged_in"]:
        app()
    else:
        login()
