# app.py  – versão com filtro “múltiplos responsáveis” de volta
import streamlit as st
import pandas as pd
import re, io, html, os
from datetime import datetime, timedelta, date
from unidecode import unidecode
from rapidfuzz import fuzz
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine

# ──────────────────────────────────────────────────────────────────────────────
SUFFIX_STATE   = "_grifar_v1"
ITENS_POR_PAGINA = 20
st.set_page_config(layout="wide", page_title="Verificador de Duplicidade")

st.markdown("""
<style>
 mark {background:#ffeb3b;padding:0 2px;}
 pre  {white-space:pre-wrap;font-family:'Courier New',monospace;}
</style>
""", unsafe_allow_html=True)

# ───────────── Textos & similaridade ─────────────────────────────────────────
def normalizar_texto(t: str | None) -> str:
    if not t or not isinstance(t, str): return ""
    t = unidecode(t.lower())
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', t)).strip()

def calcular_similaridade(a: str, b: str) -> float:
    a, b = normalizar_texto(a), normalizar_texto(b)
    if not a or not b: return 0.0
    if abs(len(a) - len(b)) > 0.3 * max(len(a), len(b)): return 0.0
    return fuzz.token_set_ratio(a, b) / 100

def cor_sim(r: float) -> str:
    return "#FF5252" if r >= .9 else "#FFB74D" if r >= .7 else "#FFD54F"

def highlight_common(t1: str, t2: str, min_len: int = 3):
    tok1 = re.findall(r'\w+', unidecode(t1.lower()))
    tok2 = re.findall(r'\w+', unidecode(t2.lower()))
    comuns = {w for w in tok1 if w in tok2 and len(w) >= min_len}

    def wrap(txt):
        def repl(m):
            p = m.group(0)
            return f"<mark>{html.escape(p)}</mark>" if unidecode(p.lower()) in comuns else html.escape(p)
        return "<pre>" + re.sub(r'\w+', repl, txt) + "</pre>"

    return wrap(t1), wrap(t2)

# ───────────── Banco de dados ────────────────────────────────────────────────
@st.cache_resource
def db_engine() -> Engine | None:
    h = st.secrets.get("database", {}).get("host") or os.getenv("DB_HOST")
    u = st.secrets.get("database", {}).get("user") or os.getenv("DB_USER")
    p = st.secrets.get("database", {}).get("password") or os.getenv("DB_PASS")
    n = st.secrets.get("database", {}).get("name") or os.getenv("DB_NAME")
    if not all([h, u, p, n]):
        st.error("Credenciais do banco ausentes em `st.secrets` ou variáveis DB_*")
        return None
    uri = f"mysql+mysqlconnector://{u}:{p}@{h}/{n}"
    try:
        eng = create_engine(uri, pool_pre_ping=True, pool_recycle=3600)
        with eng.connect(): pass
        return eng
    except exc.SQLAlchemyError as e:
        st.exception(e); return None

@st.cache_data(ttl=3600, hash_funcs={Engine: lambda _: None})
def carregar(eng: Engine):
    hoje = date.today()
    limite = hoje - timedelta(days=7)
    q_abertas = text("""
        SELECT activity_id, activity_folder, user_profile_name, activity_date,
               activity_status, Texto
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type='Verificar' AND activity_status='Aberta'
    """)
    q_hist = text("""
        SELECT activity_id, activity_folder, user_profile_name, activity_date,
               activity_status, Texto
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type='Verificar' AND DATE(activity_date) >= :lim
    """)
    try:
        with eng.connect() as c:
            df1 = pd.read_sql(q_abertas, c)
            df2 = pd.read_sql(q_hist, c, params={"lim": limite})
        df = pd.concat([df1, df2], ignore_index=True)
        if df.empty: return df, None
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
        df["Texto"] = df["Texto"].astype(str).fillna("")
        df = df.sort_values(["activity_folder", "activity_date", "activity_id"],
                            ascending=[True, False, False]).drop_duplicates("activity_id")
        return df, None
    except exc.SQLAlchemyError as e:
        return None, e

# ───────────── Similaridade (cache em session_state) ─────────────────────────
def sim_map_df(df: pd.DataFrame, min_sim: float):
    sig = (tuple(sorted(df["activity_id"])), min_sim)
    key = "simcache" + SUFFIX_STATE
    if key in st.session_state and st.session_state[key]["sig"] == sig:
        c = st.session_state[key]
        return c["map"], c["dup"]

    dup_map, ids_dup = {}, set()
    for _, g in df.groupby("activity_folder"):
        acts = g.to_dict("records")
        for i, a in enumerate(acts):
            dup_map.setdefault(a["activity_id"], [])
            for b in acts[i + 1:]:
                r = calcular_similaridade(a["Texto"], b["Texto"])
                if r >= min_sim:
                    c = cor_sim(r)
                    ids_dup.update([a["activity_id"], b["activity_id"]])
                    dup_map[a["activity_id"]].append(dict(id=b["activity_id"], ratio=r, cor=c))
                    dup_map.setdefault(b["activity_id"], []).append(dict(id=a["activity_id"], ratio=r, cor=c))
    for k in dup_map:
        dup_map[k].sort(key=lambda x: x["ratio"], reverse=True)
    st.session_state[key] = {"sig": sig, "map": dup_map, "dup": ids_dup}
    return dup_map, ids_dup

# ───────────── Links Zflow ───────────────────────────────────────────────────
link_z = lambda i: dict(
    antigo=f"https://zflow.zionbyonset.com.br/activity/3/details/{i}",
    novo  =f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={i}#/fixcol1"
)

# ───────────── Estado inicial ────────────────────────────────────────────────
defaults = {
    f"show_text{SUFFIX_STATE}": False,
    f"full_act{SUFFIX_STATE}":  None,
    f"cmp{SUFFIX_STATE}":       None,
    f"page{SUFFIX_STATE}":      0,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ───────────── Diálogo texto completo ────────────────────────────────────────
@st.dialog("Texto completo")
def dlg_full():
    d = st.session_state[f"full_act{SUFFIX_STATE}"]
    if d is None: return
    data = d["activity_date"].strftime("%d/%m/%Y %H:%M") if pd.notna(d["activity_date"]) else "N/A"
    st.markdown(f"### ID {d['activity_id']} • {data}")
    st.text_area("Texto", d["Texto"], height=400, disabled=True)
    if st.button("Fechar"):
        st.session_state[f"show_text{SUFFIX_STATE}"] = False
        st.rerun()

# ───────────── APP principal ────────────────────────────────────────────────
def app():
    st.sidebar.success(f"Logado como **{st.session_state['username']}**")
    if st.sidebar.button("Logout"):
        st.session_state.clear(); st.rerun()

    eng = db_engine()
    if not eng: st.stop()

    if st.sidebar.button("🔄 Atualizar dados"):
        carregar.clear(); st.session_state[f"cmp{SUFFIX_STATE}"] = None

    df, err = carregar(eng)
    if err: st.exception(err); st.stop()
    if df.empty: st.warning("Sem atividades."); st.stop()

    # ─── Filtros de período ────────────────────────────────────────────────
    hoje = date.today()
    d_ini_def, d_fim_def = hoje - timedelta(days=1), hoje + timedelta(days=14)
    st.sidebar.header("Período")
    d_ini = st.sidebar.date_input("Início", d_ini_def)
    d_fim = st.sidebar.date_input("Fim",    d_fim_def, min_value=d_ini)
    if d_ini > d_fim:
        st.sidebar.error("Início > fim."); st.stop()
    df_per = df[(df["activity_date"].notna()) &
                df["activity_date"].dt.date.between(d_ini, d_fim)]
    st.success(f"{len(df_per)} atividades no período selecionado.")

    # ─── Filtros de análise ────────────────────────────────────────────────
    st.sidebar.header("Filtros de análise")
    pastas   = sorted(df_per["activity_folder"].dropna().unique().tolist())
    pastas_sel  = st.sidebar.multiselect("Pastas", pastas)
    status   = sorted(df_per["activity_status"].dropna().unique().tolist())
    status_sel  = st.sidebar.multiselect("Status", status)

    df_a = df_per.copy()
    if pastas_sel: df_a = df_a[df_a["activity_folder"].isin(pastas_sel)]
    if status_sel: df_a = df_a[df_a["activity_status"].isin(status_sel)]

    # ─── Filtros de exibição ───────────────────────────────────────────────
    st.sidebar.header("Exibição")
    min_sim = st.sidebar.slider("Similaridade mínima (%)", 0, 100, 70, 5) / 100
    only_dup = st.sidebar.checkbox("Somente duplicatas", True)

    # >> Filtro “múltiplos responsáveis”
    pastas_multi = {
        p for p, g in df_a[df_a["activity_folder"].notna()].groupby("activity_folder")
        if g["user_profile_name"].nunique() > 1
    }
    only_multi = st.sidebar.checkbox("Pastas com múltiplos responsáveis", False)

    # ─── Similaridades (cache) ─────────────────────────────────────────────
    dup_map, ids_dup = sim_map_df(df_a, min_sim)

    df_v = df_a.copy()
    if only_dup:   df_v = df_v[df_v["activity_id"].isin(ids_dup)]
    if only_multi: df_v = df_v[df_v["activity_folder"].isin(pastas_multi)]

    # ─── Comparação ativa ─────────────────────────────────────────────────
    cmp = st.session_state[f"cmp{SUFFIX_STATE}"]
    if cmp:
        a, b = cmp["base"], cmp["comp"]
        st.subheader(f"Comparação {a['activity_id']} × {b['activity_id']}")
        c1, c2 = st.columns(2)
        h1, h2 = highlight_common(a["Texto"], b["Texto"])
        c1.markdown(h1, unsafe_allow_html=True)
        c2.markdown(h2, unsafe_allow_html=True)
        if st.button("Fechar comparação"):
            st.session_state[f"cmp{SUFFIX_STATE}"] = None; st.rerun()
        st.markdown("---")

    # ─── Paginação ────────────────────────────────────────────────────────
    pastas_ord = sorted(df_v["activity_folder"].dropna().unique().tolist())
    page   = st.session_state[f"page{SUFFIX_STATE}"]
    tot_pg = max(1, (len(pastas_ord) + ITENS_POR_PAGINA - 1) // ITENS_POR_PAGINA)
    page   = max(0, min(page, tot_pg - 1))
    st.session_state[f"page{SUFFIX_STATE}"] = page

    if tot_pg > 1:
        b1, _, b2 = st.columns([1, 2, 1])
        if b1.button("⬅", disabled=page == 0):
            st.session_state[f"page{SUFFIX_STATE}"] -= 1; st.rerun()
        st.markdown(f"<p style='text-align:center;'>Página {page+1}/{tot_pg}</p>",
                    unsafe_allow_html=True)
        if b2.button("➡", disabled=page == tot_pg - 1):
            st.session_state[f"page{SUFFIX_STATE}"] += 1; st.rerun()

    ini, fim = page * ITENS_POR_PAGINA, (page + 1) * ITENS_POR_PAGINA
    for pasta in pastas_ord[ini:fim]:
        df_p = df_v[df_v["activity_folder"] == pasta]
        with st.expander(f"📁 {pasta} ({len(df_p)})", expanded=len(df_p) < 10):
            for _, r in df_p.iterrows():
                c1, c2 = st.columns([.6, .4])
                with c1:
                    data = r["activity_date"].strftime("%d/%m/%Y %H:%M") if pd.notna(r["activity_date"]) else "N/A"
                    st.markdown(f"**ID** `{r['activity_id']}` • {data} • `{r['activity_status']}`")
                    st.markdown(f"**Usuário:** {r['user_profile_name']}")
                    st.text_area("Texto", r["Texto"], height=100, disabled=True,
                                 key=f"txt_{r['activity_id']}")
                    if st.button("👁 Completo", key=f"ver_{r['activity_id']}",
                                 on_click=lambda act=r: st.session_state.update(
                                     {f"full_act{SUFFIX_STATE}": act,
                                      f"show_text{SUFFIX_STATE}": True})): pass
                    links = link_z(r["activity_id"])
                    st.link_button("ZFlow v1", links["antigo"])
                    st.link_button("ZFlow v2", links["novo"])
                with c2:
                    sims = dup_map.get(r["activity_id"], [])
                    if sims:
                        st.markdown(f"**Duplicatas:** {len(sims)}")
                        for s in sims:
                            inf = df[df["activity_id"] == s["id"]].iloc[0]
                            d = inf["activity_date"].strftime("%d/%m/%y %H:%M") if pd.notna(inf["activity_date"]) else "N/A"
                            st.markdown(
                                f"<div style='background:{s['cor']};padding:3px;border-radius:5px;'>"
                                f"<b>{inf['activity_id']}</b> • {s['ratio']:.0%}<br>"
                                f"{d} • {inf['activity_status']}<br>{inf['user_profile_name']}"
                                "</div>", unsafe_allow_html=True)
                            st.button("⚖ Comparar",
                                key=f"cmp_{r['activity_id']}_{inf['activity_id']}",
                                on_click=lambda a=r, b=inf: st.session_state.update(
                                    {f"cmp{SUFFIX_STATE}": {"base": a, "comp": b}}))

    if st.session_state[f"show_text{SUFFIX_STATE}"]: dlg_full()

# ───────────── Login ─────────────────────────────────────────────────────────
def cred(u, p):
    return u in st.secrets.get("credentials", {}).get("usernames", {}) and \
           str(st.secrets["credentials"]["usernames"][u]) == p

def login():
    st.header("Login")
    with st.form("log"):
        u = st.text_input("Usuário")
        p = st.text_input("Senha", type="password")
        if st.form_submit_button("Entrar"):
            if cred(u, p):
                st.session_state.update({"logged_in": True, "username": u}); st.rerun()
            else:
                st.error("Credenciais inválidas.")

# ───────────── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not st.session_state.get("logged_in"): st.session_state["logged_in"] = False
    if st.session_state["logged_in"]:
        app()
    else:
        login()
