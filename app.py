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

# ═════════ FUNÇÕES DE TEXTO/SIMILARIDADE ═════════════════════════════════════
def normalizar_texto(t: str | None) -> str:
    if not t or not isinstance(t, str):
        return ""
    t = unidecode(t.lower())
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def calcular_similaridade(a: str, b: str) -> float:
    a, b = normalizar_texto(a), normalizar_texto(b)
    if not a or not b:
        return 0.0
    if abs(len(a) - len(b)) > 0.5 * max(len(a), len(b)):
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100

def cor_sim(r: float) -> str:
    return "#FF5252" if r >= .9 else "#FFB74D" if r >= .7 else "#FFD54F"

def highlight_common(t1: str, t2: str, min_len: int = 3):
    tok1 = re.findall(r"\w+", normalizar_texto(t1))
    tok2 = re.findall(r"\w+", normalizar_texto(t2))
    comuns = {w for w in tok1 if w in tok2 and len(w) >= min_len}

    def wrap(txt):
        parts, out = re.split(r"(\W+)", txt), []
        for p in parts:
            if not p:
                continue
            if re.match(r"\w+", p) and normalizar_texto(p) in comuns:
                out.append(f"<mark class='common'>{html.escape(p)}</mark>")
            else:
                out.append(html.escape(p))
        return "<pre class='highlighted-text'>" + "".join(out) + "</pre>"

    return wrap(t1), wrap(t2)

# ═════════ BANCO ═════════════════════════════════════════════════════════════
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
    q1 = text("""SELECT activity_id,activity_folder,user_profile_name,activity_date,
                       activity_status,Texto
                FROM ViewGrdAtividadesTarcisio
                WHERE activity_type='Verificar' AND activity_status='Aberta'""")
    q2 = text("""SELECT activity_id,activity_folder,user_profile_name,activity_date,
                       activity_status,Texto
                FROM ViewGrdAtividadesTarcisio
                WHERE activity_type='Verificar' AND DATE(activity_date)>=:lim""")

    try:
        with eng.connect() as c:
            df1 = pd.read_sql(q1, c)
            df2 = pd.read_sql(q2, c, params={"lim": lim})
        df = pd.concat([df1, df2], ignore_index=True)
        if df.empty:
            cols = ["activity_id", "activity_folder", "user_profile_name",
                    "activity_date", "activity_status", "Texto"]
            return pd.DataFrame(columns=cols), None
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
        df["Texto"] = df["Texto"].astype(str).fillna("")
        df = (df.sort_values(["activity_folder", "activity_date", "activity_id"],
                             ascending=[True, False, False])
                .drop_duplicates("activity_id"))
        return df, None
    except exc.SQLAlchemyError as e:
        return None, e

# ═════════ SIMILARIDADE CACHÊ ════════════════════════════════════════════════
def sim_map_cached(df: pd.DataFrame, min_sim: float):
    sig = (tuple(sorted(df["activity_id"])), min_sim)
    key = "simcache" + SUFFIX_STATE
    if key in st.session_state and st.session_state[key]["sig"] == sig:
        c = st.session_state[key]; return c["map"], c["dup"]

    m, ids = {}, set()
    ph = st.sidebar.empty(); bar = st.sidebar.progress(0)
    grupos = df.groupby("activity_folder")
    for i, (_, g) in enumerate(grupos):
        bar.progress((i + 1) / grupos.ngroups)
        acts = g.to_dict("records")
        for j, a in enumerate(acts):
            m.setdefault(a["activity_id"], [])
            for b in acts[j + 1:]:
                r = calcular_similaridade(a["Texto"], b["Texto"])
                if r >= min_sim:
                    c = cor_sim(r)
                    ids.update([a["activity_id"], b["activity_id"]])
                    m[a["activity_id"]].append(dict(id=b["activity_id"], ratio=r, cor=c))
                    m.setdefault(b["activity_id"], []).append(dict(id=a["activity_id"], ratio=r, cor=c))
    bar.empty(); ph.empty()
    for k in m: m[k].sort(key=lambda x: x["ratio"], reverse=True)
    st.session_state[key] = {"sig": sig, "map": m, "dup": ids}
    return m, ids

# ═════════ HELPERS UI ════════════════════════════════════════════════════════
link_z = lambda i: dict(antigo=f"https://zflow.zionbyonset.com.br/activity/3/details/{i}",
                        novo  =f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={i}#/fixcol1")

# estado inicial
for k, v in {
    f"show_text{SUFFIX_STATE}": False,
    f"full_act{SUFFIX_STATE}":  None,
    f"cmp{SUFFIX_STATE}":       None,
    f"page{SUFFIX_STATE}":      0,
    f"last_update{SUFFIX_STATE}": None,
}.items():
    st.session_state.setdefault(k, v)

@st.dialog("Texto completo")
def dlg_full():
    d = st.session_state[f"full_act{SUFFIX_STATE}"];  # dict ou None
    if d is None: return
    data = d["activity_date"].strftime("%d/%m/%Y %H:%M") if pd.notna(d["activity_date"]) else "N/A"
    st.markdown(f"### ID {d['activity_id']} • {data}")
    st.text_area("Texto", d["Texto"], height=400, disabled=True,
                 key=f"dlg_txt_{d['activity_id']}{SUFFIX_STATE}")
    if st.button("Fechar"):
        st.session_state[f"show_text{SUFFIX_STATE}"] = False

# ═════════ APP PRINCIPAL ═════════════════════════════════════════════════════
def app():
    # -------- Header / Logout ----------
    st.sidebar.success(f"Logado como **{st.session_state['username']}**")
    if st.sidebar.button("Logout"):
        st.session_state.clear(); st.rerun()

    eng = db_engine()
    if not eng: st.stop()

    if st.sidebar.button("🔄 Atualizar dados"):
        carregar_dados.clear(); sim_map_cached.clear()
        st.session_state[f"last_update{SUFFIX_STATE}"] = datetime.now()

    df, err = carregar_dados(eng)
    if err:  st.exception(err); st.stop()
    if df.empty: st.warning("Sem atividades."); st.stop()

    up = st.session_state.get(f"last_update{SUFFIX_STATE}")
    if up: st.sidebar.caption(f"Dados atualizados em {up:%d/%m/%Y %H:%M:%S}")

    # -------- Período ----------
    hoje = date.today()
    ini_def, fim_def = hoje - timedelta(days=1), hoje + timedelta(days=14)
    st.sidebar.header("Período")
    d_ini = st.sidebar.date_input("Início", ini_def)
    d_fim = st.sidebar.date_input("Fim", fim_def, min_value=d_ini)
    if d_ini > d_fim:
        st.sidebar.error("Início > fim."); st.stop()

    df_per = df[(df["activity_date"].notna()) &
                df["activity_date"].dt.date.between(d_ini, d_fim)]
    st.title(f"🔎 Duplicidades ({len(df_per)} atividades no período)")

    # -------- Filtros ----------
    st.sidebar.header("Filtros")
    pastas = sorted(df_per["activity_folder"].dropna().unique().tolist())
    pastas_sel  = st.sidebar.multiselect("Pastas", pastas)
    status = sorted(df_per["activity_status"].dropna().unique().tolist())
    status_sel  = st.sidebar.multiselect("Status", status)

    df_a = df_per.copy()
    if pastas_sel:  df_a = df_a[df_a["activity_folder"].isin(pastas_sel)]
    if status_sel:  df_a = df_a[df_a["activity_status"].isin(status_sel)]

    st.sidebar.header("Exibição")
    min_sim = st.sidebar.slider("Similaridade mínima (%)", 0, 100, 70, 5) / 100
    only_dup = st.sidebar.checkbox("Somente duplicatas", True)
    pastas_multi = {p for p, g in df_a.groupby("activity_folder")
                    if g["user_profile_name"].nunique() > 1}
    only_multi = st.sidebar.checkbox("Pastas com múltiplos responsáveis")

    users = sorted(df_a["user_profile_name"].dropna().unique())
    users_sel = st.sidebar.multiselect("Usuários", users)

    # -------- Similaridade ----------
    sim_map, ids_dup = sim_map_cached(df_a, min_sim)

    df_v = df_a.copy()
    if only_dup:   df_v = df_v[df_v["activity_id"].isin(ids_dup)]
    if only_multi: df_v = df_v[df_v["activity_folder"].isin(pastas_multi)]
    if users_sel:  df_v = df_v[df_v["user_profile_name"].isin(users_sel)]

    # -------- Exportar ----------
    st.sidebar.markdown("---")
    if st.sidebar.button("📥 Exportar XLSX"):
        if df_v.empty:
            st.sidebar.warning("Nada a exportar.")
        else:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                df_v.to_excel(w, index=False, sheet_name="Atividades")
            st.sidebar.download_button("Baixar",
                buf.getvalue(),
                f"duplicatas_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # -------- Comparação ativa ----------
    cmp = st.session_state[f"cmp{SUFFIX_STATE}"]
    if cmp:
        a, b = cmp["base"], cmp["comp"]
        ratio = next((x["ratio"] for x in sim_map[a["activity_id"]] if x["id"] == b["activity_id"]), 0)
        st.subheader(f"Comparação {a['activity_id']} × {b['activity_id']} • {ratio:.0%}")
        c1, c2 = st.columns(2)
        h1, h2 = highlight_common(a["Texto"], b["Texto"])
        c1.markdown(h1, unsafe_allow_html=True)
        c2.markdown(h2, unsafe_allow_html=True)
        if st.button("Fechar comparação"):
            st.session_state[f"cmp{SUFFIX_STATE}"] = None; st.rerun()
        st.markdown("---")

    # -------- Paginação ----------
    pastas_ord = sorted(df_v["activity_folder"].dropna().unique().tolist())
    page = st.session_state[f"page{SUFFIX_STATE}"]
    tot = max(1, (len(pastas_ord) + ITENS_POR_PAGINA - 1) // ITENS_POR_PAGINA)
    page = max(0, min(page, tot - 1)); st.session_state[f"page{SUFFIX_STATE}"] = page

    if tot > 1:
        b1, _, b2 = st.columns([1, 2, 1])
        if b1.button("⬅", disabled=page == 0):
            st.session_state[f"page{SUFFIX_STATE}"] -= 1; st.rerun()
        st.markdown(f"<p style='text-align:center;'>Página {page+1}/{tot}</p>",
                    unsafe_allow_html=True)
        if b2.button("➡", disabled=page == tot - 1):
            st.session_state[f"page{SUFFIX_STATE}"] += 1; st.rerun()

    # -------- Listagem por pasta ----------
    ini, fim = page * ITENS_POR_PAGINA, (page + 1) * ITENS_POR_PAGINA
    for pasta in pastas_ord[ini:fim]:
        df_p = df_v[df_v["activity_folder"] == pasta]
        analisadas = len(df_a[df_a["activity_folder"] == pasta])
        with st.expander(f"📁 {pasta} ({len(df_p)}/{analisadas})", expanded=False):
            for _, r in df_p.iterrows():
                c1, c2 = st.columns([.6, .4])
                with c1:
                    d = r["activity_date"].strftime("%d/%m/%Y %H:%M") if pd.notna(r["activity_date"]) else "N/A"
                    st.markdown(f"**ID** `{r['activity_id']}` • {d} • `{r['activity_status']}`")
                    st.markdown(f"**Usuário:** {r['user_profile_name']}")
                    st.text_area("Texto", r["Texto"], height=100, disabled=True,
                                 key=f"txt_{pasta}_{r['activity_id']}")
                    links = link_z(r["activity_id"])
                    st.button("👁 Completo",
                              key=f"full_{r['activity_id']}",
                              on_click=lambda act=r: (
                                  st.session_state.update({f"full_act{SUFFIX_STATE}": act,
                                                           f"show_text{SUFFIX_STATE}": True})))
                    st.link_button("ZFlow v1", links["antigo"])
                    st.link_button("ZFlow v2", links["novo"])

                with c2:
                    sims = sim_map.get(r["activity_id"], [])
                    if sims:
                        st.markdown(f"**Duplicatas:** {len(sims)}")
                        for s in sims:
                            info = df[df["activity_id"] == s["id"]].iloc[0]
                            badge = f"<div class='similarity-badge' style='background:{s['cor']};'>"
                            badge += f"<b>{info['activity_id']}</b> • {s['ratio']:.0%}<br>"
                            d2 = info["activity_date"].strftime("%d/%m/%y %H:%M") if pd.notna(info["activity_date"]) else "N/A"
                            badge += f"{d2} • {info['activity_status']}<br>{info['user_profile_name']}</div>"
                            st.markdown(badge, unsafe_allow_html=True)
                            st.button("⚖ Comparar",
                                key=f"cmp_{r['activity_id']}_{info['activity_id']}",
                                on_click=lambda a=r, b=info: st.session_state.update(
                                    {f"cmp{SUFFIX_STATE}": {"base": a, "comp": b}}))
                    elif not only_dup:
                        st.markdown("<span style='color:green;'>Sem duplicatas</span>", unsafe_allow_html=True)

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
