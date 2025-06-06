import streamlit as st
import pandas as pd
import re, io, html, os
from datetime import datetime, timedelta, date
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from unidecode import unidecode
from rapidfuzz import fuzz

# ═════════ CONFIGURAÇÃO GERAL ════════════════════════════════════════════════
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
</style>
""", unsafe_allow_html=True)

# ═════════ FUNÇÕES DE TEXTO & SIMILARIDADE ═══════════════════════════════════
def normalizar_texto(txt: str | None) -> str:
    """Remove acentos, pontuação e múltiplos espaços, retornando lower-case."""
    if not txt or not isinstance(txt, str):
        return ""
    txt = unidecode(txt.lower())
    txt = re.sub(r"[^\w\s]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()

def calcular_similaridade(txt_a: str, txt_b: str) -> float:
    """Token-set ratio (RapidFuzz) entre textos já normalizados."""
    a, b = normalizar_texto(txt_a), normalizar_texto(txt_b)
    if not a or not b:
        return 0.0
    # evita comparar blocos com tamanhos muito díspares
    if abs(len(a) - len(b)) > 0.5 * max(len(a), len(b)):
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100

def cor_sim(ratio: float) -> str:
    """Define cor do badge conforme similaridade."""
    return "#FF5252" if ratio >= .9 else "#FFB74D" if ratio >= .7 else "#FFD54F"

def highlight_common(texto1: str, texto2: str, min_len: int = 3) -> tuple[str, str]:
    """
    Devolve HTML de dois textos com palavras em comum destacadas
    usando <mark class="common">.
    """
    tokens1 = re.findall(r"\w+", normalizar_texto(texto1))
    tokens2 = re.findall(r"\w+", normalizar_texto(texto2))
    comuns  = {w for w in tokens1 if w in tokens2 and len(w) >= min_len}

    def wrap(original: str) -> str:
        parts, out = re.split(r"(\W+)", original), []
        for part in parts:
            if not part:
                continue
            if re.match(r"\w+", part) and normalizar_texto(part) in comuns:
                out.append(f"<mark class='common'>{html.escape(part)}</mark>")
            else:
                out.append(html.escape(part))
        return "<pre class='highlighted-text'>" + "".join(out) + "</pre>"

    return wrap(texto1), wrap(texto2)

# ═════════ BANCO DE DADOS (MySQL) ════════════════════════════════════════════
@st.cache_resource
def db_engine() -> Engine | None:
    """Cria engine MySQL usando credenciais em st.secrets ou variáveis de ambiente."""
    host = st.secrets.get("database", {}).get("host") or os.getenv("DB_HOST")
    user = st.secrets.get("database", {}).get("user") or os.getenv("DB_USER")
    pw   = st.secrets.get("database", {}).get("password") or os.getenv("DB_PASS")
    db   = st.secrets.get("database", {}).get("name") or os.getenv("DB_NAME")
    if not all([host, user, pw, db]):
        st.error("Credenciais do banco ausentes em st.secrets ou variáveis DB_*")
        return None
    uri = f"mysql+mysqlconnector://{user}:{pw}@{host}/{db}"
    try:
        eng = create_engine(uri, pool_pre_ping=True, pool_recycle=3600)
        with eng.connect(): pass
        return eng
    except exc.SQLAlchemyError as e:
        st.exception(e)
        return None

@st.cache_data(ttl=3600, hash_funcs={Engine: lambda _: None})
def carregar_dados(eng: Engine) -> tuple[pd.DataFrame | None, Exception | None]:
    """Lê atividades 'Verificar' Abertas + últimos 7 dias do histórico."""
    hoje, limite = date.today(), date.today() - timedelta(days=7)
    q_abertas = text("""
        SELECT activity_id, activity_folder, user_profile_name, activity_date,
               activity_status, Texto
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type='Verificar' AND activity_status='Aberta'""")
    q_hist = text("""
        SELECT activity_id, activity_folder, user_profile_name, activity_date,
               activity_status, Texto
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type='Verificar' AND DATE(activity_date) >= :limite""")
    try:
        with eng.connect() as con:
            df1 = pd.read_sql(q_abertas, con)
            df2 = pd.read_sql(q_hist, con, params={"limite": limite})
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

# ═════════ CACHE DE SIMILARIDADE (em session_state) ═════════════════════════
def sim_map_cached(df: pd.DataFrame, min_sim: float):
    """Retorna (mapa, ids_dup) utilizando cache em session_state."""
    sig = (tuple(sorted(df["activity_id"])), min_sim)
    key = "simcache" + SUFFIX_STATE
    if key in st.session_state and st.session_state[key]["sig"] == sig:
        c = st.session_state[key]; return c["map"], c["dup"]

    mapa, ids_dup = {}, set()
    barra = st.sidebar.progress(0)
    grupos = df.groupby("activity_folder")
    for idx, (_, g) in enumerate(grupos):
        barra.progress((idx+1) / grupos.ngroups)
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

# ═════════ LINKS RÁPIDOS PARA ZFLOW ══════════════════════════════════════════
link_z = lambda i: {
    "antigo": f"https://zflow.zionbyonset.com.br/activity/3/details/{i}",
    "novo"  : f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={i}#/fixcol1"
}

# ═════════ ESTADO PADRÃO DA SESSÃO ═══════════════════════════════════════════
defaults = {
    f"show_text{SUFFIX_STATE}": False,
    f"full_act{SUFFIX_STATE}":  None,
    f"cmp{SUFFIX_STATE}":       None,   # {"base_id": .., "comp_id": ..}
    f"page{SUFFIX_STATE}":      0,
    f"last_update{SUFFIX_STATE}": None,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ═════════ DIALOG TEXTO COMPLETO ═════════════════════════════════════════════
@st.dialog("Texto completo")
def dlg_full():
    d = st.session_state[f"full_act{SUFFIX_STATE}"]
    if d is None: return
    data_fmt = d["activity_date"].strftime("%d/%m/%Y %H:%M") if pd.notna(d["activity_date"]) else "N/A"
    st.markdown(f"### ID {d['activity_id']} – {data_fmt}")
    st.text_area("Texto", d["Texto"], height=400, disabled=True,
                 key=f"dlg_txt_{d['activity_id']}{SUFFIX_STATE}")
    if st.button("Fechar"):
        st.session_state[f"show_text{SUFFIX_STATE}"] = False

# ═════════ APP PRINCIPAL ═════════════════════════════════════════════════════
def app():
    # Header / logout
    st.sidebar.success(f"Logado como **{st.session_state['username']}**")
    if st.sidebar.button("Logout"):
        st.session_state.clear(); st.rerun()

    eng = db_engine()
    if not eng: st.stop()

    # Botão atualizar dados
    if st.sidebar.button("🔄 Atualizar dados"):
        carregar_dados.clear()                           # limpa cache de dados
        st.session_state.pop("simcache"+SUFFIX_STATE, None)  # limpa similaridade
        st.session_state[f"last_update{SUFFIX_STATE}"] = datetime.now()

    # Carrega dados
    df, err = carregar_dados(eng)
    if err:
        st.exception(err); st.stop()
    if df.empty:
        st.warning("Sem atividades encontradas."); st.stop()

    # Info de última atualização
    up = st.session_state[f"last_update{SUFFIX_STATE}"]
    if up: st.sidebar.caption(f"Dados atualizados em: {up:%d/%m/%Y %H:%M:%S}")
    else:  st.sidebar.caption("Dados provenientes do cache.")

    # Período
    hoje = date.today()
    d_ini = st.sidebar.date_input("Início", hoje - timedelta(days=1))
    d_fim = st.sidebar.date_input("Fim",    hoje + timedelta(days=14), min_value=d_ini)
    if d_ini > d_fim: st.sidebar.error("Início > fim."); st.stop()
    df_per = df[(df["activity_date"].notna()) & df["activity_date"].dt.date.between(d_ini, d_fim)]
    st.title(f"🔎 Duplicidades ({len(df_per)} atividades)")

    # Filtros
    pastas = sorted(df_per["activity_folder"].dropna().unique())
    pastas_sel = st.sidebar.multiselect("Pastas", pastas)
    status = sorted(df_per["activity_status"].dropna().unique())
    status_sel = st.sidebar.multiselect("Status", status)
    df_a = df_per.copy()
    if pastas_sel: df_a = df_a[df_a["activity_folder"].isin(pastas_sel)]
    if status_sel: df_a = df_a[df_a["activity_status"].isin(status_sel)]

    min_sim = st.sidebar.slider("Similaridade mínima (%)", 0, 100, 70, 5) / 100
    only_dup = st.sidebar.checkbox("Somente duplicatas", True)
    pastas_multi = {p for p, g in df_a.groupby("activity_folder")
                    if g["user_profile_name"].nunique() > 1}
    only_multi = st.sidebar.checkbox("Pastas com múltiplos responsáveis")
    users_disp = sorted(df_a["user_profile_name"].dropna().unique())
    users_sel = st.sidebar.multiselect("Usuários", users_disp)

    # Similaridade (cache)
    sim_map, ids_dup = sim_map_cached(df_a, min_sim)

    # Exibição final
    df_v = df_a.copy()
    if only_dup:   df_v = df_v[df_v["activity_id"].isin(ids_dup)]
    if only_multi: df_v = df_v[df_v["activity_folder"].isin(pastas_multi)]
    if users_sel:  df_v = df_v[df_v["user_profile_name"].isin(users_sel)]

    # Paginação
    pastas_ord = sorted(df_v["activity_folder"].dropna().unique())
    page = st.session_state[f"page{SUFFIX_STATE}"]
    total_pages = max(1, (len(pastas_ord) + ITENS_POR_PAGINA - 1) // ITENS_POR_PAGINA)
    page = max(0, min(page, total_pages - 1)); st.session_state[f"page{SUFFIX_STATE}"] = page

    if total_pages > 1:
        col_prev, col_label, col_next = st.columns([1, 2, 1])
        if col_prev.button("⬅", disabled=page == 0):
            st.session_state[f"page{SUFFIX_STATE}"] -= 1; st.rerun()
        col_label.markdown(f"<p style='text-align:center;'>Página {page+1}/{total_pages}</p>", unsafe_allow_html=True)
        if col_next.button("➡", disabled=page == total_pages - 1):
            st.session_state[f"page{SUFFIX_STATE}"] += 1; st.rerun()

    # Loop pastas
    cmp_state = st.session_state[f"cmp{SUFFIX_STATE}"]
    start, end = page * ITENS_POR_PAGINA, (page + 1) * ITENS_POR_PAGINA
    for pasta in pastas_ord[start:end]:
        df_p = df_v[df_v["activity_folder"] == pasta]
        analisadas = len(df_a[df_a["activity_folder"] == pasta])
        with st.expander(f"📁 {pasta} ({len(df_p)}/{analisadas})"):
            for _, r in df_p.iterrows():
                act_id = int(r["activity_id"])
                col1, col2 = st.columns([.6, .4])
                
                # ---- Coluna Info principal ----
                with col1:
                    data_fmt = r["activity_date"].strftime("%d/%m/%Y %H:%M") if pd.notna(r["activity_date"]) else "N/A"
                    st.markdown(f"**ID** `{act_id}` • {data_fmt} • `{r['activity_status']}`")
                    st.markdown(f"**Usuário:** {r['user_profile_name']}")
                    st.text_area("Texto", r["Texto"], height=100, disabled=True,
                                 key=f"txt_{pasta}_{act_id}")
                    st.button("👁 Completo",
                              key=f"full_{act_id}",
                              on_click=lambda act=r: st.session_state.update({
                                  f"full_act{SUFFIX_STATE}": act,
                                  f"show_text{SUFFIX_STATE}": True}))
                    lnk = link_z(act_id)
                    st.link_button("ZFlow v1", lnk["antigo"])
                    st.link_button("ZFlow v2", lnk["novo"])

                # ---- Coluna duplicatas ----
                with col2:
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

                # ---- Comparação embutida (se este item é o base selecionado) ----
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
                        if st.button("❌ Fechar comparação", key=f"cls_cmp_{act_id}"):
                            st.session_state[f"cmp{SUFFIX_STATE}"] = None
                            st.rerun()

    # Chama dialog texto completo, se ativo
    if st.session_state[f"show_text{SUFFIX_STATE}"]:
        dlg_full()

# ═════════ LOGIN ═════════════════════════════════════════════════════════════
def cred_ok(user: str, pwd: str) -> bool:
    cred = st.secrets.get("credentials", {}).get("usernames", {})
    return user in cred and str(cred[user]) == pwd

def login_form():
    st.header("Login")
    with st.form("login"):
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
        login_form()
