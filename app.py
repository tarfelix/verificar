# app.py
import streamlit as st
import pandas as pd
import re
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from datetime import datetime, timedelta, date
from unidecode import unidecode
from rapidfuzz import fuzz
import io
import html
import os

# ==============================================================================
# CONFIGURAÇÕES GERAIS
# ==============================================================================
SUFFIX_STATE = "_grifar_v1"
ITENS_POR_PAGINA = 20
st.set_page_config(layout="wide", page_title="Verificador de Duplicidade (Grifar Semelhanças)")

# CSS global (destaque <mark> e <pre> quebrando linha)
st.markdown(
    """
    <style>
      mark { background-color:#ffeb3b; padding:0 2px; }
      pre  { white-space: pre-wrap; font-family: 'Courier New', monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# FUNÇÕES DE TEXTO E SIMILARIDADE
# ==============================================================================
def normalizar_texto(txt: str | None) -> str:
    if not txt or not isinstance(txt, str):
        return ""
    txt = unidecode(txt.lower())
    txt = re.sub(r"[^\w\s]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def calcular_similaridade(texto_a: str, texto_b: str) -> float:
    a, b = normalizar_texto(texto_a), normalizar_texto(texto_b)
    if not a or not b:
        return 0.0
    if abs(len(a) - len(b)) > 0.3 * max(len(a), len(b)):
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


def obter_cor_similaridade(ratio: float) -> str:
    if ratio >= 0.90:
        return "#FF5252"  # alta
    if ratio >= 0.70:
        return "#FFB74D"  # média
    return "#FFD54F"      # baixa


def highlight_common_words(t1: str, t2: str, min_len: int = 3) -> tuple[str, str]:
    tokens1 = re.findall(r"\w+", unidecode(t1.lower()))
    tokens2 = re.findall(r"\w+", unidecode(t2.lower()))
    comuns = {w for w in tokens1 if w in tokens2 and len(w) >= min_len}

    def _wrap(txt: str):
        def repl(m):
            palavra = m.group(0)
            if unidecode(palavra.lower()) in comuns:
                return f"<mark>{html.escape(palavra)}</mark>"
            return html.escape(palavra)
        return "<pre>" + re.sub(r"\w+", repl, txt) + "</pre>"

    return _wrap(t1), _wrap(t2)


# ==============================================================================
# BANCO DE DADOS
# ==============================================================================
@st.cache_resource
def get_db_engine() -> Engine | None:
    db_host = st.secrets.get("database", {}).get("host") or os.getenv("DB_HOST")
    db_user = st.secrets.get("database", {}).get("user") or os.getenv("DB_USER")
    db_pass = st.secrets.get("database", {}).get("password") or os.getenv("DB_PASS")
    db_name = st.secrets.get("database", {}).get("name") or os.getenv("DB_NAME")

    if not all([db_host, db_user, db_pass, db_name]):
        st.error(
            "Credenciais do banco não encontradas. "
            "Configure em st.secrets ou nas variáveis DB_HOST/DB_USER/DB_PASS/DB_NAME."
        )
        return None

    uri = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_name}"
    try:
        engine = create_engine(uri, pool_pre_ping=True, pool_recycle=3600)
        with engine.connect():
            pass
        return engine
    except exc.SQLAlchemyError as e:
        st.exception(e)
        return None


@st.cache_data(ttl=3600, hash_funcs={Engine: lambda _: None})
def carregar_dados_base(eng: Engine) -> tuple[pd.DataFrame | None, Exception | None]:
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
        WHERE activity_type='Verificar' AND DATE(activity_date) >= :limite
    """)

    try:
        with eng.connect() as con:
            df_abertas = pd.read_sql(q_abertas, con)
            df_hist   = pd.read_sql(q_hist, con, params={"limite": limite})
        df = pd.concat([df_abertas, df_hist], ignore_index=True)
        if df.empty:
            cols = ["activity_id", "activity_folder", "user_profile_name",
                    "activity_date", "activity_status", "Texto"]
            return pd.DataFrame(columns=cols), None
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
        df["Texto"] = df["Texto"].astype(str).fillna("")
        df = df.sort_values(
            ["activity_folder", "activity_date", "activity_id"],
            ascending=[True, False, False],
        ).drop_duplicates("activity_id")
        return df, None
    except exc.SQLAlchemyError as e:
        return None, e


# ==============================================================================
# SIMILARIDADE (cache em session_state)
# ==============================================================================
def get_similarity_map(df: pd.DataFrame, min_sim: float):
    signature = (tuple(sorted(df["activity_id"])), float(min_sim))
    key = "similarity_cache" + SUFFIX_STATE

    if key in st.session_state and st.session_state[key]["signature"] == signature:
        return st.session_state[key]["map"], st.session_state[key]["ids_dup"]

    ids_dup, sim_map = set(), {}
    for _, grp in df.groupby("activity_folder"):
        acts = grp.to_dict("records")
        for i, a in enumerate(acts):
            sim_map.setdefault(a["activity_id"], [])
            for b in acts[i + 1 :]:
                r = calcular_similaridade(a["Texto"], b["Texto"])
                if r >= min_sim:
                    cor = obter_cor_similaridade(r)
                    ids_dup.update([a["activity_id"], b["activity_id"]])
                    sim_map[a["activity_id"]].append(dict(id_similar=b["activity_id"], ratio=r, cor=cor))
                    sim_map.setdefault(b["activity_id"], []).append(
                        dict(id_similar=a["activity_id"], ratio=r, cor=cor)
                    )
    for k in sim_map:
        sim_map[k].sort(key=lambda x: x["ratio"], reverse=True)

    st.session_state[key] = {"signature": signature, "map": sim_map, "ids_dup": ids_dup}
    return sim_map, ids_dup


# ==============================================================================
# LINKS
# ==============================================================================
def gerar_links_zflow(act_id: int):
    return {
        "antigo": f"https://zflow.zionbyonset.com.br/activity/3/details/{act_id}",
        "novo":   f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={act_id}#/fixcol1",
    }


# ==============================================================================
# ESTADO DA SESSÃO (com valores padrão válidos)
# ==============================================================================
defaults = {
    f"show_texto_dialog{SUFFIX_STATE}": False,
    f"atividade_para_texto_dialog{SUFFIX_STATE}": None,
    f"comparacao_ativa{SUFFIX_STATE}": None,
    f"pagina_atual{SUFFIX_STATE}": 0,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ==============================================================================
# DIÁLOGO DE TEXTO COMPLETO
# ==============================================================================
@st.dialog("Texto Completo da Atividade")
def dialogo_texto_completo():
    dados = st.session_state[f"atividade_para_texto_dialog{SUFFIX_STATE}"]
    if dados is None:
        return
    data_fmt = dados["activity_date"].strftime("%d/%m/%Y %H:%M") if pd.notna(dados["activity_date"]) else "N/A"
    st.markdown(f"### ID `{dados['activity_id']}` | Data: {data_fmt}")
    st.text_area("Texto", dados["Texto"], height=400, disabled=True)
    if st.button("Fechar"):
        st.session_state[f"show_texto_dialog{SUFFIX_STATE}"] = False
        st.rerun()


# ==============================================================================
# APP PRINCIPAL
# ==============================================================================
def app():

    # --- Cabeçalho / logout ---------------------------------------------------
    st.sidebar.success(f"Logado como **{st.session_state['username']}**")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    st.title("🔎 Verificador de Duplicidade (Comparação Lado a Lado)")

    # --- Dados ---------------------------------------------------------------
    eng = get_db_engine()
    if not eng:
        st.stop()

    if st.sidebar.button("🔄 Atualizar dados"):
        carregar_dados_base.clear()
        st.session_state[f"comparacao_ativa{SUFFIX_STATE}"] = None

    df_raw, err = carregar_dados_base(eng)
    if err:
        st.exception(err)
        st.stop()
    if df_raw.empty:
        st.warning("Nenhuma atividade encontrada.")
        st.stop()

    # --- Filtros -------------------------------------------------------------
    hoje = date.today()
    inicio_def = hoje - timedelta(days=1)
    fim_def = hoje + timedelta(days=14)

    st.sidebar.header("Período")
    d_ini = st.sidebar.date_input("Início", inicio_def)
    d_fim = st.sidebar.date_input("Fim", fim_def, min_value=d_ini)
    if d_ini > d_fim:
        st.sidebar.error("Data inicial maior que final.")
        st.stop()

    df_per = df_raw[
        (df_raw["activity_date"].notna()) &
        (df_raw["activity_date"].dt.date.between(d_ini, d_fim))
    ]
    st.success(f"{len(df_per)} atividades no período selecionado.")

    st.sidebar.header("Filtros de análise")
    pastas = sorted(df_per["activity_folder"].unique())
    pastas_sel = st.sidebar.multiselect("Pastas", pastas)
    status = sorted(df_per["activity_status"].unique())
    status_sel = st.sidebar.multiselect("Status", status)

    df_analise = df_per.copy()
    if pastas_sel:
        df_analise = df_analise[df_analise["activity_folder"].isin(pastas_sel)]
    if status_sel:
        df_analise = df_analise[df_analise["activity_status"].isin(status_sel)]

    st.sidebar.header("Exibição")
    min_sim = st.sidebar.slider("Similaridade mínima (%)", 0, 100, 70, 5) / 100
    apenas_dup = st.sidebar.checkbox("Somente duplicatas", True)

    # --- Similaridades -------------------------------------------------------
    sim_map, ids_dup = get_similarity_map(df_analise, min_sim)

    df_view = df_analise.copy()
    if apenas_dup:
        df_view = df_view[df_view["activity_id"].isin(ids_dup)]

    # --- Comparação ativa ----------------------------------------------------
    comp = st.session_state.get(f"comparacao_ativa{SUFFIX_STATE}")
    if comp:
        base, compa = comp["base"], comp["comparar"]
        st.subheader(f"Comparação: {base['activity_id']} × {compa['activity_id']}")
        c1, c2 = st.columns(2)
        h_a, h_b = highlight_common_words(base["Texto"], compa["Texto"])
        c1.markdown(h_a, unsafe_allow_html=True)
        c2.markdown(h_b, unsafe_allow_html=True)
        if st.button("Fechar comparação"):
            st.session_state[f"comparacao_ativa{SUFFIX_STATE}"] = None
            st.rerun()
        st.markdown("---")

    # --- Paginação -----------------------------------------------------------
    pastas_ord = sorted(df_view["activity_folder"].unique())
    pag = st.session_state[f"pagina_atual{SUFFIX_STATE}"]
    tot_pag = (len(pastas_ord) + ITENS_POR_PAGINA - 1) // ITENS_POR_PAGINA or 1
    pag = max(0, min(pag, tot_pag - 1))
    st.session_state[f"pagina_atual{SUFFIX_STATE}"] = pag

    if tot_pag > 1:
        b_ant, _, b_prox = st.columns([1, 2, 1])
        if b_ant.button("⬅️", disabled=pag == 0):
            st.session_state[f"pagina_atual{SUFFIX_STATE}"] -= 1
            st.rerun()
        st.markdown(f"<p style='text-align:center;'>Página {pag+1}/{tot_pag}</p>", unsafe_allow_html=True)
        if b_prox.button("➡️", disabled=pag == tot_pag - 1):
            st.session_state[f"pagina_atual{SUFFIX_STATE}"] += 1
            st.rerun()

    # --- Listagem por pasta --------------------------------------------------
    inicio, fim = pag * ITENS_POR_PAGINA, (pag + 1) * ITENS_POR_PAGINA
    for pasta in pastas_ord[inicio:fim]:
        df_p = df_view[df_view["activity_folder"] == pasta]
        with st.expander(f"📁 {pasta} ({len(df_p)})", expanded=len(df_p) < 10):
            for _, row in df_p.iterrows():
                col1, col2 = st.columns([0.6, 0.4])
                with col1:
                    dstr = row["activity_date"].strftime("%d/%m/%Y %H:%M") if pd.notna(row["activity_date"]) else "N/A"
                    st.markdown(f"**ID:** `{row['activity_id']}` • **Data:** {dstr} • **Status:** `{row['activity_status']}`")
                    st.markdown(f"**Usuário:** {row['user_profile_name']}")
                    st.text_area("Texto", row["Texto"], height=100, disabled=True,
                                 key=f"txt_{row['activity_id']}")
                    if st.button("👁️ Ver completo", key=f"ver_{row['activity_id']}",
                                 on_click=lambda r=row: (
                                     st.session_state.update({
                                         f"atividade_para_texto_dialog{SUFFIX_STATE}": r,
                                         f"show_texto_dialog{SUFFIX_STATE}": True
                                     }),
                                     None)[1]):
                        pass
                    links = gerar_links_zflow(row["activity_id"])
                    st.link_button("🔗 ZFlow v1", links["antigo"])
                    st.link_button("🔗 ZFlow v2", links["novo"])

                with col2:
                    sims = sim_map.get(row["activity_id"], [])
                    if sims:
                        st.markdown(f"**Duplicatas:** {len(sims)}", unsafe_allow_html=True)
                        for s in sims:
                            info = df_raw[df_raw["activity_id"] == s["id_similar"]].iloc[0]
                            bg = s["cor"]
                            ddup = info["activity_date"].strftime("%d/%m/%y %H:%M") if pd.notna(info["activity_date"]) else "N/A"
                            st.markdown(
                                f"""<div style="background:{bg};padding:3px;border-radius:5px;">
                                   <b>{info['activity_id']}</b> • {s['ratio']:.0%}<br>
                                   {ddup} • {info['activity_status']}<br>
                                   {info['user_profile_name']}
                                   </div>""",
                                unsafe_allow_html=True,
                            )
                            st.button(
                                "⚖️ Comparar",
                                key=f"cmp_{row['activity_id']}_{info['activity_id']}",
                                on_click=lambda a=row, b=info: (
                                    st.session_state.update({f"comparacao_ativa{SUFFIX_STATE}": {"base": a, "comparar": b}}),
                                    None)[1],
                            )

    # --- Diálogo de texto completo ------------------------------------------
    if st.session_state[f"show_texto_dialog{SUFFIX_STATE}"]:
        dialogo_texto_completo()


# ==============================================================================
# LOGIN / EXECUÇÃO
# ==============================================================================
def check_credentials(user, pwd):
    return user in st.secrets.get("credentials", {}).get("usernames", {}) and \
           str(st.secrets["credentials"]["usernames"][user]) == pwd


def login_form():
    st.header("Login")
    with st.form("login"):
        u = st.text_input("Usuário")
        p = st.text_input("Senha", type="password")
        if st.form_submit_button("Entrar"):
            if check_credentials(u, p):
                st.session_state.update({"logged_in": True, "username": u})
                st.rerun()
            else:
                st.error("Credenciais inválidas.")


if __name__ == "__main__":
    if not st.session_state.get("logged_in"):
        st.session_state["logged_in"] = False
    if st.session_state["logged_in"]:
        app()
    else:
        login_form()
