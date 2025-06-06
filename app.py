import streamlit as st
import pandas as pd
import re, io, html, os
from datetime import datetime, timedelta, date
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from unidecode import unidecode
from rapidfuzz import fuzz

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
SUFFIX_STATE = "_final_v4" # Sufixo atualizado para esta versão
ITENS_POR_PAGINA = 20
HIGHLIGHT_COLOR = "#a8d1ff" # Azul claro para destacar semelhanças

st.set_page_config(layout="wide", page_title="Verificador de Duplicidade (Refinado)")

st.markdown(
    f"""
    <style>
    mark.common {{ background-color:{HIGHLIGHT_COLOR}; padding:0 2px; font-weight: bold;}}
    pre.highlighted-text {{ 
        white-space: pre-wrap; 
        word-wrap: break-word; 
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
        font-size: 0.9em;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        background-color: #f9f9f9;
    }}
    .similarity-badge {{
        padding: 3px 6px; 
        border-radius: 5px; 
        color: black; 
        margin-bottom: 5px; 
        font-weight: 500;
        display: inline-block;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================
def normalizar_texto(txt: str | None) -> str:
    if not txt or not isinstance(txt, str): return ""
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

def highlight_common(t1: str, t2: str, min_len: int = 3) -> tuple[str, str]:
    tok1 = re.findall(r'\w+', normalizar_texto(t1))
    tok2 = re.findall(r'\w+', normalizar_texto(t2))
    comuns = {w for w in tok1 if w in tok2 and len(w) >= min_len}

    def wrap(txt):
        parts = re.split(r"(\W+)", txt)
        out = []
        for part in parts:
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
def carregar_dados(eng: Engine) -> tuple[pd.DataFrame | None, Exception | None]:
    hoje = date.today()
    limite = hoje - timedelta(days=7)
    q_abertas = text("""SELECT activity_id, activity_folder, user_profile_name, activity_date, activity_status, Texto FROM ViewGrdAtividadesTarcisio WHERE activity_type='Verificar' AND activity_status='Aberta'""")
    q_hist = text("""SELECT activity_id, activity_folder, user_profile_name, activity_date, activity_status, Texto FROM ViewGrdAtividadesTarcisio WHERE activity_type='Verificar' AND DATE(activity_date) >= :limite""")
    try:
        with eng.connect() as c:
            df1 = pd.read_sql(q_abertas, c)
            df2 = pd.read_sql(q_hist, c, params={"limite": limite})
        df = pd.concat([df1, df2], ignore_index=True)
        if df.empty: return pd.DataFrame(), None
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
        df["Texto"] = df["Texto"].astype(str).fillna("")
        df_sorted = df.sort_values(["activity_id", "activity_status"], ascending=[True, True])
        df_deduped = df_sorted.drop_duplicates("activity_id", keep="first")
        return df_deduped.sort_values(["activity_folder", "activity_date", "activity_id"], ascending=[True, False, False]).reset_index(drop=True), None
    except exc.SQLAlchemyError as e:
        return None, e

# ==============================================================================
# CACHE DE SIMILARIDADE
# ==============================================================================
def get_similarity_map_cached(df: pd.DataFrame, min_sim: float):
    if df.empty: return {}, set() # Retorna vazio se não houver dados para analisar
    ids_tuple = tuple(sorted(df["activity_id"]))
    signature = (ids_tuple, min_sim)
    cache_key = "simcache" + SUFFIX_STATE
    if cache_key in st.session_state and st.session_state[cache_key].get("sig") == signature:
        c = st.session_state[cache_key]; return c["map"], c["dup"]

    dup_map, ids_dup = {}, set()
    prog_placeholder = st.sidebar.empty(); prog_bar = st.sidebar.progress(0, text="Analisando similaridades...")
    grupos = df.groupby("activity_folder"); total_grupos = len(grupos)
    for idx, (_, g) in enumerate(grupos):
        prog_bar.progress((idx + 1) / total_grupos, text=f"Analisando {total_grupos} pastas...")
        acts = g.to_dict("records")
        if len(acts) < 2: continue
        for i, a in enumerate(acts):
            dup_map.setdefault(a["activity_id"], [])
            for b in acts[i + 1:]:
                r = calcular_similaridade(a["Texto"], b["Texto"])
                if r >= min_sim:
                    c_color = cor_sim(r)
                    ids_dup.update([a["activity_id"], b["activity_id"]])
                    dup_map[a["activity_id"]].append(dict(id_similar=b["activity_id"], ratio=r, cor=c_color))
                    dup_map.setdefault(b["activity_id"], []).append(dict(id_similar=a["activity_id"], ratio=r, cor=c_color))
    prog_bar.empty(); prog_placeholder.empty()
    for k in dup_map: dup_map[k].sort(key=lambda x: x["ratio"], reverse=True)
    st.session_state[cache_key] = {"sig": signature, "map": dup_map, "dup": ids_dup}
    return dup_map, ids_dup

# ==============================================================================
# FUNÇÕES DE UI E ESTADO
# ==============================================================================
link_z = lambda i: {"antigo": f"https://zflow.zionbyonset.com.br/activity/3/details/{i}","novo": f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={i}#/fixcol1"}

for key_base in ["show_text_dialog", "full_act", "comparacao_ativa", "pagina_atual", "last_db_update_time"]:
    full_key = f"{key_base}{SUFFIX_STATE}"
    st.session_state.setdefault(full_key, False if "show" in key_base else (0 if "pagina" in key_base else None))

@st.dialog("Texto completo")
def dlg_full_text():
    d = st.session_state[f"full_act{SUFFIX_STATE}"]
    if d is None: return
    data_fmt = d["activity_date"].strftime("%d/%m/%Y %H:%M") if pd.notna(d["activity_date"]) else "N/A"
    st.markdown(f"### ID {d['activity_id']} • {data_fmt}")
    st.text_area("Texto", d["Texto"], height=400, disabled=True, key=f"dlg_txt_{d['activity_id']}{SUFFIX_STATE}")
    if st.button("Fechar", key=f"dlg_close_btn{SUFFIX_STATE}"):
        st.session_state[f"show_text_dialog{SUFFIX_STATE}"] = False; st.rerun()

# ==============================================================================
# APLICAÇÃO PRINCIPAL
# ==============================================================================
def app():
    st.sidebar.success(f"Logado como **{st.session_state['username']}**")
    if st.sidebar.button("Logout", key=f"logout_btn{SUFFIX_STATE}"):
        st.session_state.clear(); st.rerun()

    eng = db_engine();
    if not eng: st.stop()

    if st.sidebar.button("🔄 Atualizar dados", key=f"update_data_btn{SUFFIX_STATE}"):
        carregar_dados.clear()
        st.session_state.pop("simcache" + SUFFIX_STATE, None)
        st.session_state[f"comparacao_ativa{SUFFIX_STATE}"] = None
        st.session_state[f"last_db_update_time{SUFFIX_STATE}"] = datetime.now()

    df_raw_total, err = carregar_dados(eng)
    if err: st.exception(err); st.stop()
    if df_raw_total.empty: st.warning("Sem atividades base carregadas."); st.stop()
    
    if not st.session_state.get(f"last_db_update_time{SUFFIX_STATE}"):
        st.session_state[f"last_db_update_time{SUFFIX_STATE}"] = datetime.now()
    last_update = st.session_state[f"last_db_update_time{SUFFIX_STATE}"]
    st.sidebar.caption(f"Dados do banco atualizados em: {last_update:%d/%m/%Y %H:%M:%S}")

    st.sidebar.header("Período")
    hoje = date.today()
    data_inicio_padrao = hoje - timedelta(days=1)
    df_abertas_futuras = df_raw_total[(df_raw_total["activity_status"] == "Aberta") & (df_raw_total["activity_date"].notna()) & (df_raw_total["activity_date"].dt.date > hoje)]
    data_fim_padrao = df_abertas_futuras["activity_date"].dt.date.max() if not df_abertas_futuras.empty else hoje + timedelta(days=14)
    if data_inicio_padrao > data_fim_padrao: data_inicio_padrao = data_fim_padrao - timedelta(days=1)
    
    d_ini = st.sidebar.date_input("Início", data_inicio_padrao, key=f"di_{SUFFIX_STATE}")
    d_fim = st.sidebar.date_input("Fim", d_fim_padrao, min_value=d_ini, key=f"df_{SUFFIX_STATE}")
    
    df_periodo = df_raw_total[(df_raw_total["activity_date"].notna()) & df_raw_total["activity_date"].dt.date.between(d_ini, d_fim)]
    st.title(f"🔎 Verificador de Duplicidade ({len(df_periodo)} atividades no período)")
    
    st.sidebar.header("Filtros de Análise")
    pastas_disp = sorted(df_periodo["activity_folder"].dropna().unique().tolist())
    pastas_sel_analise = st.sidebar.multiselect("Pastas para Análise:", pastas_disp, key=f"psel_analise_{SUFFIX_STATE}")

    # --- LÓGICA DE FILTRO DE STATUS AJUSTADA ---
    # O DataFrame para análise de similaridade agora IGNORA o filtro de status da UI
    df_para_analise = df_periodo.copy()
    if pastas_sel_analise:
        df_para_analise = df_para_analise[df_para_analise["activity_folder"].isin(pastas_sel_analise)]
    
    # O filtro de status agora é apenas para EXIBIÇÃO
    st.sidebar.header("Filtros de Exibição")
    status_disp = sorted(df_para_analise["activity_status"].dropna().unique())
    status_sel_exibicao = st.sidebar.multiselect("Status para Exibição:", status_disp, key=f"ssel_exib_{SUFFIX_STATE}")
    
    min_sim = st.sidebar.slider("Similaridade mínima (%)", 0, 100, 70, 5, key=f"sim_slider_{SUFFIX_STATE}") / 100
    only_dup = st.sidebar.checkbox("Exibir Somente com Duplicatas", True, key=f"only_dup_{SUFFIX_STATE}")
    
    pastas_multi_analisadas = {p for p, g in df_para_analise.groupby("activity_folder") if g["user_profile_name"].nunique() > 1}
    only_multi = st.sidebar.checkbox("Pastas com Múltiplos Responsáveis (na análise)", False, key=f"only_multi_{SUFFIX_STATE}")
    
    usuarios_disp_exibicao = sorted(df_para_analise["user_profile_name"].dropna().unique())
    usuarios_sel_exibicao = st.sidebar.multiselect("Exibir Usuário(s):", usuarios_disp_exibicao, key=f"user_sel_{SUFFIX_STATE}")

    # --- Análise de Similaridade ---
    map_id_para_similaridades, ids_com_duplicatas = get_similarity_map_cached(df_para_analise, min_sim)

    # --- DataFrame Final para Exibição ---
    df_exibir = df_para_analise.copy() # Começa com os dados filtrados por pasta (mas não por status de exibição ainda)
    if status_sel_exibicao: df_exibir = df_exibir[df_exibir["activity_status"].isin(status_sel_exibicao)]
    if only_dup: df_exibir = df_exibir[df_exibir["activity_id"].isin(ids_com_duplicatas)]
    if only_multi: df_exibir = df_exibir[df_exibir["activity_folder"].isin(pastas_multi_analisadas)]
    if usuarios_sel_exibicao: df_exibir = df_exibir[df_exibir["user_profile_name"].isin(usuarios_sel_exibicao)]
    
    st.sidebar.markdown("---")
    if st.sidebar.button("📥 Exportar para XLSX", key=f"export_btn{SUFFIX_STATE}"):
        # ... (Lógica de exportação como na sua versão)
        if df_exibir.empty: st.sidebar.warning("Nenhum dado para exportar.")
        else:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_exibir.to_excel(writer, index=False, sheet_name="Atividades_Exibidas")
                lista_export_dup = []
                for id_base, lista_sim in map_id_para_similaridades.items():
                    if id_base in df_exibir["activity_id"].values:
                        for sim_info in lista_sim:
                            detalhes_similar_rows = df_raw_total[df_raw_total["activity_id"] == sim_info["id_similar"]]
                            if not detalhes_similar_rows.empty:
                                detalhes_similar = detalhes_similar_rows.iloc[0]
                                data_dup_str = (detalhes_similar["activity_date"].strftime("%Y-%m-%d %H:%M") if pd.notna(detalhes_similar["activity_date"]) else None)
                                lista_export_dup.append({
                                    "ID_Base": id_base, "ID_Duplicata": sim_info["id_similar"],"Similaridade": sim_info["ratio"],"Cor": sim_info["cor"],
                                    "Data_Dup": data_dup_str, "Usuario_Dup": detalhes_similar["user_profile_name"],"Status_Dup": detalhes_similar["activity_status"],
                                })
                if lista_export_dup: pd.DataFrame(lista_export_dup).to_excel(writer, index=False, sheet_name="Detalhes_Duplicatas")
            st.sidebar.download_button("Baixar XLSX", output.getvalue(),f"duplicatas_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    # --- Seção de Comparação Lado a Lado ---
    cmp_data = st.session_state.get(f"cmp{SUFFIX_STATE}")
    if cmp_data:
        # CORREÇÃO: Usar df_raw_total para garantir que sempre encontramos os dados
        base_rows = df_raw_total[df_raw_total.activity_id == cmp_data["base_id"]]
        comp_rows = df_raw_total[df_raw_total.activity_id == cmp_data["comp_id"]]
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
        else:
            st.warning("Não foi possível encontrar os dados para uma das atividades da comparação. Limpando seleção.")
            st.session_state[f"cmp{SUFFIX_STATE}"] = None; st.rerun()


    st.header("Análise Detalhada por Pasta")
    if df_exibir.empty: st.info("Nenhuma atividade para os filtros de exibição selecionados.")

    # --- Paginação e Listagem ---
    pastas_ord = sorted(df_exibir["activity_folder"].dropna().unique())
    pagina_atual = st.session_state[f"page{SUFFIX_STATE}"]
    total_paginas = max(1, (len(pastas_ord) + ITENS_POR_PAGINA - 1) // ITENS_POR_PAGINA)
    pagina_atual = max(0, min(pagina_atual, total_paginas - 1)); st.session_state[f"page{SUFFIX_STATE}"] = pagina_atual

    if total_paginas > 1:
        c1,c2,c3 = st.columns([1,2,1])
        if c1.button("⬅️", disabled=pagina_atual==0, key=f"prev_{SUFFIX_STATE}"): st.session_state[f"page{SUFFIX_STATE}"]-=1; st.rerun()
        c2.markdown(f"<p style='text-align:center'>Página {pagina_atual+1}/{total_paginas}</p>", unsafe_allow_html=True)
        if c3.button("➡️", disabled=pagina_atual>=total_paginas-1, key=f"next_{SUFFIX_STATE}"): st.session_state[f"page{SUFFIX_STATE}"]+=1; st.rerun()

    ini, fim = pagina_atual * ITENS_POR_PAGINA, (pagina_atual+1) * ITENS_POR_PAGINA
    for pasta in pastas_ord[ini:fim]:
        df_p = df_exibir[df_exibir["activity_folder"] == pasta]
        # O total analisado vem de df_para_analise, que já foi filtrado por pasta na sidebar
        analisadas = len(df_para_analise[df_para_analise["activity_folder"] == pasta])
        with st.expander(f"📁 {pasta} ({len(df_p)} exibidas / {analisadas} analisadas)", expanded=False):
            for _, r in df_p.iterrows():
                act_id = int(r["activity_id"])
                c1_item, c2_item = st.columns([.6, .4]) # Renomeado para evitar conflito
                with c1_item:
                    data = r["activity_date"].strftime("%d/%m/%Y %H:%M") if pd.notna(r["activity_date"]) else "N/A"
                    st.markdown(f"**ID** `{act_id}` • {data} • `{r['activity_status']}`")
                    st.markdown(f"**Usuário:** {r['user_profile_name']}")
                    st.text_area("Texto", r["Texto"], height=100, disabled=True, key=f"txt_{pasta}_{act_id}")
                    btn1_item, btn2_item, btn3_item = st.columns(3) # Renomeado
                    btn1_item.button("👁 Completo", key=f"full_{act_id}", on_click=lambda act=r: st.session_state.update({f"full_act{SUFFIX_STATE}": act, f"show_text{SUFFIX_STATE}": True}))
                    lnk = link_z(act_id)
                    btn2_item.link_button("ZFlow v1", lnk["antigo"])
                    btn3_item.link_button("ZFlow v2", lnk["novo"])
                with c2_item:
                    sims = map_id_para_similaridades.get(act_id, [])
                    if sims:
                        st.markdown(f"**Duplicatas:** {len(sims)}")
                        for s_info in sims: # Renomeado
                            # CORREÇÃO: Busca no df_para_analise que contém todos os status para o par similar
                            info_rows = df_para_analise[df_para_analise["activity_id"] == s_info["id_similar"]]
                            if not info_rows.empty:
                                info = info_rows.iloc[0]
                                d = info["activity_date"].strftime("%d/%m/%y %H:%M") if pd.notna(info["activity_date"]) else "N/A"
                                badge = (f"<div class='similarity-badge' style='background:{s_info['cor']};'>"
                                         f"<b>{info['activity_id']}</b> • {s_info['ratio']:.0%}<br>"
                                         f"{d} • {info['activity_status']}<br>{info['user_profile_name']}</div>")
                                st.markdown(badge, unsafe_allow_html=True)
                                st.button("⚖ Comparar", key=f"cmp_{act_id}_{info['activity_id']}",
                                          on_click=lambda a=r, b=info: st.session_state.update({f"cmp{SUFFIX_STATE}": {"base_id": a.activity_id, "comp_id": b.activity_id}}))
                    elif not only_dup:
                        st.markdown("<span style='color:green;'>Sem duplicatas</span>", unsafe_allow_html=True)
    if st.session_state.get(f"show_text_dialog{SUFFIX_STATE}"): dlg_full_text()

# ==============================================================================
# LOGIN
# ==============================================================================
def cred_ok(user: str, pwd: str) -> bool:
    cred = st.secrets.get("credentials", {}).get("usernames", {})
    return user in cred and str(cred[user]) == pwd

def login_form():
    st.header("Login")
    with st.form("login"):
        u = st.text_input("Usuário")
        p = st.text_input("Senha", type="password")
        if st.form_submit_button("Entrar"):
            if cred_ok(u, p): st.session_state.update({"logged_in": True, "username": u}); st.rerun()
            else: st.error("Credenciais inválidas.")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    if not st.session_state.get("logged_in"): st.session_state["logged_in"] = False
    if st.session_state["logged_in"]: app()
    else: login_form()
