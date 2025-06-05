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
# difflib não é mais usado diretamente se highlight_common_words for a única abordagem de diff
# Mas pode ser útil para HtmlDiff se você decidir reintroduzir essa opção no futuro.

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
SUFFIX_STATE = "_grifar_v2" # Atualizado para nova versão de estado
ITENS_POR_PAGINA = 20
HIGHLIGHT_COLOR = "#a8d1ff" # Cor para highlight_common_words, pode ajustar

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
    /* Estilo para os badges de similaridade */
    .similarity-badge {{
        padding: 3px 6px; 
        border-radius: 5px; 
        color: black; 
        margin-bottom: 5px; 
        font-weight: 500;
        display: inline-block; /* Para que o margin-bottom funcione bem */
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================
def normalizar_texto(txt: str | None) -> str: # Renomeado parâmetro para txt
    if not txt or not isinstance(txt, str): return ""
    txt = unidecode(txt.lower())
    txt = re.sub(r"[^\w\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def calcular_similaridade(texto_a: str, texto_b: str) -> float:
    norm_a, norm_b = normalizar_texto(texto_a), normalizar_texto(texto_b)
    if not norm_a or not norm_b: return 0.0
    if abs(len(norm_a) - len(norm_b)) > 0.3 * max(len(norm_a), len(norm_b)) and min(len(norm_a),len(norm_b)) > 0: # Heurística de tamanho
        return 0.0
    return fuzz.token_set_ratio(norm_a, norm_b) / 100.0

def obter_cor_similaridade(ratio: float) -> str: # Renomeado parâmetro para ratio
    LIMIAR_ALTA, LIMIAR_MEDIA = 0.90, 0.70
    CORES = {"alta": "#FF5252", "media": "#FFB74D", "baixa": "#FFD54F"}
    if ratio >= LIMIAR_ALTA: return CORES["alta"]
    if ratio >= LIMIAR_MEDIA: return CORES["media"]
    return CORES["baixa"]

def highlight_common_words(text1: str, text2: str, min_len: int = 3) -> tuple[str, str]:
    # Normaliza os textos uma vez para encontrar palavras comuns
    norm_text1_tokens = re.findall(r"\w+", normalizar_texto(text1))
    norm_text2_tokens = re.findall(r"\w+", normalizar_texto(text2))
    
    comuns_normalizadas = {w for w in norm_text1_tokens if w in norm_text2_tokens and len(w) >= min_len}

    def _wrap_text_with_highlights(original_text: str) -> str:
        # Divide o texto original mantendo os delimitadores (palavras e não-palavras)
        parts = re.split(r"(\W+)", original_text)
        highlighted_parts = []
        for part in parts:
            if not part: continue # Pular partes vazias
            # Normaliza a parte atual (que deve ser uma palavra se não for delimitador)
            # para verificar se está no conjunto de comuns_normalizadas
            if re.match(r"\w+", part) and normalizar_texto(part) in comuns_normalizadas:
                highlighted_parts.append(f'<mark class="common">{html.escape(part)}</mark>')
            else:
                highlighted_parts.append(html.escape(part))
        return "<pre class='highlighted-text'>" + "".join(highlighted_parts) + "</pre>"

    return _wrap_text_with_highlights(text1), _wrap_text_with_highlights(text2)

@st.cache_resource
def db_engine() -> Engine | None: # Nome da função como na sua versão
    h = st.secrets.get("database", {}).get("host") or os.getenv("DB_HOST")
    u = st.secrets.get("database", {}).get("user") or os.getenv("DB_USER")
    p = st.secrets.get("database", {}).get("password") or os.getenv("DB_PASS")
    n = st.secrets.get("database", {}).get("name") or os.getenv("DB_NAME")
    if not all([h, u, p, n]):
        st.error("Credenciais do banco ausentes."); return None
    uri = f"mysql+mysqlconnector://{u}:{p}@{h}/{n}"
    try:
        eng = create_engine(uri, pool_pre_ping=True, pool_recycle=3600); 
        with eng.connect(): pass
        return eng
    except exc.SQLAlchemyError as e: st.exception(e); return None

@st.cache_data(ttl=3600, hash_funcs={Engine: lambda _: None})
def carregar_dados(eng: Engine) -> tuple[pd.DataFrame | None, Exception | None]: # Nome da função como na sua versão
    hoje = date.today()
    limite = hoje - timedelta(days=7)
    q_abertas = text("SELECT activity_id, activity_folder, user_profile_name, activity_date, activity_status, Texto, activity_type FROM ViewGrdAtividadesTarcisio WHERE activity_type='Verificar' AND activity_status='Aberta'")
    q_hist = text("SELECT activity_id, activity_folder, user_profile_name, activity_date, activity_status, Texto, activity_type FROM ViewGrdAtividadesTarcisio WHERE activity_type='Verificar' AND DATE(activity_date) >= :lim")
    try:
        with eng.connect() as c:
            df1 = pd.read_sql(q_abertas, c)
            df2 = pd.read_sql(q_hist, c, params={"lim": limite})
        df = pd.concat([df1, df2], ignore_index=True)
        if df.empty: 
            cols = ["activity_id","activity_folder","user_profile_name","activity_date","activity_status","Texto","activity_type"]
            df_ret = pd.DataFrame(columns=cols); df_ret["activity_date"]=pd.Series(dtype="datetime64[ns]"); df_ret["Texto"]=pd.Series(dtype="object")
            return df_ret, None # Retorna DataFrame vazio com tipos corretos
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
        df["Texto"] = df["Texto"].astype(str).fillna("")
        # Priorizar 'Aberta' se houver activity_id duplicado
        df = df.sort_values(["activity_id", "activity_status"], ascending=[True, True]).drop_duplicates("activity_id", keep="first")
        df = df.sort_values(["activity_folder", "activity_date", "activity_id"], ascending=[True, False, False]).reset_index(drop=True) # Reset index
        return df, None
    except exc.SQLAlchemyError as e: return None, e

def get_similarity_map_cached(df: pd.DataFrame, min_sim: float): # Renomeado para clareza
    # Usar uma signature mais robusta que inclua os IDs das atividades e o min_sim
    # Ordenar os IDs para garantir que a ordem não afete a signature
    ids_tuple = tuple(df["activity_id"].sort_values().unique()) if not df.empty else tuple()
    signature = (ids_tuple, float(min_sim))
    cache_key = f"simcache{SUFFIX_STATE}"

    if cache_key in st.session_state and st.session_state[cache_key]["signature"] == signature:
        cached_data = st.session_state[cache_key]
        return cached_data["map"], cached_data["ids_dup"]

    dup_map, ids_dup = {}, set()
    prog_placeholder = st.sidebar.empty(); prog_bar = st.sidebar.progress(0)
    total_grupos = df.groupby("activity_folder").ngroups if not df.empty else 0
    prog_count = 0

    for _, grp in df.groupby("activity_folder"):
        prog_count += 1
        if total_grupos > 0: prog_bar.progress(prog_count / total_grupos)
        
        atividades = grp.to_dict("records")
        if len(atividades) < 2: continue
        for i, base_act in enumerate(atividades): # Renomeado para clareza
            dup_map.setdefault(base_act["activity_id"], [])
            for comp_act in atividades[i + 1 :]: # Renomeado para clareza
                ratio = calcular_similaridade(base_act["Texto"], comp_act["Texto"])
                if ratio >= min_sim:
                    cor = obter_cor_similaridade(ratio)
                    ids_dup.update([base_act["activity_id"], comp_act["activity_id"]])
                    dup_map[base_act["activity_id"]].append(dict(id_similar=comp_act["activity_id"], ratio=ratio, cor=cor))
                    dup_map.setdefault(comp_act["activity_id"], []).append(dict(id_similar=base_act["activity_id"], ratio=ratio, cor=cor))
    
    prog_bar.empty(); prog_placeholder.text("Análise de similaridade concluída.")
    for k_map in dup_map: # Renomeado para clareza
        dup_map[k_map].sort(key=lambda x: x["ratio"], reverse=True)
    
    st.session_state[cache_key] = {"signature": signature, "map": dup_map, "ids_dup": ids_dup}
    return dup_map, ids_dup

# Links Zflow (como na sua versão)
link_z = lambda i: dict(
    antigo=f"https://zflow.zionbyonset.com.br/activity/3/details/{i}",
    novo  =f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={i}#/fixcol1"
)

# Estado inicial (como na sua versão, adaptado)
for key_base in ["show_text_dialog", "full_act_dialog", "comparacao_ativa", "pagina_atual"]:
    full_key = f"{key_base}{SUFFIX_STATE}"
    if full_key not in st.session_state:
        st.session_state[full_key] = False if "show" in key_base else (0 if "pagina" in key_base else None)

@st.dialog("Texto completo")
def dlg_full_text(): # Renomeado para clareza
    d = st.session_state[f"full_act_dialog{SUFFIX_STATE}"]
    if d is None: return
    data_fmt = d["activity_date"].strftime("%d/%m/%Y %H:%M") if pd.notna(d["activity_date"]) else "N/A"
    st.markdown(f"### ID {d['activity_id']} • {data_fmt}")
    st.text_area("Texto", d["Texto"], height=400, disabled=True, key=f"dlg_txt_content_{d['activity_id']}{SUFFIX_STATE}")
    if st.button("Fechar", key=f"dlg_txt_close_btn{SUFFIX_STATE}"):
        st.session_state[f"show_text_dialog{SUFFIX_STATE}"] = False; st.rerun()

# ==============================================================================
# APLICAÇÃO PRINCIPAL
# ==============================================================================
def app():
    st.sidebar.success(f"Logado como **{st.session_state['username']}**")
    if st.sidebar.button("Logout", key=f"logout_btn{SUFFIX_STATE}_app"): # Chave única
        st.session_state.clear(); st.rerun()

    eng = db_engine()
    if not eng: st.stop()

    if st.sidebar.button("🔄 Atualizar dados", key=f"update_data_btn{SUFFIX_STATE}"):
        carregar_dados.clear()
        # Limpar cache de similaridade também
        cache_key_sim = f"simcache{SUFFIX_STATE}"
        if cache_key_sim in st.session_state: del st.session_state[cache_key_sim]
        st.session_state[f"comparacao_ativa{SUFFIX_STATE}"] = None # Limpa comparação ativa
        st.toast("Dados base recarregados! Refaça a análise de similaridade se necessário.", icon="🔄")
        # O st.rerun é implícito após limpar o cache de carregar_dados

    df_raw_total, erro_db = carregar_dados(eng)
    if erro_db: st.exception(erro_db); st.stop()
    if df_raw_total.empty: st.warning("Sem atividades base carregadas."); st.stop()

    # --- Filtros de período ---
    st.sidebar.header("Período (Exibição)")
    hoje = date.today()
    data_inicio_padrao = hoje - timedelta(days=1)
    df_abertas_futuras = df_raw_total[(df_raw_total["activity_status"] == "Aberta") & (df_raw_total["activity_date"].notna()) & (df_raw_total["activity_date"].dt.date > hoje)]
    data_fim_padrao = df_abertas_futuras["activity_date"].dt.date.max() if not df_abertas_futuras.empty else hoje + timedelta(days=14)
    if data_inicio_padrao > data_fim_padrao: data_inicio_padrao = data_fim_padrao - timedelta(days=1) if data_fim_padrao > hoje else hoje - timedelta(days=1)
    
    d_ini = st.sidebar.date_input("Início", data_inicio_padrao, key=f"di{SUFFIX_STATE}")
    d_fim = st.sidebar.date_input("Fim", data_fim_padrao, min_value=d_ini, key=f"df{SUFFIX_STATE}")
    if d_ini > d_fim: st.sidebar.error("Início > fim."); st.stop()
    
    df_periodo = df_raw_total[(df_raw_total["activity_date"].notna()) & df_raw_total["activity_date"].dt.date.between(d_ini, d_fim)]
    st.success(f"{len(df_periodo)} atividades no período selecionado (de {len(df_raw_total)} na base).")

    # --- Filtros de análise ---
    st.sidebar.header("Filtros de Análise (sobre o período)")
    pastas_disp = sorted(df_periodo["activity_folder"].dropna().unique())
    pastas_sel_analise = st.sidebar.multiselect("Analisar Pasta(s):", pastas_disp, default=[], key=f"pasta_analise{SUFFIX_STATE}")
    status_disp_analise = sorted(df_periodo["activity_status"].dropna().unique())
    status_sel_analise = st.sidebar.multiselect("Analisar Status:", status_disp_analise, default=[], key=f"status_analise{SUFFIX_STATE}")
    
    df_para_analise_final = df_periodo.copy()
    if pastas_sel_analise: df_para_analise_final = df_para_analise_final[df_para_analise_final["activity_folder"].isin(pastas_sel_analise)]
    if status_sel_analise: df_para_analise_final = df_para_analise_final[df_para_analise_final["activity_status"].isin(status_sel_analise)]

    # --- Filtros de exibição ---
    st.sidebar.header("Filtros de Exibição (Pós-Análise)")
    min_sim_exib = st.sidebar.slider("Similaridade mínima (%):", 0, 100, 70, 5, key=f"sim_exib{SUFFIX_STATE}") / 100
    only_dup_exib = st.sidebar.checkbox("Somente com duplicatas", value=True, key=f"only_dup_exib{SUFFIX_STATE}")
    pastas_multi_exib = {p for p, g in df_para_analise_final.groupby("activity_folder") if g["user_profile_name"].nunique() > 1}
    only_multi_exib = st.sidebar.checkbox("Pastas com múltiplos responsáveis", False, key=f"only_multi_exib{SUFFIX_STATE}")
    usuarios_disp_exib = sorted(df_para_analise_final["user_profile_name"].dropna().unique())
    usuarios_sel_exib = st.sidebar.multiselect("Exibir Usuário(s):", usuarios_disp_exib, default=[], key=f"user_exib{SUFFIX_STATE}")

    # --- Cálculo de Similaridade (usando sua função cacheada) ---
    if not df_para_analise_final.empty and len(df_para_analise_final) > 1:
        map_id_para_similaridades, ids_com_duplicatas_calculados = get_similarity_map_cached(df_para_analise_final, min_sim_exib)
    else:
        map_id_para_similaridades, ids_com_duplicatas_calculados = {}, set()
    
    # --- DataFrame para Exibição Final ---
    df_exibir_final = df_para_analise_final.copy() # Começa com o que foi filtrado para análise
    if only_dup_exib: df_exibir_final = df_exibir_final[df_exibir_final["activity_id"].isin(ids_com_duplicatas_calculados)]
    if only_multi_exib: df_exibir_final = df_exibir_final[df_exibir_final["activity_folder"].isin(pastas_multi_exib)]
    if usuarios_sel_exib: df_exibir_final = df_exibir_final[df_exibir_final["user_profile_name"].isin(usuarios_sel_exib)]
    
    # --- Exportar para XLSX ---
    st.sidebar.markdown("---")
    if st.sidebar.button("📥 Exportar para XLSX", key=f"export_btn{SUFFIX_STATE}"):
        if df_exibir_final.empty: st.sidebar.warning("Nenhum dado para exportar.")
        else:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_exibir_final.to_excel(writer, index=False, sheet_name="Atividades_Exibidas")
                lista_export_dup = []
                for id_base, lista_sim in map_id_para_similaridades.items():
                    if id_base in df_exibir_final["activity_id"].values:
                        for sim_info in lista_sim:
                            det_sim_rows = df_raw_total[df_raw_total["activity_id"] == sim_info["id_similar"]]
                            if not det_sim_rows.empty:
                                det_sim = det_sim_rows.iloc[0]
                                data_str = det_sim["activity_date"].strftime("%Y-%m-%d %H:%M") if pd.notna(det_sim["activity_date"]) else None
                                lista_export_dup.append({
                                    "ID_Base": id_base, "ID_Duplicata": sim_info["id_similar"],
                                    "Similaridade": sim_info["ratio"], "Cor": sim_info["cor"],
                                    "Data_Dup": data_str, "Usuario_Dup": det_sim["user_profile_name"],
                                    "Status_Dup": det_sim["activity_status"],
                                })
                if lista_export_dup: pd.DataFrame(lista_export_dup).to_excel(writer, index=False, sheet_name="Detalhes_Duplicatas")
            st.sidebar.download_button("Baixar XLSX", output.getvalue(),f"duplicatas_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # --- Seção de Comparação Lado-a-Lado (direto na página) ---
    cmp_data = st.session_state.get(f"comparacao_ativa{SUFFIX_STATE}")
    if cmp_data:
        a_base, b_comp = cmp_data["base"], cmp_data["comp"]
        with st.container(border=True):
            st.subheader(f"🔎 Comparação: ID `{a_base['activity_id']}` vs `{b_comp['activity_id']}`")
            col1_cmp, col2_cmp = st.columns(2)
            html_a, html_b = highlight_common_words(a_base["Texto"], b_comp["Texto"])
            with col1_cmp: st.markdown(f"**ID {a_base['activity_id']}**<br>{html_a}", unsafe_allow_html=True)
            with col2_cmp: st.markdown(f"**ID {b_comp['activity_id']}**<br>{html_b}", unsafe_allow_html=True)
            if st.button("Ocultar Comparação", key=f"fechar_comp_btn{SUFFIX_STATE}"):
                fechar_comparacao_textos(); st.rerun()
        st.markdown("---")

    # --- Listagem por Pasta (com Paginação) ---
    st.header("Análise Detalhada por Pasta")
    if df_exibir_final.empty: st.info("Nenhuma atividade para os filtros de exibição selecionados.")

    pastas_ordenadas = sorted(df_exibir_final["activity_folder"].dropna().unique()) if not df_exibir_final.empty else []
    pagina_atual = st.session_state.get(f"pagina_atual{SUFFIX_STATE}", 0)

    if pastas_ordenadas:
        # (Lógica de paginação como na sua versão, adaptada para df_exibir_final)
        total_pastas_exibiveis = len(pastas_ordenadas)
        total_paginas = max(1, (total_pastas_exibiveis + ITENS_POR_PAGINA - 1) // ITENS_POR_PAGINA)
        pagina_atual = max(0, min(pagina_atual, total_paginas - 1)) # Garante que a página_atual é válida
        st.session_state[f"pagina_atual{SUFFIX_STATE}"] = pagina_atual

        if total_paginas > 1:
            cols_pag = st.columns([1,2,1]) # Renomeado para evitar conflito
            if cols_pag[0].button("⬅", disabled=(pagina_atual == 0), key=f"prev_btn{SUFFIX_STATE}"):
                st.session_state[f"pagina_atual{SUFFIX_STATE}"] -=1; st.rerun()
            cols_pag[1].markdown(f"<p style='text-align:center;'>Página {pagina_atual+1} / {total_paginas}</p>", unsafe_allow_html=True)
            if cols_pag[2].button("➡", disabled=(pagina_atual >= total_paginas -1), key=f"next_btn{SUFFIX_STATE}"):
                st.session_state[f"pagina_atual{SUFFIX_STATE}"] +=1; st.rerun()

        inicio_idx = pagina_atual * ITENS_POR_PAGINA
        fim_idx = inicio_idx + ITENS_POR_PAGINA
        pastas_para_render = pastas_ordenadas[inicio_idx:fim_idx] # Renomeado

        for nome_pasta_render in pastas_para_render: # Renomeado
            df_pasta_exib = df_exibir_final[df_exibir_final["activity_folder"] == nome_pasta_render] # Renomeado
            if df_pasta_exib.empty: continue
            
            total_analisado = len(df_para_analise_final[df_para_analise_final["activity_folder"] == nome_pasta_render]) # Renomeado
            titulo_exp = f"📁 {nome_pasta_render} ({len(df_pasta_exib)} exibidas / {total_analisado} analisadas)" # Renomeado
            
            with st.expander(titulo_exp, expanded=len(df_pasta_exib) < 10):
                for _, r_atividade in df_pasta_exib.iterrows(): # Renomeado
                    atividade_dict = r_atividade.to_dict() # Renomeado
                    st.markdown("---")
                    col_info_item, col_sim_item = st.columns([.6, .4]) # Renomeado
                    with col_info_item:
                        data_str_item = atividade_dict["activity_date"].strftime("%d/%m/%Y %H:%M") if pd.notna(atividade_dict["activity_date"]) else "N/A"
                        st.markdown(f"**ID** `{atividade_dict['activity_id']}` • {data_str_item} • `{atividade_dict['activity_status']}`")
                        st.markdown(f"**Usuário:** {atividade_dict['user_profile_name']}")
                        st.text_area("Texto", atividade_dict["Texto"], height=100, disabled=True, key=f"txt_item{SUFFIX_STATE}_{atividade_dict['activity_id']}")
                        
                        btns_item = st.columns(3) # Renomeado
                        links_item = link_z(atividade_dict["activity_id"]) # Renomeado
                        btns_item[0].button("👁 Completo", key=f"ver_item{SUFFIX_STATE}_{atividade_dict['activity_id']}", on_click=on_click_ver_texto_completo, args=(atividade_dict,))
                        btns_item[1].link_button("ZFlow v1", links_item["antigo"])
                        btns_item[2].link_button("ZFlow v2", links_item["novo"])
                    
                    with col_sim_item:
                        sims_item = map_id_para_similaridades.get(atividade_dict["activity_id"], []) # Renomeado
                        if sims_item:
                            st.markdown(f"**Duplicatas:** {len(sims_item)}")
                            for s_item in sims_item: # Renomeado
                                inf_item_rows = df_raw_total[df_raw_total["activity_id"] == s_item["id_similar"]] # Renomeado
                                if not inf_item_rows.empty:
                                    inf_item = inf_item_rows.iloc[0].to_dict() # Renomeado
                                    cont_dup_item = st.container(border=True) # Renomeado
                                    d_item_str = inf_item["activity_date"].strftime("%d/%m/%y %H:%M") if pd.notna(inf_item["activity_date"]) else "N/A"
                                    cont_dup_item.markdown(f"<div class='similarity-badge' style='background-color:{s_item['cor']};'>"
                                                           f"<b>ID: {inf_item['activity_id']}</b> ({s_item['ratio']:.0%})<br>"
                                                           f"{d_item_str} • {inf_item['activity_status']}<br>{inf_item['user_profile_name']}"
                                                           "</div>", unsafe_allow_html=True)
                                    cont_dup_item.button("⚖ Comparar", key=f"cmp_item{SUFFIX_STATE}_{atividade_dict['activity_id']}_{inf_item['activity_id']}", on_click=on_click_comparar_textos, args=(atividade_dict, inf_item))
                        elif not only_dup_exib: # Mostrar "Sem duplicatas" apenas se o filtro não estiver ativo
                            st.markdown("<span style='color:green;'>Sem duplicatas (nesta análise)</span>", unsafe_allow_html=True)


    if st.session_state.get(f"show_text_dialog{SUFFIX_STATE}"): dlg_full_text() # Renomeado para corresponder

# ==============================================================================
# LOGIN (como na sua versão)
# ==============================================================================
def check_credentials(username, password): # Renomeado parâmetros para clareza
    try:
        user_creds = st.secrets.get("credentials", {}).get("usernames", {})
        return username in user_creds and str(user_creds[username]) == password
    except Exception: # Captura genérica para erros de secrets
        return False

def login_form():
    st.header("Login")
    with st.form(f"login_form{SUFFIX_STATE}"): # Chave única
        u_login = st.text_input("Usuário", key=f"user_login_input{SUFFIX_STATE}") # Renomeado
        p_login = st.text_input("Senha", type="password", key=f"pass_login_input{SUFFIX_STATE}") # Renomeado
        if st.form_submit_button("Entrar"):
            if check_credentials(u_login, p_login):
                st.session_state.update({"logged_in": True, "username": u_login}); st.rerun()
            else: st.error("Credenciais inválidas.")

# ==============================================================================
# MAIN (como na sua versão)
# ==============================================================================
if __name__ == "__main__":
    if not st.session_state.get("logged_in"): st.session_state["logged_in"] = False
    if st.session_state["logged_in"]: app()
    else: login_form()
