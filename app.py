import streamlit as st
import pandas as pd
import re
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from datetime import datetime, timedelta, date
from unidecode import unidecode # Para remover acentos
from rapidfuzz import fuzz # Para cálculo de similaridade
import io # Para exportação XLSX
import html # Para escapar caracteres HTML na função de highlight
# difflib não está sendo usado na sua última versão fornecida, removido para limpeza
# import difflib 

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================
SUFFIX_STATE = "_v_timestamp_exp" # Sufixo atualizado para esta versão
ITENS_POR_PAGINA = 20 
HIGHLIGHT_COLOR = "#a8d1ff" 

st.set_page_config(layout="wide", page_title="Verificador de Duplicidade (Atualizado)")

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
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def calcular_similaridade(texto_a: str, texto_b: str) -> float:
    norm_a, norm_b = normalizar_texto(texto_a), normalizar_texto(texto_b)
    if not norm_a or not norm_b: return 0.0
    if abs(len(norm_a) - len(norm_b)) > 0.5 * max(len(norm_a), len(norm_b)) and min(len(norm_a),len(norm_b)) > 0:
        return 0.0
    return fuzz.token_set_ratio(norm_a, norm_b) / 100.0

def obter_cor_similaridade(ratio: float) -> str:
    LIMIAR_ALTA, LIMIAR_MEDIA = 0.90, 0.70
    CORES = {"alta": "#FF5252", "media": "#FFB74D", "baixa": "#FFD54F"}
    if ratio >= LIMIAR_ALTA: return CORES["alta"]
    if ratio >= LIMIAR_MEDIA: return CORES["media"]
    return CORES["baixa"]

def highlight_common_words(text1: str, text2: str, min_len: int = 3) -> tuple[str, str]:
    norm_text1_tokens = re.findall(r"\w+", normalizar_texto(text1))
    norm_text2_tokens = re.findall(r"\w+", normalizar_texto(text2))
    comuns_normalizadas = {w for w in norm_text1_tokens if w in norm_text2_tokens and len(w) >= min_len}

    def _wrap_text_with_highlights(original_text: str) -> str:
        parts = re.split(r"(\W+)", original_text)
        highlighted_parts = []
        for part in parts:
            if not part: continue
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
    if not all([h, u, p, n]): st.error("Credenciais do banco ausentes."); return None
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
            return df_ret, None
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
        df["Texto"] = df["Texto"].astype(str).fillna("")
        df = df.sort_values(["activity_id", "activity_status"], ascending=[True, True]).drop_duplicates("activity_id", keep="first")
        df = df.sort_values(["activity_folder", "activity_date", "activity_id"], ascending=[True, False, False]).reset_index(drop=True)
        return df, None
    except exc.SQLAlchemyError as e: return None, e

def get_similarity_map_cached(df: pd.DataFrame, min_sim: float):
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
        for i, base_act in enumerate(atividades):
            dup_map.setdefault(base_act["activity_id"], [])
            for comp_act in atividades[i + 1 :]:
                ratio = calcular_similaridade(base_act["Texto"], comp_act["Texto"])
                if ratio >= min_sim:
                    cor = obter_cor_similaridade(ratio)
                    ids_dup.update([base_act["activity_id"], comp_act["activity_id"]])
                    dup_map[base_act["activity_id"]].append(dict(id_similar=comp_act["activity_id"], ratio=ratio, cor=cor))
                    dup_map.setdefault(comp_act["activity_id"], []).append(dict(id_similar=base_act["activity_id"], ratio=ratio, cor=cor))
    prog_bar.empty(); prog_placeholder.text("Análise de similaridade concluída.")
    for k_map in dup_map: dup_map[k_map].sort(key=lambda x: x["ratio"], reverse=True)
    st.session_state[cache_key] = {"signature": signature, "map": dup_map, "ids_dup": ids_dup}
    return dup_map, ids_dup

link_z = lambda i: dict(antigo=f"https://zflow.zionbyonset.com.br/activity/3/details/{i}", novo =f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={i}#/fixcol1")

# ==============================================================================
# ESTADO INICIAL E FUNÇÕES DE UI PARA DIALOGS E COMPARAÇÃO
# ==============================================================================
for key_base in ["show_text_dialog", "full_act_dialog", "comparacao_ativa", "pagina_atual", "last_db_update_time", "force_timestamp_update"]:
    full_key = f"{key_base}{SUFFIX_STATE}"
    if full_key not in st.session_state:
        st.session_state[full_key] = False if "show" in key_base else (0 if "pagina" in key_base else None)

@st.dialog("Texto completo")
def dlg_full_text():
    d = st.session_state[f"full_act_dialog{SUFFIX_STATE}"]
    if d is None: return
    data_fmt = d["activity_date"].strftime("%d/%m/%Y %H:%M") if pd.notna(d["activity_date"]) else "N/A"
    st.markdown(f"### ID {d['activity_id']} • {data_fmt}")
    st.text_area("Texto", d["Texto"], height=400, disabled=True, key=f"dlg_txt_content_{d['activity_id']}{SUFFIX_STATE}")
    if st.button("Fechar", key=f"dlg_txt_close_btn{SUFFIX_STATE}"):
        st.session_state[f"show_text_dialog{SUFFIX_STATE}"] = False; st.rerun()

def on_click_ver_texto_completo(atividade):
    st.session_state[f"full_act_dialog{SUFFIX_STATE}"] = atividade # Corrigido de 'atividade_para_texto_dialog'
    st.session_state[f"show_text_dialog{SUFFIX_STATE}"] = True

def on_click_comparar_textos(atividade_base, atividade_comparar):
    st.session_state[f"comparacao_ativa{SUFFIX_STATE}"] = {"base": atividade_base, "comp": atividade_comparar} # Corrigido 'comp'

def fechar_comparacao_textos():
    st.session_state[f"comparacao_ativa{SUFFIX_STATE}"] = None
# ==============================================================================
# APLICAÇÃO PRINCIPAL
# ==============================================================================
def app():
    st.sidebar.success(f"Logado como **{st.session_state['username']}**")
    if st.sidebar.button("Logout", key=f"logout_btn{SUFFIX_STATE}_app"):
        st.session_state.clear(); st.rerun()

    eng = db_engine()
    if not eng: st.stop()

    if st.sidebar.button("🔄 Atualizar Dados Base", key=f"update_data_btn{SUFFIX_STATE}"):
        carregar_dados.clear()
        cache_key_sim = f"simcache{SUFFIX_STATE}"
        if cache_key_sim in st.session_state: del st.session_state[cache_key_sim]
        st.session_state[f"comparacao_ativa{SUFFIX_STATE}"] = None
        st.session_state[f"force_timestamp_update{SUFFIX_STATE}"] = True # Sinaliza para atualizar o timestamp
        st.toast("Dados base sendo recarregados...", icon="🔄")
        # O rerun é implícito

    df_raw_total, erro_db = carregar_dados(eng)
    if erro_db: st.exception(erro_db); st.stop()
    if df_raw_total.empty: st.warning("Sem atividades base carregadas."); st.stop()

    # Atualizar timestamp da última carga
    if st.session_state.get(f"force_timestamp_update{SUFFIX_STATE}", False) or st.session_state[f"last_db_update_time{SUFFIX_STATE}"] is None:
        if not erro_db and not df_raw_total.empty: # Só atualiza se a carga foi bem sucedida
            st.session_state[f"last_db_update_time{SUFFIX_STATE}"] = datetime.now()
            st.session_state[f"force_timestamp_update{SUFFIX_STATE}"] = False
    
    last_update_time = st.session_state[f"last_db_update_time{SUFFIX_STATE}"]
    if last_update_time:
        st.sidebar.caption(f"Dados carregados do banco em: {last_update_time.strftime('%d/%m/%Y %H:%M:%S')}")
    else:
        st.sidebar.caption("Dados ainda não carregados ou falha na carga.")


    st.sidebar.header("Período (Exibição)") # Movido para antes dos seletores de data
    hoje = date.today()
    data_inicio_padrao = hoje - timedelta(days=1)
    df_abertas_futuras = df_raw_total[(df_raw_total["activity_status"] == "Aberta") & (df_raw_total["activity_date"].notna()) & (df_raw_total["activity_date"].dt.date > hoje)]
    data_fim_padrao = df_abertas_futuras["activity_date"].dt.date.max() if not df_abertas_futuras.empty else hoje + timedelta(days=14)
    if data_inicio_padrao > data_fim_padrao: data_inicio_padrao = data_fim_padrao - timedelta(days=1) if data_fim_padrao > hoje else hoje - timedelta(days=1)
    
    d_ini = st.sidebar.date_input("Início", data_inicio_padrao, key=f"di{SUFFIX_STATE}")
    d_fim = st.sidebar.date_input("Fim", data_fim_padrao, min_value=d_ini, key=f"df{SUFFIX_STATE}")
    if d_ini > d_fim: st.sidebar.error("Início > fim."); st.stop()
    
    df_periodo = df_raw_total[(df_raw_total["activity_date"].notna()) & df_raw_total["activity_date"].dt.date.between(d_ini, d_fim)]
    st.title(f"🔎 Verificador de Duplicidade ({len(df_periodo)} atividades no período)") # Título dinâmico

    st.sidebar.header("Filtros de Análise")
    pastas_disp = sorted(df_periodo["activity_folder"].dropna().unique().tolist())
    pastas_sel = st.sidebar.multiselect("Pastas", pastas_disp, key=f"psel{SUFFIX_STATE}") # Chave simplificada
    status_disp = sorted(df_periodo["activity_status"].dropna().unique().tolist()) # Renomeado
    status_sel = st.sidebar.multiselect("Status", status_disp, key=f"ssel{SUFFIX_STATE}") # Chave simplificada

    df_a = df_periodo.copy()
    if pastas_sel: df_a = df_a[df_a["activity_folder"].isin(pastas_sel)]
    if status_sel: df_a = df_a[df_a["activity_status"].isin(status_sel)]

    st.sidebar.header("Exibição")
    min_sim_exib = st.sidebar.slider("Similaridade mínima (%)", 0, 100, 70, 5, key=f"sim_exib{SUFFIX_STATE}") / 100
    only_dup = st.sidebar.checkbox("Somente com duplicatas", True, key=f"only_dup{SUFFIX_STATE}")
    pastas_multi_exib = {p for p, g in df_a.groupby("activity_folder") if g["user_profile_name"].nunique() > 1} if not df_a.empty else set() # Baseado em df_a
    only_multi = st.sidebar.checkbox("Pastas com múltiplos responsáveis", False, key=f"only_multi{SUFFIX_STATE}")
    
    # Similaridade calculada sobre df_a (filtrado por período, pasta de análise, status de análise)
    if not df_a.empty and len(df_a) > 1:
        map_id_para_similaridades, ids_com_duplicatas_calculados = get_similarity_map_cached(df_a, min_sim_exib)
    else:
        map_id_para_similaridades, ids_com_duplicatas_calculados = {}, set()
    
    df_v = df_a.copy() # Começa com df_a para aplicar filtros de exibição
    if only_dup: df_v = df_v[df_v["activity_id"].isin(ids_com_duplicatas_calculados)]
    if only_multi: df_v = df_v[df_v["activity_folder"].isin(pastas_multi_exib)]
    # Filtro de usuário para exibição (faltava na sua versão, adicionado aqui)
    usuarios_disp_final = sorted(df_v["user_profile_name"].dropna().unique()) if not df_v.empty else []
    usuarios_sel_final = st.sidebar.multiselect("Exibir Usuário(s) (final):", usuarios_disp_final, default=[], key=f"user_final{SUFFIX_STATE}")
    if usuarios_sel_final: df_v = df_v[df_v["user_profile_name"].isin(usuarios_sel_final)]


    st.sidebar.markdown("---")
    if st.sidebar.button("📥 Exportar para XLSX", key=f"export_btn{SUFFIX_STATE}"):
        # ... (Lógica de exportação como antes, usando df_v e map_id_para_similaridades) ...
        if df_v.empty: st.sidebar.warning("Nenhum dado para exportar.")
        else:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_v.to_excel(writer, index=False, sheet_name="Atividades_Exibidas") # Exporta df_v
                lista_export_dup = []
                # Usar map_id_para_similaridades que foi calculado sobre df_a
                for id_base_export, lista_similares_export in map_id_para_similaridades.items():
                    if id_base_export in df_v["activity_id"].values: # Apenas para atividades que estão em df_v
                        for sim_info_export in lista_similares_export:
                            # Buscar detalhes da duplicata no df_raw_total para garantir que temos a info completa
                            detalhes_similar_export_rows = df_raw_total[df_raw_total["activity_id"] == sim_info_export["id_similar"]]
                            if not detalhes_similar_export_rows.empty:
                                detalhes_similar_export = detalhes_similar_export_rows.iloc[0]
                                data_dup_exp_str = (detalhes_similar_export["activity_date"].strftime("%Y-%m-%d %H:%M") if pd.notna(detalhes_similar_export["activity_date"]) else None)
                                lista_export_dup.append({
                                    "ID_Base": id_base_export, "ID_Duplicata": sim_info_export["id_similar"], # Corrigido nome da coluna
                                    "Similaridade": sim_info_export["ratio"],"Cor": sim_info_export["cor"],
                                    "Data_Dup": data_dup_exp_str, "Usuario_Dup": detalhes_similar_export["user_profile_name"],
                                    "Status_Dup": detalhes_similar_export["activity_status"],
                                })
                if lista_export_dup: pd.DataFrame(lista_export_dup).to_excel(writer, index=False, sheet_name="Detalhes_Duplicatas")
            st.sidebar.download_button("Baixar XLSX", output.getvalue(),f"duplicatas_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


    cmp_data = st.session_state.get(f"comparacao_ativa{SUFFIX_STATE}")
    if cmp_data:
        a_base, b_comp = cmp_data["base"], cmp_data["comp"]
        with st.container(border=True):
            # Encontrar o ratio para exibir no título
            ratio_atual = 0
            if a_base["activity_id"] in map_id_para_similaridades:
                for sim_entry in map_id_para_similaridades[a_base["activity_id"]]:
                    if sim_entry["id_similar"] == b_comp["activity_id"]:
                        ratio_atual = sim_entry["ratio"]
                        break
            st.subheader(f"🔎 Comparação: ID `{a_base['activity_id']}` vs `{b_comp['activity_id']}` (Similaridade: {ratio_atual:.0%})")
            col1_cmp, col2_cmp = st.columns(2)
            html_a, html_b = highlight_common_words(a_base["Texto"], b_comp["Texto"])
            with col1_cmp: st.markdown(f"**ID {a_base['activity_id']} (Base)**<br>{html_a}", unsafe_allow_html=True)
            with col2_cmp: st.markdown(f"**ID {b_comp['activity_id']} (Similar)**<br>{html_b}", unsafe_allow_html=True)
            if st.button("Ocultar Comparação", key=f"fechar_comp{SUFFIX_STATE}"):
                fechar_comparacao_textos(); st.rerun()
        st.markdown("---")

    st.header("Análise Detalhada por Pasta")
    if df_v.empty: st.info("Nenhuma atividade para os filtros de exibição selecionados.")

    pastas_ord = sorted(df_v["activity_folder"].dropna().unique().tolist())
    page = st.session_state.get(f"page{SUFFIX_STATE}", 0) # Usar .get para segurança

    if pastas_ord:
        # ... (Lógica de paginação como na sua versão) ...
        total_pastas_exibiveis = len(pastas_ord)
        total_paginas = max(1, (total_pastas_exibiveis + ITENS_POR_PAGINA - 1) // ITENS_POR_PAGINA)
        page = max(0, min(page, total_paginas - 1)) 
        st.session_state[f"page{SUFFIX_STATE}"] = page
        if total_paginas > 1:
            b1, _, b2 = st.columns([1, 2, 1]) # Ajuste de colunas para paginação
            if b1.button("⬅", disabled=page == 0, key=f"prev_btn{SUFFIX_STATE}"): # Corrigido nome
                st.session_state[f"page{SUFFIX_STATE}"] -=1; st.rerun()
            st.markdown(f"<p style='text-align:center;'>Página {page+1}/{total_paginas}</p>", unsafe_allow_html=True)
            if b2.button("➡", disabled=(page >= total_paginas -1), key=f"next_btn{SUFFIX_STATE}"): # Corrigido nome
                st.session_state[f"page{SUFFIX_STATE}"] +=1; st.rerun()

        ini, fim = page * ITENS_POR_PAGINA, (page + 1) * ITENS_POR_PAGINA
        for pasta_nome_loop in pastas_ord[ini:fim]: # Renomeado para evitar conflito
            df_p_loop = df_v[df_v["activity_folder"] == pasta_nome_loop] # Renomeado
            if df_p_loop.empty: continue
            
            # Recalcular total_analisado_pasta com base em df_para_analise_final (que foi usado para similaridade)
            total_analisado_pasta_loop = len(df_para_analise_final[df_para_analise_final["activity_folder"] == pasta_nome_loop]) # Renomeado
            titulo_exp_loop = f"📁 {pasta_nome_loop} ({len(df_p_loop)} exibidas / {total_analisado_pasta_loop} analisadas)" # Renomeado
            
            with st.expander(titulo_exp_loop, expanded=False): # ALTERADO PARA expanded=False
                for _, r_item_loop in df_p_loop.iterrows(): # Renomeado
                    atividade_item_dict = r_item_loop.to_dict() # Renomeado
                    st.markdown("---")
                    col_info_render, col_sim_render = st.columns([.6, .4]) # Renomeado
                    with col_info_render:
                        data_render_str = atividade_item_dict["activity_date"].strftime("%d/%m/%Y %H:%M") if pd.notna(atividade_item_dict["activity_date"]) else "N/A"
                        st.markdown(f"**ID** `{atividade_item_dict['activity_id']}` • {data_render_str} • `{atividade_item_dict['activity_status']}`")
                        st.markdown(f"**Usuário:** {atividade_item_dict['user_profile_name']}")
                        st.text_area("Texto", atividade_item_dict["Texto"], height=100, disabled=True, key=f"txt_render{SUFFIX_STATE}_{atividade_item_dict['activity_id']}")
                        btns_render = st.columns(3) # Renomeado
                        links_render = link_z(atividade_item_dict["activity_id"]) # Renomeado
                        btns_render[0].button("👁 Completo", key=f"ver_render{SUFFIX_STATE}_{atividade_item_dict['activity_id']}", on_click=on_click_ver_texto_completo, args=(atividade_item_dict,))
                        btns_render[1].link_button("ZFlow v1", links_render["antigo"])
                        btns_render[2].link_button("ZFlow v2", links_render["novo"])
                    with col_sim_render:
                        sims_render = map_id_para_similaridades.get(atividade_item_dict["activity_id"], []) # Renomeado
                        if sims_render:
                            st.markdown(f"**Duplicatas:** {len(sims_render)}")
                            for s_render in sims_render: # Renomeado
                                inf_render_rows = df_raw_total[df_raw_total["activity_id"] == s_render["id_similar"]] # Renomeado
                                if not inf_render_rows.empty:
                                    inf_render = inf_render_rows.iloc[0].to_dict() # Renomeado
                                    cont_dup_render = st.container(border=True) # Renomeado
                                    d_render_str = inf_render["activity_date"].strftime("%d/%m/%y %H:%M") if pd.notna(inf_render["activity_date"]) else "N/A"
                                    cont_dup_render.markdown(f"<div class='similarity-badge' style='background-color:{s_render['cor']};'>" # Usando a classe CSS
                                                           f"<b>ID: {inf_render['activity_id']}</b> ({s_render['ratio']:.0%})<br>"
                                                           f"{d_render_str} • {inf_render['activity_status']}<br>{inf_render['user_profile_name']}"
                                                           "</div>", unsafe_allow_html=True)
                                    cont_dup_render.button("⚖ Comparar", key=f"cmp_render{SUFFIX_STATE}_{atividade_item_dict['activity_id']}_{inf_render['activity_id']}", on_click=on_click_comparar_textos, args=(atividade_item_dict, inf_render))
                        elif not only_dup: st.markdown("<span style='color:green;'>Sem duplicatas</span>", unsafe_allow_html=True)

    if st.session_state.get(f"show_text_dialog{SUFFIX_STATE}"): dlg_full_text()

# ==============================================================================
# LOGIN (como na sua versão)
# ==============================================================================
def check_credentials(username, password): # Renomeado para clareza
    try:
        user_creds_login = st.secrets.get("credentials", {}).get("usernames", {}) # Renomeado
        return username in user_creds_login and str(user_creds_login[username]) == password
    except Exception: return False # Mais genérico para erros de secrets

def login_form():
    st.header("Login")
    with st.form(f"login_f{SUFFIX_STATE}"): # Renomeado
        u_form = st.text_input("Usuário", key=f"u_form_in{SUFFIX_STATE}") # Renomeado
        p_form = st.text_input("Senha", type="password", key=f"p_form_in{SUFFIX_STATE}") # Renomeado
        if st.form_submit_button("Entrar"):
            if check_credentials(u_form, p_form):
                st.session_state.update({"logged_in": True, "username": u_form}); st.rerun()
            else: st.error("Credenciais inválidas.")

# ==============================================================================
# MAIN (como na sua versão)
# ==============================================================================
if __name__ == "__main__":
    if not st.session_state.get("logged_in"): st.session_state["logged_in"] = False
    if st.session_state["logged_in"]: app()
    else: login_form()
