import streamlit as st
import pandas as pd
import re
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine # Importar Engine para hash_funcs
from datetime import datetime, timedelta, date 
from unidecode import unidecode
from rapidfuzz import fuzz
import io
import difflib 

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================
st.set_page_config(layout="wide", page_title="Verificador de Duplicidade (Estável)")

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================
def normalizar_texto(txt: str | None) -> str:
    if not txt or not isinstance(txt, str): return ""
    txt = unidecode(txt.lower())
    txt = re.sub(r'[^\w\s]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

def calcular_similaridade(texto_a: str, texto_b: str) -> float:
    norm_a = normalizar_texto(texto_a)
    norm_b = normalizar_texto(texto_b)
    if not norm_a or not norm_b: return 0.0
    if abs(len(norm_a) - len(norm_b)) > 0.3 * max(len(norm_a), len(norm_b)):
        return 0.0
    return fuzz.token_set_ratio(norm_a, norm_b) / 100.0

def obter_cor_similaridade(ratio: float) -> str:
    LIMIAR_ALTA, LIMIAR_MEDIA = 0.90, 0.70
    CORES = {'alta': '#FF5252', 'media': '#FFB74D', 'baixa': '#FFD54F'}
    if ratio >= LIMIAR_ALTA: return CORES['alta']
    if ratio >= LIMIAR_MEDIA: return CORES['media']
    return CORES['baixa']

def gerar_links_zflow(activity_id: int) -> dict:
    return {
        "antigo": f"https://zflow.zionbyonset.com.br/activity/3/details/{activity_id}",
        "novo": f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={activity_id}#/fixcol1"
    }

@st.cache_resource
def get_db_engine() -> Engine | None:
    try:
        db_host = st.secrets["database"]["host"]
        db_user = st.secrets["database"]["user"]
        db_pass = st.secrets["database"]["password"]
        db_name = st.secrets["database"]["name"]
    except KeyError:
        st.warning("Credenciais do banco não encontradas em st.secrets. Usando fallback local.")
        db_user, db_pass, db_host, db_name = "tarcisio", "123qwe", "40.88.40.110", "zion_flow"
        if not all([db_host, db_user, db_pass, db_name]):
             st.error("Credenciais do banco não definidas.")
             return None
    db_uri = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_name}"
    try:
        engine = create_engine(db_uri, pool_pre_ping=True, pool_recycle=3600)
        with engine.connect(): pass
        return engine
    except exc.SQLAlchemyError:
        return None 

@st.cache_data(ttl=3600, hash_funcs={Engine: lambda _: None})
def carregar_dados(eng_param: Engine) -> tuple[pd.DataFrame | None, Exception | None]:
    hoje_dt = date.today()
    data_limite_historico = hoje_dt - timedelta(days=7)
    query_abertas = text("""
        SELECT activity_id, activity_folder, user_profile_name, activity_date, activity_status, Texto, activity_type
        FROM ViewGrdAtividadesTarcisio WHERE activity_type = 'Verificar' AND activity_status = 'Aberta'
    """)
    query_historico = text("""
        SELECT activity_id, activity_folder, user_profile_name, activity_date, activity_status, Texto, activity_type
        FROM ViewGrdAtividadesTarcisio WHERE activity_type = 'Verificar' AND DATE(activity_date) >= :data_limite
    """)
    try:
        with eng_param.connect() as connection:
            df_abertas = pd.read_sql(query_abertas, connection)
            df_historico = pd.read_sql(query_historico, connection, params={"data_limite": data_limite_historico})
        
        df_combinado = pd.concat([df_abertas, df_historico], ignore_index=True)

        if df_combinado.empty:
            # Retorna um DataFrame vazio com as colunas e tipos corretos se não houver dados
            cols = ['activity_id', 'activity_folder', 'user_profile_name', 'activity_date', 'activity_status', 'Texto', 'activity_type']
            df_final = pd.DataFrame(columns=cols)
            df_final['activity_date'] = pd.Series(dtype='datetime64[ns]')
            df_final['Texto'] = pd.Series(dtype='object')
            # Garanta que outras colunas importantes tenham tipos definidos se necessário
            for col in ['activity_id', 'user_profile_name', 'activity_folder', 'activity_status', 'activity_type']:
                if col not in df_final.columns: # Adicionado para garantir que a coluna existe
                    df_final[col] = pd.Series(dtype='object') # Ou o tipo apropriado
            df_final['Texto'] = df_final['Texto'].astype(str).fillna('')
            return df_final, None

        df_combinado_sorted = df_combinado.sort_values(by=['activity_id', 'activity_status'], ascending=[True, True])
        df_final_temp = df_combinado_sorted.drop_duplicates(subset=['activity_id'], keep='first')
        
        df_final = df_final_temp.sort_values(by=['activity_folder', 'activity_date', 'activity_id'], ascending=[True, False, False]).copy()
        
        # --- CORREÇÃO APLICADA AQUI ---
        # Forçar a conversão para datetime, transformando erros em NaT (Not a Time)
        df_final['activity_date'] = pd.to_datetime(df_final['activity_date'], errors='coerce')
        df_final['Texto'] = df_final['Texto'].astype(str).fillna('')
        
        return df_final, None
    except exc.SQLAlchemyError as e:
        return None, e

# ==============================================================================
# Estado da Sessão para Dialogs
# ==============================================================================
SUFFIX_DIALOG = "_dialog_datetime_fix" 
if f'show_texto_dialog{SUFFIX_DIALOG}' not in st.session_state:
    st.session_state[f'show_texto_dialog{SUFFIX_DIALOG}'] = False
if f'atividade_para_texto_dialog{SUFFIX_DIALOG}' not in st.session_state:
    st.session_state[f'atividade_para_texto_dialog{SUFFIX_DIALOG}'] = None

if f'show_comparacao_dialog{SUFFIX_DIALOG}' not in st.session_state:
    st.session_state[f'show_comparacao_dialog{SUFFIX_DIALOG}'] = False
if f'atividades_para_comparacao{SUFFIX_DIALOG}' not in st.session_state:
    st.session_state[f'atividades_para_comparacao{SUFFIX_DIALOG}'] = None 

# ==============================================================================
# Funções Decoradas com @st.dialog
# ==============================================================================
@st.dialog("Texto Completo da Atividade")
def mostrar_texto_completo_dialog():
    atividade_data = st.session_state[f'atividade_para_texto_dialog{SUFFIX_DIALOG}']
    if atividade_data:
        # Verifica se 'activity_date' é um objeto datetime antes de formatar
        data_formatada = atividade_data['activity_date'].strftime('%d/%m/%Y %H:%M') if pd.notna(atividade_data['activity_date']) and isinstance(atividade_data['activity_date'], (datetime, pd.Timestamp)) else "Data Inválida"
        st.markdown(f"### Texto Completo - ID: `{atividade_data['activity_id']}`")
        st.markdown(f"**Pasta:** {atividade_data['activity_folder']} | **Data:** {data_formatada} | **Usuário:** {atividade_data['user_profile_name']} | **Status:** {atividade_data['activity_status']}")
        st.text_area("Texto:", value=str(atividade_data['Texto']), height=400, disabled=True, key=f"dialog_txt_content{SUFFIX_DIALOG}_{atividade_data['activity_id']}")
        if st.button("Fechar Texto", key=f"fechar_dialog_txt_btn{SUFFIX_DIALOG}"):
            st.session_state[f'show_texto_dialog{SUFFIX_DIALOG}'] = False; st.rerun()

@st.dialog("Comparação Detalhada de Textos", width="large")
def mostrar_comparacao_html_diff_dialog():
    atividades_comp_data = st.session_state[f'atividades_para_comparacao{SUFFIX_DIALOG}']
    if atividades_comp_data:
        base_comp = atividades_comp_data['base']
        comparar_comp = atividades_comp_data['comparar']
        st.markdown(f"### Comparando ID `{base_comp['activity_id']}` com ID `{comparar_comp['activity_id']}`")
        texto_base_comp = str(base_comp['Texto'])
        texto_comparar_comp = str(comparar_comp['Texto'])
        differ = difflib.HtmlDiff(wrapcolumn=65, linejunk=difflib.IS_LINE_JUNK, charjunk=difflib.IS_CHARACTER_JUNK) 
        html_comparison = differ.make_table(texto_base_comp.splitlines(), texto_comparar_comp.splitlines(),
                                             fromdesc=f"ID: {base_comp['activity_id']}", 
                                             todesc=f"ID: {comparar_comp['activity_id']}")
        st.components.v1.html(html_comparison, height=550, scrolling=True)
        if st.button("Fechar Comparação", key=f"fechar_dialog_html_comp_btn{SUFFIX_DIALOG}"):
            st.session_state[f'show_comparacao_dialog{SUFFIX_DIALOG}'] = False; st.rerun()

# ==============================================================================
# Funções para abrir os dialogs
# ==============================================================================
def on_click_ver_texto_completo(atividade):
    st.session_state[f'atividade_para_texto_dialog{SUFFIX_DIALOG}'] = atividade
    st.session_state[f'show_texto_dialog{SUFFIX_DIALOG}'] = True

def on_click_comparar_textos_html_dialog(atividade_base, atividade_comparar):
    st.session_state[f'atividades_para_comparacao{SUFFIX_DIALOG}'] = {'base': atividade_base, 'comparar': atividade_comparar}
    st.session_state[f'show_comparacao_dialog{SUFFIX_DIALOG}'] = True

# ==============================================================================
# INTERFACE PRINCIPAL DO APP (RENOMEADA PARA app)
# ==============================================================================
def app(): 
    st.sidebar.success(f"Logado como: **{st.session_state['username']}**")
    if st.sidebar.button("Logout", key=f"logout_btn{SUFFIX_DIALOG}"): 
        for key_state in list(st.session_state.keys()): del st.session_state[key_state]
        st.rerun()

    st.title("🔎 Verificador de Duplicidade (HtmlDiff no Dialog)")
    st.markdown("Análise de atividades 'Verificar' para identificar potenciais duplicidades.")

    eng = get_db_engine() 
    if not eng: st.error("Falha crítica na conexão com o banco."); st.stop()

    st.sidebar.header("⚙️ Filtros e Opções")
    
    if st.sidebar.button("🔄 Atualizar Dados Base", help="Busca os dados mais recentes.", key=f"buscar_btn_base{SUFFIX_DIALOG}"):
        carregar_dados.clear(); st.toast("Buscando dados atualizados...", icon="🔄")
    
    df_raw_total, erro_db = carregar_dados(eng) 

    if erro_db: st.error("Erro ao carregar dados."); st.exception(erro_db); st.stop()
    if df_raw_total is None or df_raw_total.empty: st.warning("Nenhuma atividade 'Verificar' retornada."); st.stop()

    st.sidebar.markdown("---"); st.sidebar.subheader("1. Filtro de Período (Exibição)")
    hoje_ref = date.today() # Usar date.today()
    
    # Garantir que activity_date é datetime antes de usar .dt
    if 'activity_date' not in df_raw_total.columns or not pd.api.types.is_datetime64_any_dtype(df_raw_total['activity_date']):
        st.error("Coluna 'activity_date' não está no formato datetime esperado após o carregamento.")
        st.stop()
        
    data_inicio_padrao = hoje_ref - timedelta(days=1)
    # Filtrar por NaT antes de aplicar .dt.date e depois .max()
    datas_abertas_futuras_series = df_raw_total[
        (df_raw_total['activity_status'] == 'Aberta') & 
        (df_raw_total['activity_date'].notna()) & # Checa se não é NaT
        (df_raw_total['activity_date'].dt.date > hoje_ref)
    ]['activity_date'].dt.date
    
    data_fim_padrao = datas_abertas_futuras_series.max() if not datas_abertas_futuras_series.empty else hoje_ref + timedelta(days=14)
    if data_inicio_padrao > data_fim_padrao: data_inicio_padrao = data_fim_padrao - timedelta(days=1)

    data_inicio_selecionada = st.sidebar.date_input("Data de Início (Exibição)", value=data_inicio_padrao, key=f"di_exib{SUFFIX_DIALOG}")
    data_fim_selecionada = st.sidebar.date_input("Data de Fim (Exibição)", value=data_fim_padrao, key=f"df_exib{SUFFIX_DIALOG}")

    if data_inicio_selecionada > data_fim_selecionada: st.sidebar.error("Data de início > data de fim."); st.stop()
    
    df_atividades_periodo_ui = df_raw_total[
        (df_raw_total['activity_date'].notna()) & # Adicionado para segurança
        (df_raw_total['activity_date'].dt.date >= data_inicio_selecionada) & 
        (df_raw_total['activity_date'].dt.date <= data_fim_selecionada)
    ]

    if df_atividades_periodo_ui.empty: 
        st.info(f"Nenhuma atividade para o período de exibição de {data_inicio_selecionada.strftime('%d/%m/%Y')} a {data_fim_selecionada.strftime('%d/%m/%Y')}.")
    else: 
        st.success(f"**{len(df_atividades_periodo_ui)}** atividades no período de exibição (de {len(df_raw_total)} total carregado na base).")
    
    # O restante do código continua aqui, usando df_atividades_periodo_ui como base para mais filtros e análise
    # ... (filtros de análise, filtros de exibição final, análise de similaridade, exportação, exibição de resultados)
    # (Cole o bloco de código da versão anterior que começa com "st.sidebar.markdown("---"); st.sidebar.subheader("2. Filtros de Análise")"
    #  até antes da chamada aos dialogs no final da função app() )

    # --- Filtros de Análise (sobre o período de exibição) ---
    st.sidebar.markdown("---"); st.sidebar.subheader("2. Filtros de Análise (sobre o período de exibição)")
    pastas_disp = sorted(df_atividades_periodo_ui['activity_folder'].dropna().unique()) if not df_atividades_periodo_ui.empty else []
    pastas_sel = st.sidebar.multiselect("Analisar Pasta(s):", pastas_disp, default=[], key=f"pasta_sel{SUFFIX_DIALOG}")
    status_disp_analise = sorted(df_atividades_periodo_ui['activity_status'].dropna().unique()) if not df_atividades_periodo_ui.empty else []
    status_sel_analise = st.sidebar.multiselect("Analisar Status:", status_disp_analise, default=[], key=f"status_sel{SUFFIX_DIALOG}")

    df_para_analise = df_atividades_periodo_ui.copy() 
    if pastas_sel: df_para_analise = df_para_analise[df_para_analise['activity_folder'].isin(pastas_sel)]
    if status_sel_analise: df_para_analise = df_para_analise[df_para_analise['activity_status'].isin(status_sel_analise)]
    
    # --- Filtros de Exibição Final ---
    st.sidebar.markdown("---"); st.sidebar.subheader("3. Filtros de Exibição Final")
    min_sim = st.sidebar.slider("Similaridade ≥ que (%):", 0, 100, 70, 5, key=f"sim_slider{SUFFIX_DIALOG}") / 100.0
    apenas_dup = st.sidebar.checkbox("Exibir apenas com duplicatas", value=True, key=f"dup_cb{SUFFIX_DIALOG}")
    pastas_multi_user = {nome for nome, grupo in df_para_analise.groupby('activity_folder') if grupo['user_profile_name'].nunique() > 1} if not df_para_analise.empty else set()
    apenas_multi = st.sidebar.checkbox("Exibir pastas com múltiplos usuários", False, key=f"multi_cb{SUFFIX_DIALOG}")
    usuarios_disp_ex = sorted(df_para_analise['user_profile_name'].dropna().unique()) if not df_para_analise.empty else []
    usuarios_sel = st.sidebar.multiselect("Exibir Usuário(s):", usuarios_disp_ex, default=[], key=f"user_sel{SUFFIX_DIALOG}")

    ids_com_duplicatas = set()
    map_id_para_similaridades = {} 

    if not df_para_analise.empty and len(df_para_analise) > 1:
        prog_placeholder = st.sidebar.empty(); prog_bar = st.sidebar.progress(0)
        total_pastas_analise = df_para_analise['activity_folder'].nunique(); pastas_processadas_analise = 0
        for nome_pasta_calc, df_pasta_calc in df_para_analise.groupby('activity_folder'):
            prog_placeholder.text(f"Analisando Pasta: {nome_pasta_calc}...")
            if len(df_pasta_calc) < 2:
                pastas_processadas_analise += 1
                if total_pastas_analise > 0: prog_bar.progress(pastas_processadas_analise / total_pastas_analise)
                continue
            atividades_nesta_pasta = df_pasta_calc.to_dict('records')
            for i in range(len(atividades_nesta_pasta)):
                base = atividades_nesta_pasta[i]
                if base['activity_id'] not in map_id_para_similaridades: map_id_para_similaridades[base['activity_id']] = []
                for j in range(i + 1, len(atividades_nesta_pasta)):
                    comparar = atividades_nesta_pasta[j]
                    similaridade = calcular_similaridade(base['Texto'], comparar['Texto'])
                    if similaridade >= min_sim:
                        ids_com_duplicatas.add(base['activity_id']); ids_com_duplicatas.add(comparar['activity_id'])
                        cor = obter_cor_similaridade(similaridade)
                        map_id_para_similaridades[base['activity_id']].append({'id_similar': comparar['activity_id'], 'ratio': similaridade, 'cor': cor})
                        if comparar['activity_id'] not in map_id_para_similaridades: map_id_para_similaridades[comparar['activity_id']] = []
                        map_id_para_similaridades[comparar['activity_id']].append({'id_similar': base['activity_id'], 'ratio': similaridade, 'cor': cor})
            pastas_processadas_analise += 1
            if total_pastas_analise > 0: prog_bar.progress(pastas_processadas_analise / total_pastas_analise)
        prog_bar.empty(); prog_placeholder.text("Análise de similaridade concluída.")
        for act_id_sort_map in map_id_para_similaridades:
            map_id_para_similaridades[act_id_sort_map] = sorted(map_id_para_similaridades[act_id_sort_map], key=lambda x: x['ratio'], reverse=True)

    df_exibir = df_para_analise.copy()
    if apenas_dup: df_exibir = df_exibir[df_exibir['activity_id'].isin(ids_com_duplicatas)]
    if apenas_multi: df_exibir = df_exibir[df_exibir['activity_folder'].isin(pastas_multi_user)]
    if usuarios_sel: df_exibir = df_exibir[df_exibir['user_profile_name'].isin(usuarios_sel)]

    st.sidebar.markdown("---")
    if st.sidebar.button("📥 Exportar para XLSX", key=f"export_btn{SUFFIX_DIALOG}"):
        if not df_exibir.empty:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_exibir.to_excel(writer, index=False, sheet_name='Atividades_Exibidas')
                lista_export_duplicatas = []
                if map_id_para_similaridades:
                    for id_base_export, lista_similares_export in map_id_para_similaridades.items():
                        if id_base_export in df_exibir['activity_id'].values: 
                            for sim_info_export in lista_similares_export:
                                detalhes_similar_export_rows = df_raw_total[df_raw_total['activity_id'] == sim_info_export['id_similar']] # Usar df_raw_total para detalhes
                                if not detalhes_similar_export_rows.empty:
                                    detalhes_similar_export = detalhes_similar_export_rows.iloc[0]
                                    lista_export_duplicatas.append({
                                        'ID_Base': id_base_export,
                                        'ID_Duplicata_Potencial': sim_info_export['id_similar'],
                                        'Percentual_Similaridade': sim_info_export['ratio'],
                                        'Cor_Similaridade': sim_info_export['cor'],
                                        'Data_Duplicata': detalhes_similar_export['activity_date'].strftime('%Y-%m-%d %H:%M') if pd.notna(detalhes_similar_export['activity_date']) else None,
                                        'Usuario_Duplicata': detalhes_similar_export['user_profile_name'],
                                        'Status_Duplicata': detalhes_similar_export['activity_status']
                                    })
                    if lista_export_duplicatas:
                        pd.DataFrame(lista_export_duplicatas).to_excel(writer, index=False, sheet_name='Detalhes_Duplicatas')
            st.sidebar.download_button("Baixar XLSX", output.getvalue(), f"duplicatas_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else: st.sidebar.warning("Nenhum dado para exportar.")

    st.header("Análise Detalhada por Pasta")
    if df_exibir.empty: st.info("Nenhuma atividade para os filtros selecionados.")
    
    pastas_ordenadas = sorted(df_exibir['activity_folder'].unique()) if not df_exibir.empty else []

    for nome_pasta in pastas_ordenadas:
        df_pasta_exibicao = df_exibir[df_exibir['activity_folder'] == nome_pasta]
        if df_pasta_exibicao.empty: continue
        total_analisado_pasta = len(df_para_analise[df_para_analise['activity_folder'] == nome_pasta])
        titulo = f"📁 Pasta: {nome_pasta} ({len(df_pasta_exibicao)} exibidas / {total_analisado_pasta} analisadas)"
        
        with st.expander(titulo, expanded=len(df_pasta_exibicao) < 10):
            for _, atividade_row in df_pasta_exibicao.iterrows():
                atividade = atividade_row.to_dict()
                st.markdown("---")
                col_info, col_sim_display = st.columns([0.6, 0.4])
                with col_info:
                    data_atividade_str = atividade['activity_date'].strftime('%d/%m/%Y %H:%M') if pd.notna(atividade['activity_date']) else "Data Inválida"
                    st.markdown(f"**ID:** `{atividade['activity_id']}` | **Data:** {data_atividade_str} | **Status:** `{atividade['activity_status']}`")
                    st.markdown(f"**Usuário:** {atividade['user_profile_name']}")
                    st.text_area("Texto:", str(atividade['Texto']), height=100, key=f"texto_exp{SUFFIX_DIALOG}_{nome_pasta}_{atividade['activity_id']}", disabled=True)
                    btn_cols = st.columns(3)
                    links = gerar_links_zflow(atividade['activity_id'])
                    if btn_cols[0].button("👁️ Ver Completo", key=f"ver_completo_btn{SUFFIX_DIALOG}_{atividade['activity_id']}", on_click=on_click_ver_texto_completo, args=(atividade,)): pass
                    btn_cols[1].link_button("🔗 ZFlow v1", links['antigo'])
                    btn_cols[2].link_button("� ZFlow v2", links['novo'])

                with col_sim_display:
                    similares_para_esta_atividade = map_id_para_similaridades.get(atividade['activity_id'], [])
                    if similares_para_esta_atividade:
                        st.markdown(f"**<span style='color:red;'>Duplicatas (Intra-Pasta):</span>** ({len(similares_para_esta_atividade)})", unsafe_allow_html=True)
                        for sim_data in similares_para_esta_atividade:
                            info_dupe_rows = df_raw_total[df_raw_total['activity_id'] == sim_data['id_similar']] # Usar df_raw_total
                            if not info_dupe_rows.empty:
                                info_dupe = info_dupe_rows.iloc[0].to_dict()
                                container_dup = st.container(border=True)
                                data_dupe_str = info_dupe['activity_date'].strftime('%d/%m/%y %H:%M') if pd.notna(info_dupe['activity_date']) else "Data Inválida"
                                container_dup.markdown(f"""<small><div style='background-color:{sim_data['cor']}; padding: 3px 6px; border-radius: 5px; color: black; margin-bottom: 5px; font-weight: 500;'>
                                <b>ID: {info_dupe['activity_id']} ({sim_data['ratio']:.0%})</b><br>
                                Data: {data_dupe_str} | Status: {info_dupe['activity_status']}<br>
                                Usuário: {info_dupe['user_profile_name']}
                                </div></small>""", unsafe_allow_html=True)
                                if container_dup.button("⚖️ Comparar (Detalhado)", key=f"comp_html_dialog_btn{SUFFIX_DIALOG}_{atividade['activity_id']}_{info_dupe['activity_id']}", on_click=on_click_comparar_textos_html_dialog, args=(atividade, info_dupe)): pass
                            else: st.caption(f"Detalhes da ID {sim_data['id_similar']} não disponíveis.")
                    else:
                        if not apenas_dup: st.markdown("**<span style='color:green;'>Sem duplicatas (nesta análise)</span>**", unsafe_allow_html=True)

    if st.session_state.get(f'show_texto_dialog{SUFFIX_DIALOG}', False):
        mostrar_texto_completo_dialog()
    if st.session_state.get(f'show_comparacao_dialog{SUFFIX_DIALOG}', False):
        mostrar_comparacao_html_diff_dialog()

# ==============================================================================
# LÓGICA DE LOGIN
# ==============================================================================
def check_credentials(username, password):
    try:
        user_creds = st.secrets["credentials"]["usernames"]
        if username in user_creds and str(user_creds[username]) == password: return True
    except KeyError: return False
    except Exception: return False
    return False

def login_form():
    st.header("Login - Verificador de Duplicidade")
    with st.form(f"login_form{SUFFIX_DIALOG}_main"): 
        username = st.text_input("Usuário", key=f"login_username{SUFFIX_DIALOG}_main")
        password = st.text_input("Senha", key=f"login_password{SUFFIX_DIALOG}_main", type="password")
        submitted = st.form_submit_button("Entrar")
        if submitted:
            if check_credentials(username, password):
                st.session_state["logged_in"] = True; st.session_state["username"] = username; st.rerun()
            else: st.error("Usuário ou senha inválidos.")
    st.info("Use as credenciais do secrets.toml.")

if __name__ == "__main__":
    if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
    
    if st.session_state["logged_in"]:
        app() 
    else:
        login_form()
�
