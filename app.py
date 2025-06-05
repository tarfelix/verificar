import streamlit as st
import pandas as pd
import re
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from datetime import datetime, timedelta, date 
from unidecode import unidecode
from rapidfuzz import fuzz
import io
import difflib

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================
st.set_page_config(layout="wide", page_title="Verificador de Duplicidade Avançado")

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

@st.cache_data(ttl=3600)
def buscar_dados_do_banco(_engine: Engine, data_inicio_req: date, data_fim_req: date) -> tuple[pd.DataFrame | None, Exception | None]:
    query = text("""
        SELECT activity_id, activity_folder, activity_subject, user_id, user_profile_name,
               activity_date, activity_fatal, activity_status, activity_type,
               activity_publish_date, Texto, observacoes, tags,
               activity_created_at, activity_updated_at
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type = :tipo_atividade
          AND activity_status = :status_atividade 
          AND DATE(activity_date) BETWEEN :data_inicio_query AND :data_fim_query
        ORDER BY activity_folder, activity_date DESC, activity_id DESC
    """)
    try:
        with _engine.connect() as connection:
            df = pd.read_sql(query, connection, params={
                "tipo_atividade": "Verificar", "status_atividade": "Aberta",
                "data_inicio_query": data_inicio_req, "data_fim_query": data_fim_req
            })
        df['activity_date'] = pd.to_datetime(df['activity_date'])
        df['Texto'] = df['Texto'].astype(str).fillna('')
        return df, None
    except exc.SQLAlchemyError as e:
        return None, e

# ==============================================================================
# Estado da Sessão para Dialogs
# ==============================================================================
SUFFIX_DIALOG = "_v_dialog_simple_diff" 
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
        st.markdown(f"### Texto Completo - Atividade ID: `{atividade_data['activity_id']}`")
        st.markdown(f"**Pasta:** {atividade_data['activity_folder']} | **Data:** {atividade_data['activity_date'].strftime('%d/%m/%Y %H:%M')} | **Usuário:** {atividade_data['user_profile_name']} | **Status:** {atividade_data['activity_status']}")
        st.text_area("Texto da Publicação:", value=str(atividade_data['Texto']), height=400, disabled=True, key=f"dialog_texto_content{SUFFIX_DIALOG}_{atividade_data['activity_id']}")
        if st.button("Fechar Texto", key=f"fechar_dialog_texto_btn{SUFFIX_DIALOG}"):
            st.session_state[f'show_texto_dialog{SUFFIX_DIALOG}'] = False
            st.rerun()

@st.dialog("Comparação de Textos (Simplificada)")
def mostrar_comparacao_dialog_simplificada():
    atividades_comp_data = st.session_state[f'atividades_para_comparacao{SUFFIX_DIALOG}']
    if atividades_comp_data:
        base_comp = atividades_comp_data['base']
        comparar_comp = atividades_comp_data['comparar']
    
        st.markdown(f"### Comparando ID `{base_comp['activity_id']}` com ID `{comparar_comp['activity_id']}`")
        
        texto_base_comp = str(base_comp['Texto'])
        texto_comparar_comp = str(comparar_comp['Texto'])
        
        # Usar unified_diff para uma saída de texto mais leve
        diff_result = list(difflib.unified_diff(
            texto_base_comp.splitlines(keepends=True),
            texto_comparar_comp.splitlines(keepends=True),
            fromfile=f"ID_{base_comp['activity_id']}",
            tofile=f"ID_{comparar_comp['activity_id']}",
            lineterm='' # Evita linhas em branco extras entre as linhas do diff
        ))
        
        if not diff_result:
            st.info("Os textos são idênticos (após normalização, se aplicável pela similaridade já calculada).")
        else:
            # Exibir o diff em um bloco de código para melhor formatação
            st.code("".join(diff_result), language='diff', line_numbers=False)

        # Opcional: mostrar os textos completos abaixo do diff se ainda for útil
        # with st.expander("Ver textos completos originais"):
        #     col1, col2 = st.columns(2)
        #     with col1:
        #         st.subheader(f"Texto ID: {base_comp['activity_id']}")
        #         st.text(texto_base_comp)
        #     with col2:
        #         st.subheader(f"Texto ID: {comparar_comp['activity_id']}")
        #         st.text(texto_comparar_comp)

        if st.button("Fechar Comparação", key=f"fechar_dialog_comparacao_btn_simple{SUFFIX_DIALOG}"):
            st.session_state[f'show_comparacao_dialog{SUFFIX_DIALOG}'] = False
            st.rerun()

# ==============================================================================
# Funções para abrir os dialogs
# ==============================================================================
def on_click_ver_texto_completo(atividade):
    st.session_state[f'atividade_para_texto_dialog{SUFFIX_DIALOG}'] = atividade
    st.session_state[f'show_texto_dialog{SUFFIX_DIALOG}'] = True

def on_click_comparar_textos_dialog(atividade_base, atividade_comparar): # Renomeado para clareza
    st.session_state[f'atividades_para_comparacao{SUFFIX_DIALOG}'] = {'base': atividade_base, 'comparar': atividade_comparar}
    st.session_state[f'show_comparacao_dialog{SUFFIX_DIALOG}'] = True

# ==============================================================================
# INTERFACE PRINCIPAL DO APP
# ==============================================================================
def app_principal():
    # ... (código da sidebar e carregamento de dados como na versão anterior) ...
    # ... (certifique-se de que as chaves dos widgets na sidebar são únicas se copiar e colar) ...
    st.sidebar.success(f"Logado como: **{st.session_state['username']}**")
    if st.sidebar.button("Logout", key=f"logout_button{SUFFIX_DIALOG}_main"): # Chave única
        for key_state in list(st.session_state.keys()): del st.session_state[key_state]
        st.rerun()

    st.title("🔎 Verificador de Duplicidade Avançado")
    st.markdown("Análise de atividades 'Verificar' para identificar potenciais duplicidades.")

    engine = get_db_engine()
    if not engine: st.error("Falha crítica na conexão com o banco."); st.stop()

    st.sidebar.header("⚙️ Filtros e Opções")
    st.sidebar.markdown("---")
    st.sidebar.subheader("1. Filtro de Período")
    hoje = date.today() # Usar date de datetime
    data_inicio_selecionada = st.sidebar.date_input("Data de Início", hoje - timedelta(days=7), key=f"di{SUFFIX_DIALOG}_main")
    data_fim_selecionada = st.sidebar.date_input("Data de Fim", hoje, key=f"df{SUFFIX_DIALOG}_main")

    if data_inicio_selecionada > data_fim_selecionada: st.sidebar.error("Data de início > data de fim."); st.stop()
    
    if st.sidebar.button("🔎 Buscar/Atualizar Dados", help="Busca dados do MySQL.", key=f"buscar{SUFFIX_DIALOG}_main"):
        buscar_dados_do_banco.clear(); st.toast("Buscando dados atualizados...", icon="�")
    
    df_atividades_periodo, erro_db = buscar_dados_do_banco(engine, data_inicio_selecionada, data_fim_selecionada)

    if erro_db: st.error("Erro ao buscar atividades."); st.exception(erro_db); st.stop()
    
    if df_atividades_periodo is None or df_atividades_periodo.empty:
        st.warning(f"Nenhuma atividade para o período de {data_inicio_selecionada.strftime('%d/%m/%Y')} a {data_fim_selecionada.strftime('%d/%m/%Y')}.")
    else:
        st.success(f"**{len(df_atividades_periodo)}** atividades 'Verificar' (Abertas) carregadas.")
    
    st.sidebar.markdown("---"); st.sidebar.subheader("2. Filtros de Análise")
    pastas_disp = sorted(df_atividades_periodo['activity_folder'].dropna().unique()) if not df_atividades_periodo.empty else []
    pastas_sel = st.sidebar.multiselect("Analisar Pasta(s):", pastas_disp, default=[], key=f"pasta_sel{SUFFIX_DIALOG}_main")
    status_disp_analise = sorted(df_atividades_periodo['activity_status'].dropna().unique()) if not df_atividades_periodo.empty else []
    status_sel_analise = st.sidebar.multiselect("Analisar Status:", status_disp_analise, default=[], key=f"status_sel{SUFFIX_DIALOG}_main")

    df_para_analise = df_atividades_periodo.copy()
    if pastas_sel: df_para_analise = df_para_analise[df_para_analise['activity_folder'].isin(pastas_sel)]
    if status_sel_analise: df_para_analise = df_para_analise[df_para_analise['activity_status'].isin(status_sel_analise)]
    
    st.sidebar.markdown("---"); st.sidebar.subheader("3. Filtros de Exibição")
    min_sim = st.sidebar.slider("Similaridade ≥ que (%):", 0, 100, 70, 5, key=f"sim_slider{SUFFIX_DIALOG}_main") / 100.0
    apenas_dup = st.sidebar.checkbox("Exibir apenas com duplicatas", value=True, key=f"dup_cb{SUFFIX_DIALOG}_main")
    pastas_multi_user = {nome for nome, grupo in df_para_analise.groupby('activity_folder') if grupo['user_profile_name'].nunique() > 1} if not df_para_analise.empty else set()
    apenas_multi = st.sidebar.checkbox("Exibir pastas com múltiplos usuários", False, key=f"multi_cb{SUFFIX_DIALOG}_main")
    usuarios_disp_ex = sorted(df_para_analise['user_profile_name'].dropna().unique()) if not df_para_analise.empty else []
    usuarios_sel = st.sidebar.multiselect("Exibir Usuário(s):", usuarios_disp_ex, default=[], key=f"user_sel{SUFFIX_DIALOG}_main")

    ids_com_duplicatas = set()
    todas_similaridades = []

    if not df_para_analise.empty and len(df_para_analise) > 1:
        prog_placeholder = st.sidebar.empty()
        prog_bar = st.sidebar.progress(0)
        total_a_analisar = df_para_analise['activity_folder'].nunique() 
        prog_count = 0
        for nome_pasta_iter, df_pasta_iter in df_para_analise.groupby('activity_folder'):
            prog_placeholder.text(f"Analisando pasta: {nome_pasta_iter}...")
            if len(df_pasta_iter) < 2: 
                prog_count +=1
                if total_a_analisar > 0: prog_bar.progress(prog_count/total_a_analisar)
                continue
            atividades_lista = df_pasta_iter.to_dict('records')
            for i in range(len(atividades_lista)):
                for j in range(i + 1, len(atividades_lista)):
                    base, comparar = atividades_lista[i], atividades_lista[j]
                    similaridade = calcular_similaridade(base['Texto'], comparar['Texto'])
                    if similaridade >= min_sim:
                        cor = obter_cor_similaridade(similaridade)
                        todas_similaridades.append({'id_base': base['activity_id'], 'id_similar': comparar['activity_id'], 'ratio': similaridade, 'cor': cor})
                        ids_com_duplicatas.add(base['activity_id'])
                        ids_com_duplicatas.add(comparar['activity_id'])
            prog_count +=1
            if total_a_analisar > 0: prog_bar.progress(prog_count/total_a_analisar)
        prog_bar.empty(); prog_placeholder.text("Análise de similaridade concluída.")
    df_similaridades = pd.DataFrame(todas_similaridades)

    df_exibir = df_para_analise.copy()
    if apenas_dup: df_exibir = df_exibir[df_exibir['activity_id'].isin(ids_com_duplicatas)]
    if apenas_multi: df_exibir = df_exibir[df_exibir['activity_folder'].isin(pastas_multi_user)]
    if usuarios_sel: df_exibir = df_exibir[df_exibir['user_profile_name'].isin(usuarios_sel)]

    st.sidebar.markdown("---")
    if st.sidebar.button("📥 Exportar para XLSX", key=f"export_btn{SUFFIX_DIALOG}_main"):
        if not df_exibir.empty:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_exibir.to_excel(writer, index=False, sheet_name='Atividades_Exibidas')
                if not df_similaridades.empty:
                    df_sim_export = df_similaridades[df_similaridades['id_base'].isin(df_exibir['activity_id'])]
                    if not df_sim_export.empty:
                        df_sim_export = pd.merge(df_sim_export, df_atividades_periodo[['activity_id', 'activity_date', 'user_profile_name', 'activity_status']].rename(
                            columns={'activity_id': 'id_similar', 'activity_date': 'data_similar',
                                     'user_profile_name': 'usuario_similar', 'activity_status': 'status_similar'}),
                            on='id_similar', how='left')
                        df_sim_export.to_excel(writer, index=False, sheet_name='Detalhes_Duplicatas')
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
        
        with st.expander(titulo, expanded=False):
            for _, atividade_row in df_pasta_exibicao.iterrows():
                atividade = atividade_row.to_dict()
                st.markdown("---")
                col_info, col_sim_display = st.columns([0.6, 0.4])
                with col_info:
                    st.markdown(f"**ID:** `{atividade['activity_id']}` | **Data:** {atividade['activity_date'].strftime('%d/%m/%Y %H:%M')} | **Status:** `{atividade['activity_status']}`")
                    st.markdown(f"**Usuário:** {atividade['user_profile_name']}")
                    st.text_area("Texto:", str(atividade['Texto']), height=100, key=f"texto_exp{SUFFIX_DIALOG}_{nome_pasta}_{atividade['activity_id']}", disabled=True)
                    btn_cols = st.columns(3)
                    links = gerar_links_zflow(atividade['activity_id'])
                    if btn_cols[0].button("👁️ Ver Completo", key=f"ver_completo_btn{SUFFIX_DIALOG}_{atividade['activity_id']}", on_click=on_click_ver_texto_completo, args=(atividade,)): pass
                    btn_cols[1].link_button("🔗 ZFlow v1", links['antigo'])
                    btn_cols[2].link_button("🔗 ZFlow v2", links['novo'])

                with col_sim_display:
                    similares = df_similaridades[df_similaridades['id_base'] == atividade['activity_id']] if not df_similaridades.empty else pd.DataFrame()
                    if not similares.empty:
                        st.markdown(f"**<span style='color:red;'>Duplicatas:</span>** ({len(similares)})", unsafe_allow_html=True)
                        for _, sim_data in similares.sort_values(by='ratio', ascending=False).iterrows():
                            info_dupe_rows = df_atividades_periodo[df_atividades_periodo['activity_id'] == sim_data['id_similar']]
                            if not info_dupe_rows.empty:
                                info_dupe = info_dupe_rows.iloc[0].to_dict()
                                container_dup = st.container(border=True)
                                container_dup.markdown(f"""<small><div style='background-color:{sim_data['cor']}; padding: 3px 6px; border-radius: 5px; color: black; margin-bottom: 5px; font-weight: 500;'>
                                <b>ID: {info_dupe['activity_id']} ({sim_data['ratio']:.0%})</b><br>
                                Data: {info_dupe['activity_date'].strftime('%d/%m/%y')} | Status: {info_dupe['activity_status']}<br>
                                Usuário: {info_dupe['user_profile_name']}
                                </div></small>""", unsafe_allow_html=True)
                                if container_dup.button("⚖️ Comparar (Simplificado)", key=f"comp_dialog_btn_simple{SUFFIX_DIALOG}_{atividade['activity_id']}_{info_dupe['activity_id']}", on_click=on_click_comparar_textos_dialog, args=(atividade, info_dupe)): pass # Mudado para on_click_comparar_textos_dialog
                            else: st.caption(f"Detalhes da ID {sim_data['id_similar']} não disponíveis.")
                    else:
                        if not apenas_dup: st.markdown("**<span style='color:green;'>Sem duplicatas</span>**", unsafe_allow_html=True)

    if st.session_state.get(f'show_texto_dialog{SUFFIX_DIALOG}', False):
        mostrar_texto_completo_dialog()
    if st.session_state.get(f'show_comparacao_dialog{SUFFIX_DIALOG}', False):
        mostrar_comparacao_dialog_simplificada() # Chama a nova função de dialog simplificado

# ==============================================================================
# LÓGICA DE LOGIN
# ==============================================================================
def check_credentials(username, password):
    try:
        user_creds = st.secrets["credentials"]["usernames"] # Renomeado para clareza
        if username in user_creds and str(user_creds[username]) == password: return True
    except KeyError: return False
    except Exception: return False
    return False

def login_form():
    st.header("Login - Verificador de Duplicidade")
    with st.form(f"login_form{SUFFIX_DIALOG}_main"): # Chave única
        username = st.text_input("Usuário", key=f"login_username{SUFFIX_DIALOG}_main")
        password = st.text_input("Senha", key=f"login_password{SUFFIX_DIALOG}_main", type="password")
        submitted = st.form_submit_button("Entrar")
        if submitted:
            if check_credentials(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.rerun()
            else: st.error("Usuário ou senha inválidos.")
    st.info("Use as credenciais do secrets.toml.")

if __name__ == "__main__":
    if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
    if st.session_state["logged_in"]: app_principal()
    else: login_form()
�
