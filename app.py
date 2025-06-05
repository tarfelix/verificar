import streamlit as st
import pandas as pd
import re
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from datetime import datetime, timedelta
from unidecode import unidecode
from rapidfuzz import fuzz
import io
import difflib # Para HtmlDiff

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
def buscar_dados_do_banco(_engine: Engine, data_inicio_req: datetime.date, data_fim_req: datetime.date) -> tuple[pd.DataFrame | None, Exception | None]:
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
                "tipo_atividade": "Verificar",
                "status_atividade": "Aberta",
                "data_inicio_query": data_inicio_req,
                "data_fim_query": data_fim_req
            })
        df['activity_date'] = pd.to_datetime(df['activity_date'])
        df['Texto'] = df['Texto'].astype(str).fillna('')
        return df, None
    except exc.SQLAlchemyError as e:
        return None, e

# ==============================================================================
# Estado da Sessão para Dialogs
# ==============================================================================
# Usar sufixos distintos para as chaves de estado da sessão
if 'show_comparacao_dialog_v_final' not in st.session_state:
    st.session_state.show_comparacao_dialog_v_final = False
if 'atividades_para_comparacao_v_final' not in st.session_state:
    st.session_state.atividades_para_comparacao_v_final = None 

# ==============================================================================
# Funções para abrir e fechar dialog de COMPARAÇÃO
# ==============================================================================
def abrir_dialog_comparacao_final(atividade_base, atividade_comparar):
    st.session_state.atividades_para_comparacao_v_final = {'base': atividade_base, 'comparar': atividade_comparar}
    st.session_state.show_comparacao_dialog_v_final = True

def fechar_dialog_comparacao_final():
    st.session_state.show_comparacao_dialog_v_final = False
    st.session_state.atividades_para_comparacao_v_final = None
    # Opcional: st.rerun() se o dialog não desaparecer visualmente apenas com a mudança de estado.
    # Geralmente, o Streamlit lida bem com isso quando o bloco condicional do dialog não é mais renderizado.

# ==============================================================================
# INTERFACE PRINCIPAL DO APP
# ==============================================================================
def app_principal():
    st.sidebar.success(f"Logado como: **{st.session_state['username']}**")
    if st.sidebar.button("Logout", key="logout_button_v_final"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

    st.title("🔎 Verificador de Duplicidade Avançado")
    st.markdown("Análise de atividades 'Verificar' para identificar potenciais duplicidades.")

    engine = get_db_engine()
    if not engine:
        st.error("Falha crítica na conexão com o banco. Verifique as configurações e o status do servidor.")
        st.stop()

    st.sidebar.header("⚙️ Filtros e Opções")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("1. Filtro de Período")
    hoje = datetime.today().date()
    data_inicio_selecionada = st.sidebar.date_input("Data de Início", hoje - timedelta(days=7), key="di_v_final")
    data_fim_selecionada = st.sidebar.date_input("Data de Fim", hoje, key="df_v_final")

    if data_inicio_selecionada > data_fim_selecionada:
        st.sidebar.error("A data de início não pode ser posterior à data de fim.")
        st.stop()
    
    if st.sidebar.button("🔎 Buscar/Atualizar Dados", help="Busca dados do MySQL para o período selecionado.", key="buscar_v_final"):
        buscar_dados_do_banco.clear() 
        st.toast("Buscando dados atualizados do banco...", icon="🔄")
    
    df_atividades_periodo, erro_db = buscar_dados_do_banco(engine, data_inicio_selecionada, data_fim_selecionada)

    if erro_db:
        st.error("Erro ao buscar atividades no banco de dados.")
        st.exception(erro_db)
        st.stop()
    
    if df_atividades_periodo is None or df_atividades_periodo.empty:
        st.warning(f"Nenhuma atividade 'Verificar' (Aberta) encontrada para o período de {data_inicio_selecionada.strftime('%d/%m/%Y')} a {data_fim_selecionada.strftime('%d/%m/%Y')}.")
    else:
        st.success(f"**{len(df_atividades_periodo)}** atividades 'Verificar' (Abertas) carregadas para o período.")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Filtros de Análise")
    pastas_disp = sorted(df_atividades_periodo['activity_folder'].dropna().unique()) if not df_atividades_periodo.empty else []
    pastas_sel = st.sidebar.multiselect("Analisar Pasta(s):", pastas_disp, default=[], key="pasta_sel_v_final")

    status_disp_analise = sorted(df_atividades_periodo['activity_status'].dropna().unique()) if not df_atividades_periodo.empty else []
    status_sel_analise = st.sidebar.multiselect("Analisar Status:", status_disp_analise, default=[], key="status_sel_v_final")

    df_para_analise = df_atividades_periodo.copy()
    if pastas_sel: df_para_analise = df_para_analise[df_para_analise['activity_folder'].isin(pastas_sel)]
    if status_sel_analise: df_para_analise = df_para_analise[df_para_analise['activity_status'].isin(status_sel_analise)]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Filtros de Exibição")
    min_sim = st.sidebar.slider("Similaridade ≥ que (%):", 0, 100, 70, 5, key="sim_slider_v_final") / 100.0
    apenas_dup = st.sidebar.checkbox("Exibir apenas com duplicatas", value=True, key="dup_cb_v_final")
    
    pastas_multi_user = {nome for nome, grupo in df_para_analise.groupby('activity_folder') if grupo['user_profile_name'].nunique() > 1} if not df_para_analise.empty else set()
    apenas_multi = st.sidebar.checkbox("Exibir pastas com múltiplos usuários", False, key="multi_cb_v_final")

    usuarios_disp_ex = sorted(df_para_analise['user_profile_name'].dropna().unique()) if not df_para_analise.empty else []
    usuarios_sel = st.sidebar.multiselect("Exibir Usuário(s):", usuarios_disp_ex, default=[], key="user_sel_v_final")

    ids_com_duplicatas = set()
    todas_similaridades = []

    if not df_para_analise.empty and len(df_para_analise) > 1:
        # Feedback de progresso
        progresso_analise_placeholder = st.sidebar.empty() 
        progresso_analise_placeholder.text("Analisando similaridades...")
        barra_progresso = st.sidebar.progress(0)
        total_pastas_para_analisar = df_para_analise['activity_folder'].nunique()
        pastas_analisadas_count = 0

        for nome_pasta_iter, df_pasta_iter in df_para_analise.groupby('activity_folder'):
            if len(df_pasta_iter) < 2: 
                pastas_analisadas_count +=1
                if total_pastas_para_analisar > 0:
                    barra_progresso.progress(pastas_analisadas_count / total_pastas_para_analisar)
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
            
            pastas_analisadas_count +=1
            if total_pastas_para_analisar > 0:
                 barra_progresso.progress(pastas_analisadas_count / total_pastas_para_analisar)
        
        barra_progresso.empty() # Limpa a barra de progresso
        progresso_analise_placeholder.text("Análise de similaridade concluída.")


    df_similaridades = pd.DataFrame(todas_similaridades)

    df_exibir = df_para_analise.copy()
    if apenas_dup: df_exibir = df_exibir[df_exibir['activity_id'].isin(ids_com_duplicatas)]
    if apenas_multi: df_exibir = df_exibir[df_exibir['activity_folder'].isin(pastas_multi_user)]
    if usuarios_sel: df_exibir = df_exibir[df_exibir['user_profile_name'].isin(usuarios_sel)]

    st.sidebar.markdown("---")
    if st.sidebar.button("📥 Exportar para XLSX", key="export_btn_v_final"):
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
            for _, atividade_row in df_pasta_exibicao.iterrows(): # Renomeado para evitar conflito
                atividade = atividade_row.to_dict() # Converter para dict para passar para o dialog
                st.markdown("---")
                col_info, col_sim_display = st.columns([0.6, 0.4])
                with col_info:
                    st.markdown(f"**ID:** `{atividade['activity_id']}` | **Data:** {atividade['activity_date'].strftime('%d/%m/%Y %H:%M')} | **Status:** `{atividade['activity_status']}`")
                    st.markdown(f"**Usuário:** {atividade['user_profile_name']}")
                    st.text_area("Texto:", str(atividade['Texto']), height=100, key=f"texto_exp_v_final_{nome_pasta}_{atividade['activity_id']}", disabled=True)
                    btn_cols = st.columns(2)
                    links = gerar_links_zflow(atividade['activity_id'])
                    btn_cols[0].link_button("🔗 ZFlow v1", links['antigo'])
                    btn_cols[1].link_button("🔗 ZFlow v2", links['novo'])
                with col_sim_display:
                    similares = df_similaridades[df_similaridades['id_base'] == atividade['activity_id']] if not df_similaridades.empty else pd.DataFrame()
                    if not similares.empty:
                        st.markdown(f"**<span style='color:red;'>Duplicatas:</span>** ({len(similares)})", unsafe_allow_html=True)
                        for _, sim_data in similares.sort_values(by='ratio', ascending=False).iterrows():
                            info_dupe_rows = df_atividades_periodo[df_atividades_periodo['activity_id'] == sim_data['id_similar']]
                            if not info_dupe_rows.empty:
                                info_dupe = info_dupe_rows.iloc[0].to_dict() # Converter para dict
                                container_dup = st.container(border=True)
                                container_dup.markdown(f"""<small><div style='background-color:{sim_data['cor']}; padding: 3px 6px; border-radius: 5px; color: black; margin-bottom: 5px; font-weight: 500;'>
                                <b>ID: {info_dupe['activity_id']} ({sim_data['ratio']:.0%})</b><br>
                                Data: {info_dupe['activity_date'].strftime('%d/%m/%y')} | Status: {info_dupe['activity_status']}<br>
                                Usuário: {info_dupe['user_profile_name']}
                                </div></small>""", unsafe_allow_html=True)
                                # Botão para abrir o dialog de comparação
                                if container_dup.button("⚖️ Comparar Textos", 
                                                        key=f"comp_dialog_btn_{atividade['activity_id']}_{info_dupe['activity_id']}",
                                                        on_click=abrir_dialog_comparacao_final,
                                                        args=(atividade, info_dupe)): # Passa os dicionários das atividades
                                    pass # Ação é feita pelo on_click
                            else: st.caption(f"Detalhes da ID {sim_data['id_similar']} não disponíveis.")
                    else:
                        if not apenas_dup: st.markdown("**<span style='color:green;'>Sem duplicatas</span>**", unsafe_allow_html=True)

    # --- Renderização Condicional do Dialog de COMPARAÇÃO (fora do loop principal) ---
    if st.session_state.show_comparacao_dialog_v_final and st.session_state.atividades_para_comparacao_v_final:
        with st.dialog("Comparação Detalhada de Textos"):
            atividades_comp = st.session_state.atividades_para_comparacao_v_final
            base_comp = atividades_comp['base']
            comparar_comp = atividades_comp['comparar']
        
            st.markdown(f"### Comparando ID `{base_comp['activity_id']}` com ID `{comparar_comp['activity_id']}`")
            
            texto_base_comp = str(base_comp['Texto'])
            texto_comparar_comp = str(comparar_comp['Texto'])
            
            differ = difflib.HtmlDiff(wrapcolumn=70)
            html_comparison = differ.make_table(texto_base_comp.splitlines(), texto_comparar_comp.splitlines(),
                                                 fromdesc=f"ID: {base_comp['activity_id']}", 
                                                 todesc=f"ID: {comparar_comp['activity_id']}")
            st.components.v1.html(html_comparison, height=600, scrolling=True)

            if st.button("Fechar Comparação", key="fechar_comp_dialog_v_final", on_click=fechar_dialog_comparacao_final):
                pass # Ação é feita pelo on_click

# ==============================================================================
# LÓGICA DE LOGIN
# ==============================================================================
def check_credentials(username, password):
    try:
        user_credentials = st.secrets["credentials"]["usernames"]
        if username in user_credentials and str(user_credentials[username]) == password:
            return True
    except KeyError: return False
    except Exception: return False
    return False

def login_form():
    st.header("Login - Verificador de Duplicidade")
    with st.form("login_form_v_final"):
        username = st.text_input("Usuário", key="login_username_v_final")
        password = st.text_input("Senha", key="login_password_v_final", type="password")
        submitted = st.form_submit_button("Entrar")
        if submitted:
            if check_credentials(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.rerun()
            else: st.error("Usuário ou senha inválidos.")
    st.info("Use as credenciais definidas no arquivo secrets.toml.")

if __name__ == "__main__":
    if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
    if st.session_state["logged_in"]: app_principal()
    else: login_form()
