import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import difflib

# ==============================================================================
# CONFIGURAÇÕES E FUNÇÕES AUXILIARES
# ==============================================================================
st.set_page_config(layout="wide", page_title="Verificador de Duplicidade Avançado")

def calcular_similaridade(texto_a, texto_b):
    if texto_a is None or texto_b is None: return 0.0
    return difflib.SequenceMatcher(None, str(texto_a), str(texto_b)).ratio()

def obter_cor_similaridade(ratio):
    if ratio >= 0.91: return "red"
    elif ratio >= 0.71: return "orange"
    elif ratio >= 0.50: return "gold"
    return "grey"

@st.cache_resource
def get_db_engine():
    db_user, db_pass, db_host, db_name = "tarcisio", "123qwe", "40.88.40.110", "zion_flow"
    if not all([db_user, db_pass, db_host, db_name]):
        st.error("Credenciais do banco não definidas no código.")
        return None
    db_uri = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_name}"
    try:
        engine_instance = create_engine(db_uri)
        with engine_instance.connect() as conn: conn.execute(text("SELECT 1"))
        return engine_instance
    except Exception as e:
        st.error(f"Erro ao conectar ao banco: {e}")
        return None

@st.cache_data(ttl=300)
def buscar_atividades_completas(_engine, data_inicio, data_fim):
    if _engine is None: return pd.DataFrame()
    query = text("""
        SELECT activity_id, activity_folder, user_profile_name, activity_date, 
               activity_status, Texto, activity_type
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type = :tipo_atividade AND activity_date BETWEEN :data_inicio AND :data_fim
        ORDER BY activity_folder, activity_date DESC, activity_id DESC
    """)
    try:
        with _engine.connect() as connection:
            df = pd.read_sql(query, connection, params={
                "tipo_atividade": "Verificar",
                "data_inicio": data_inicio, "data_fim": data_fim
            })
        if 'activity_date' in df.columns:
            df['activity_date'] = pd.to_datetime(df['activity_date']).dt.date
        if 'Texto' in df.columns:
            df['Texto'] = df['Texto'].astype(str).fillna('')
        return df
    except Exception as e:
        st.error(f"Erro ao buscar atividades: {e}")
        return pd.DataFrame()

def gerar_links_zflow(activity_id):
    link_antigo = f"https://zflow.zionbyonset.com.br/activity/3/details/{activity_id}"
    link_novo = f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={activity_id}#/fixcol1"
    return {"antigo": link_antigo, "novo": link_novo}

# ==============================================================================
# Inicialização e Gerenciamento do Estado da Sessão para Dialogs
# ==============================================================================
if 'show_texto_dialog' not in st.session_state:
    st.session_state.show_texto_dialog = False
if 'atividade_para_texto_dialog' not in st.session_state:
    st.session_state.atividade_para_texto_dialog = None

if 'show_comparacao_dialog' not in st.session_state:
    st.session_state.show_comparacao_dialog = False
if 'atividades_para_comparacao_dialog' not in st.session_state:
    st.session_state.atividades_para_comparacao_dialog = None

# ==============================================================================
# Funções para abrir e fechar dialogs
# ==============================================================================
def abrir_dialog_texto(atividade):
    st.session_state.atividade_para_texto_dialog = atividade
    st.session_state.show_texto_dialog = True

def fechar_dialog_texto():
    st.session_state.show_texto_dialog = False
    st.session_state.atividade_para_texto_dialog = None

def abrir_dialog_comparacao(atividade_base, atividade_comparar):
    st.session_state.atividades_para_comparacao_dialog = {'base': atividade_base, 'comparar': atividade_comparar}
    st.session_state.show_comparacao_dialog = True

def fechar_dialog_comparacao():
    st.session_state.show_comparacao_dialog = False
    st.session_state.atividades_para_comparacao_dialog = None

# ==============================================================================
# INTERFACE DO USUÁRIO (Streamlit)
# ==============================================================================
st.title("🔎 Verificador de Duplicidade Avançado")
st.markdown("Análise de atividades 'Verificar' para identificar potenciais duplicidades.")
st.warning("⚠️ Credenciais do banco no código. Não seguro para produção.")

engine = get_db_engine()

if engine:
    # --- Sidebar: Filtros e Opções ---
    st.sidebar.header("⚙️ Filtros e Opções")
    hoje = datetime.today().date()
    
    periodo_selecionado = st.sidebar.radio("Período das atividades:", ("Hoje, Ontem e Amanhã", "Intervalo Personalizado"), key="periodo_radio_v3")
    data_inicio_filtro, data_fim_filtro = (hoje - timedelta(days=1), hoje + timedelta(days=1)) if periodo_selecionado == "Hoje, Ontem e Amanhã" else \
                                          (st.sidebar.date_input("Data Início", hoje - timedelta(days=1), key="data_inicio_v3"), st.sidebar.date_input("Data Fim", hoje + timedelta(days=1), key="data_fim_v3"))

    if data_inicio_filtro > data_fim_filtro:
        st.sidebar.error("Data de início posterior à data de fim.")
        st.stop()

    df_atividades_raw = buscar_atividades_completas(engine, data_inicio_filtro, data_fim_filtro)

    if df_atividades_raw.empty:
        st.info(f"Nenhuma atividade 'Verificar' no período de {data_inicio_filtro.strftime('%d/%m/%Y')} a {data_fim_filtro.strftime('%d/%m/%Y')}.")
    else:
        st.success(f"{len(df_atividades_raw)} atividades 'Verificar' carregadas.")

        usuarios_disponiveis = sorted(df_atividades_raw['user_profile_name'].dropna().unique())
        usuarios_selecionados_exibicao = st.sidebar.multiselect("Filtrar exibição por Usuário(s):", usuarios_disponiveis, default=[], key="user_filter_v3")
        
        min_similaridade_display = st.sidebar.slider("Similaridade de texto mínima (%):", 0, 100, 50, 5, key="sim_slider_v3") / 100.0
        
        apenas_potenciais_duplicatas_cb = st.sidebar.checkbox("Mostrar apenas com potenciais duplicatas", False, key="dup_cb_v3")
        apenas_usuarios_diferentes_cb = st.sidebar.checkbox("Mostrar apenas pastas com múltiplos usuários", False, key="multiuser_cb_v3")

        ordem_pastas = st.sidebar.selectbox(
            "Ordenar pastas por:",
            ("Nome da Pasta (A-Z)", "Mais Atividades Primeiro", "Mais Potenciais Duplicatas Primeiro (beta)"),
            key="ordem_pastas_v3"
        )
        st.sidebar.markdown("---")
        if st.sidebar.button("Exportar Potenciais Duplicatas para CSV", key="export_btn_v3"):
            st.sidebar.info("Funcionalidade de exportação em desenvolvimento.")

        # --- Pré-processamento e Análise de Duplicidade ---
        similaridades_globais = {}
        atividades_com_duplicatas_ids = set()
        pastas_com_multiplos_usuarios_set = set()

        for nome_pasta, df_pasta_analise in df_atividades_raw.groupby('activity_folder'):
            if df_pasta_analise['user_profile_name'].nunique() > 1:
                pastas_com_multiplos_usuarios_set.add(nome_pasta)

            atividades_lista = df_pasta_analise.to_dict('records')
            for i in range(len(atividades_lista)):
                base = atividades_lista[i]
                if base['activity_id'] not in similaridades_globais: similaridades_globais[base['activity_id']] = []
                
                for j in range(i + 1, len(atividades_lista)):
                    comparar = atividades_lista[j]
                    sim = calcular_similaridade(base['Texto'], comparar['Texto'])
                    if sim >= min_similaridade_display:
                        atividades_com_duplicatas_ids.add(base['activity_id'])
                        atividades_com_duplicatas_ids.add(comparar['activity_id'])
                        
                        cor = obter_cor_similaridade(sim)
                        similaridades_globais[base['activity_id']].append({
                            'id_similar': comparar['activity_id'], 'ratio': sim, 'cor': cor, 
                            'data_similar': comparar['activity_date'], 'usuario_similar': comparar['user_profile_name'],
                            'status_similar': comparar['activity_status']
                        })
                        if comparar['activity_id'] not in similaridades_globais: similaridades_globais[comparar['activity_id']] = []
                        similaridades_globais[comparar['activity_id']].append({
                            'id_similar': base['activity_id'], 'ratio': sim, 'cor': cor,
                            'data_similar': base['activity_date'], 'usuario_similar': base['user_profile_name'],
                            'status_similar': base['activity_status']
                        })
        
        for key in similaridades_globais:
            similaridades_globais[key] = sorted(similaridades_globais[key], key=lambda x: x['ratio'], reverse=True)

        # --- Filtragem para Exibição ---
        df_exibir = df_atividades_raw.copy()
        if usuarios_selecionados_exibicao:
            df_exibir = df_exibir[df_exibir['user_profile_name'].isin(usuarios_selecionados_exibicao)]
        if apenas_potenciais_duplicatas_cb:
            df_exibir = df_exibir[df_exibir['activity_id'].isin(atividades_com_duplicatas_ids)]
        if apenas_usuarios_diferentes_cb:
            df_exibir = df_exibir[df_exibir['activity_folder'].isin(pastas_com_multiplos_usuarios_set)]

        # --- Lógica de Ordenação das Pastas ---
        lista_pastas_para_exibir = []
        if not df_exibir.empty:
            pastas_agrupadas_exibicao = df_exibir.groupby('activity_folder')
            if ordem_pastas == "Nome da Pasta (A-Z)":
                lista_pastas_para_exibir = sorted(pastas_agrupadas_exibicao.groups.keys())
            elif ordem_pastas == "Mais Atividades Primeiro":
                lista_pastas_para_exibir = pastas_agrupadas_exibicao.size().sort_values(ascending=False).index.tolist()
            elif ordem_pastas == "Mais Potenciais Duplicatas Primeiro (beta)":
                contagem_duplicatas_pasta = {}
                for nome_pasta, df_p in pastas_agrupadas_exibicao:
                    count = 0
                    for act_id_p in df_p['activity_id']: # Renomeado para evitar conflito
                        if act_id_p in similaridades_globais and similaridades_globais[act_id_p]:
                            count +=1 
                            break 
                    if count > 0: contagem_duplicatas_pasta[nome_pasta] = df_p[df_p['activity_id'].isin(atividades_com_duplicatas_ids)].shape[0]
                lista_pastas_para_exibir = sorted(contagem_duplicatas_pasta, key=contagem_duplicatas_pasta.get, reverse=True)
                pastas_sem_duplicatas = [p for p in pastas_agrupadas_exibicao.groups.keys() if p not in lista_pastas_para_exibir]
                lista_pastas_para_exibir.extend(sorted(pastas_sem_duplicatas))

        # --- Exibição Principal ---
        if not lista_pastas_para_exibir and not df_exibir.empty:
            st.info("Nenhuma pasta corresponde a todos os critérios de filtro de exibição selecionados.")
        elif df_exibir.empty :
             st.info("Nenhuma atividade 'Verificar' corresponde aos filtros de data e exibição selecionados.")

        for nome_pasta in lista_pastas_para_exibir:
            df_pasta_exibicao_atual = df_exibir[df_exibir['activity_folder'] == nome_pasta]
            multi_user_info = " (Múltiplos Usuários na Análise)" if nome_pasta in pastas_com_multiplos_usuarios_set else ""
            
            with st.expander(f"📁 Pasta: {nome_pasta} ({len(df_pasta_exibicao_atual)} atividades nesta exibição){multi_user_info}", expanded=True):
                if nome_pasta in pastas_com_multiplos_usuarios_set:
                     nomes_originais = df_atividades_raw[df_atividades_raw['activity_folder'] == nome_pasta]['user_profile_name'].unique()
                     st.caption(f"👥 Usuários na análise completa desta pasta: {', '.join(nomes_originais)}")

                for _, atividade_row in df_pasta_exibicao_atual.iterrows(): # Renomeado para evitar conflito
                    atividade = atividade_row.to_dict() # Converter para dict para passar para as funções de dialog
                    act_id = atividade['activity_id']
                    links = gerar_links_zflow(act_id)
                    
                    st.markdown("---")
                    cols_main = st.columns([0.6, 0.4]) 
                    
                    with cols_main[0]:
                        st.markdown(f"**ID:** `{act_id}` | **Data:** {atividade['activity_date'].strftime('%d/%m/%Y')} | **Status:** `{atividade['activity_status']}`")
                        st.markdown(f"**Usuário:** {atividade['user_profile_name']}")
                        
                        btn_cols_actions = st.columns(3)
                        if btn_cols_actions[0].button("👁️ Ver Texto", key=f"ver_texto_{act_id}", help="Abrir texto completo da publicação", on_click=abrir_dialog_texto, args=(atividade,)):
                            pass # Ação é feita pelo on_click
                        btn_cols_actions[1].link_button("🔗 ZFlow v1", links['antigo'], help="Abrir no ZFlow (versão antiga)")
                        btn_cols_actions[2].link_button("🔗 ZFlow v2", links['novo'], help="Abrir no ZFlow (versão nova)")

                    with cols_main[1]:
                        duplicatas_da_atividade = similaridades_globais.get(act_id, [])
                        if duplicatas_da_atividade:
                            st.markdown(f"**<span style='color:red;'>Potenciais Duplicatas:</span>** ({len(duplicatas_da_atividade)})", unsafe_allow_html=True)
                            for dup_info in duplicatas_da_atividade:
                                container_dup_display = st.container(border=True) # Renomeado para evitar conflito
                                container_dup_display.markdown(
                                    f"<small><span style='background-color:{dup_info['cor']}; padding: 1px 3px; border-radius: 3px; color: black;'>"
                                    f"ID: {dup_info['id_similar']} ({dup_info['ratio']:.0%})</span><br>"
                                    f"Data: {dup_info['data_similar'].strftime('%d/%m')} | Status: `{dup_info['status_similar']}`<br>"
                                    f"Usuário: {dup_info['usuario_similar']}</small>",
                                    unsafe_allow_html=True
                                )
                                atividade_comparar_obj_dict = df_atividades_raw[df_atividades_raw['activity_id'] == dup_info['id_similar']].iloc[0].to_dict()
                                if container_dup_display.button("⚖️ Comparar Textos", key=f"comparar_{act_id}_com_{dup_info['id_similar']}", help="Comparar textos lado a lado", on_click=abrir_dialog_comparacao, args=(atividade, atividade_comparar_obj_dict)):
                                    pass # Ação é feita pelo on_click
                        elif apenas_potenciais_duplicatas_cb:
                            pass 
                        else:
                            st.markdown(f"<small style='color:green;'>Sem duplicatas (acima de {min_similaridade_display:.0%})</small>", unsafe_allow_html=True)
        
# --- Renderização Condicional dos Dialogs (fora do loop principal) ---
if st.session_state.show_texto_dialog and st.session_state.atividade_para_texto_dialog:
    with st.dialog("Texto Completo da Atividade",  on_dismiss=fechar_dialog_texto): # Usar on_dismiss para fechar ao clicar fora
        atividade_selecionada = st.session_state.atividade_para_texto_dialog
        st.markdown(f"### Texto Completo - Atividade ID: `{atividade_selecionada['activity_id']}`")
        st.markdown(f"**Pasta:** {atividade_selecionada['activity_folder']} | **Data:** {atividade_selecionada['activity_date'].strftime('%d/%m/%Y')} | **Usuário:** {atividade_selecionada['user_profile_name']} | **Status:** {atividade_selecionada['activity_status']}")
        st.text_area("Texto da Publicação:", value=str(atividade_selecionada['Texto']), height=400, disabled=True, key=f"full_text_dialog_content_{atividade_selecionada['activity_id']}")
        if st.button("Fechar Texto", key="fechar_btn_texto_dialog", on_click=fechar_dialog_texto):
            pass

elif st.session_state.show_comparacao_dialog and st.session_state.atividades_para_comparacao_dialog:
    atividades_para_comparar = st.session_state.atividades_para_comparacao_dialog
    base = atividades_para_comparar['base']
    comparar = atividades_para_comparar['comparar']
    
    with st.dialog("Comparação Detalhada de Textos", on_dismiss=fechar_dialog_comparacao): # Usar on_dismiss
        texto_base_str = str(base['Texto']) # Renomeado para evitar conflito
        texto_comparar_str = str(comparar['Texto']) # Renomeado para evitar conflito
        
        d_compare = difflib.HtmlDiff(wrapcolumn=70) # Renomeado para evitar conflito
        html_diff_content = d_compare.make_table(texto_base_str.splitlines(), texto_comparar_str.splitlines(),
                                         fromdesc=f"Texto Atividade ID: {base['activity_id']}",
                                         todesc=f"Texto Atividade ID: {comparar['activity_id']}")

        st.markdown(f"### Comparando Atividade `{base['activity_id']}` com `{comparar['activity_id']}`")
        st.components.v1.html(html_diff_content, height=600, scrolling=True)
        if st.button("Fechar Comparação", key="fechar_btn_comparacao_dialog", on_click=fechar_dialog_comparacao):
            pass
else:
    st.error("Conexão com o banco falhou. Verifique as credenciais e o status do banco.")

st.sidebar.info("Verificador de Duplicidade v2.1")
