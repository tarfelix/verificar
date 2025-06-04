import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import difflib
import io # Para exportação XLSX

# ==============================================================================
# CONFIGURAÇÕES E FUNÇÕES AUXILIARES
# ==============================================================================
st.set_page_config(layout="wide", page_title="Verificador de Duplicidade Melhorado")

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
def buscar_atividades_raw(_engine, data_inicio, data_fim): # Renomeado para clareza
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
# Estado da Sessão para Comparação
# ==============================================================================
if 'comparacao_ativa' not in st.session_state:
    st.session_state.comparacao_ativa = None # Guarda {'id_base': ..., 'id_comparar': ...}
if 'atividade_base_comparacao' not in st.session_state:
    st.session_state.atividade_base_comparacao = None
if 'atividade_comparar_comparacao' not in st.session_state:
    st.session_state.atividade_comparar_comparacao = None


# ==============================================================================
# INTERFACE DO USUÁRIO (Streamlit)
# ==============================================================================
st.title("🔎 Verificador de Duplicidade Melhorado")
st.markdown("Análise de atividades 'Verificar' para identificar potenciais duplicidades.")
# Aviso de credenciais removido conforme solicitado

engine = get_db_engine()

if engine:
    # --- Sidebar: Filtros e Opções ---
    st.sidebar.header("⚙️ Filtros e Opções")
    hoje = datetime.today().date()
    
    periodo_selecionado = st.sidebar.radio("Período das atividades:", ("Hoje, Ontem e Amanhã", "Intervalo Personalizado"), key="periodo_radio_v4")
    data_inicio_filtro, data_fim_filtro = (hoje - timedelta(days=1), hoje + timedelta(days=1)) if periodo_selecionado == "Hoje, Ontem e Amanhã" else \
                                          (st.sidebar.date_input("Data Início", hoje - timedelta(days=1), key="data_inicio_v4"), st.sidebar.date_input("Data Fim", hoje + timedelta(days=1), key="data_fim_v4"))

    if data_inicio_filtro > data_fim_filtro:
        st.sidebar.error("Data de início posterior à data de fim.")
        st.stop()

    df_atividades_inicial = buscar_atividades_raw(engine, data_inicio_filtro, data_fim_filtro)

    if df_atividades_inicial.empty:
        st.info(f"Nenhuma atividade 'Verificar' no período de {data_inicio_filtro.strftime('%d/%m/%Y')} a {data_fim_filtro.strftime('%d/%m/%Y')}.")
    else:
        st.success(f"{len(df_atividades_inicial)} atividades 'Verificar' (todos os status) carregadas para o período inicial.")

        # --- Filtros Adicionais na Sidebar ---
        pastas_disponiveis = sorted(df_atividades_inicial['activity_folder'].dropna().unique())
        pastas_selecionadas = st.sidebar.multiselect("Filtrar por Pasta(s):", pastas_disponiveis, default=[], key="pasta_filter_v4")

        status_disponiveis = sorted(df_atividades_inicial['activity_status'].dropna().unique())
        status_selecionados = st.sidebar.multiselect("Filtrar por Status:", status_disponiveis, default=[], key="status_filter_v4")
        
        usuarios_disponiveis = sorted(df_atividades_inicial['user_profile_name'].dropna().unique())
        usuarios_selecionados_exibicao = st.sidebar.multiselect("Filtrar exibição por Usuário(s):", usuarios_disponiveis, default=[], key="user_filter_v4")
        
        min_similaridade_display = st.sidebar.slider("Similaridade de texto mínima (%):", 0, 100, 50, 5, key="sim_slider_v4") / 100.0
        
        apenas_potenciais_duplicatas_cb = st.sidebar.checkbox("Mostrar apenas com potenciais duplicatas", False, key="dup_cb_v4")
        apenas_usuarios_diferentes_cb = st.sidebar.checkbox("Mostrar apenas pastas com múltiplos usuários (na análise)", False, key="multiuser_cb_v4")

        ordem_pastas = st.sidebar.selectbox(
            "Ordenar pastas por:",
            ("Nome da Pasta (A-Z)", "Mais Atividades Primeiro", "Mais Potenciais Duplicatas Primeiro (beta)"),
            key="ordem_pastas_v4"
        )
        st.sidebar.markdown("---")
        
        # Aplicar filtros de pasta e status ANTES da análise de duplicidade
        df_analise = df_atividades_inicial.copy()
        if pastas_selecionadas:
            df_analise = df_analise[df_analise['activity_folder'].isin(pastas_selecionadas)]
        if status_selecionados:
            df_analise = df_analise[df_analise['activity_status'].isin(status_selecionados)]

        # --- Pré-processamento e Análise de Duplicidade (sobre df_analise) ---
        similaridades_globais = {} 
        atividades_com_duplicatas_ids = set()
        pastas_com_multiplos_usuarios_set_analise = set() # Baseado no df_analise

        if not df_analise.empty:
            for nome_pasta_analise, df_pasta_para_analise in df_analise.groupby('activity_folder'):
                if df_pasta_para_analise['user_profile_name'].nunique() > 1:
                    pastas_com_multiplos_usuarios_set_analise.add(nome_pasta_analise)

                atividades_lista_analise = df_pasta_para_analise.to_dict('records')
                for i in range(len(atividades_lista_analise)):
                    base = atividades_lista_analise[i]
                    if base['activity_id'] not in similaridades_globais: similaridades_globais[base['activity_id']] = []
                    
                    for j in range(i + 1, len(atividades_lista_analise)):
                        comparar = atividades_lista_analise[j]
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
            
            for key_sim in similaridades_globais: # Renomeado para evitar conflito
                similaridades_globais[key_sim] = sorted(similaridades_globais[key_sim], key=lambda x: x['ratio'], reverse=True)

        # --- Filtragem para Exibição (sobre df_analise) ---
        df_exibir_final = df_analise.copy() # Começa com o df já filtrado por pasta e status
        if usuarios_selecionados_exibicao: # Filtro de usuário é apenas para exibição
            df_exibir_final = df_exibir_final[df_exibir_final['user_profile_name'].isin(usuarios_selecionados_exibicao)]
        if apenas_potenciais_duplicatas_cb:
            df_exibir_final = df_exibir_final[df_exibir_final['activity_id'].isin(atividades_com_duplicatas_ids)]
        if apenas_usuarios_diferentes_cb: # Este filtro agora se baseia nas pastas identificadas em df_analise
            df_exibir_final = df_exibir_final[df_exibir_final['activity_folder'].isin(pastas_com_multiplos_usuarios_set_analise)]


        # --- Botão de Exportação ---
        if st.sidebar.button("Exportar Dados Exibidos para XLSX", key="export_xlsx_btn_v4"):
            if not df_exibir_final.empty:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_exibir_final.to_excel(writer, index=False, sheet_name='Atividades_Filtradas')
                    
                    # Criar uma aba para duplicatas detalhadas (opcional, mas útil)
                    duplicatas_export_list = []
                    for act_id_export, dups in similaridades_globais.items():
                        if act_id_export in df_exibir_final['activity_id'].values: # Apenas para atividades exibidas
                            for dup_info_export in dups:
                                duplicatas_export_list.append({
                                    'ID_Base': act_id_export,
                                    'ID_Duplicata_Potencial': dup_info_export['id_similar'],
                                    'Percentual_Similaridade': dup_info_export['ratio'],
                                    'Data_Duplicata': dup_info_export['data_similar'],
                                    'Usuario_Duplicata': dup_info_export['usuario_similar'],
                                    'Status_Duplicata': dup_info_export['status_similar']
                                })
                    if duplicatas_export_list:
                        df_duplicatas_export = pd.DataFrame(duplicatas_export_list)
                        df_duplicatas_export.to_excel(writer, index=False, sheet_name='Potenciais_Duplicatas')

                st.sidebar.download_button(
                    label="Baixar XLSX",
                    data=output.getvalue(),
                    file_name=f"atividades_verificar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.sidebar.warning("Nenhum dado para exportar com os filtros atuais.")


        # --- Lógica de Ordenação das Pastas (sobre df_exibir_final) ---
        lista_pastas_para_renderizar = [] # Renomeado para evitar conflito
        if not df_exibir_final.empty:
            pastas_agrupadas_renderizar = df_exibir_final.groupby('activity_folder') # Renomeado
            
            if ordem_pastas == "Nome da Pasta (A-Z)":
                lista_pastas_para_renderizar = sorted(pastas_agrupadas_renderizar.groups.keys())
            elif ordem_pastas == "Mais Atividades Primeiro":
                lista_pastas_para_renderizar = pastas_agrupadas_renderizar.size().sort_values(ascending=False).index.tolist()
            elif ordem_pastas == "Mais Potenciais Duplicatas Primeiro (beta)":
                contagem_duplicatas_pasta_render = {} # Renomeado
                for nome_pasta_render, df_p_render in pastas_agrupadas_renderizar: # Renomeado
                    count_render = 0 # Renomeado
                    for act_id_render in df_p_render['activity_id']: # Renomeado
                        if act_id_render in similaridades_globais and similaridades_globais[act_id_render]:
                            count_render +=1 
                            break 
                    if count_render > 0: contagem_duplicatas_pasta_render[nome_pasta_render] = df_p_render[df_p_render['activity_id'].isin(atividades_com_duplicatas_ids)].shape[0]
                lista_pastas_para_renderizar = sorted(contagem_duplicatas_pasta_render, key=contagem_duplicatas_pasta_render.get, reverse=True)
                pastas_sem_duplicatas_render = [p_render for p_render in pastas_agrupadas_renderizar.groups.keys() if p_render not in lista_pastas_para_renderizar] # Renomeado
                lista_pastas_para_renderizar.extend(sorted(pastas_sem_duplicatas_render))

        # --- Exibição Principal ---
        st.header("Resultados da Análise")
        if not lista_pastas_para_renderizar and not df_exibir_final.empty:
            st.info("Nenhuma pasta corresponde a todos os critérios de filtro de exibição selecionados.")
        elif df_exibir_final.empty :
             st.info("Nenhuma atividade 'Verificar' corresponde aos filtros aplicados.")

        for nome_pasta_render_loop in lista_pastas_para_renderizar: # Renomeado
            df_pasta_render_loop = df_exibir_final[df_exibir_final['activity_folder'] == nome_pasta_render_loop] # Renomeado
            multi_user_info_display = " (Múltiplos Usuários na Análise)" if nome_pasta_render_loop in pastas_com_multiplos_usuarios_set_analise else "" # Renomeado
            
            with st.expander(f"📁 Pasta: {nome_pasta_render_loop} ({len(df_pasta_render_loop)} atividades nesta exibição){multi_user_info_display}", expanded=True):
                if nome_pasta_render_loop in pastas_com_multiplos_usuarios_set_analise:
                     # Mostra usuários da análise original (df_analise) para esta pasta
                     nomes_originais_analise = df_analise[df_analise['activity_folder'] == nome_pasta_render_loop]['user_profile_name'].unique()
                     st.caption(f"👥 Usuários nesta pasta (considerando filtros de pasta/status): {', '.join(nomes_originais_analise)}")

                for _, atividade_render_loop in df_pasta_render_loop.iterrows(): # Renomeado
                    act_id_loop = atividade_render_loop['activity_id'] # Renomeado
                    links_loop = gerar_links_zflow(act_id_loop) # Renomeado
                    
                    st.markdown("---")
                    main_cols_display = st.columns([0.6, 0.4])  # Renomeado
                    
                    with main_cols_display[0]:
                        st.markdown(f"**ID:** `{act_id_loop}` | **Data:** {atividade_render_loop['activity_date'].strftime('%d/%m/%Y')} | **Status:** `{atividade_render_loop['activity_status']}`")
                        st.markdown(f"**Usuário:** {atividade_render_loop['user_profile_name']}")
                        st.text_area("Texto da Publicação:", value=str(atividade_render_loop['Texto']), height=100, key=f"texto_area_{act_id_loop}", disabled=True)
                        
                        action_btn_cols = st.columns(2) # Renomeado
                        action_btn_cols[0].link_button("🔗 ZFlow v1", links_loop['antigo'], help="Abrir no ZFlow (versão antiga)", key=f"zflow1_{act_id_loop}")
                        action_btn_cols[1].link_button("🔗 ZFlow v2", links_loop['novo'], help="Abrir no ZFlow (versão nova)", key=f"zflow2_{act_id_loop}")

                    with main_cols_display[1]:
                        duplicatas_da_atividade_loop = similaridades_globais.get(act_id_loop, []) # Renomeado
                        if duplicatas_da_atividade_loop:
                            st.markdown(f"**<span style='color:red;'>Potenciais Duplicatas:</span>** ({len(duplicatas_da_atividade_loop)})", unsafe_allow_html=True)
                            for dup_info_loop in duplicatas_da_atividade_loop: # Renomeado
                                dup_container_display = st.container(border=True) # Renomeado
                                dup_container_display.markdown(
                                    f"<small><span style='background-color:{dup_info_loop['cor']}; padding: 1px 3px; border-radius: 3px; color: black;'>"
                                    f"ID: {dup_info_loop['id_similar']} ({dup_info_loop['ratio']:.0%})</span><br>"
                                    f"Data: {dup_info_loop['data_similar'].strftime('%d/%m')} | Status: `{dup_info_loop['status_similar']}`<br>"
                                    f"Usuário: {dup_info_loop['usuario_similar']}</small>",
                                    unsafe_allow_html=True
                                )
                                
                                # Botão para ativar a comparação
                                if dup_container_display.button("⚖️ Comparar Textos", key=f"comparar_btn_{act_id_loop}_com_{dup_info_loop['id_similar']}", help="Comparar textos lado a lado"):
                                    st.session_state.comparacao_ativa = {'id_base': act_id_loop, 'id_comparar': dup_info_loop['id_similar']}
                                    # Encontrar os dados completos das atividades para comparação
                                    st.session_state.atividade_base_comparacao = df_atividades_inicial[df_atividades_inicial['activity_id'] == act_id_loop].iloc[0].to_dict()
                                    st.session_state.atividade_comparar_comparacao = df_atividades_inicial[df_atividades_inicial['activity_id'] == dup_info_loop['id_similar']].iloc[0].to_dict()
                                    st.rerun() # Força o rerun para exibir a comparação

                        elif apenas_potenciais_duplicatas_cb:
                            pass 
                        else:
                            st.markdown(f"<small style='color:green;'>Sem duplicatas (acima de {min_similaridade_display:.0%})</small>", unsafe_allow_html=True)

                    # Exibir a comparação se ativa para esta atividade base
                    if st.session_state.comparacao_ativa and st.session_state.comparacao_ativa['id_base'] == act_id_loop:
                        base_comp = st.session_state.atividade_base_comparacao
                        comparar_comp = st.session_state.atividade_comparar_comparacao
                        id_comparado_atual = st.session_state.comparacao_ativa['id_comparar']

                        st.markdown("---")
                        st.subheader(f"🔎 Comparação Detalhada: ID `{base_comp['activity_id']}` vs ID `{comparar_comp['activity_id']}`")
                        
                        texto_base_comp = str(base_comp['Texto'])
                        texto_comparar_comp = str(comparar_comp['Texto'])
                        
                        html_differ = difflib.HtmlDiff(wrapcolumn=70)
                        html_comparison = html_differ.make_table(texto_base_comp.splitlines(), texto_comparar_comp.splitlines(),
                                                             fromdesc=f"Atividade ID: {base_comp['activity_id']}",
                                                             todesc=f"Atividade ID: {comparar_comp['activity_id']}")
                        st.components.v1.html(html_comparison, height=600, scrolling=True)
                        if st.button("Fechar Comparação", key=f"fechar_comp_{act_id_loop}_{id_comparado_atual}"):
                            st.session_state.comparacao_ativa = None
                            st.session_state.atividade_base_comparacao = None
                            st.session_state.atividade_comparar_comparacao = None
                            st.rerun()
                        st.markdown("---")


        # Limpar estado de comparação se nenhuma comparação estiver ativa (caso o usuário navegue ou mude filtros)
        # Esta lógica pode precisar de refinamento para não fechar a comparação inesperadamente.
        # Por enquanto, a comparação é fechada explicitamente pelo botão "Fechar Comparação".

else:
    st.error("Conexão com o banco falhou. Verifique as credenciais e o status do banco.")

st.sidebar.info("Verificador de Duplicidade v3")

