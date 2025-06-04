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
def buscar_atividades_raw(_engine, data_inicio, data_fim):
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
# Estado da Sessão para Dialogs e Comparação
# ==============================================================================
if 'show_texto_dialog_v4' not in st.session_state: # Chaves únicas para esta versão
    st.session_state.show_texto_dialog_v4 = False
if 'atividade_para_texto_dialog_v4' not in st.session_state:
    st.session_state.atividade_para_texto_dialog_v4 = None

if 'show_comparacao_dialog_v4' not in st.session_state:
    st.session_state.show_comparacao_dialog_v4 = False
if 'atividades_para_comparacao_dialog_v4' not in st.session_state:
    st.session_state.atividades_para_comparacao_dialog_v4 = None

# --- Estado para comparação direta na página (substituindo dialog para comparação) ---
if 'comparacao_direta_ativa' not in st.session_state:
    st.session_state.comparacao_direta_ativa = None # Guarda {'id_base': ..., 'id_comparar': ...}
if 'atividade_base_comparacao_direta' not in st.session_state:
    st.session_state.atividade_base_comparacao_direta = None
if 'atividade_comparar_comparacao_direta' not in st.session_state:
    st.session_state.atividade_comparar_comparacao_direta = None


# ==============================================================================
# Funções para abrir e fechar dialog de TEXTO
# ==============================================================================
def abrir_dialog_texto_v4(atividade):
    st.session_state.atividade_para_texto_dialog_v4 = atividade
    st.session_state.show_texto_dialog_v4 = True

def fechar_dialog_texto_v4():
    st.session_state.show_texto_dialog_v4 = False
    st.session_state.atividade_para_texto_dialog_v4 = None

# ==============================================================================
# INTERFACE DO USUÁRIO (Streamlit)
# ==============================================================================
st.title("🔎 Verificador de Duplicidade Melhorado")
st.markdown("Análise de atividades 'Verificar' para identificar potenciais duplicidades.")

engine = get_db_engine()

if engine:
    # --- Sidebar: Filtros e Opções ---
    st.sidebar.header("⚙️ Filtros e Opções")
    hoje = datetime.today().date()
    
    periodo_selecionado = st.sidebar.radio("Período das atividades:", ("Hoje, Ontem e Amanhã", "Intervalo Personalizado"), key="periodo_radio_v5")
    data_inicio_filtro, data_fim_filtro = (hoje - timedelta(days=1), hoje + timedelta(days=1)) if periodo_selecionado == "Hoje, Ontem e Amanhã" else \
                                          (st.sidebar.date_input("Data Início", hoje - timedelta(days=1), key="data_inicio_v5"), st.sidebar.date_input("Data Fim", hoje + timedelta(days=1), key="data_fim_v5"))

    if data_inicio_filtro > data_fim_filtro:
        st.sidebar.error("Data de início posterior à data de fim.")
        st.stop()

    df_atividades_inicial = buscar_atividades_raw(engine, data_inicio_filtro, data_fim_filtro)

    if df_atividades_inicial.empty:
        st.info(f"Nenhuma atividade 'Verificar' no período de {data_inicio_filtro.strftime('%d/%m/%Y')} a {data_fim_filtro.strftime('%d/%m/%Y')}.")
    else:
        st.success(f"{len(df_atividades_inicial)} atividades 'Verificar' (todos os status) carregadas para o período inicial.")

        pastas_disponiveis = sorted(df_atividades_inicial['activity_folder'].dropna().unique())
        pastas_selecionadas = st.sidebar.multiselect("Filtrar por Pasta(s):", pastas_disponiveis, default=[], key="pasta_filter_v5")

        status_disponiveis = sorted(df_atividades_inicial['activity_status'].dropna().unique())
        status_selecionados = st.sidebar.multiselect("Filtrar por Status:", status_disponiveis, default=[], key="status_filter_v5")
        
        usuarios_disponiveis = sorted(df_atividades_inicial['user_profile_name'].dropna().unique())
        usuarios_selecionados_exibicao = st.sidebar.multiselect("Filtrar exibição por Usuário(s):", usuarios_disponiveis, default=[], key="user_filter_v5")
        
        min_similaridade_display = st.sidebar.slider("Similaridade de texto mínima (%):", 0, 100, 50, 5, key="sim_slider_v5") / 100.0
        
        apenas_potenciais_duplicatas_cb = st.sidebar.checkbox("Mostrar apenas com potenciais duplicatas", False, key="dup_cb_v5")
        apenas_usuarios_diferentes_cb = st.sidebar.checkbox("Mostrar apenas pastas com múltiplos usuários (na análise)", False, key="multiuser_cb_v5")

        ordem_pastas = st.sidebar.selectbox(
            "Ordenar pastas por:",
            ("Nome da Pasta (A-Z)", "Mais Atividades Primeiro", "Mais Potenciais Duplicatas Primeiro (beta)"),
            key="ordem_pastas_v5"
        )
        st.sidebar.markdown("---")
        
        df_analise = df_atividades_inicial.copy()
        if pastas_selecionadas:
            df_analise = df_analise[df_analise['activity_folder'].isin(pastas_selecionadas)]
        if status_selecionados:
            df_analise = df_analise[df_analise['activity_status'].isin(status_selecionados)]

        similaridades_globais = {} 
        atividades_com_duplicatas_ids = set()
        pastas_com_multiplos_usuarios_set_analise = set()

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
            for key_sim_v5 in similaridades_globais:
                similaridades_globais[key_sim_v5] = sorted(similaridades_globais[key_sim_v5], key=lambda x: x['ratio'], reverse=True)

        df_exibir_final = df_analise.copy()
        if usuarios_selecionados_exibicao:
            df_exibir_final = df_exibir_final[df_exibir_final['user_profile_name'].isin(usuarios_selecionados_exibicao)]
        if apenas_potenciais_duplicatas_cb:
            df_exibir_final = df_exibir_final[df_exibir_final['activity_id'].isin(atividades_com_duplicatas_ids)]
        if apenas_usuarios_diferentes_cb:
            df_exibir_final = df_exibir_final[df_exibir_final['activity_folder'].isin(pastas_com_multiplos_usuarios_set_analise)]

        if st.sidebar.button("Exportar Dados Exibidos para XLSX", key="export_xlsx_btn_v5"):
            if not df_exibir_final.empty:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_exibir_final.to_excel(writer, index=False, sheet_name='Atividades_Filtradas')
                    duplicatas_export_list = []
                    for act_id_export_v5, dups_v5 in similaridades_globais.items():
                        if act_id_export_v5 in df_exibir_final['activity_id'].values:
                            for dup_info_export_v5 in dups_v5:
                                duplicatas_export_list.append({
                                    'ID_Base': act_id_export_v5,
                                    'ID_Duplicata_Potencial': dup_info_export_v5['id_similar'],
                                    'Percentual_Similaridade': dup_info_export_v5['ratio'],
                                    'Data_Duplicata': dup_info_export_v5['data_similar'],
                                    'Usuario_Duplicata': dup_info_export_v5['usuario_similar'],
                                    'Status_Duplicata': dup_info_export_v5['status_similar']
                                })
                    if duplicatas_export_list:
                        df_duplicatas_export = pd.DataFrame(duplicatas_export_list)
                        df_duplicatas_export.to_excel(writer, index=False, sheet_name='Potenciais_Duplicatas')
                st.sidebar.download_button(
                    label="Baixar XLSX", data=output.getvalue(),
                    file_name=f"atividades_verificar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.sidebar.warning("Nenhum dado para exportar com os filtros atuais.")

        lista_pastas_para_renderizar = []
        if not df_exibir_final.empty:
            pastas_agrupadas_renderizar = df_exibir_final.groupby('activity_folder')
            if ordem_pastas == "Nome da Pasta (A-Z)":
                lista_pastas_para_renderizar = sorted(pastas_agrupadas_renderizar.groups.keys())
            elif ordem_pastas == "Mais Atividades Primeiro":
                lista_pastas_para_renderizar = pastas_agrupadas_renderizar.size().sort_values(ascending=False).index.tolist()
            elif ordem_pastas == "Mais Potenciais Duplicatas Primeiro (beta)":
                contagem_duplicatas_pasta_render = {}
                for nome_pasta_render_v5, df_p_render_v5 in pastas_agrupadas_renderizar:
                    count_render_v5 = 0
                    for act_id_render_v5 in df_p_render_v5['activity_id']:
                        if act_id_render_v5 in similaridades_globais and similaridades_globais[act_id_render_v5]:
                            count_render_v5 +=1 
                            break 
                    if count_render_v5 > 0: contagem_duplicatas_pasta_render[nome_pasta_render_v5] = df_p_render_v5[df_p_render_v5['activity_id'].isin(atividades_com_duplicatas_ids)].shape[0]
                lista_pastas_para_renderizar = sorted(contagem_duplicatas_pasta_render, key=contagem_duplicatas_pasta_render.get, reverse=True)
                pastas_sem_duplicatas_render = [p_render_v5 for p_render_v5 in pastas_agrupadas_renderizar.groups.keys() if p_render_v5 not in lista_pastas_para_renderizar]
                lista_pastas_para_renderizar.extend(sorted(pastas_sem_duplicatas_render))

        st.header("Resultados da Análise")
        if not lista_pastas_para_renderizar and not df_exibir_final.empty:
            st.info("Nenhuma pasta corresponde a todos os critérios de filtro de exibição selecionados.")
        elif df_exibir_final.empty :
             st.info("Nenhuma atividade 'Verificar' corresponde aos filtros aplicados.")

        # --- Seção de Comparação (fora do loop de pastas) ---
        if st.session_state.comparacao_direta_ativa:
            base_comp_direct = st.session_state.atividade_base_comparacao_direta
            comparar_comp_direct = st.session_state.atividade_comparar_comparacao_direta
            
            st.markdown("---")
            st.subheader(f"🔎 Comparação Detalhada: ID `{base_comp_direct['activity_id']}` vs ID `{comparar_comp_direct['activity_id']}`")
            
            texto_base_comp_direct = str(base_comp_direct['Texto'])
            texto_comparar_comp_direct = str(comparar_comp_direct['Texto'])
            
            html_differ_direct = difflib.HtmlDiff(wrapcolumn=70)
            html_comparison_direct = html_differ_direct.make_table(texto_base_comp_direct.splitlines(), texto_comparar_comp_direct.splitlines(),
                                                         fromdesc=f"Atividade ID: {base_comp_direct['activity_id']}",
                                                         todesc=f"Atividade ID: {comparar_comp_direct['activity_id']}")
            st.components.v1.html(html_comparison_direct, height=600, scrolling=True)
            if st.button("Fechar Comparação", key=f"fechar_comp_direta_{base_comp_direct['activity_id']}_{comparar_comp_direct['activity_id']}"):
                st.session_state.comparacao_direta_ativa = None
                st.session_state.atividade_base_comparacao_direta = None
                st.session_state.atividade_comparar_comparacao_direta = None
                st.rerun()
            st.markdown("---")


        for nome_pasta_render_loop_v5 in lista_pastas_para_renderizar:
            df_pasta_render_loop_v5 = df_exibir_final[df_exibir_final['activity_folder'] == nome_pasta_render_loop_v5]
            multi_user_info_display_v5 = " (Múltiplos Usuários na Análise)" if nome_pasta_render_loop_v5 in pastas_com_multiplos_usuarios_set_analise else ""
            
            with st.expander(f"📁 Pasta: {nome_pasta_render_loop_v5} ({len(df_pasta_render_loop_v5)} atividades nesta exibição){multi_user_info_display_v5}", expanded=True):
                if nome_pasta_render_loop_v5 in pastas_com_multiplos_usuarios_set_analise:
                     nomes_originais_analise_v5 = df_analise[df_analise['activity_folder'] == nome_pasta_render_loop_v5]['user_profile_name'].unique()
                     st.caption(f"👥 Usuários nesta pasta (considerando filtros de pasta/status): {', '.join(nomes_originais_analise_v5)}")

                for _, atividade_render_loop_v5_row in df_pasta_render_loop_v5.iterrows():
                    atividade_dict_v5 = atividade_render_loop_v5_row.to_dict() # Para passar para on_click
                    act_id_loop_v5 = atividade_dict_v5['activity_id']
                    links_loop_v5 = gerar_links_zflow(act_id_loop_v5)
                    
                    st.markdown("---")
                    main_cols_display_v5 = st.columns([0.6, 0.4])  
                    
                    with main_cols_display_v5[0]:
                        st.markdown(f"**ID:** `{act_id_loop_v5}` | **Data:** {atividade_dict_v5['activity_date'].strftime('%d/%m/%Y')} | **Status:** `{atividade_dict_v5['activity_status']}`")
                        st.markdown(f"**Usuário:** {atividade_dict_v5['user_profile_name']}")
                        st.text_area("Texto da Publicação:", value=str(atividade_dict_v5['Texto']), height=100, key=f"texto_area_{act_id_loop_v5}", disabled=True)
                        
                        action_btn_cols_v5 = st.columns(3)
                        # Botão Ver Texto agora usa o dialog
                        if action_btn_cols_v5[0].button("👁️ Ver Texto", key=f"ver_texto_btn_{act_id_loop_v5}", help="Abrir texto completo da publicação", on_click=abrir_dialog_texto_v4, args=(atividade_dict_v5,)):
                            pass # Ação é feita pelo on_click
                        action_btn_cols_v5[1].link_button("� ZFlow v1", links_loop_v5['antigo'], help="Abrir no ZFlow (versão antiga)") # Removido 'key'
                        action_btn_cols_v5[2].link_button("🔗 ZFlow v2", links_loop_v5['novo'], help="Abrir no ZFlow (versão nova)") # Removido 'key'

                    with main_cols_display_v5[1]:
                        duplicatas_da_atividade_loop_v5 = similaridades_globais.get(act_id_loop_v5, [])
                        if duplicatas_da_atividade_loop_v5:
                            st.markdown(f"**<span style='color:red;'>Potenciais Duplicatas:</span>** ({len(duplicatas_da_atividade_loop_v5)})", unsafe_allow_html=True)
                            for dup_info_loop_v5 in duplicatas_da_atividade_loop_v5:
                                dup_container_display_v5 = st.container(border=True)
                                dup_container_display_v5.markdown(
                                    f"<small><span style='background-color:{dup_info_loop_v5['cor']}; padding: 1px 3px; border-radius: 3px; color: black;'>"
                                    f"ID: {dup_info_loop_v5['id_similar']} ({dup_info_loop_v5['ratio']:.0%})</span><br>"
                                    f"Data: {dup_info_loop_v5['data_similar'].strftime('%d/%m')} | Status: `{dup_info_loop_v5['status_similar']}`<br>"
                                    f"Usuário: {dup_info_loop_v5['usuario_similar']}</small>",
                                    unsafe_allow_html=True
                                )
                                
                                if dup_container_display_v5.button("⚖️ Comparar Textos", key=f"comparar_btn_direto_{act_id_loop_v5}_com_{dup_info_loop_v5['id_similar']}", help="Comparar textos lado a lado"):
                                    st.session_state.comparacao_direta_ativa = {'id_base': act_id_loop_v5, 'id_comparar': dup_info_loop_v5['id_similar']}
                                    st.session_state.atividade_base_comparacao_direta = df_atividades_inicial[df_atividades_inicial['activity_id'] == act_id_loop_v5].iloc[0].to_dict()
                                    st.session_state.atividade_comparar_comparacao_direta = df_atividades_inicial[df_atividades_inicial['activity_id'] == dup_info_loop_v5['id_similar']].iloc[0].to_dict()
                                    st.rerun()
                        elif apenas_potenciais_duplicatas_cb:
                            pass 
                        else:
                            st.markdown(f"<small style='color:green;'>Sem duplicatas (acima de {min_similaridade_display:.0%})</small>", unsafe_allow_html=True)

        # --- Renderização Condicional do Dialog de TEXTO (fora do loop principal) ---
        if st.session_state.show_texto_dialog_v4 and st.session_state.atividade_para_texto_dialog_v4:
            # Removido on_dismiss pois não era suportado
            with st.dialog("Texto Completo da Atividade"): 
                atividade_selecionada_v5 = st.session_state.atividade_para_texto_dialog_v4
                st.markdown(f"### Texto Completo - Atividade ID: `{atividade_selecionada_v5['activity_id']}`")
                st.markdown(f"**Pasta:** {atividade_selecionada_v5['activity_folder']} | **Data:** {atividade_selecionada_v5['activity_date'].strftime('%d/%m/%Y')} | **Usuário:** {atividade_selecionada_v5['user_profile_name']} | **Status:** {atividade_selecionada_v5['activity_status']}")
                st.text_area("Texto da Publicação:", value=str(atividade_selecionada_v5['Texto']), height=400, disabled=True, key=f"full_text_dialog_content_v5_{atividade_selecionada_v5['activity_id']}")
                if st.button("Fechar Texto", key="fechar_btn_texto_dialog_v5", on_click=fechar_dialog_texto_v4):
                    pass # Ação é feita pelo on_click, que chama st.rerun() implicitamente
else:
    st.error("Conexão com o banco falhou. Verifique as credenciais e o status do banco.")

st.sidebar.info("Verificador de Duplicidade v3.1")
