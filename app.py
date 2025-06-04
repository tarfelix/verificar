import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import io # Para exportação XLSX

# ==============================================================================
# CONFIGURAÇÕES E FUNÇÕES AUXILIARES
# ==============================================================================
st.set_page_config(layout="wide", page_title="Visualizador de Atividades 'Verificar'")

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

@st.cache_data(ttl=300) # Cache dos dados
def buscar_atividades_db(_engine, data_inicio, data_fim):
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
            df['Texto'] = df['Texto'].astype(str).fillna('') # Garantir que seja string
        return df
    except Exception as e:
        st.error(f"Erro ao buscar atividades: {e}")
        return pd.DataFrame()

def gerar_links_zflow(activity_id):
    link_antigo = f"https://zflow.zionbyonset.com.br/activity/3/details/{activity_id}"
    link_novo = f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={activity_id}#/fixcol1"
    return {"antigo": link_antigo, "novo": link_novo}

# ==============================================================================
# INTERFACE DO USUÁRIO (Streamlit)
# ==============================================================================
st.title("👁️ Visualizador de Atividades 'Verificar'")
st.markdown("Filtre e visualize atividades 'Verificar' para identificar possíveis duplicidades manualmente.")

engine = get_db_engine()

if engine:
    # --- Sidebar: Filtros e Opções ---
    st.sidebar.header("⚙️ Filtros e Opções")
    hoje = datetime.today().date()
    
    periodo_selecionado = st.sidebar.radio(
        "Período das atividades:", 
        ("Hoje, Ontem e Amanhã", "Intervalo Personalizado"), 
        key="periodo_radio_s1" # Chave única
    )
    data_inicio_filtro, data_fim_filtro = (hoje - timedelta(days=1), hoje + timedelta(days=1)) if periodo_selecionado == "Hoje, Ontem e Amanhã" else \
                                          (st.sidebar.date_input("Data Início", hoje - timedelta(days=1), key="data_inicio_s1"), 
                                           st.sidebar.date_input("Data Fim", hoje + timedelta(days=1), key="data_fim_s1"))

    if data_inicio_filtro > data_fim_filtro:
        st.sidebar.error("Data de início não pode ser posterior à data de fim.")
        st.stop()

    df_atividades_inicial = buscar_atividades_db(engine, data_inicio_filtro, data_fim_filtro)

    if df_atividades_inicial.empty:
        st.info(f"Nenhuma atividade 'Verificar' encontrada no período de {data_inicio_filtro.strftime('%d/%m/%Y')} a {data_fim_filtro.strftime('%d/%m/%Y')}.")
    else:
        st.success(f"{len(df_atividades_inicial)} atividades 'Verificar' carregadas para o período inicial.")

        # --- Filtros Adicionais na Sidebar ---
        pastas_disponiveis = sorted(df_atividades_inicial['activity_folder'].dropna().unique())
        pastas_selecionadas = st.sidebar.multiselect("Filtrar por Pasta(s):", pastas_disponiveis, default=[], key="pasta_filter_s1")

        status_disponiveis = sorted(df_atividades_inicial['activity_status'].dropna().unique())
        status_selecionados = st.sidebar.multiselect("Filtrar por Status:", status_disponiveis, default=[], key="status_filter_s1")
        
        usuarios_disponiveis = sorted(df_atividades_inicial['user_profile_name'].dropna().unique())
        usuarios_selecionados_exibicao = st.sidebar.multiselect("Filtrar exibição por Usuário(s):", usuarios_disponiveis, default=[], key="user_filter_s1")
        
        # Checkbox para pastas com múltiplos usuários (aplicado após filtros de pasta/status)
        apenas_pastas_multi_usuarios_cb = st.sidebar.checkbox(
            "Mostrar apenas pastas com múltiplos usuários (após filtros de pasta/status)", 
            False, 
            key="multiuser_cb_s1"
        )

        ordem_pastas = st.sidebar.selectbox(
            "Ordenar pastas por:",
            ("Nome da Pasta (A-Z)", "Mais Atividades Primeiro"), # Removida ordenação por duplicatas
            key="ordem_pastas_s1"
        )
        st.sidebar.markdown("---")
        
        # Aplicar filtros principais (pasta, status) para criar o df_base_para_exibicao
        df_base_para_exibicao = df_atividades_inicial.copy()
        if pastas_selecionadas:
            df_base_para_exibicao = df_base_para_exibicao[df_base_para_exibicao['activity_folder'].isin(pastas_selecionadas)]
        if status_selecionados:
            df_base_para_exibicao = df_base_para_exibicao[df_base_para_exibicao['activity_status'].isin(status_selecionados)]

        # Identificar pastas com múltiplos usuários DENTRO do df_base_para_exibicao
        pastas_com_multiplos_usuarios_filtrado = set()
        if not df_base_para_exibicao.empty:
            for nome_pasta, df_pasta_temp in df_base_para_exibicao.groupby('activity_folder'):
                if df_pasta_temp['user_profile_name'].nunique() > 1:
                    pastas_com_multiplos_usuarios_filtrado.add(nome_pasta)

        # Aplicar filtros de exibição (usuário, apenas pastas com múltiplos usuários)
        df_exibir_final = df_base_para_exibicao.copy()
        if usuarios_selecionados_exibicao:
            df_exibir_final = df_exibir_final[df_exibir_final['user_profile_name'].isin(usuarios_selecionados_exibicao)]
        if apenas_pastas_multi_usuarios_cb:
            df_exibir_final = df_exibir_final[df_exibir_final['activity_folder'].isin(pastas_com_multiplos_usuarios_filtrado)]


        # --- Botão de Exportação ---
        if st.sidebar.button("Exportar Dados Exibidos para XLSX", key="export_xlsx_btn_s1"):
            if not df_exibir_final.empty:
                output = io.BytesIO()
                # Usar 'with' garante que o writer seja fechado corretamente
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_exibir_final.to_excel(writer, index=False, sheet_name='Atividades_Filtradas')
                
                st.sidebar.download_button(
                    label="Baixar XLSX",
                    data=output.getvalue(), # Obter bytes do buffer
                    file_name=f"atividades_verificar_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.sidebar.warning("Nenhum dado para exportar com os filtros atuais.")

        # --- Lógica de Ordenação das Pastas (sobre df_exibir_final) ---
        lista_pastas_para_renderizar = []
        if not df_exibir_final.empty:
            pastas_agrupadas_renderizar = df_exibir_final.groupby('activity_folder')
            if ordem_pastas == "Nome da Pasta (A-Z)":
                lista_pastas_para_renderizar = sorted(pastas_agrupadas_renderizar.groups.keys())
            elif ordem_pastas == "Mais Atividades Primeiro":
                lista_pastas_para_renderizar = pastas_agrupadas_renderizar.size().sort_values(ascending=False).index.tolist()
        
        # --- Exibição Principal ---
        st.header("Lista de Atividades 'Verificar'")
        if not lista_pastas_para_renderizar and not df_exibir_final.empty:
            st.info("Nenhuma pasta corresponde a todos os critérios de filtro de exibição selecionados.")
        elif df_exibir_final.empty :
             st.info("Nenhuma atividade 'Verificar' corresponde aos filtros aplicados.")

        for nome_pasta_render in lista_pastas_para_renderizar:
            df_pasta_atual_render = df_exibir_final[df_exibir_final['activity_folder'] == nome_pasta_render]
            
            # Verificar se esta pasta (já filtrada para exibição) tem múltiplos usuários
            tem_multi_usuarios_na_exibicao = df_pasta_atual_render['user_profile_name'].nunique() > 1
            multi_user_info_display = " (Múltiplos Usuários nesta exibição)" if tem_multi_usuarios_na_exibicao else ""
            
            with st.expander(f"📁 Pasta: {nome_pasta_render} ({len(df_pasta_atual_render)} atividades nesta exibição){multi_user_info_display}", expanded=True):
                if tem_multi_usuarios_na_exibicao:
                     usuarios_nesta_pasta_exibicao = df_pasta_atual_render['user_profile_name'].unique()
                     st.caption(f"👥 Usuários nesta pasta (exibição atual): {', '.join(usuarios_nesta_pasta_exibicao)}")

                for _, atividade_row in df_pasta_atual_render.iterrows():
                    act_id = atividade_row['activity_id']
                    links = gerar_links_zflow(act_id)
                    
                    st.markdown("---")
                    # Layout simplificado, sem coluna para duplicatas
                    st.markdown(f"**ID:** `{act_id}` | **Data:** {atividade_row['activity_date'].strftime('%d/%m/%Y')} | **Status:** `{atividade_row['activity_status']}`")
                    st.markdown(f"**Usuário:** {atividade_row['user_profile_name']}")
                    st.text_area("Texto da Publicação:", value=str(atividade_row['Texto']), height=100, key=f"texto_area_s1_{act_id}", disabled=True)
                    
                    action_btn_cols = st.columns(2)
                    action_btn_cols[0].link_button("🔗 ZFlow v1", links['antigo'], help="Abrir no ZFlow (versão antiga)")
                    action_btn_cols[1].link_button("🔗 ZFlow v2", links['novo'], help="Abrir no ZFlow (versão nova)")
else:
    st.error("Conexão com o banco falhou. Verifique as credenciais e o status do banco.")

st.sidebar.info("Visualizador de Atividades v1 - Simplificado")
