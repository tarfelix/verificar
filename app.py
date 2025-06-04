import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from difflib import SequenceMatcher

# ==============================================================================
# CONFIGURAÇÕES E FUNÇÕES AUXILIARES
# ==============================================================================

# Configuração da página do Streamlit
st.set_page_config(layout="wide", page_title="Verificador de Duplicidade de Atividades")

# Função para calcular similaridade entre duas strings
def calcular_similaridade(texto_a, texto_b):
    if texto_a is None or texto_b is None:
        return 0.0
    return SequenceMatcher(None, str(texto_a), str(texto_b)).ratio()

# Função para determinar a cor com base no percentual de similaridade
def obter_cor_similaridade(ratio):
    if ratio >= 0.91:
        return "red"
    elif ratio >= 0.71:
        return "orange"
    elif ratio >= 0.50:
        return "gold"
    return "grey"

# Função para conectar ao banco de dados (COM CREDENCIAIS HARDCODED)
@st.cache_resource
def get_db_engine():
    # ==========================================================================
    # ATENÇÃO: CREDENCIAIS DO BANCO DE DADOS DIRETAMENTE NO CÓDIGO!
    # ISSO NÃO É SEGURO PARA PRODUÇÃO OU CÓDIGO COMPARTILHADO.
    # ==========================================================================
    db_user = "tarcisio"
    db_pass = "123qwe"
    db_host = "40.88.40.110"
    db_name = "zion_flow"
    # ==========================================================================

    if not all([db_user, db_pass, db_host, db_name]):
        st.error("Credenciais do banco de dados não definidas completamente no código.")
        return None

    db_uri = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_name}"
    try:
        engine_instance = create_engine(db_uri)
        with engine_instance.connect() as connection:
            pass
        return engine_instance
    except Exception as e:
        st.error(f"Erro ao conectar ao banco de dados: {e}")
        return None

# Função para buscar dados do banco
@st.cache_data(ttl=600)
def buscar_atividades(_engine, data_inicio, data_fim):
    if _engine is None:
        st.error("Engine do banco de dados não inicializada.")
        return pd.DataFrame()

    query = text("""
        SELECT activity_id, activity_folder, activity_subject, user_id, user_profile_name,
               activity_date, activity_fatal, activity_status, activity_type,
               activity_publish_date, Texto, observacoes, tags,
               activity_created_at, activity_updated_at
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type = :tipo_atividade
          AND activity_date BETWEEN :data_inicio AND :data_fim
        ORDER BY activity_folder, activity_date, activity_id
    """)
    try:
        with _engine.connect() as connection:
            df = pd.read_sql(query, connection, params={
                "tipo_atividade": "Verificar", # Mantém o filtro por tipo "Verificar"
                "data_inicio": data_inicio,
                "data_fim": data_fim
            })
        if 'activity_date' in df.columns:
             df['activity_date'] = pd.to_datetime(df['activity_date']).dt.date
        return df
    except Exception as e:
        st.error(f"Erro ao buscar atividades: {e}")
        return pd.DataFrame()

# ==============================================================================
# INTERFACE DO USUÁRIO (Streamlit)
# ==============================================================================

st.title("🔎 Verificador de Duplicidade de Atividades")
st.markdown("Análise de atividades 'Verificar' para identificar potenciais duplicidades com base na pasta e similaridade do texto.")
st.warning("⚠️ As credenciais do banco de dados estão diretamente no código. Esta abordagem não é segura para ambientes de produção ou código compartilhado.")

engine = get_db_engine()

if engine:
    # --- Filtros na Sidebar ---
    st.sidebar.header("🗓️ Filtros de Análise")
    hoje = datetime.today().date()
    
    periodo_selecionado = st.sidebar.radio(
        "Selecionar período das atividades:",
        ("Hoje, Ontem e Amanhã (Padrão)", "Intervalo Personalizado"),
        key="periodo_radio",
    )

    if periodo_selecionado == "Hoje, Ontem e Amanhã (Padrão)":
        data_inicio_filtro = hoje - timedelta(days=1)
        data_fim_filtro = hoje + timedelta(days=1)
        st.sidebar.markdown(f"Período padrão: **{data_inicio_filtro.strftime('%d/%m/%Y')}** a **{data_fim_filtro.strftime('%d/%m/%Y')}**")
    else:
        data_inicio_filtro = st.sidebar.date_input("Data de Início", hoje - timedelta(days=1))
        data_fim_filtro = st.sidebar.date_input("Data de Fim", hoje + timedelta(days=1))

    if data_inicio_filtro > data_fim_filtro:
        st.sidebar.error("A data de início não pode ser posterior à data de fim.")
        st.stop()

    df_atividades_raw = buscar_atividades(engine, data_inicio_filtro, data_fim_filtro)

    if df_atividades_raw.empty:
        st.info(f"Nenhuma atividade 'Verificar' encontrada para o período de {data_inicio_filtro.strftime('%d/%m/%Y')} a {data_fim_filtro.strftime('%d/%m/%Y')}.")
    else:
        st.success(f"{len(df_atividades_raw)} atividades 'Verificar' (todos os status) carregadas para o período.")

        # --- Filtros de Exibição na Sidebar ---
        usuarios_disponiveis = sorted(df_atividades_raw['user_profile_name'].dropna().unique())
        usuarios_selecionados_exibicao = st.sidebar.multiselect(
            "Filtrar exibição por Usuário(s):",
            options=usuarios_disponiveis,
            default=[],
            help="Selecione usuários para filtrar a lista de atividades exibidas. A análise de duplicidade ainda considera todos."
        )

        min_similaridade_display = st.sidebar.slider(
            "Exibir similaridades de texto acima de (%):",
            min_value=0, max_value=100, value=50, step=5,
            key="min_similaridade_slider",
        ) / 100.0
        
        mostrar_apenas_com_duplicatas = st.sidebar.checkbox(
            "Mostrar apenas atividades com potenciais duplicatas",
            value=False,
            help="Marque para exibir apenas as atividades que têm alguma similaridade com outra(s)."
        )
        st.sidebar.markdown("---")
        
        # Aplicar filtro de usuário para exibição (não afeta a análise de duplicatas)
        df_atividades_para_exibicao = df_atividades_raw.copy()
        if usuarios_selecionados_exibicao:
            df_atividades_para_exibicao = df_atividades_para_exibicao[df_atividades_para_exibicao['user_profile_name'].isin(usuarios_selecionados_exibicao)]


        # --- Análise e Exibição ---
        st.header("Análise por Pasta")
        
        pastas_agrupadas_raw = df_atividades_raw.groupby('activity_folder') # Para análise de duplicatas
        pastas_agrupadas_exibicao = df_atividades_para_exibicao.groupby('activity_folder') # Para iterar na exibição

        nenhuma_duplicata_para_mostrar = True

        for nome_pasta, df_pasta_exibicao in pastas_agrupadas_exibicao:
            # Pegar todas as atividades da pasta do DataFrame original para análise completa de duplicatas
            df_pasta_analise_completa = pastas_agrupadas_raw.get_group(nome_pasta)

            if len(df_pasta_analise_completa) < 2 and len(df_pasta_exibicao) < 2 : # Se a pasta original tem menos de 2, não há o que comparar
                # Se o filtro de usuário removeu todas menos uma, e a original só tinha uma, não mostra se "apenas duplicatas"
                if not mostrar_apenas_com_duplicatas:
                    with st.expander(f"📁 Pasta: {nome_pasta} (1 atividade nesta exibição - sem comparação interna)"):
                        st.dataframe(df_pasta_exibicao[['activity_id', 'activity_date', 'user_profile_name', 'activity_status', 'Texto']], use_container_width=True)
                        nenhuma_duplicata_para_mostrar = False # Algo foi mostrado
                continue

            # Calcular similaridades para TODAS as atividades na pasta (usando df_pasta_analise_completa)
            atividades_lista_analise = df_pasta_analise_completa.to_dict('records')
            similaridades_por_id_analise = {atividade['activity_id']: [] for atividade in atividades_lista_analise}

            for i in range(len(atividades_lista_analise)):
                atividade_base = atividades_lista_analise[i]
                for j in range(i + 1, len(atividades_lista_analise)):
                    atividade_comparar = atividades_lista_analise[j]
                    similaridade = calcular_similaridade(atividade_base['Texto'], atividade_comparar['Texto'])
                    
                    if similaridade >= min_similaridade_display:
                        similaridades_por_id_analise[atividade_base['activity_id']].append({
                            'id_similar': atividade_comparar['activity_id'],
                            'data_similar': atividade_comparar['activity_date'],
                            'usuario_similar': atividade_comparar['user_profile_name'],
                            'status_similar': atividade_comparar['activity_status'],
                            'ratio': similaridade, 'cor': obter_cor_similaridade(similaridade)
                        })
                        similaridades_por_id_analise[atividade_comparar['activity_id']].append({
                            'id_similar': atividade_base['activity_id'],
                            'data_similar': atividade_base['activity_date'],
                            'usuario_similar': atividade_base['user_profile_name'],
                            'status_similar': atividade_base['activity_status'],
                            'ratio': similaridade, 'cor': obter_cor_similaridade(similaridade)
                        })
            
            # Filtrar quais atividades da pasta serão exibidas com base no checkbox "mostrar_apenas_com_duplicatas"
            # e no filtro de usuário (já aplicado em df_pasta_exibicao)
            atividades_para_mostrar_na_pasta = []
            for _, row_exibicao in df_pasta_exibicao.iterrows():
                id_exibicao = row_exibicao['activity_id']
                if mostrar_apenas_com_duplicatas:
                    if id_exibicao in similaridades_por_id_analise and similaridades_por_id_analise[id_exibicao]:
                        atividades_para_mostrar_na_pasta.append(row_exibicao)
                else:
                    atividades_para_mostrar_na_pasta.append(row_exibicao)
            
            if not atividades_para_mostrar_na_pasta:
                continue # Pula para a próxima pasta se não houver nada para mostrar nesta

            nenhuma_duplicata_para_mostrar = False # Algo será mostrado

            # Montar título do expander
            expander_title = f"📁 Pasta: {nome_pasta} ({len(df_pasta_exibicao)} atividades nesta exibição / {len(df_pasta_analise_completa)} total na pasta)"
            usuarios_na_pasta_analise = df_pasta_analise_completa['user_profile_name'].nunique()
            nomes_usuarios_unicos_analise = df_pasta_analise_completa['user_profile_name'].unique()
            aviso_usuarios = ""
            if usuarios_na_pasta_analise > 1:
                nomes_usuarios_str = ", ".join(map(str, nomes_usuarios_unicos_analise))
                aviso_usuarios = f"👥 **Info:** Múltiplos usuários na pasta (análise completa): {nomes_usuarios_str}"
                expander_title += f" (Múltiplos Usuários na Análise)"

            with st.expander(expander_title, expanded=True):
                if aviso_usuarios:
                    st.markdown(aviso_usuarios)
                
                for atividade_info_dict in pd.DataFrame(atividades_para_mostrar_na_pasta).to_dict('records'):
                    row_id = atividade_info_dict['activity_id']
                    st.markdown(f"---")
                    col_info, col_similar = st.columns([0.6, 0.4])

                    with col_info:
                        st.markdown(f"**ID:** `{row_id}` | **Data:** {atividade_info_dict['activity_date'].strftime('%d/%m/%Y')} | **Status:** `{atividade_info_dict['activity_status']}`")
                        st.markdown(f"**Usuário:** {atividade_info_dict['user_profile_name']}")
                        with st.container():
                             st.text_area("Texto da Publicação:", value=str(atividade_info_dict['Texto']), height=100, key=f"texto_{row_id}", disabled=True)
                    
                    with col_similar:
                        # Usar as similaridades calculadas com base na análise completa da pasta
                        lista_similares = similaridades_por_id_analise.get(row_id, [])
                        if lista_similares:
                            st.markdown(f"**<span style='color:red;'>Potenciais Duplicatas:</span>** ({len(lista_similares)})", unsafe_allow_html=True)
                            similares_ordenados = sorted(lista_similares, key=lambda x: x['ratio'], reverse=True)
                            for sim_info in similares_ordenados:
                                st.markdown(
                                    f"<small><span style='background-color:{sim_info['cor']}; padding: 2px 5px; border-radius: 3px; color: black; display: inline-block; margin-bottom: 3px;'>"
                                    f"ID: {sim_info['id_similar']} ({sim_info['ratio']:.0%}) <br>"
                                    f"Data: {sim_info['data_similar'].strftime('%d/%m')} | Status: `{sim_info['status_similar']}` <br>"
                                    f"Usuário: {sim_info['usuario_similar']}"
                                    f"</span></small>",
                                    unsafe_allow_html=True
                                )
                        elif mostrar_apenas_com_duplicatas:
                             pass # Não mostra nada se o filtro é para apenas duplicatas e não há
                        else:
                            st.markdown(f"**<span style='color:green;'>Sem duplicatas (acima de {min_similaridade_display:.0%})</span>**", unsafe_allow_html=True)
        
        if nenhuma_duplicata_para_mostrar and mostrar_apenas_com_duplicatas:
            st.info("Nenhuma atividade com potenciais duplicatas encontradas para os filtros selecionados.")
        elif df_atividades_para_exibicao.empty and usuarios_selecionados_exibicao :
             st.info(f"Nenhuma atividade encontrada para o(s) usuário(s) selecionado(s) na exibição: {', '.join(usuarios_selecionados_exibicao)}.")


else:
    st.error("A conexão com o banco de dados não pôde ser estabelecida.")
    st.info("Verifique as credenciais diretamente no código (função get_db_engine) e o status do servidor do banco de dados.")

st.sidebar.markdown("---")
st.sidebar.info("Desenvolvido para auxiliar na identificação de duplicidades.")
