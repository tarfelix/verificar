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
    # Garante que ambos sejam strings para SequenceMatcher
    return SequenceMatcher(None, str(texto_a), str(texto_b)).ratio()

# Função para determinar a cor com base no percentual de similaridade
def obter_cor_similaridade(ratio):
    if ratio >= 0.91:
        return "red"  # Muito Alta Similaridade
    elif ratio >= 0.71:
        return "orange" # Alta Similaridade
    elif ratio >= 0.50:
        return "gold"  # Média Similaridade
    return "grey" # Baixa Similaridade (não será mostrado por padrão, mas para referência)

# Função para conectar ao banco de dados (COM CREDENCIAIS HARDCODED)
@st.cache_resource # Cache da conexão para performance
def get_db_engine():
    # ==========================================================================
    # ATENÇÃO: CREDENCIAIS DO BANCO DE DADOS DIRETAMENTE NO CÓDIGO!
    # ISSO NÃO É SEGURO PARA PRODUÇÃO OU CÓDIGO COMPARTILHADO.
    # Considere os riscos antes de usar desta forma.
    # ==========================================================================
    db_user = "tarcisio"    # SUBSTITUA PELO SEU USUÁRIO REAL
    db_pass = "123qwe"      # SUBSTITUA PELA SUA SENHA REAL
    db_host = "40.88.40.110" # SUBSTITUA PELO SEU HOST REAL
    db_name = "zion_flow"   # SUBSTITUA PELO SEU NOME DE BANCO REAL
    # ==========================================================================

    if not all([db_user, db_pass, db_host, db_name]):
        st.error("Credenciais do banco de dados não definidas completamente no código. Verifique a função get_db_engine.")
        return None

    db_uri = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_name}"
    try:
        engine = create_engine(db_uri)
        # Testar conexão
        with engine.connect() as connection:
            pass # Teste de conexão bem-sucedido
        return engine
    except Exception as e:
        st.error(f"Erro ao conectar ao banco de dados: {e}")
        st.error("Verifique se as credenciais hardcoded estão corretas e se o servidor do banco está acessível.")
        return None

# Função para buscar dados do banco
@st.cache_data(ttl=600) # Cache dos dados por 10 minutos
def buscar_atividades(engine, data_inicio, data_fim):
    if engine is None:
        st.error("Engine do banco de dados não inicializada.")
        return pd.DataFrame()

    # Query SQL parametrizada para segurança
    query = text("""
        SELECT activity_id, activity_folder, activity_subject, user_id, user_profile_name,
               activity_date, activity_fatal, activity_status, activity_type,
               activity_publish_date, Texto, observacoes, tags,
               activity_created_at, activity_updated_at
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type = :tipo_atividade
          AND activity_status = :status_atividade
          AND activity_date BETWEEN :data_inicio AND :data_fim
        ORDER BY activity_folder, activity_date, activity_id
    """)
    try:
        with engine.connect() as connection:
            df = pd.read_sql(query, connection, params={
                "tipo_atividade": "Verificar",
                "status_atividade": "Aberta",
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
st.markdown("Este aplicativo ajuda a identificar atividades do tipo 'Verificar' que podem ser duplicadas, com base na pasta e similaridade do texto da publicação.")
st.warning("⚠️ As credenciais do banco de dados estão diretamente no código. Esta abordagem não é segura para ambientes de produção ou código compartilhado.")


# Conectar ao banco
engine = get_db_engine()

if engine:
    # --- Seleção de Datas ---
    st.sidebar.header("🗓️ Filtro de Período")
    hoje = datetime.today().date()
    
    periodo_selecionado = st.sidebar.radio(
        "Selecionar período:",
        ("Hoje, Ontem e Amanhã (Padrão)", "Intervalo Personalizado"),
        key="periodo_radio",
        help="Define o intervalo de datas para buscar as atividades 'Verificar'."
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
        st.stop() # Impede a execução do restante do script se as datas forem inválidas

    # --- Carregar Dados ---
    df_atividades = buscar_atividades(engine, data_inicio_filtro, data_fim_filtro)

    if df_atividades.empty:
        st.info(f"Nenhuma atividade 'Verificar' (Aberta) encontrada para o período de {data_inicio_filtro.strftime('%d/%m/%Y')} a {data_fim_filtro.strftime('%d/%m/%Y')}.")
    else:
        st.success(f"{len(df_atividades)} atividades 'Verificar' (Abertas) carregadas para o período.")
        
        # --- Análise e Exibição ---
        st.header("Análise por Pasta")

        min_similaridade_display = st.sidebar.slider(
            "Exibir similaridades acima de (%):",
            min_value=0, max_value=100, value=50, step=5,
            key="min_similaridade_slider",
            help="Apenas atividades com similaridade de texto acima deste valor serão destacadas como potenciais duplicatas."
        ) / 100.0
        
        # Agrupar por pasta para processamento
        pastas_agrupadas = df_atividades.groupby('activity_folder')

        for nome_pasta, df_pasta_original in pastas_agrupadas:
            df_pasta = df_pasta_original.copy() # Trabalhar com uma cópia para evitar SettingWithCopyWarning
            
            if len(df_pasta) < 2:
                with st.expander(f"📁 Pasta: {nome_pasta} (1 atividade - sem comparação interna)"):
                    st.dataframe(df_pasta[['activity_id', 'activity_date', 'user_profile_name', 'Texto']], use_container_width=True)
                continue

            expander_title = f"📁 Pasta: {nome_pasta} ({len(df_pasta)} atividades)"
            
            usuarios_na_pasta = df_pasta['user_profile_name'].nunique()
            nomes_usuarios_unicos = df_pasta['user_profile_name'].unique() 
            aviso_usuarios = ""
            if usuarios_na_pasta > 1:
                nomes_usuarios_str = ", ".join(map(str, nomes_usuarios_unicos))
                aviso_usuarios = f"⚠️ **Atenção:** Múltiplos usuários nesta pasta: {nomes_usuarios_str}"
                expander_title += f" (Múltiplos Usuários: {', '.join(nomes_usuarios_unicos[:2])}{'...' if len(nomes_usuarios_unicos) > 2 else ''})"


            with st.expander(expander_title, expanded=True):
                if aviso_usuarios:
                    st.markdown(aviso_usuarios)

                atividades_lista = df_pasta.to_dict('records')
                similaridades_por_id = {atividade['activity_id']: [] for atividade in atividades_lista}

                for i in range(len(atividades_lista)):
                    atividade_base = atividades_lista[i]
                    for j in range(i + 1, len(atividades_lista)):
                        atividade_comparar = atividades_lista[j]
                        similaridade = calcular_similaridade(atividade_base['Texto'], atividade_comparar['Texto'])
                        
                        if similaridade >= min_similaridade_display:
                            similaridades_por_id[atividade_base['activity_id']].append({
                                'id_similar': atividade_comparar['activity_id'],
                                'data_similar': atividade_comparar['activity_date'],
                                'usuario_similar': atividade_comparar['user_profile_name'],
                                'ratio': similaridade,
                                'cor': obter_cor_similaridade(similaridade)
                            })
                            similaridades_por_id[atividade_comparar['activity_id']].append({
                                'id_similar': atividade_base['activity_id'],
                                'data_similar': atividade_base['activity_date'],
                                'usuario_similar': atividade_base['user_profile_name'],
                                'ratio': similaridade,
                                'cor': obter_cor_similaridade(similaridade)
                            })

                for atividade_info in atividades_lista: 
                    row_id = atividade_info['activity_id']
                    st.markdown(f"---")
                    col_info, col_similar = st.columns([0.6, 0.4])

                    with col_info:
                        st.markdown(f"**ID:** `{row_id}` | **Data:** {atividade_info['activity_date'].strftime('%d/%m/%Y')} | **Usuário:** {atividade_info['user_profile_name']}")
                        with st.container():
                             st.text_area("Texto da Publicação:", value=str(atividade_info['Texto']), height=100, key=f"texto_{row_id}", disabled=True)
                    
                    with col_similar:
                        lista_similares = similaridades_por_id[row_id]
                        if lista_similares:
                            st.markdown(f"**<span style='color:red;'>Potenciais Duplicatas:</span>** ({len(lista_similares)})", unsafe_allow_html=True)
                            similares_ordenados = sorted(lista_similares, key=lambda x: x['ratio'], reverse=True)
                            for sim_info in similares_ordenados:
                                st.markdown(
                                    f"<small><span style='background-color:{sim_info['cor']}; padding: 2px 5px; border-radius: 3px; color: black; display: inline-block; margin-bottom: 3px;'>"
                                    f"ID: {sim_info['id_similar']} ({sim_info['ratio']:.0%}) <br>"
                                    f"Data: {sim_info['data_similar'].strftime('%d/%m')} | Usuário: {sim_info['usuario_similar']}"
                                    f"</span></small>",
                                    unsafe_allow_html=True
                                )
                        else:
                            st.markdown(f"**<span style='color:green;'>Sem duplicatas acima de {min_similaridade_display:.0%}</span>**", unsafe_allow_html=True)
else:
    st.error("A conexão com o banco de dados não pôde ser estabelecida.")
    st.info("Verifique as credenciais diretamente no código (função get_db_engine) e o status do servidor do banco de dados.")

st.sidebar.markdown("---")
st.sidebar.info("Desenvolvido para auxiliar na identificação de duplicidades.")
