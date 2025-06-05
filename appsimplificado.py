import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine # Importação explícita para type hinting
from datetime import datetime, timedelta
import io # Para exportação XLSX

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================

st.set_page_config(layout="wide", page_title="Verificador Leve de Atividades")

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def gerar_links_zflow(activity_id: int) -> dict:
    """Gera os links para as versões do ZFlow."""
    link_antigo = f"https://zflow.zionbyonset.com.br/activity/3/details/{activity_id}"
    link_novo = f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={activity_id}#/fixcol1"
    return {"antigo": link_antigo, "novo": link_novo}

@st.cache_resource
def get_db_engine() -> Engine | None:
    """Cria e retorna uma engine de conexão com o banco."""
    db_user, db_pass, db_host, db_name = "tarcisio", "123qwe", "40.88.40.110", "zion_flow"
    if not all([db_user, db_pass, db_host, db_name]):
        st.error("Credenciais do banco de dados não definidas completamente no código.")
        return None
    db_uri = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_name}"
    try:
        engine_instance = create_engine(db_uri, pool_pre_ping=True, pool_recycle=3600)
        with engine_instance.connect() as conn: # Testa a conexão
            conn.execute(text("SELECT 1"))
        return engine_instance
    except exc.SQLAlchemyError as e:
        st.error("Erro ao conectar ao banco de dados.")
        st.exception(e) # Mostra o erro completo para debug
        return None

@st.cache_data(ttl=300, hash_funcs={Engine: lambda _: None}) # TTL reduzido para atualizações mais rápidas se necessário
def buscar_atividades_leve(_engine: Engine, data_inicio: datetime.date, data_fim: datetime.date) -> pd.DataFrame:
    """Busca atividades no banco de dados, sem o campo Texto e outros campos pesados."""
    # Query otimizada para não buscar campos de texto longos
    query = text("""
        SELECT 
            activity_id, 
            activity_folder, 
            activity_subject,  -- Mantido para identificação básica
            user_id, 
            user_profile_name,
            activity_date, 
            activity_status, 
            activity_type
            -- Removidos: Texto, observacoes, tags, activity_publish_date, activity_fatal, activity_created_at, activity_updated_at
            -- para tornar a query mais leve. Adicione de volta se necessário.
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type = :tipo_atividade
          AND DATE(activity_date) BETWEEN :data_inicio AND :data_fim
        ORDER BY activity_folder, activity_date DESC, activity_id DESC
    """)
    try:
        with _engine.connect() as connection:
            df = pd.read_sql(query, connection, params={
                "tipo_atividade": "Verificar",
                "data_inicio": data_inicio,
                "data_fim": data_fim
            })
        if 'activity_date' in df.columns:
            # Converte para objeto de data, removendo informações de tempo se houver
            df['activity_date'] = pd.to_datetime(df['activity_date']).dt.normalize()
        return df
    except exc.SQLAlchemyError as e:
        st.error("Erro ao buscar atividades no banco (versão leve).")
        st.exception(e)
        return pd.DataFrame()

# ==============================================================================
# INTERFACE PRINCIPAL (Streamlit)
# ==============================================================================

def main():
    """Função principal que renderiza a página do Streamlit."""
    st.title("👁️ Verificador Leve de Atividades 'Verificar'")
    st.markdown("Identifique rapidamente pastas com múltiplas atividades 'Verificar'.")

    engine = get_db_engine()
    if not engine:
        st.stop() # Interrompe a execução se não houver conexão com o banco

    # --- Filtros na Sidebar ---
    st.sidebar.header("⚙️ Filtros")
    hoje = datetime.today().date()
    data_inicio_filtro = st.sidebar.date_input("Data de Início", hoje - timedelta(days=1), key="data_inicio_leve")
    data_fim_filtro = st.sidebar.date_input("Data de Fim", hoje + timedelta(days=1), key="data_fim_leve")

    if data_inicio_filtro > data_fim_filtro:
        st.sidebar.error("A data de início não pode ser posterior à data de fim.")
        st.stop()

    df_atividades_raw = buscar_atividades_leve(engine, data_inicio_filtro, data_fim_filtro)

    if df_atividades_raw.empty:
        st.info(f"Nenhuma atividade 'Verificar' encontrada para o período de {data_inicio_filtro.strftime('%d/%m/%Y')} a {data_fim_filtro.strftime('%d/%m/%Y')}.")
        st.stop()

    st.success(f"**{len(df_atividades_raw)}** atividades 'Verificar' carregadas para o período (sem conteúdo de texto).")
    
    # --- Filtros Adicionais ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtros Adicionais:")
    
    # Filtro por Pasta
    pastas_disponiveis = sorted(df_atividades_raw['activity_folder'].dropna().unique())
    pastas_selecionadas = st.sidebar.multiselect("Filtrar por Pasta(s):", pastas_disponiveis, default=[], key="pasta_filter_leve")

    # Filtro por Status
    status_disponiveis = sorted(df_atividades_raw['activity_status'].dropna().unique())
    status_selecionados = st.sidebar.multiselect("Filtrar por Status:", status_disponiveis, default=[], key="status_filter_leve")

    # Filtro por Usuário (para exibição)
    usuarios_disponiveis = sorted(df_atividades_raw['user_profile_name'].dropna().unique())
    usuarios_selecionados_exibicao = st.sidebar.multiselect(
        "Filtrar por Usuário(s):", usuarios_disponiveis, default=[], key="user_filter_leve"
    )
    
    # Checkbox para mostrar apenas pastas com múltiplas atividades ou múltiplos usuários
    mostrar_apenas_pastas_com_recorrencia = st.sidebar.checkbox(
        "Mostrar apenas pastas com >1 atividade", 
        value=True, 
        key="recorrencia_pasta_cb_leve",
        help="Exibe apenas pastas que têm mais de uma atividade 'Verificar' após os filtros acima."
    )
    
    mostrar_apenas_pastas_multi_usuarios = st.sidebar.checkbox(
        "Mostrar apenas pastas com múltiplos usuários", 
        False, 
        key="multiuser_pasta_cb_leve",
        help="Exibe apenas pastas que têm atividades de mais de um usuário, após os filtros acima."
    )
    
    # --- Aplicar filtros ---
    df_filtrado = df_atividades_raw.copy()
    if pastas_selecionadas:
        df_filtrado = df_filtrado[df_filtrado['activity_folder'].isin(pastas_selecionadas)]
    if status_selecionados:
        df_filtrado = df_filtrado[df_filtrado['activity_status'].isin(status_selecionados)]
    if usuarios_selecionados_exibicao:
        df_filtrado = df_filtrado[df_filtrado['user_profile_name'].isin(usuarios_selecionados_exibicao)]

    # --- Contagem e identificação para filtros de checkbox ---
    contagem_por_pasta = df_filtrado.groupby('activity_folder')['activity_id'].count()
    pastas_com_recorrencia = contagem_por_pasta[contagem_por_pasta > 1].index.tolist()
    
    pastas_com_multi_usuarios_set = {
        nome for nome, df_grupo in df_filtrado.groupby('activity_folder')
        if df_grupo['user_profile_name'].nunique() > 1
    }

    # Aplicar filtros de checkbox
    if mostrar_apenas_pastas_com_recorrencia:
        df_filtrado = df_filtrado[df_filtrado['activity_folder'].isin(pastas_com_recorrencia)]
    if mostrar_apenas_pastas_multi_usuarios:
        df_filtrado = df_filtrado[df_filtrado['activity_folder'].isin(pastas_com_multi_usuarios_set)]


    # --- Botão de Exportação ---
    st.sidebar.markdown("---")
    if st.sidebar.button("📥 Exportar Dados Exibidos para XLSX", key="export_xlsx_btn_leve"):
        if not df_filtrado.empty:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Selecionar colunas para exportação (sem 'Texto')
                colunas_export = ['activity_id', 'activity_folder', 'activity_subject', 'user_profile_name', 'activity_date', 'activity_status']
                df_export = df_filtrado[colunas_export].copy()
                df_export.to_excel(writer, index=False, sheet_name='Atividades_Verificar_Leve')
            
            st.sidebar.download_button(
                label="Baixar XLSX",
                data=output.getvalue(),
                file_name=f"atividades_verificar_leve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.sidebar.warning("Nenhum dado para exportar com os filtros atuais.")

    # --- Ordenação de Pastas ---
    ordem_pastas_opcao = st.sidebar.selectbox( # Renomeado para evitar conflito com variável
        "Ordenar pastas por:",
        ("Nome da Pasta (A-Z)", "Mais Atividades Primeiro"),
        key="ordem_pastas_leve"
    )

    # --- Exibição dos Resultados ---
    st.header("Lista de Atividades 'Verificar'")
    
    if df_filtrado.empty:
        st.info("Nenhuma atividade 'Verificar' corresponde a todos os filtros selecionados.")
        st.stop()

    pastas_agrupadas_exibicao = df_filtrado.groupby('activity_folder')
    
    # Aplicar ordenação
    if ordem_pastas_opcao == "Nome da Pasta (A-Z)":
        nomes_pastas_ordenados = sorted(pastas_agrupadas_exibicao.groups.keys())
    elif ordem_pastas_opcao == "Mais Atividades Primeiro":
        nomes_pastas_ordenados = pastas_agrupadas_exibicao.size().sort_values(ascending=False).index.tolist()
    else: # Fallback
        nomes_pastas_ordenados = sorted(pastas_agrupadas_exibicao.groups.keys())


    for nome_pasta in nomes_pastas_ordenados:
        df_pasta_exibicao = pastas_agrupadas_exibicao.get_group(nome_pasta) # Usar get_group para obter o DataFrame da pasta
        
        # Info se a pasta (já filtrada) tem múltiplos usuários
        multi_user_info = " (Múltiplos Usuários nesta pasta)" if nome_pasta in pastas_com_multi_usuarios_set else ""

        with st.expander(f"📁 Pasta: {nome_pasta} ({len(df_pasta_exibicao)} atividades){multi_user_info}", expanded=True):
            if nome_pasta in pastas_com_multi_usuarios_set:
                 usuarios_na_pasta = df_pasta_exibicao['user_profile_name'].unique()
                 st.caption(f"👥 Usuários: {', '.join(usuarios_na_pasta)}")

            for _, atividade in df_pasta_exibicao.iterrows():
                activity_id = atividade['activity_id']
                links = gerar_links_zflow(activity_id)
                st.markdown("---")
                
                # Exibição simplificada sem a coluna de duplicatas
                st.markdown(f"**ID:** `{activity_id}` | **Data:** {atividade['activity_date'].strftime('%d/%m/%Y')} | **Status:** `{atividade['activity_status']}`")
                st.markdown(f"**Usuário:** {atividade['user_profile_name']}")
                if 'activity_subject' in atividade and pd.notna(atividade['activity_subject']):
                    st.caption(f"Assunto: {atividade['activity_subject']}") # Mostrar assunto se disponível

                # Botões de link para ZFlow
                link_cols = st.columns(2)
                link_cols[0].link_button("🔗 ZFlow v1", links['antigo'], help="Abrir no ZFlow (versão antiga)")
                link_cols[1].link_button("🔗 ZFlow v2", links['novo'], help="Abrir no ZFlow (versão nova)")
else:
    st.error("A conexão com o banco de dados não pôde ser estabelecida.")
    st.info("Verifique as credenciais no código e o status do servidor do banco de dados.")

if __name__ == "__main__":
    main()
