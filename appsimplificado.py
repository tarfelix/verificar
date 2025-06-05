import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from datetime import datetime, timedelta
import io

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================

st.set_page_config(layout="wide", page_title="Verificador Leve de Atividades")

# ==============================================================================
# FUNÇÕES DE LÓGICA E DADOS (AGORA USANDO ST.SECRETS)
# ==============================================================================

@st.cache_resource
def get_db_engine() -> Engine | None:
    """Cria e retorna uma engine de conexão com o banco usando st.secrets."""
    try:
        db_creds = st.secrets["database"]
        db_uri = (
            f"mysql+mysqlconnector://{db_creds['user']}:{db_creds['password']}"
            f"@{db_creds['host']}/{db_creds['name']}"
        )
        engine_instance = create_engine(db_uri, pool_pre_ping=True, pool_recycle=3600)
        # Testa a conexão para garantir que as credenciais funcionam
        with engine_instance.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine_instance
    except KeyError:
        st.error("As credenciais do banco de dados não foram encontradas nos 'Secrets'. Verifique a seção [database].")
        return None
    except exc.SQLAlchemyError as e:
        st.error("Erro ao conectar ao banco de dados. Verifique as credenciais e o status do servidor.")
        st.exception(e)
        return None

@st.cache_data(ttl=300, hash_funcs={Engine: lambda _: None})
def buscar_atividades_leve(_engine: Engine, data_inicio: datetime.date, data_fim: datetime.date) -> pd.DataFrame:
    """Busca atividades no banco de dados, sem o campo Texto e outros campos pesados."""
    # A query continua a mesma
    query = text("""
        SELECT 
            activity_id, activity_folder, activity_subject, user_id, 
            user_profile_name, activity_date, activity_status, activity_type
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
            df['activity_date'] = pd.to_datetime(df['activity_date']).dt.normalize()
        return df
    except exc.SQLAlchemyError as e:
        st.error("Erro ao buscar atividades no banco.")
        st.exception(e)
        return pd.DataFrame()

def gerar_links_zflow(activity_id: int) -> dict:
    """Gera os links para as versões do ZFlow."""
    link_antigo = f"https://zflow.zionbyonset.com.br/activity/3/details/{activity_id}"
    link_novo = f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={activity_id}#/fixcol1"
    return {"antigo": link_antigo, "novo": link_novo}

# ==============================================================================
# LÓGICA DE LOGIN (SEGURA, USANDO ST.SECRETS)
# ==============================================================================

def verify_user(username, password):
    """Verifica as credenciais de um usuário a partir do st.secrets."""
    try:
        users = st.secrets["credentials"]["usernames"]
        if username in users and users[username] == password:
            return True
    except KeyError:
        st.error("A estrutura de logins de usuários não foi encontrada nos 'Secrets'. Verifique a seção [credentials].")
        return False
    return False

def check_login():
    """Renderiza a tela de login ou a aplicação principal."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Se já logado, mostra o app principal e encerra a função aqui
    if st.session_state.logged_in:
        main_app()
        return

    # Se não, mostra o formulário de login
    st.title("Login de Acesso")
    st.markdown("Por favor, insira suas credenciais para continuar.")
    
    username = st.text_input("Usuário", key="login_user")
    password = st.text_input("Senha", type="password", key="login_pass")

    if st.button("Entrar", key="login_button"):
        if verify_user(username, password):
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Usuário ou senha incorretos.")

# ==============================================================================
# INTERFACE PRINCIPAL DA APLICAÇÃO
# ==============================================================================

def main_app():
    """Função que renderiza a página principal após o login."""
    engine = get_db_engine()
    
    if not engine:
        # A mensagem de erro já é mostrada dentro de get_db_engine()
        st.stop()

    # --- Barra Lateral ---
    st.sidebar.header("⚙️ Filtros")
    
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    hoje = datetime.today().date()
    data_inicio_filtro = st.sidebar.date_input("Data de Início", hoje - timedelta(days=1), key="data_inicio_leve")
    data_fim_filtro = st.sidebar.date_input("Data de Fim", hoje + timedelta(days=1), key="data_fim_leve")

    # (O restante do código da interface principal continua exatamente o mesmo)
    if data_inicio_filtro > data_fim_filtro:
        st.sidebar.error("A data de início não pode ser posterior à data de fim.")
        st.stop()

    df_atividades_raw = buscar_atividades_leve(engine, data_inicio_filtro, data_fim_filtro)

    if df_atividades_raw.empty:
        st.title("👁️ Verificador Leve de Atividades 'Verificar'")
        st.info(f"Nenhuma atividade 'Verificar' encontrada para o período.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtros Adicionais:")
    
    pastas_disponiveis = sorted(df_atividades_raw['activity_folder'].dropna().unique())
    pastas_selecionadas = st.sidebar.multiselect("Filtrar por Pasta(s):", pastas_disponiveis, default=[], key="pasta_filter_leve")

    status_disponiveis = sorted(df_atividades_raw['activity_status'].dropna().unique())
    status_selecionados = st.sidebar.multiselect("Filtrar por Status:", status_disponiveis, default=[], key="status_filter_leve")

    usuarios_disponiveis = sorted(df_atividades_raw['user_profile_name'].dropna().unique())
    usuarios_selecionados_exibicao = st.sidebar.multiselect("Filtrar por Usuário(s):", usuarios_disponiveis, default=[], key="user_filter_leve")
    
    mostrar_apenas_pastas_com_recorrencia = st.sidebar.checkbox("Apenas pastas com >1 atividade", value=True, key="recorrencia_pasta_cb_leve")
    mostrar_apenas_pastas_multi_usuarios = st.sidebar.checkbox("Apenas pastas com múltiplos usuários", False, key="multiuser_pasta_cb_leve")
    
    df_filtrado = df_atividades_raw.copy()
    if pastas_selecionadas: df_filtrado = df_filtrado[df_filtrado['activity_folder'].isin(pastas_selecionadas)]
    if status_selecionados: df_filtrado = df_filtrado[df_filtrado['activity_status'].isin(status_selecionados)]
    if usuarios_selecionados_exibicao: df_filtrado = df_filtrado[df_filtrado['user_profile_name'].isin(usuarios_selecionados_exibicao)]

    contagem_por_pasta = df_filtrado.groupby('activity_folder')['activity_id'].count()
    pastas_com_recorrencia = contagem_por_pasta[contagem_por_pasta > 1].index.tolist()
    pastas_com_multi_usuarios_set = {nome for nome, df_grupo in df_filtrado.groupby('activity_folder') if df_grupo['user_profile_name'].nunique() > 1}

    if mostrar_apenas_pastas_com_recorrencia: df_filtrado = df_filtrado[df_filtrado['activity_folder'].isin(pastas_com_recorrencia)]
    if mostrar_apenas_pastas_multi_usuarios: df_filtrado = df_filtrado[df_filtrado['activity_folder'].isin(pastas_com_multi_usuarios_set)]

    st.sidebar.markdown("---")
    if not df_filtrado.empty:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            colunas_export = ['activity_id', 'activity_folder', 'activity_subject', 'user_profile_name', 'activity_date', 'activity_status']
            df_filtrado[colunas_export].to_excel(writer, index=False, sheet_name='Atividades_Verificar_Leve')
        st.sidebar.download_button("📥 Baixar Dados (XLSX)", output.getvalue(), f"atividades_verificar_leve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    else:
        st.sidebar.warning("Nenhum dado para exportar.")

    ordem_pastas_opcao = st.sidebar.selectbox("Ordenar pastas por:", ("Nome da Pasta (A-Z)", "Mais Atividades Primeiro"), key="ordem_pastas_leve")

    st.title("👁️ Verificador Leve de Atividades 'Verificar'")
    st.markdown(f"Exibindo **{len(df_filtrado)}** atividades em **{df_filtrado['activity_folder'].nunique()}** pastas.")

    if df_filtrado.empty:
        st.info("Nenhuma atividade corresponde a todos os filtros selecionados.")
        st.stop()

    pastas_agrupadas_exibicao = df_filtrado.groupby('activity_folder')
    if ordem_pastas_opcao == "Nome da Pasta (A-Z)":
        nomes_pastas_ordenados = sorted(pastas_agrupadas_exibicao.groups.keys())
    else:
        nomes_pastas_ordenados = pastas_agrupadas_exibicao.size().sort_values(ascending=False).index.tolist()

    for nome_pasta in nomes_pastas_ordenados:
        df_pasta_exibicao = pastas_agrupadas_exibicao.get_group(nome_pasta)
        multi_user_info = " (Múltiplos Usuários)" if nome_pasta in pastas_com_multi_usuarios_set else ""
        with st.expander(f"📁 Pasta: {nome_pasta} ({len(df_pasta_exibicao)} atividades){multi_user_info}", expanded=True):
            if nome_pasta in pastas_com_multi_usuarios_set:
                st.caption(f"👥 Usuários: {', '.join(df_pasta_exibicao['user_profile_name'].unique())}")
            for _, atividade in df_pasta_exibicao.iterrows():
                st.markdown("---")
                st.markdown(f"**ID:** `{atividade['activity_id']}` | **Data:** {atividade['activity_date'].strftime('%d/%m/%Y')} | **Status:** `{atividade['activity_status']}`")
                st.markdown(f"**Usuário:** {atividade['user_profile_name']}")
                if 'activity_subject' in atividade and pd.notna(atividade['activity_subject']):
                    st.caption(f"Assunto: {atividade['activity_subject']}")
                link_cols = st.columns(2)
                links = gerar_links_zflow(atividade['activity_id'])
                link_cols[0].link_button("🔗 ZFlow v1", links['antigo'], key=f"v1_{atividade['activity_id']}")
                link_cols[1].link_button("🔗 ZFlow v2", links['novo'], key=f"v2_{atividade['activity_id']}")

# ==============================================================================
# PONTO DE ENTRADA DA APLICAÇÃO
# ==============================================================================

if __name__ == "__main__":
    check_login()
