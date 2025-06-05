import streamlit as st
import pandas as pd
import re
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from datetime import datetime, timedelta
from unidecode import unidecode
from rapidfuzz import fuzz
import io

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================
st.set_page_config(layout="wide", page_title="Verificador de Duplicidade Otimizado")

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
    db_user, db_pass, db_host, db_name = "tarcisio", "123qwe", "40.88.40.110", "zion_flow"
    if not all([db_user, db_pass, db_host, db_name]):
        # Este st.error está OK aqui, pois @st.cache_resource é diferente de @st.cache_data
        # e geralmente é chamado uma vez na inicialização.
        st.error("Credenciais do banco não definidas.")
        return None
    db_uri = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_name}"
    try:
        engine = create_engine(db_uri, pool_pre_ping=True, pool_recycle=3600)
        with engine.connect(): pass
        return engine
    except exc.SQLAlchemyError as e:
        st.error(f"Erro ao conectar ao banco: {e}") # OK aqui também
        return None

@st.cache_data(ttl=7200)
def carregar_dados_iniciais_do_banco(_engine: Engine) -> pd.DataFrame | None:
    """Carrega atividades 'Verificar' ABERTAS dos últimos 7 dias.
    Retorna DataFrame em sucesso, None em erro."""
    # st.toast removido daqui
    data_limite = datetime.today().date() - timedelta(days=7)
    query = text("""
        SELECT activity_id, activity_folder, activity_subject, user_id, user_profile_name,
               activity_date, activity_fatal, activity_status, activity_type,
               activity_publish_date, Texto, observacoes, tags,
               activity_created_at, activity_updated_at
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type = :tipo_atividade
          AND activity_status = :status_atividade 
          AND DATE(activity_date) >= :data_limite 
        ORDER BY activity_folder, activity_date DESC, activity_id DESC
    """)
    try:
        with _engine.connect() as connection:
            df = pd.read_sql(query, connection, params={
                "tipo_atividade": "Verificar",
                "status_atividade": "Aberta",
                "data_limite": data_limite
            })
        df['activity_date'] = pd.to_datetime(df['activity_date'])
        df['Texto'] = df['Texto'].astype(str).fillna('')
        return df
    except exc.SQLAlchemyError as e:
        # Não usar st.error ou st.exception aqui. Apenas logar e retornar None.
        print(f"Erro SQLAlchemy ao buscar atividades: {e}") # Log para o console do servidor
        return None # Indica erro
    except Exception as e:
        print(f"Erro geral ao buscar atividades: {e}") # Log para o console do servidor
        return None # Indica erro

# ==============================================================================
# INTERFACE PRINCIPAL
# ==============================================================================
def main():
    st.title("🔎 Verificador de Duplicidade Otimizado")
    st.markdown("Análise de atividades 'Verificar' para identificar potenciais duplicidades.")

    engine = get_db_engine()
    if not engine:
        st.stop()

    st.sidebar.header("⚙️ Filtros e Opções")
    
    if st.sidebar.button("🔄 Atualizar Dados do Banco", help="Limpa o cache e busca os dados mais recentes do MySQL."):
        carregar_dados_iniciais_do_banco.clear()
        st.toast("Cache de dados limpo! Recarregando do banco...")
        # O st.rerun() não é estritamente necessário aqui, pois o Streamlit reexecutará
        # e a função sem cache será chamada. Mas pode ajudar a forçar.
        st.rerun() 

    # Tentativa de carregar os dados
    # Exibe o toast *antes* de chamar a função cacheada, mas só faz sentido se não for um cache hit.
    # Para simplificar, vamos exibir um spinner enquanto carrega se não for cache.
    # No entanto, a forma mais simples é aceitar que o toast da função clear já avisa.
    
    df_raw_total = carregar_dados_iniciais_do_banco(engine)

    if df_raw_total is None: # Erro ao carregar dados do banco
        st.error("Falha ao carregar dados do banco de dados. Verifique os logs do servidor ou tente atualizar.")
        st.stop()
    elif df_raw_total.empty:
        st.warning("Nenhuma atividade 'Verificar' (Aberta, últimos 7 dias) retornada do banco de dados.")
        # Não parar aqui, permitir que o usuário veja a interface e tente atualizar.
    
    # --- Filtro de data ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("1. Filtro de Período")
    
    if not df_raw_total.empty:
        min_data_disponivel = df_raw_total['activity_date'].dt.date.min()
        max_data_disponivel = df_raw_total['activity_date'].dt.date.max()
        default_start = min_data_disponivel
        default_end = max_data_disponivel
    else: # Fallback se df_raw_total estiver vazio após a tentativa de carregamento
        hoje = datetime.today().date()
        default_start = hoje - timedelta(days=7)
        default_end = hoje
        min_data_disponivel = default_start
        max_data_disponivel = default_end

    data_inicio_filtro = st.sidebar.date_input("Data de Início", value=default_start, 
                                               min_value=min_data_disponivel if not df_raw_total.empty else None, 
                                               max_value=max_data_disponivel if not df_raw_total.empty else None, 
                                               key="di_cache_v2")
    data_fim_filtro = st.sidebar.date_input("Data de Fim", value=default_end, 
                                            min_value=min_data_disponivel if not df_raw_total.empty else None, 
                                            max_value=max_data_disponivel if not df_raw_total.empty else None, 
                                            key="df_cache_v2")

    if data_inicio_filtro > data_fim_filtro:
        st.sidebar.error("A data de início não pode ser posterior à data de fim.")
        st.stop()
    
    df_atividades_periodo = pd.DataFrame() # Inicializa como DataFrame vazio
    if not df_raw_total.empty:
        mask_data = (df_raw_total['activity_date'].dt.date >= data_inicio_filtro) & \
                    (df_raw_total['activity_date'].dt.date <= data_fim_filtro)
        df_atividades_periodo = df_raw_total[mask_data]

    if df_atividades_periodo.empty and not df_raw_total.empty: # Só mostra se havia dados totais mas o filtro de período zerou
        st.info("Nenhuma atividade encontrada para o período selecionado dentro dos dados carregados.")
    elif not df_atividades_periodo.empty:
        st.success(f"**{len(df_atividades_periodo)}** atividades no período selecionado (de {len(df_raw_total)} carregadas).")
    
    # --- Outros filtros e lógica de exibição (adaptar nomes de DataFrames) ---
    
    # --- Filtros de Análise (aplicados ao df_atividades_periodo) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Filtros de Análise")
    pastas_disponiveis = sorted(df_atividades_periodo['activity_folder'].dropna().unique()) if not df_atividades_periodo.empty else []
    pastas_selecionadas = st.sidebar.multiselect("Analisar apenas Pasta(s):", pastas_disponiveis, default=[])

    status_disponiveis_analise = sorted(df_atividades_periodo['activity_status'].dropna().unique()) if not df_atividades_periodo.empty else []
    status_selecionados_analise = st.sidebar.multiselect("Analisar apenas Status (dos dados carregados):", status_disponiveis_analise, default=[])

    df_para_analise = df_atividades_periodo.copy()
    if pastas_selecionadas:
        df_para_analise = df_para_analise[df_para_analise['activity_folder'].isin(pastas_selecionadas)]
    if status_selecionados_analise:
        df_para_analise = df_para_analise[df_para_analise['activity_status'].isin(status_selecionados_analise)]
    
    # --- Filtros de Exibição (aplicados após o cálculo de similaridade) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Filtros de Exibição")
    min_similaridade = st.sidebar.slider("Exibir similaridades ≥ que (%):", 0, 100, 70, 5, key="sim_slider_cache_v2") / 100.0
    apenas_com_duplicatas = st.sidebar.checkbox("Exibir apenas atividades com duplicatas", value=True, key="dup_cb_cache_v2")
    
    pastas_com_multiplos_usuarios = {
        nome for nome, grupo in df_para_analise.groupby('activity_folder') 
        if grupo['user_profile_name'].nunique() > 1
    } if not df_para_analise.empty else set()
    
    apenas_pastas_multi_usuarios = st.sidebar.checkbox("Exibir apenas pastas com múltiplos usuários", False, key="multiuser_cb_cache_v2")

    usuarios_disponiveis_exibicao = sorted(df_para_analise['user_profile_name'].dropna().unique()) if not df_para_analise.empty else []
    usuarios_selecionados = st.sidebar.multiselect("Exibir apenas Usuário(s):", usuarios_disponiveis_exibicao, default=[], key="user_select_cache_v2")

    # --- Análise de Similaridade ---
    ids_com_duplicatas = set()
    todas_similaridades = [] 

    if not df_para_analise.empty:
        with st.spinner(f"Analisando {len(df_para_analise)} atividades para similaridade... (pode levar um momento)"):
            for nome_pasta, df_pasta in df_para_analise.groupby('activity_folder'):
                if len(df_pasta) < 2: continue
                atividades_lista = df_pasta.to_dict('records')
                for i in range(len(atividades_lista)):
                    for j in range(i + 1, len(atividades_lista)):
                        base, comparar = atividades_lista[i], atividades_lista[j]
                        similaridade = calcular_similaridade(base['Texto'], comparar['Texto'])
                        if similaridade >= min_similaridade:
                            cor = obter_cor_similaridade(similaridade)
                            todas_similaridades.append({'id_base': base['activity_id'], 'id_similar': comparar['activity_id'], 'ratio': similaridade, 'cor': cor})
                            ids_com_duplicatas.add(base['activity_id'])
                            ids_com_duplicatas.add(comparar['activity_id'])
        df_similaridades = pd.DataFrame(todas_similaridades)
    else:
        df_similaridades = pd.DataFrame() # DataFrame vazio se não houver análise


    # --- Aplicação dos Filtros de Exibição ---
    df_exibir = df_para_analise.copy() 
    if apenas_com_duplicatas: df_exibir = df_exibir[df_exibir['activity_id'].isin(ids_com_duplicatas)]
    if apenas_pastas_multi_usuarios: df_exibir = df_exibir[df_exibir['activity_folder'].isin(pastas_com_multiplos_usuarios)]
    if usuarios_selecionados: df_exibir = df_exibir[df_exibir['user_profile_name'].isin(usuarios_selecionados)]

    # --- Botão de Exportação ---
    st.sidebar.markdown("---")
    if st.sidebar.button("📥 Exportar para XLSX", key="export_btn_cache_v2"):
        if not df_exibir.empty:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_exibir.to_excel(writer, index=Fal
