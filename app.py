import streamlit as st
import pandas as pd
import re
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from datetime import datetime, timedelta
from unidecode import unidecode
from rapidfuzz import fuzz

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================

# Configuração da página do Streamlit
st.set_page_config(layout="wide", page_title="Verificador de Duplicidade")

# Cores de alto contraste para os badges de similaridade (texto preto funciona bem)
CORES_SIMILARIDADE = {
    'alta': '#FF5252',  # Vermelho
    'media': '#FFB74D', # Laranja
    'baixa': '#FFD54F', # Amarelo
}
LIMIAR_ALTA = 0.90
LIMIAR_MEDIA = 0.70

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def normalizar_texto(txt: str | None) -> str:
    """Limpa e padroniza o texto para uma comparação mais eficaz."""
    if not txt or not isinstance(txt, str):
        return ""
    txt = unidecode(txt.lower())          # Remove acentos e converte para minúsculas
    txt = re.sub(r'[^\w\s]', ' ', txt)     # Remove pontuação, substituindo por espaço
    txt = re.sub(r'\s+', ' ', txt).strip() # Remove espaços múltiplos e nas pontas
    return txt

def calcular_similaridade(texto_a: str, texto_b: str) -> float:
    """Calcula a similaridade usando rapidfuzz após normalizar os textos."""
    norm_a = normalizar_texto(texto_a)
    norm_b = normalizar_texto(texto_b)

    if not norm_a or not norm_b:
        return 0.0
    
    if abs(len(norm_a) - len(norm_b)) > 0.3 * max(len(norm_a), len(norm_b)):
        return 0.0

    return fuzz.token_set_ratio(norm_a, norm_b) / 100.0

def obter_cor_similaridade(ratio: float) -> str:
    """Retorna a cor correspondente com base no nível de similaridade."""
    if ratio >= LIMIAR_ALTA:
        return CORES_SIMILARIDADE['alta']
    if ratio >= LIMIAR_MEDIA:
        return CORES_SIMILARIDADE['media']
    return CORES_SIMILARIDADE['baixa']

# --- Função de conexão com credenciais hardcoded, como solicitado ---
@st.cache_resource
def get_db_engine() -> Engine | None:
    """Cria e retorna uma engine de conexão com o banco, com credenciais no código."""
    # ==========================================================================
    # ATENÇÃO: CREDENCIAIS DO BANCO DE DADOS DIRETAMENTE NO CÓDIGO!
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
        # pool_pre_ping verifica a conexão antes de usá-la, evitando "MySQL has gone away"
        engine_instance = create_engine(db_uri, pool_pre_ping=True, pool_recycle=3600)
        with engine_instance.connect():
            pass # Testa a conexão
        return engine_instance
    except exc.SQLAlchemyError as e:
        st.error("Erro ao conectar ao banco de dados.")
        st.exception(e) # Mostra o erro completo para debug
        return None

@st.cache_data(ttl=600, hash_funcs={Engine: lambda _: None})
def buscar_atividades(_engine: Engine, data_inicio: datetime.date, data_fim: datetime.date) -> pd.DataFrame:
    """Busca atividades no banco de dados para o período especificado."""
    query = text("""
        SELECT activity_id, activity_folder, activity_subject, user_id, user_profile_name,
               activity_date, activity_fatal, activity_status, activity_type,
               activity_publish_date, Texto, observacoes, tags,
               activity_created_at, activity_updated_at
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type = :tipo_atividade
          AND DATE(activity_date) BETWEEN :data_inicio AND :data_fim
        ORDER BY activity_folder, activity_date, activity_id
    """)
    try:
        with _engine.connect() as connection:
            df = pd.read_sql(query, connection, params={
                "tipo_atividade": "Verificar",
                "data_inicio": data_inicio,
                "data_fim": data_fim
            })
        if 'activity_date' in df.columns:
            df['activity_date'] = pd.to_datetime(df['activity_date'])
        return df
    except exc.SQLAlchemyError as e:
        st.error("Erro ao buscar atividades no banco.")
        st.exception(e)
        return pd.DataFrame()

# ==============================================================================
# INTERFACE PRINCIPAL (Streamlit)
# ==============================================================================

def main():
    """Função principal que renderiza a página do Streamlit."""
    st.title("🔎 Verificador de Duplicidade de Atividades")
    # Adicionando o aviso de segurança diretamente na interface, como no original
    st.warning("⚠️ As credenciais do banco de dados estão diretamente no código. Esta abordagem não é segura para ambientes de produção ou código compartilhado.")

    engine = get_db_engine()
    if not engine:
        st.stop()

    # --- Filtros na Sidebar ---
    st.sidebar.header("🗓️ Filtros de Análise")
    hoje = datetime.today().date()
    data_inicio_filtro = st.sidebar.date_input("Data de Início", hoje - timedelta(days=1))
    data_fim_filtro = st.sidebar.date_input("Data de Fim", hoje + timedelta(days=1))

    if data_inicio_filtro > data_fim_filtro:
        st.sidebar.error("A data de início não pode ser posterior à data de fim.")
        st.stop()

    df_atividades_raw = buscar_atividades(engine, data_inicio_filtro, data_fim_filtro)

    if df_atividades_raw.empty:
        st.info(f"Nenhuma atividade 'Verificar' encontrada para o período selecionado.")
        st.stop()

    st.success(f"**{len(df_atividades_raw)}** atividades carregadas para o período.")
    
    with st.expander("Resumo de Atividades por Pasta", expanded=True):
        st.dataframe(
            df_atividades_raw.groupby('activity_folder')['activity_id']
            .count().rename('Qtd. Atividades')
            .reset_index().sort_values('Qtd. Atividades', ascending=False),
            use_container_width=True
        )

    min_similaridade_display = st.sidebar.slider(
        "Exibir similaridades de texto ≥ que (%):", min_value=0, max_value=100, value=70, step=5
    ) / 100.0
    
    mostrar_apenas_com_duplicatas = st.sidebar.checkbox(
        "Mostrar apenas atividades com duplicatas", value=True
    )
    st.sidebar.markdown("---")

    st.header("Análise Detalhada por Pasta")
    
    pastas_agrupadas = df_atividades_raw.groupby('activity_folder')
    ids_com_duplicatas = set()
    todas_similaridades = []

    with st.spinner("Analisando similaridades... Este processo pode levar alguns segundos."):
        for nome_pasta, df_pasta in pastas_agrupadas:
            if len(df_pasta) < 2:
                continue

            atividades_lista = df_pasta.to_dict('records')
            for i in range(len(atividades_lista)):
                for j in range(i + 1, len(atividades_lista)):
                    base = atividades_lista[i]
                    comparar = atividades_lista[j]
                    
                    similaridade = calcular_similaridade(base['Texto'], comparar['Texto'])
                    
                    if similaridade >= min_similaridade_display:
                        cor = obter_cor_similaridade(similaridade)
                        todas_similaridades.append({'id_base': base['activity_id'], 'id_similar': comparar['activity_id'], 'ratio': similaridade, 'cor': cor})
                        todas_similaridades.append({'id_base': comparar['activity_id'], 'id_similar': base['activity_id'], 'ratio': similaridade, 'cor': cor})
                        ids_com_duplicatas.add(base['activity_id'])
                        ids_com_duplicatas.add(comparar['activity_id'])
    
    df_similaridades = pd.DataFrame(todas_similaridades)
    
    if mostrar_apenas_com_duplicatas and not ids_com_duplicatas:
        st.info("Nenhuma atividade com potencial de duplicata encontrada com os filtros atuais.")
        st.stop()

    df_para_exibir = df_atividades_raw.copy()
    if mostrar_apenas_com_duplicatas:
        df_para_exibir = df_atividades_raw[df_atividades_raw['activity_id'].isin(ids_com_duplicatas)]

    if not df_para_exibir.empty and ids_com_duplicatas:
         df_dupes_export = df_atividades_raw[df_atividades_raw['activity_id'].isin(ids_com_duplicatas)]
         st.download_button(
             "📥 Exportar Duplicatas (.csv)", 
             df_dupes_export.to_csv(index=False, sep=';'), 
             "duplicatas.csv",
             "text/csv"
         )

    pastas_agrupadas_exibicao = df_para_exibir.groupby('activity_folder')
    for nome_pasta, df_pasta_exibicao in pastas_agrupadas_exibicao:
        total_na_pasta_original = len(df_atividades_raw[df_atividades_raw['activity_folder'] == nome_pasta])
        expander_title = f"📁 Pasta: {nome_pasta} ({len(df_pasta_exibicao)} atividades com duplicatas / {total_na_pasta_original} no total)"
        
        with st.expander(expander_title, expanded=False):
            for _, atividade in df_pasta_exibicao.iterrows():
                row_id = atividade['activity_id']
                st.markdown("---")
                col_info, col_similar = st.columns([0.6, 0.4])

                with col_info:
                    st.markdown(f"**ID:** `{row_id}` | **Data:** {atividade['activity_date'].strftime('%d/%m/%Y %H:%M')} | **Status:** `{atividade['activity_status']}`")
                    st.markdown(f"**Usuário:** {atividade['user_profile_name']}")
                    st.text_area(
                        "Texto da Publicação:", value=str(atividade['Texto']), height=100, 
                        key=f"texto_{nome_pasta}_{row_id}", disabled=True
                    )
                
                with col_similar:
                    similares = df_similaridades[df_similaridades['id_base'] == row_id] if not df_similaridades.empty else pd.DataFrame()
                    if not similares.empty:
                        st.markdown(f"**<span style='color:red;'>Potenciais Duplicatas:</span>** ({len(similares)})", unsafe_allow_html=True)
                        similares_ordenados = similares.sort_values(by='ratio', ascending=False)
                        
                        for _, sim_info in similares_ordenados.iterrows():
                            info_dupe = df_atividades_raw[df_atividades_raw['activity_id'] == sim_info['id_similar']].iloc[0]
                            st.markdown(
                                f"""<small><div style='background-color:{sim_info['cor']}; padding: 3px 6px; border-radius: 5px; color: black; margin-bottom: 5px; font-weight: 500;'>
                                <b>ID: {info_dupe['activity_id']} ({sim_info['ratio']:.0%})</b><br>
                                Data: {info_dupe['activity_date'].strftime('%d/%m/%y')} | Status: {info_dupe['activity_status']}<br>
                                Usuário: {info_dupe['user_profile_name']}
                                </div></small>""",
                                unsafe_allow_html=True
                            )
                    else:
                        st.markdown("**<span style='color:green;'>Sem duplicatas (nesta exibição)</span>**", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
