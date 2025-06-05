import streamlit as st
import pandas as pd
import re
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from datetime import datetime, timedelta
from unidecode import unidecode
from rapidfuzz import fuzz
import io # Para exportação XLSX

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================

st.set_page_config(layout="wide", page_title="Verificador de Duplicidade Avançado")

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
    if not txt or not isinstance(txt, str):
        return ""
    txt = unidecode(txt.lower())
    txt = re.sub(r'[^\w\s]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

def calcular_similaridade(texto_a: str, texto_b: str) -> float:
    norm_a = normalizar_texto(texto_a)
    norm_b = normalizar_texto(texto_b)
    if not norm_a or not norm_b:
        return 0.0
    # Heurística para evitar comparações desnecessárias
    if abs(len(norm_a) - len(norm_b)) > 0.3 * max(len(norm_a), len(norm_b)):
        return 0.0
    return fuzz.token_set_ratio(norm_a, norm_b) / 100.0

def obter_cor_similaridade(ratio: float) -> str:
    if ratio >= LIMIAR_ALTA: return CORES_SIMILARIDADE['alta']
    if ratio >= LIMIAR_MEDIA: return CORES_SIMILARIDADE['media']
    return CORES_SIMILARIDADE['baixa']

def gerar_links_zflow(activity_id: int) -> dict:
    link_antigo = f"https://zflow.zionbyonset.com.br/activity/3/details/{activity_id}"
    link_novo = f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={activity_id}#/fixcol1"
    return {"antigo": link_antigo, "novo": link_novo}

@st.cache_resource
def get_db_engine() -> Engine | None:
    db_user, db_pass, db_host, db_name = "tarcisio", "123qwe", "40.88.40.110", "zion_flow"
    if not all([db_user, db_pass, db_host, db_name]):
        st.error("Credenciais do banco de dados não definidas completamente no código.")
        return None
    db_uri = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_name}"
    try:
        engine_instance = create_engine(db_uri, pool_pre_ping=True, pool_recycle=3600)
        with engine_instance.connect(): pass
        return engine_instance
    except exc.SQLAlchemyError as e:
        st.error("Erro ao conectar ao banco de dados.")
        st.exception(e)
        return None

@st.cache_data(ttl=600, hash_funcs={Engine: lambda _: None})
def buscar_atividades(_engine: Engine, data_inicio: datetime.date, data_fim: datetime.date) -> pd.DataFrame:
    query = text("""
        SELECT activity_id, activity_folder, activity_subject, user_id, user_profile_name,
               activity_date, activity_fatal, activity_status, activity_type,
               activity_publish_date, Texto, observacoes, tags,
               activity_created_at, activity_updated_at
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type = :tipo_atividade
          AND DATE(activity_date) BETWEEN :data_inicio AND :data_fim
        ORDER BY activity_folder, activity_date DESC, activity_id DESC
    """)
    try:
        with _engine.connect() as connection:
            df = pd.read_sql(query, connection, params={
                "tipo_atividade": "Verificar",
                "data_inicio": data_inicio, "data_fim": data_fim
            })
        if 'activity_date' in df.columns:
            df['activity_date'] = pd.to_datetime(df['activity_date'])
        if 'Texto' in df.columns:
            df['Texto'] = df['Texto'].astype(str).fillna('')
        return df
    except exc.SQLAlchemyError as e:
        st.error("Erro ao buscar atividades no banco.")
        st.exception(e)
        return pd.DataFrame()

# ==============================================================================
# INTERFACE PRINCIPAL
# ==============================================================================
def main():
    st.title("🔎 Verificador de Duplicidade de Atividades")
    
    engine = get_db_engine()
    if not engine:
        st.stop()

    # --- Sidebar: Filtros ---
    st.sidebar.header("⚙️ Filtros e Opções")
    hoje = datetime.today().date()
    data_inicio_filtro = st.sidebar.date_input("Data de Início", hoje - timedelta(days=1))
    data_fim_filtro = st.sidebar.date_input("Data de Fim", hoje + timedelta(days=1))

    if data_inicio_filtro > data_fim_filtro:
        st.sidebar.error("A data de início não pode ser posterior à data de fim.")
        st.stop()

    df_atividades_raw = buscar_atividades(engine, data_inicio_filtro, data_fim_filtro)

    if df_atividades_raw.empty:
        st.info("Nenhuma atividade 'Verificar' encontrada para o período selecionado.")
        st.stop()

    st.success(f"**{len(df_atividades_raw)}** atividades carregadas para o período.")
    
    # --- Filtros de Análise (aplicados antes do cálculo de similaridade) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("1. Filtros de Análise")
    pastas_disponiveis = sorted(df_atividades_raw['activity_folder'].dropna().unique())
    pastas_selecionadas = st.sidebar.multiselect("Analisar apenas Pasta(s):", pastas_disponiveis, default=[])

    status_disponiveis = sorted(df_atividades_raw['activity_status'].dropna().unique())
    status_selecionados = st.sidebar.multiselect("Analisar apenas Status:", status_disponiveis, default=[])

    df_para_analise = df_atividades_raw.copy()
    if pastas_selecionadas:
        df_para_analise = df_para_analise[df_para_analise['activity_folder'].isin(pastas_selecionadas)]
    if status_selecionados:
        df_para_analise = df_para_analise[df_para_analise['activity_status'].isin(status_selecionados)]

    # --- Filtros de Exibição (aplicados após o cálculo) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Filtros de Exibição")
    min_similaridade_display = st.sidebar.slider(
        "Exibir similaridades ≥ que (%):", min_value=0, max_value=100, value=70, step=5
    ) / 100.0
    
    mostrar_apenas_com_duplicatas = st.sidebar.checkbox(
        "Exibir apenas atividades com duplicatas", value=True
    )
    
    pastas_com_multiplos_usuarios_set = {
        nome for nome, df_grupo in df_para_analise.groupby('activity_folder')
        if df_grupo['user_profile_name'].nunique() > 1
    }
    
    apenas_pastas_multi_usuarios_cb = st.sidebar.checkbox(
        "Exibir apenas pastas com múltiplos usuários", 
        False, 
        help="Mostra pastas que, após os filtros de análise, contêm atividades de mais de um usuário."
    )

    usuarios_disponiveis = sorted(df_para_analise['user_profile_name'].dropna().unique()) if not df_para_analise.empty else []
    usuarios_selecionados_exibicao = st.sidebar.multiselect(
        "Exibir apenas Usuário(s):", usuarios_disponiveis, default=[]
    )

    # --- Análise de Similaridade ---
    ids_com_duplicatas = set()
    todas_similaridades = []

    with st.spinner(f"Analisando {len(df_para_analise)} atividades..."):
        for nome_pasta, df_pasta in df_para_analise.groupby('activity_folder'):
            if len(df_pasta) < 2: continue
            
            atividades_lista = df_pasta.to_dict('records')
            for i in range(len(atividades_lista)):
                for j in range(i + 1, len(atividades_lista)):
                    base, comparar = atividades_lista[i], atividades_lista[j]
                    similaridade = calcular_similaridade(base['Texto'], comparar['Texto'])
                    if similaridade >= min_similaridade_display:
                        cor = obter_cor_similaridade(similaridade)
                        # Adiciona o par para ambas as atividades
                        todas_similaridades.append({'id_base': base['activity_id'], 'id_similar': comparar['activity_id'], 'ratio': similaridade, 'cor': cor})
                        todas_similaridades.append({'id_base': comparar['activity_id'], 'id_similar': base['activity_id'], 'ratio': similaridade, 'cor': cor})
                        ids_com_duplicatas.add(base['activity_id'])
                        ids_com_duplicatas.add(comparar['activity_id'])
    
    df_similaridades = pd.DataFrame(todas_similaridades).drop_duplicates()

    # --- Aplicação dos Filtros de Exibição ---
    df_para_exibir = df_para_analise.copy()
    if mostrar_apenas_com_duplicatas:
        df_para_exibir = df_para_exibir[df_para_exibir['activity_id'].isin(ids_com_duplicatas)]
    if apenas_pastas_multi_usuarios_cb:
        df_para_exibir = df_para_exibir[df_para_exibir['activity_folder'].isin(pastas_com_multiplos_usuarios_set)]
    if usuarios_selecionados_exibicao:
        df_para_exibir = df_para_exibir[df_para_exibir['user_profile_name'].isin(usuarios_selecionados_exibicao)]

    # --- Botão de Exportação (agora para XLSX) ---
    st.sidebar.markdown("---")
    if st.sidebar.button("📥 Exportar para XLSX"):
        if not df_para_exibir.empty:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Aba 1: Atividades exibidas
                df_para_exibir.to_excel(writer, index=False, sheet_name='Atividades_Exibidas')
                # Aba 2: Detalhes das duplicatas
                if not df_similaridades.empty:
                    df_export_sim = df_similaridades[df_similaridades['id_base'].isin(df_para_exibir['activity_id'])]
                    df_export_sim.to_excel(writer, index=False, sheet_name='Detalhes_Duplicatas')
            
            st.sidebar.download_button(
                "Baixar XLSX", output.getvalue(), "duplicatas.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.sidebar.warning("Nenhum dado para exportar com os filtros atuais.")

    # --- Exibição dos Resultados ---
    st.header("Análise Detalhada por Pasta")
    
    if df_para_exibir.empty:
        st.info("Nenhuma atividade corresponde a todos os filtros selecionados.")
        st.stop()
    
    pastas_agrupadas_exibicao = df_para_exibir.groupby('activity_folder')
    for nome_pasta, df_pasta_exibicao in pastas_agrupadas_exibicao:
        total_na_pasta_original = len(df_atividades_raw[df_atividades_raw['activity_folder'] == nome_pasta])
        expander_title = f"📁 Pasta: {nome_pasta} ({len(df_pasta_exibicao)} atividades na exibição / {total_na_pasta_original} no total)"
        
        with st.expander(expander_title, expanded=False):
            for _, atividade in df_pasta_exibicao.iterrows():
                row_id = atividade['activity_id']
                links = gerar_links_zflow(row_id)
                st.markdown("---")
                col_info, col_similar = st.columns([0.6, 0.4])

                with col_info:
                    st.markdown(f"**ID:** `{row_id}` | **Data:** {atividade['activity_date'].strftime('%d/%m/%Y %H:%M')} | **Status:** `{atividade['activity_status']}`")
                    st.markdown(f"**Usuário:** {atividade['user_profile_name']}")
                    st.text_area(
                        "Texto da Publicação:", value=str(atividade['Texto']), height=100, 
                        key=f"texto_{nome_pasta}_{row_id}", disabled=True
                    )
                    
                    btn_cols = st.columns(2)
                    btn_cols[0].link_button("🔗 ZFlow v1", links['antigo'], help="Abrir no ZFlow (versão antiga)")
                    btn_cols[1].link_button("🔗 ZFlow v2", links['novo'], help="Abrir no ZFlow (versão nova)")

                with col_similar:
                    similares = df_similaridades[df_similaridades['id_base'] == row_id] if not df_similaridades.empty else pd.DataFrame()
                    if not similares.empty:
                        st.markdown(f"**<span style='color:red;'>Potenciais Duplicatas:</span>** ({len(similares)})", unsafe_allow_html=True)
                        similares_ordenados = similares.sort_values(by='ratio', ascending=False)
                        
                        for _, sim_info in similares_ordenados.iterrows():
                            # Usar df_raw para garantir que a info da duplicata seja encontrada mesmo que ela tenha sido filtrada da exibição
                            info_dupe_rows = df_atividades_raw[df_atividades_raw['activity_id'] == sim_info['id_similar']]
                            if not info_dupe_rows.empty:
                                info_dupe = info_dupe_rows.iloc[0]
                                st.markdown(
                                    f"""<small><div style='background-color:{sim_info['cor']}; padding: 3px 6px; border-radius: 5px; color: black; margin-bottom: 5px; font-weight: 500;'>
                                    <b>ID: {info_dupe['activity_id']} ({sim_info['ratio']:.0%})</b><br>
                                    Data: {info_dupe['activity_date'].strftime('%d/%m/%y')} | Status: {info_dupe['activity_status']}<br>
                                    Usuário: {info_dupe['user_profile_name']}
                                    </div></small>""",
                                    unsafe_allow_html=True
                                )
                    else:
                        st.markdown("**<span style='color:green;'>Sem duplicatas (nesta análise)</span>**", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
