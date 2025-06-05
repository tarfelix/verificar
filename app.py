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
    CORES = {'alta': '#FF5252', 'media': '#FFB74D', 'baixa': '#FFD54F'} # Cores para os badges
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
    # Credenciais diretamente no código - ATENÇÃO para segurança em produção
    db_user, db_pass, db_host, db_name = "tarcisio", "123qwe", "40.88.40.110", "zion_flow"
    if not all([db_user, db_pass, db_host, db_name]):
        st.error("Credenciais do banco não definidas.")
        return None
    db_uri = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_name}"
    try:
        engine = create_engine(db_uri, pool_pre_ping=True, pool_recycle=3600)
        with engine.connect(): pass # Testa a conexão
        return engine
    except exc.SQLAlchemyError as e:
        st.error(f"Erro ao conectar ao banco: {e}")
        return None

@st.cache_data(ttl=7200) # Cache de 2 horas para os dados carregados
def carregar_dados_iniciais(_engine: Engine) -> pd.DataFrame:
    """Carrega atividades 'Verificar' ABERTAS dos últimos 7 dias."""
    st.toast("Buscando dados no banco MySQL...")
    # Query para buscar atividades 'Verificar' ABERTAS dos últimos 7 dias
    # (desde 7 dias atrás até a data atual)
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
                "status_atividade": "Aberta", # Filtra por status "Aberta"
                "data_limite": data_limite     # Filtra pelos últimos 7 dias
            })
        # Assegura que 'activity_date' seja datetime e 'Texto' seja string
        df['activity_date'] = pd.to_datetime(df['activity_date'])
        df['Texto'] = df['Texto'].astype(str).fillna('')
        return df
    except exc.SQLAlchemyError as e:
        st.error("Erro ao buscar atividades no banco.")
        st.exception(e) # Mostra o traceback completo do erro SQL para debug
        return pd.DataFrame()

# ==============================================================================
# INTERFACE PRINCIPAL
# ==============================================================================
def main():
    st.title("🔎 Verificador de Duplicidade Otimizado")
    st.markdown("Análise de atividades 'Verificar' para identificar potenciais duplicidades.")

    engine = get_db_engine()
    if not engine:
        st.stop() # Para a execução se não houver conexão com o banco

    # --- Sidebar: Filtros e Opções ---
    st.sidebar.header("⚙️ Filtros e Opções")
    
    # Botão para atualizar os dados do banco
    if st.sidebar.button("🔄 Atualizar Dados do Banco", help="Limpa o cache e busca os dados mais recentes (últimos 7 dias, status 'Aberta') do MySQL."):
        carregar_dados_iniciais.clear() # Limpa o cache da função de carregamento
        st.toast("Cache limpo! Os dados serão recarregados do banco.")
        st.rerun() # Força o recarregamento da página para buscar novos dados

    # Carregar dados (usará o cache se não for atualizado)
    df_raw_total = carregar_dados_iniciais(engine)

    if df_raw_total.empty:
        st.warning("Nenhuma atividade 'Verificar' (Aberta, últimos 7 dias) retornada do banco de dados.")
        st.stop() # Para se nenhum dado inicial for carregado
    
    # --- Filtro de data agora é em memória, sobre os dados já carregados ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("1. Filtro de Período (dentro dos dados carregados)")
    
    # Define as datas mínimas e máximas para os seletores com base nos dados carregados
    # Isso garante que o usuário só possa selecionar datas presentes no conjunto de dados em memória
    min_data_disponivel = df_raw_total['activity_date'].dt.date.min()
    max_data_disponivel = df_raw_total['activity_date'].dt.date.max()

    data_inicio_filtro = st.sidebar.date_input(
        "Data de Início", 
        value=min_data_disponivel, # Padrão para a data mais antiga nos dados carregados
        min_value=min_data_disponivel,
        max_value=max_data_disponivel,
        key="di_cache" # Chave única
    )
    data_fim_filtro = st.sidebar.date_input(
        "Data de Fim", 
        value=max_data_disponivel, # Padrão para a data mais recente
        min_value=min_data_disponivel,
        max_value=max_data_disponivel,
        key="df_cache" # Chave única
    )

    if data_inicio_filtro > data_fim_filtro:
        st.sidebar.error("A data de início não pode ser posterior à data de fim.")
        st.stop() # Para a execução se as datas forem inválidas
    
    # Aplica o filtro de data ao DataFrame carregado em memória
    mask_data = (df_raw_total['activity_date'].dt.date >= data_inicio_filtro) & \
                (df_raw_total['activity_date'].dt.date <= data_fim_filtro)
    df_atividades_periodo = df_raw_total[mask_data]

    if df_atividades_periodo.empty:
        st.info("Nenhuma atividade encontrada para o período selecionado dentro dos dados carregados.")
        # Não usar st.stop() aqui, para permitir que o usuário ajuste os filtros
    else:
        st.success(f"**{len(df_atividades_periodo)}** atividades carregadas para o período selecionado.")
    
    # --- Filtros de Análise (aplicados ao df_atividades_periodo) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Filtros de Análise")
    pastas_disponiveis = sorted(df_atividades_periodo['activity_folder'].dropna().unique()) if not df_atividades_periodo.empty else []
    pastas_selecionadas = st.sidebar.multiselect("Analisar apenas Pasta(s):", pastas_disponiveis, default=[])

    # O filtro de Status agora é aplicado sobre os dados já filtrados por "Aberta" no carregamento inicial.
    # Se quiser permitir ver outros status, a query inicial teria que ser mais abrangente.
    # Por agora, como o carregamento inicial é só de "Aberta", este filtro pode não ser tão útil
    # a menos que a query inicial mude.
    status_disponiveis_analise = sorted(df_atividades_periodo['activity_status'].dropna().unique()) if not df_atividades_periodo.empty else []
    status_selecionados_analise = st.sidebar.multiselect("Analisar apenas Status (dos dados carregados):", status_disponiveis_analise, default=[])

    df_para_analise = df_atividades_periodo.copy()
    if pastas_selecionadas:
        df_para_analise = df_para_analise[df_para_analise['activity_folder'].isin(pastas_selecionadas)]
    if status_selecionados_analise: # Se o usuário selecionar algum status aqui
        df_para_analise = df_para_analise[df_para_analise['activity_status'].isin(status_selecionados_analise)]
    
    # --- Filtros de Exibição (aplicados após o cálculo de similaridade) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Filtros de Exibição")
    min_similaridade = st.sidebar.slider("Exibir similaridades ≥ que (%):", 0, 100, 70, 5, key="sim_slider_cache") / 100.0
    apenas_com_duplicatas = st.sidebar.checkbox("Exibir apenas atividades com duplicatas", value=True, key="dup_cb_cache")
    
    pastas_com_multiplos_usuarios = {
        nome for nome, grupo in df_para_analise.groupby('activity_folder') 
        if grupo['user_profile_name'].nunique() > 1
    } if not df_para_analise.empty else set()
    
    apenas_pastas_multi_usuarios = st.sidebar.checkbox(
        "Exibir apenas pastas com múltiplos usuários", 
        False, 
        key="multiuser_cb_cache"
    )

    usuarios_disponiveis_exibicao = sorted(df_para_analise['user_profile_name'].dropna().unique()) if not df_para_analise.empty else []
    usuarios_selecionados = st.sidebar.multiselect(
        "Exibir apenas Usuário(s):", 
        usuarios_disponiveis_exibicao, 
        default=[], 
        key="user_select_cache"
    )

    # --- Análise de Similaridade ---
    ids_com_duplicatas = set()
    todas_similaridades = [] # Lista para armazenar os pares de similaridade

    # Só executa a análise se houver dados e o dataframe para análise não estiver vazio
    if not df_para_analise.empty:
        with st.spinner(f"Analisando {len(df_para_analise)} atividades para similaridade..."):
            for nome_pasta, df_pasta in df_para_analise.groupby('activity_folder'):
                if len(df_pasta) < 2: continue # Pula se não houver pelo menos 2 atividades para comparar
                
                atividades_lista = df_pasta.to_dict('records')
                for i in range(len(atividades_lista)):
                    for j in range(i + 1, len(atividades_lista)):
                        base, comparar = atividades_lista[i], atividades_lista[j]
                        similaridade = calcular_similaridade(base['Texto'], comparar['Texto'])
                        if similaridade >= min_similaridade:
                            cor = obter_cor_similaridade(similaridade)
                            # Adiciona informação do par para a atividade base
                            todas_similaridades.append({
                                'id_base': base['activity_id'], 
                                'id_similar': comparar['activity_id'], 
                                'ratio': similaridade, 
                                'cor': cor
                            })
                            ids_com_duplicatas.add(base['activity_id'])
                            ids_com_duplicatas.add(comparar['activity_id'])
        
    df_similaridades = pd.DataFrame(todas_similaridades)

    # --- Aplicação dos Filtros de Exibição ---
    df_exibir = df_para_analise.copy() # Começa com os dados já filtrados por data, pasta e status
    if apenas_com_duplicatas:
        df_exibir = df_exibir[df_exibir['activity_id'].isin(ids_com_duplicatas)]
    if apenas_pastas_multi_usuarios:
        df_exibir = df_exibir[df_exibir['activity_folder'].isin(pastas_com_multiplos_usuarios)]
    if usuarios_selecionados:
        df_exibir = df_exibir[df_exibir['user_profile_name'].isin(usuarios_selecionados)]

    # --- Botão de Exportação ---
    st.sidebar.markdown("---")
    if st.sidebar.button("📥 Exportar para XLSX", key="export_btn_cache"):
        if not df_exibir.empty:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_exibir.to_excel(writer, index=False, sheet_name='Atividades_Exibidas')
                # Adiciona detalhes das duplicatas apenas para as atividades exibidas
                if not df_similaridades.empty:
                    # Filtra df_similaridades para incluir apenas aquelas onde id_base está em df_exibir
                    df_sim_export = df_similaridades[df_similaridades['id_base'].isin(df_exibir['activity_id'])]
                    # Merge para adicionar detalhes da atividade similar
                    if not df_sim_export.empty:
                        df_sim_export = pd.merge(
                            df_sim_export,
                            df_raw_total[['activity_id', 'activity_date', 'user_profile_name', 'activity_status']].rename(
                                columns={
                                    'activity_id': 'id_similar', 
                                    'activity_date': 'data_similar',
                                    'user_profile_name': 'usuario_similar',
                                    'activity_status': 'status_similar'
                                }
                            ),
                            on='id_similar',
                            how='left'
                        )
                        df_sim_export.to_excel(writer, index=False, sheet_name='Detalhes_Duplicatas')
            
            st.sidebar.download_button(
                "Baixar XLSX", 
                output.getvalue(), 
                f"duplicatas_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.sidebar.warning("Nenhum dado para exportar com os filtros atuais.")

    # --- Exibição dos Resultados ---
    st.header("Análise Detalhada por Pasta")
    if df_exibir.empty:
        st.info("Nenhuma atividade corresponde a todos os filtros selecionados.")
    
    # Ordenação de pastas para exibição
    pastas_para_exibir_ordenadas = []
    if not df_exibir.empty:
        # Adicionar aqui a lógica de ordenação das pastas se necessário, 
        # por enquanto, usa a ordem do groupby
        pastas_para_exibir_ordenadas = sorted(df_exibir['activity_folder'].unique())


    for nome_pasta in pastas_para_exibir_ordenadas:
        df_pasta_exibicao = df_exibir[df_exibir['activity_folder'] == nome_pasta]
        if df_pasta_exibicao.empty: continue # Pula se a pasta ficou vazia após filtros

        total_analisado_na_pasta = len(df_para_analise[df_para_analise['activity_folder'] == nome_pasta])
        titulo_expander = f"📁 Pasta: {nome_pasta} ({len(df_pasta_exibicao)} exibidas / {total_analisado_na_pasta} analisadas)"
        
        with st.expander(titulo_expander, expanded=False):
            for _, atividade in df_pasta_exibicao.iterrows():
                st.markdown("---")
                col_info, col_similar_display = st.columns([0.6, 0.4]) # Renomeado col_similar
                with col_info:
                    st.markdown(f"**ID:** `{atividade['activity_id']}` | **Data:** {atividade['activity_date'].strftime('%d/%m/%Y %H:%M')} | **Status:** `{atividade['activity_status']}`")
                    st.markdown(f"**Usuário:** {atividade['user_profile_name']}")
                    st.text_area(
                        "Texto:", 
                        str(atividade['Texto']), 
                        height=100, 
                        key=f"texto_exp_{nome_pasta}_{atividade['activity_id']}", # Chave única
                        disabled=True
                    )
                    btn_cols_links = st.columns(2) # Renomeado
                    links = gerar_links_zflow(atividade['activity_id'])
                    btn_cols_links[0].link_button("🔗 ZFlow v1", links['antigo'])
                    btn_cols_links[1].link_button("🔗 ZFlow v2", links['novo'])

                with col_similar_display:
                    # Filtra as similaridades para a atividade atual
                    similares_info = df_similaridades[df_similaridades['id_base'] == atividade['activity_id']] if not df_similaridades.empty else pd.DataFrame()
                    
                    if not similares_info.empty:
                        st.markdown(f"**<span style='color:red;'>Potenciais Duplicatas:</span>** ({len(similares_info)})", unsafe_allow_html=True)
                        # Ordena as duplicatas pela maior similaridade primeiro
                        for _, sim_data in similares_info.sort_values(by='ratio', ascending=False).iterrows():
                            # Busca os detalhes da atividade similar no DataFrame original (df_raw_total ou df_para_analise)
                            # para garantir que temos os dados mesmo que ela não esteja em df_exibir
                            info_dupe_rows = df_para_analise[df_para_analise['activity_id'] == sim_data['id_similar']]
                            if not info_dupe_rows.empty:
                                info_dupe = info_dupe_rows.iloc[0]
                                st.markdown(
                                    f"""<small><div style='background-color:{sim_data['cor']}; padding: 3px 6px; border-radius: 5px; color: black; margin-bottom: 5px; font-weight: 500;'>
                                    <b>ID: {info_dupe['activity_id']} ({sim_data['ratio']:.0%})</b><br>
                                    Data: {info_dupe['activity_date'].strftime('%d/%m/%y')} | Status: {info_dupe['activity_status']}<br>
                                    Usuário: {info_dupe['user_profile_name']}
                                    </div></small>""",
                                    unsafe_allow_html=True
                                )
                            else:
                                st.caption(f"Detalhes da duplicata ID {sim_data['id_similar']} não disponíveis nos dados filtrados para análise.")
                    else:
                        # Só mostra "Sem duplicatas" se o checkbox "apenas com duplicatas" NÃO estiver marcado
                        if not apenas_com_duplicatas :
                            st.markdown("**<span style='color:green;'>Sem duplicatas (nesta análise)</span>**", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
