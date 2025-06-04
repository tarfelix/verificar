import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import difflib # Usaremos difflib.HtmlDiff

# ==============================================================================
# CONFIGURAÇÕES E FUNÇÕES AUXILIARES
# ==============================================================================
st.set_page_config(layout="wide", page_title="Verificador de Duplicidade Avançado")

# --- Funções de Similaridade e Cor (semelhantes às anteriores) ---
def calcular_similaridade(texto_a, texto_b):
    if texto_a is None or texto_b is None: return 0.0
    return difflib.SequenceMatcher(None, str(texto_a), str(texto_b)).ratio()

def obter_cor_similaridade(ratio):
    if ratio >= 0.91: return "red"
    elif ratio >= 0.71: return "orange"
    elif ratio >= 0.50: return "gold"
    return "grey"

# --- Conexão com Banco (Hardcoded - Mantenha o aviso de segurança) ---
@st.cache_resource
def get_db_engine():
    db_user, db_pass, db_host, db_name = "tarcisio", "123qwe", "40.88.40.110", "zion_flow"
    if not all([db_user, db_pass, db_host, db_name]):
        st.error("Credenciais do banco não definidas no código.")
        return None
    db_uri = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_name}"
    try:
        engine_instance = create_engine(db_uri)
        with engine_instance.connect() as conn: conn.execute(text("SELECT 1")) # Test connection
        return engine_instance
    except Exception as e:
        st.error(f"Erro ao conectar ao banco: {e}")
        return None

# --- Busca de Atividades (sem filtro de status) ---
@st.cache_data(ttl=300) # Cache reduzido para refletir mudanças mais rapidamente se necessário
def buscar_atividades_completas(_engine, data_inicio, data_fim):
    if _engine is None: return pd.DataFrame()
    query = text("""
        SELECT activity_id, activity_folder, user_profile_name, activity_date, 
               activity_status, Texto, activity_type
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type = :tipo_atividade AND activity_date BETWEEN :data_inicio AND :data_fim
        ORDER BY activity_folder, activity_date DESC, activity_id DESC
    """) # Ordenação inicial
    try:
        with _engine.connect() as connection:
            df = pd.read_sql(query, connection, params={
                "tipo_atividade": "Verificar",
                "data_inicio": data_inicio, "data_fim": data_fim
            })
        if 'activity_date' in df.columns:
            df['activity_date'] = pd.to_datetime(df['activity_date']).dt.date
        # Garantir que 'Texto' seja string para evitar problemas com None mais tarde
        if 'Texto' in df.columns:
            df['Texto'] = df['Texto'].astype(str).fillna('')
        return df
    except Exception as e:
        st.error(f"Erro ao buscar atividades: {e}")
        return pd.DataFrame()

# --- Geração de Links ZFlow ---
def gerar_links_zflow(activity_id):
    link_antigo = f"https://zflow.zionbyonset.com.br/activity/3/details/{activity_id}"
    # Corrigindo a lógica para o link novo conforme a reinterpretação da solicitação
    link_novo = f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={activity_id}#/fixcol1"
    return {"antigo": link_antigo, "novo": link_novo}

# --- Função para o Dialog de Comparação de Textos ---
def mostrar_dialog_comparacao(atividade_base, atividade_comparar):
    texto_base = str(atividade_base['Texto'])
    texto_comparar = str(atividade_comparar['Texto'])
    
    d = difflib.HtmlDiff(wrapcolumn=70)
    html_diff = d.make_table(texto_base.splitlines(), texto_comparar.splitlines(),
                             fromdesc=f"Texto Atividade ID: {atividade_base['activity_id']}",
                             todesc=f"Texto Atividade ID: {atividade_comparar['activity_id']}")

    st.markdown(f"### Comparando Atividade `{atividade_base['activity_id']}` com `{atividade_comparar['activity_id']}`")
    st.components.v1.html(html_diff, height=600, scrolling=True)

# --- Função para o Dialog de Visualização de Texto ---
def mostrar_dialog_texto_completo(atividade):
    st.markdown(f"### Texto Completo - Atividade ID: `{atividade['activity_id']}`")
    st.markdown(f"**Pasta:** {atividade['activity_folder']} | **Data:** {atividade['activity_date'].strftime('%d/%m/%Y')} | **Usuário:** {atividade['user_profile_name']} | **Status:** {atividade['activity_status']}")
    st.text_area("Texto da Publicação:", value=str(atividade['Texto']), height=400, disabled=True, key=f"full_text_dialog_{atividade['activity_id']}")


# ==============================================================================
# Inicialização do Estado da Sessão (Session State)
# ==============================================================================
if 'dialog_aberto' not in st.session_state:
    st.session_state.dialog_aberto = None # Pode ser 'texto' ou 'comparacao'
if 'atividade_dialog_texto' not in st.session_state:
    st.session_state.atividade_dialog_texto = None # Guarda a atividade para o dialog de texto
if 'atividades_dialog_comparacao' not in st.session_state:
    st.session_state.atividades_dialog_comparacao = None # Guarda {'base': ..., 'comparar': ...}

# ==============================================================================
# INTERFACE DO USUÁRIO (Streamlit)
# ==============================================================================
st.title("🔎 Verificador de Duplicidade Avançado")
st.markdown("Análise de atividades 'Verificar' para identificar potenciais duplicidades.")

engine = get_db_engine()

if engine:
    # --- Sidebar: Filtros e Opções ---
    st.sidebar.header("⚙️ Filtros e Opções")
    hoje = datetime.today().date()
    
    periodo_selecionado = st.sidebar.radio("Período das atividades:", ("Hoje, Ontem e Amanhã", "Intervalo Personalizado"), key="periodo_radio")
    data_inicio_filtro, data_fim_filtro = (hoje - timedelta(days=1), hoje + timedelta(days=1)) if periodo_selecionado == "Hoje, Ontem e Amanhã" else \
                                          (st.sidebar.date_input("Data Início", hoje - timedelta(days=1)), st.sidebar.date_input("Data Fim", hoje + timedelta(days=1)))

    if data_inicio_filtro > data_fim_filtro:
        st.sidebar.error("Data de início posterior à data de fim.")
        st.stop()

    df_atividades_raw = buscar_atividades_completas(engine, data_inicio_filtro, data_fim_filtro)

    if df_atividades_raw.empty:
        st.info(f"Nenhuma atividade 'Verificar' no período de {data_inicio_filtro.strftime('%d/%m/%Y')} a {data_fim_filtro.strftime('%d/%m/%Y')}.")
    else:
        st.success(f"{len(df_atividades_raw)} atividades 'Verificar' carregadas.")

        usuarios_disponiveis = sorted(df_atividades_raw['user_profile_name'].dropna().unique())
        usuarios_selecionados_exibicao = st.sidebar.multiselect("Filtrar exibição por Usuário(s):", usuarios_disponiveis, default=[])
        
        min_similaridade_display = st.sidebar.slider("Similaridade de texto mínima (%):", 0, 100, 50, 5) / 100.0
        
        apenas_potenciais_duplicatas_cb = st.sidebar.checkbox("Mostrar apenas com potenciais duplicatas", False)
        apenas_usuarios_diferentes_cb = st.sidebar.checkbox("Mostrar apenas pastas com múltiplos usuários", False)

        ordem_pastas = st.sidebar.selectbox(
            "Ordenar pastas por:",
            ("Nome da Pasta (A-Z)", "Mais Atividades Primeiro", "Mais Potenciais Duplicatas Primeiro (beta)"), # Ordenação de duplicatas é mais complexa
            key="ordem_pastas_select"
        )
        st.sidebar.markdown("---")
        if st.sidebar.button("Exportar Potenciais Duplicatas para CSV"):
            # Lógica de exportação será adicionada aqui
            # Precisamos primeiro ter a lista de duplicatas
            # Placeholder:
            # df_export = ...
            # st.download_button("Baixar CSV", df_export.to_csv(index=False, sep=';', encoding='utf-8-sig'), "potenciais_duplicatas.csv", "text/csv")
            st.sidebar.info("Funcionalidade de exportação em desenvolvimento.")


        # --- Pré-processamento e Análise de Duplicidade ---
        # Esta análise é feita em TODOS os dados carregados (df_atividades_raw)
        # para garantir que as duplicatas sejam encontradas independentemente dos filtros de exibição.
        similaridades_globais = {} # {id_base: [{id_similar, ratio, cor, ...}]}
        atividades_com_duplicatas_ids = set()
        pastas_com_multiplos_usuarios_set = set()

        # Calcular todas as similaridades primeiro
        for nome_pasta, df_pasta_analise in df_atividades_raw.groupby('activity_folder'):
            if df_pasta_analise['user_profile_name'].nunique() > 1:
                pastas_com_multiplos_usuarios_set.add(nome_pasta)

            atividades_lista = df_pasta_analise.to_dict('records')
            for i in range(len(atividades_lista)):
                base = atividades_lista[i]
                if base['activity_id'] not in similaridades_globais: similaridades_globais[base['activity_id']] = []
                
                for j in range(i + 1, len(atividades_lista)):
                    comparar = atividades_lista[j]
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
        
        # Ordenar as listas de similaridades por 'ratio' descendente
        for key in similaridades_globais:
            similaridades_globais[key] = sorted(similaridades_globais[key], key=lambda x: x['ratio'], reverse=True)


        # --- Filtragem para Exibição ---
        df_exibir = df_atividades_raw.copy()
        if usuarios_selecionados_exibicao:
            df_exibir = df_exibir[df_exibir['user_profile_name'].isin(usuarios_selecionados_exibicao)]
        if apenas_potenciais_duplicatas_cb:
            df_exibir = df_exibir[df_exibir['activity_id'].isin(atividades_com_duplicatas_ids)]
        if apenas_usuarios_diferentes_cb:
            df_exibir = df_exibir[df_exibir['activity_folder'].isin(pastas_com_multiplos_usuarios_set)]

        # --- Lógica de Ordenação das Pastas ---
        lista_pastas_para_exibir = []
        if not df_exibir.empty:
            pastas_agrupadas_exibicao = df_exibir.groupby('activity_folder')
            
            if ordem_pastas == "Nome da Pasta (A-Z)":
                lista_pastas_para_exibir = sorted(pastas_agrupadas_exibicao.groups.keys())
            elif ordem_pastas == "Mais Atividades Primeiro":
                lista_pastas_para_exibir = pastas_agrupadas_exibicao.size().sort_values(ascending=False).index.tolist()
            elif ordem_pastas == "Mais Potenciais Duplicatas Primeiro (beta)":
                # Contar duplicatas por pasta (considerando o df_exibir)
                contagem_duplicatas_pasta = {}
                for nome_pasta, df_p in pastas_agrupadas_exibicao:
                    count = 0
                    for act_id in df_p['activity_id']:
                        if act_id in similaridades_globais and similaridades_globais[act_id]:
                            count +=1 # Conta atividades que TEM duplicatas
                            break # Só precisa de uma para contar a pasta
                    if count > 0: contagem_duplicatas_pasta[nome_pasta] = df_p[df_p['activity_id'].isin(atividades_com_duplicatas_ids)].shape[0]

                lista_pastas_para_exibir = sorted(contagem_duplicatas_pasta, key=contagem_duplicatas_pasta.get, reverse=True)
                # Adicionar pastas sem duplicatas no final, ordenadas por nome
                pastas_sem_duplicatas = [p for p in pastas_agrupadas_exibicao.groups.keys() if p not in lista_pastas_para_exibir]
                lista_pastas_para_exibir.extend(sorted(pastas_sem_duplicatas))


        # --- Exibição Principal ---
        if not lista_pastas_para_exibir and not df_exibir.empty: # Caso especial onde filtros resultam em nenhuma pasta mas havia dados
            st.info("Nenhuma pasta corresponde a todos os critérios de filtro de exibição selecionados.")
        elif df_exibir.empty :
             st.info("Nenhuma atividade 'Verificar' corresponde aos filtros de data e exibição selecionados.")

        for nome_pasta in lista_pastas_para_exibir:
            df_pasta_exibicao_atual = df_exibir[df_exibir['activity_folder'] == nome_pasta]
            
            # Info sobre múltiplos usuários na pasta (baseado na análise completa)
            multi_user_info = " (Múltiplos Usuários na Análise)" if nome_pasta in pastas_com_multiplos_usuarios_set else ""
            
            with st.expander(f"📁 Pasta: {nome_pasta} ({len(df_pasta_exibicao_atual)} atividades nesta exibição){multi_user_info}", expanded=True):
                if nome_pasta in pastas_com_multiplos_usuarios_set:
                     nomes_originais = df_atividades_raw[df_atividades_raw['activity_folder'] == nome_pasta]['user_profile_name'].unique()
                     st.caption(f"👥 Usuários na análise completa desta pasta: {', '.join(nomes_originais)}")

                for _, atividade in df_pasta_exibicao_atual.iterrows():
                    act_id = atividade['activity_id']
                    links = gerar_links_zflow(act_id)
                    
                    st.markdown("---")
                    cols = st.columns([0.6, 0.4]) # Coluna principal e coluna de ações/duplicatas
                    
                    with cols[0]: # Informações da Atividade
                        st.markdown(f"**ID:** `{act_id}` | **Data:** {atividade['activity_date'].strftime('%d/%m/%Y')} | **Status:** `{atividade['activity_status']}`")
                        st.markdown(f"**Usuário:** {atividade['user_profile_name']}")
                        
                        btn_cols = st.columns(3)
                        if btn_cols[0].button("👁️ Ver Texto", key=f"ver_texto_{act_id}", help="Abrir texto completo da publicação"):
                            st.session_state.atividade_dialog_texto = atividade.to_dict()
                            st.session_state.dialog_aberto = 'texto'
                            st.rerun() # Força o rerun para abrir o dialog

                        btn_cols[1].link_button("🔗 ZFlow v1", links['antigo'], help="Abrir no ZFlow (versão antiga)")
                        btn_cols[2].link_button("🔗 ZFlow v2", links['novo'], help="Abrir no ZFlow (versão nova)")

                    with cols[1]: # Potenciais Duplicatas
                        duplicatas_da_atividade = similaridades_globais.get(act_id, [])
                        if duplicatas_da_atividade:
                            st.markdown(f"**<span style='color:red;'>Potenciais Duplicatas:</span>** ({len(duplicatas_da_atividade)})", unsafe_allow_html=True)
                            for dup_info in duplicatas_da_atividade:
                                container_dup = st.container(border=True)
                                container_dup.markdown(
                                    f"<small><span style='background-color:{dup_info['cor']}; padding: 1px 3px; border-radius: 3px; color: black;'>"
                                    f"ID: {dup_info['id_similar']} ({dup_info['ratio']:.0%})</span><br>"
                                    f"Data: {dup_info['data_similar'].strftime('%d/%m')} | Status: `{dup_info['status_similar']}`<br>"
                                    f"Usuário: {dup_info['usuario_similar']}</small>",
                                    unsafe_allow_html=True
                                )
                                # Botão para comparar esta duplicata específica
                                if container_dup.button("⚖️ Comparar Textos", key=f"comparar_{act_id}_com_{dup_info['id_similar']}", help="Comparar textos lado a lado"):
                                    atividade_comparar_obj = df_atividades_raw[df_atividades_raw['activity_id'] == dup_info['id_similar']].iloc[0].to_dict()
                                    st.session_state.atividades_dialog_comparacao = {'base': atividade.to_dict(), 'comparar': atividade_comparar_obj}
                                    st.session_state.dialog_aberto = 'comparacao'
                                    st.rerun()
                        elif apenas_potenciais_duplicatas_cb:
                            pass # Não mostra nada se o filtro é para apenas duplicatas e não há
                        else:
                            st.markdown(f"<small style='color:green;'>Sem duplicatas (acima de {min_similaridade_display:.0%})</small>", unsafe_allow_html=True)
        
        # --- Lógica dos Dialogs ---
        if st.session_state.dialog_aberto == 'texto' and st.session_state.atividade_dialog_texto:
            with st.dialog("Texto Completo da Atividade",  dismissed=(lambda: setattr(st.session_state, 'dialog_aberto', None))):
                 mostrar_dialog_texto_completo(st.session_state.atividade_dialog_texto)
                 if st.button("Fechar Texto", key="fechar_dialog_texto"):
                     st.session_state.dialog_aberto = None
                     st.rerun()
        
        elif st.session_state.dialog_aberto == 'comparacao' and st.session_state.atividades_dialog_comparacao:
            atividades_para_comparar = st.session_state.atividades_dialog_comparacao
            with st.dialog("Comparação Detalhada de Textos", dismissed=(lambda: setattr(st.session_state, 'dialog_aberto', None))):
                mostrar_dialog_comparacao(atividades_para_comparar['base'], atividades_para_comparar['comparar'])
                if st.button("Fechar Comparação", key="fechar_dialog_comparacao"):
                    st.session_state.dialog_aberto = None
                    st.rerun()

else:
    st.error("Conexão com o banco falhou. Verifique as credenciais e o status do banco.")

st.sidebar.info("Verificador de Duplicidade v2")
