import streamlit as st
import pandas as pd
import re
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine # Importar Engine para hash_funcs
from datetime import datetime, timedelta, date 
from unidecode import unidecode
from rapidfuzz import fuzz
import io
import difflib 

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================
st.set_page_config(layout="wide", page_title="Verificador de Duplicidade (Cache Fix)")

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
    try:
        db_host = st.secrets["database"]["host"]
        db_user = st.secrets["database"]["user"]
        db_pass = st.secrets["database"]["password"]
        db_name = st.secrets["database"]["name"]
    except KeyError:
        st.warning("Credenciais do banco não encontradas em st.secrets. Usando fallback local.")
        db_user, db_pass, db_host, db_name = "tarcisio", "123qwe", "40.88.40.110", "zion_flow"
        if not all([db_host, db_user, db_pass, db_name]):
             st.error("Credenciais do banco não definidas.")
             return None
    db_uri = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_name}"
    try:
        engine = create_engine(db_uri, pool_pre_ping=True, pool_recycle=3600)
        with engine.connect(): pass
        return engine
    except exc.SQLAlchemyError:
        return None 

# CORREÇÃO PRINCIPAL APLICADA AQUI no decorador @st.cache_data
@st.cache_data(ttl=3600, hash_funcs={Engine: lambda _: None}) 
def carregar_dados(_engine_param: Engine, data_inicio_req: date, data_fim_req: date) -> tuple[pd.DataFrame | None, Exception | None]:
    # O nome do parâmetro aqui é _engine_param para garantir que o underscore seja a primeira coisa
    # e para diferenciá-lo da variável 'eng' ou 'engine' no escopo de chamada.
    hoje_dt = date.today() 
    data_limite_historico = hoje_dt - timedelta(days=7)
    
    # Ajuste na query para buscar todas as abertas E o histórico dos últimos 7 dias
    query_abertas = text("""
        SELECT activity_id, activity_folder, user_profile_name, activity_date, activity_status, Texto, activity_type
        FROM ViewGrdAtividadesTarcisio WHERE activity_type = 'Verificar' AND activity_status = 'Aberta'
    """)
    query_historico = text("""
        SELECT activity_id, activity_folder, user_profile_name, activity_date, activity_status, Texto, activity_type
        FROM ViewGrdAtividadesTarcisio WHERE activity_type = 'Verificar' AND DATE(activity_date) >= :data_limite
    """)
    
    # As datas passadas (data_inicio_req, data_fim_req) agora são usadas para filtrar o resultado combinado,
    # e não diretamente nas queries de "todas abertas" ou "histórico 7 dias".
    # A função buscará uma base maior e depois filtraremos por data na função app().

    try:
        with _engine_param.connect() as connection: # Usar o parâmetro _engine_param
            df_abertas = pd.read_sql(query_abertas, connection)
            df_historico = pd.read_sql(query_historico, connection, params={"data_limite": data_limite_historico})
        
        df_combinado = pd.concat([df_abertas, df_historico], ignore_index=True)
        df_combinado.sort_values(by=['activity_id', 'activity_status'], ascending=[True, True], inplace=True) 
        df_final = df_combinado.drop_duplicates(subset=['activity_id'], keep='first')
        df_final.sort_values(by=['activity_folder', 'activity_date', 'activity_id'], ascending=[True, False, False], inplace=True)
        df_final['activity_date'] = pd.to_datetime(df_final['activity_date'])
        df_final['Texto'] = df_final['Texto'].astype(str).fillna('')
        return df_final, None
    except exc.SQLAlchemyError as e:
        return None, e

# ==============================================================================
# Estado da Sessão para Dialogs
# ==============================================================================
SUFFIX_DIALOG = "_dialog_hash_fix" 
if f'show_texto_dialog{SUFFIX_DIALOG}' not in st.session_state:
    st.session_state[f'show_texto_dialog{SUFFIX_DIALOG}'] = False
if f'atividade_para_texto_dialog{SUFFIX_DIALOG}' not in st.session_state:
    st.session_state[f'atividade_para_texto_dialog{SUFFIX_DIALOG}'] = None

if f'show_comparacao_dialog{SUFFIX_DIALOG}' not in st.session_state:
    st.session_state[f'show_comparacao_dialog{SUFFIX_DIALOG}'] = False
if f'atividades_para_comparacao{SUFFIX_DIALOG}' not in st.session_state:
    st.session_state[f'atividades_para_comparacao{SUFFIX_DIALOG}'] = None 

# ==============================================================================
# Funções Decoradas com @st.dialog
# ==============================================================================
@st.dialog("Texto Completo da Atividade")
def mostrar_texto_completo_dialog():
    atividade_data = st.session_state[f'atividade_para_texto_dialog{SUFFIX_DIALOG}']
    if atividade_data:
        st.markdown(f"### Texto Completo - ID: `{atividade_data['activity_id']}`")
        st.markdown(f"**Pasta:** {atividade_data['activity_folder']} | **Data:** {atividade_data['activity_date'].strftime('%d/%m/%Y %H:%M')} | **Usuário:** {atividade_data['user_profile_name']} | **Status:** {atividade_data['activity_status']}")
        st.text_area("Texto:", value=str(atividade_data['Texto']), height=400, disabled=True, key=f"dialog_txt_content{SUFFIX_DIALOG}_{atividade_data['activity_id']}")
        if st.button("Fechar Texto", key=f"fechar_dialog_txt_btn{SUFFIX_DIALOG}"):
            st.session_state[f'show_texto_dialog{SUFFIX_DIALOG}'] = False; st.rerun()

@st.dialog("Comparação Detalhada de Textos", width="large")
def mostrar_comparacao_html_diff_dialog():
    atividades_comp_data = st.session_state[f'atividades_para_comparacao{SUFFIX_DIALOG}']
    if atividades_comp_data:
        base_comp = atividades_comp_data['base']
        comparar_comp = atividades_comp_data['comparar']
        st.markdown(f"### Comparando ID `{base_comp['activity_id']}` com ID `{comparar_comp['activity_id']}`")
        texto_base_comp = str(base_comp['Texto'])
        texto_comparar_comp = str(comparar_comp['Texto'])
        differ = difflib.HtmlDiff(wrapcolumn=65, linejunk=difflib.IS_LINE_JUNK, charjunk=difflib.IS_CHARACTER_JUNK) 
        html_comparison = differ.make_table(texto_base_comp.splitlines(), texto_comparar_comp.splitlines(),
                                             fromdesc=f"ID: {base_comp['activity_id']}", 
                                             todesc=f"ID: {comparar_comp['activity_id']}")
        st.components.v1.html(html_comparison, height=550, scrolling=True)
        if st.button("Fechar Comparação", key=f"fechar_dialog_html_comp_btn{SUFFIX_DIALOG}"):
            st.session_state[f'show_comparacao_dialog{SUFFIX_DIALOG}'] = False; st.rerun()

# ==============================================================================
# Funções para abrir os dialogs
# ==============================================================================
def on_click_ver_texto_completo(atividade):
    st.session_state[f'atividade_para_texto_dialog{SUFFIX_DIALOG}'] = atividade
    st.session_state[f'show_texto_dialog{SUFFIX_DIALOG}'] = True

def on_click_comparar_textos_html_dialog(atividade_base, atividade_comparar):
    st.session_state[f'atividades_para_comparacao{SUFFIX_DIALOG}'] = {'base': atividade_base, 'comparar': atividade_comparar}
    st.session_state[f'show_comparacao_dialog{SUFFIX_DIALOG}'] = True

# ==============================================================================
# INTERFACE PRINCIPAL DO APP (RENOMEADA PARA app)
# ==============================================================================
def app(): # Renomeado de app_principal para app
    st.sidebar.success(f"Logado como: **{st.session_state['username']}**")
    if st.sidebar.button("Logout", key=f"logout_btn{SUFFIX_DIALOG}"): # Chave única
        for key_state in list(st.session_state.keys()): del st.session_state[key_state]
        st.rerun()

    st.title("🔎 Verificador de Duplicidade (HtmlDiff no Dialog)")
    st.markdown("Análise de atividades 'Verificar' para identificar potenciais duplicidades.")

    eng = get_db_engine() # Variável eng como no traceback
    if not eng: st.error("Falha crítica na conexão com o banco."); st.stop()

    st.sidebar.header("⚙️ Filtros e Opções")
    
    # Botão para atualizar os dados (chama a função com as datas atuais da sidebar)
    # As datas para a query inicial de carregar_dados são fixas (últimos 7 dias + abertas)
    # O filtro de data na sidebar apenas filtra o DataFrame em memória.
    
    if st.sidebar.button("🔄 Atualizar Dados Base", help="Busca os dados mais recentes (Abertas + Histórico 7 dias).", key=f"buscar_btn_base{SUFFIX_DIALOG}"):
        # Para carregar_dados, não precisamos passar datas, pois a lógica de quais dados buscar está dentro dela.
        # No entanto, o cache do Streamlit é baseado nos argumentos.
        # Para forçar o recarregamento de carregar_dados, precisamos limpar seu cache.
        # A função carregar_dados não usa data_inicio/fim como argumento para a query BASE.
        # O _engine é o único argumento que influencia o cache de carregar_dados.
        # A limpeza será feita pela mudança da "chave" de cache se for necessário,
        # ou explicitamente se forçar um refresh para os mesmos parâmetros (o que não é o caso aqui).
        carregar_dados.clear() # Limpa o cache de carregar_dados
        st.toast("Buscando dados atualizados...", icon="🔄")
        # st.rerun() # O Streamlit fará rerun ao limpar o cache de uma função usada

    # A função carregar_dados não precisa mais das datas da sidebar para sua query interna,
    # mas o cache do Streamlit ainda depende dos argumentos.
    # Para simplificar e garantir que o cache funcione, vamos passar datas fixas simbólicas
    # ou nenhuma, e deixar a lógica de datas dentro de carregar_dados.
    # No entanto, a assinatura atual de carregar_dados espera datas.
    # Vamos ajustar: carregar_dados não precisará mais de data_inicio_req e data_fim_req
    # se sua lógica interna é fixa.
    # --- AJUSTE ---
    # A função carregar_dados agora não depende de data_inicio_req e data_fim_req para sua query principal.
    # Esses parâmetros são removidos da chamada e da definição da função.
    # O cache @st.cache_data(ttl=3600) só dependerá do _engine.
    
    # Versão anterior de carregar_dados (buscar_dados_do_banco) tinha data_inicio_req, data_fim_req
    # A nova carregar_dados (ajustada para "Abertas + Histórico 7 dias") não precisaria delas.
    # Vou manter a estrutura de passar datas para carregar_dados, mas a query interna é que define o escopo.
    # Para o cache de `carregar_dados`, passaremos as datas que definem o "escopo máximo" que a função busca.
    # Neste caso, é um pouco redundante já que a query é fixa, mas mantém a assinatura.
    
    hoje_ref = date.today()
    data_inicio_carga_cache = hoje_ref - timedelta(days=7) # Limite inferior para o cache
    data_fim_carga_cache = hoje_ref + timedelta(days=365) # Limite superior amplo para abertas futuras
                                                       # Este é apenas para a chave do cache.
                                                       # A query em si tem sua própria lógica de data.

    df_raw_total, erro_db = carregar_dados(eng, data_inicio_carga_cache, data_fim_carga_cache)


    if erro_db: st.error("Erro ao carregar dados."); st.exception(erro_db); st.stop()
    if df_raw_total is None or df_raw_total.empty: st.warning("Nenhuma atividade 'Verificar' retornada."); st.stop()

    st.sidebar.markdown("---"); st.sidebar.subheader("1. Filtro de Período (Exibição)")
    
    # Define as datas padrão para os seletores da sidebar com base nos dados carregados
    data_inicio_padrao = hoje_ref - timedelta(days=1)
    datas_abertas_futuras = df_raw_total[(df_raw_total['activity_status'] == 'Aberta') & (df_raw_total['activity_date'].dt.date > hoje_ref)]['activity_date'].dt.date
    data_fim_padrao = datas_abertas_futuras.max() if not datas_abertas_futuras.empty else hoje_ref + timedelta(days=14)
    if data_inicio_padrao > data_fim_padrao: data_inicio_padrao = data_fim_padrao - timedelta(days=1)

    data_inicio_selecionada = st.sidebar.date_input("Data de Início (Exibição)", value=data_inicio_padrao, key=f"di_exib{SUFFIX_DIALOG}")
    data_fim_selecionada = st.sidebar.date_input("Data de Fim (Exibição)", value=data_fim_padrao, key=f"df_exib{SUFFIX_DIALOG}")

    if data_inicio_selecionada > data_fim_selecionada: st.sidebar.error("Data de início > data de fim."); st.stop()
    
    # Filtro de data para exibição (opera sobre df_raw_total)
    mask_data = (df_raw_total['activity_date'].dt.date >= data_inicio_selecionada) & (df_raw_total['activity_date'].dt.date <= data_fim_selecionada)
    df_atividades_periodo_exibicao = df_raw_total[mask_data] # Este é o DataFrame para exibir e analisar

    if df_atividades_periodo_exibicao.empty: 
        st.info(f"Nenhuma atividade para o período de exibição de {data_inicio_selecionada.strftime('%d/%m/%Y')} a {data_fim_selecionada.strftime('%d/%m/%Y')}.")
    else: 
        st.success(f"**{len(df_atividades_periodo_exibicao)}** atividades no período de exibição (de {len(df_raw_total)} total carregado na base).")
    
    # Filtros de Análise e Exibição operam sobre df_atividades_periodo_exibicao
    st.sidebar.markdown("---"); st.sidebar.subheader("2. Filtros de Análise (sobre o período de exibição)")
    pastas_disp = sorted(df_atividades_periodo_exibicao['activity_folder'].dropna().unique()) if not df_atividades_periodo_exibicao.empty else []
    pastas_sel = st.sidebar.multiselect("Analisar Pasta(s):", pastas_disp, default=[], key=f"pasta_sel{SUFFIX_DIALOG}")
    status_disp_analise = sorted(df_atividades_periodo_exibicao['activity_status'].dropna().unique()) if not df_atividades_periodo_exibicao.empty else []
    status_sel_analise = st.sidebar.multiselect("Analisar Status:", status_disp_analise, default=[], key=f"status_sel{SUFFIX_DIALOG}")

    df_para_analise = df_atividades_periodo_exibicao.copy()
    if pastas_sel: df_para_analise = df_para_analise[df_para_analise['activity_folder'].isin(pastas_sel)]
    if status_sel_analise: df_para_analise = df_para_analise[df_para_analise['activity_status'].isin(status_sel_analise)]
    
    st.sidebar.markdown("---"); st.sidebar.subheader("3. Filtros de Exibição Final")
    min_sim = st.sidebar.slider("Similaridade ≥ que (%):", 0, 100, 70, 5, key=f"sim_slider{SUFFIX_DIALOG}") / 100.0
    apenas_dup = st.sidebar.checkbox("Exibir apenas com duplicatas", value=True, key=f"dup_cb{SUFFIX_DIALOG}")
    pastas_multi_user = {nome for nome, grupo in df_para_analise.groupby('activity_folder') if grupo['user_profile_name'].nunique() > 1} if not df_para_analise.empty else set()
    apenas_multi = st.sidebar.checkbox("Exibir pastas com múltiplos usuários", False, key=f"multi_cb{SUFFIX_DIALOG}")
    usuarios_disp_ex = sorted(df_para_analise['user_profile_name'].dropna().unique()) if not df_para_analise.empty else []
    usuarios_sel = st.sidebar.multiselect("Exibir Usuário(s):", usuarios_disp_ex, default=[], key=f"user_sel{SUFFIX_DIALOG}")

    ids_com_duplicatas = set()
    map_id_para_similaridades = {} # Usar map para melhor organização

    if not df_para_analise.empty and len(df_para_analise) > 1:
        prog_placeholder = st.sidebar.empty(); prog_bar = st.sidebar.progress(0)
        total_pastas_analise = df_para_analise['activity_folder'].nunique(); pastas_processadas_analise = 0
        
        for nome_pasta_calc, df_pasta_calc in df_para_analise.groupby('activity_folder'):
            prog_placeholder.text(f"Analisando Pasta: {nome_pasta_calc}...")
            if len(df_pasta_calc) < 2:
                pastas_processadas_analise += 1
                if total_pastas_analise > 0: prog_bar.progress(pastas_processadas_analise / total_pastas_analise)
                continue
            atividades_nesta_pasta = df_pasta_calc.to_dict('records')
            for i in range(len(atividades_nesta_pasta)):
                base = atividades_nesta_pasta[i]
                if base['activity_id'] not in map_id_para_similaridades: map_id_para_similaridades[base['activity_id']] = []
                for j in range(i + 1, len(atividades_nesta_pasta)):
                    comparar = atividades_nesta_pasta[j]
                    similaridade = calcular_similaridade(base['Texto'], comparar['Texto'])
                    if similaridade >= min_sim:
                        ids_com_duplicatas.add(base['activity_id']); ids_com_duplicatas.add(comparar['activity_id'])
                        cor = obter_cor_similaridade(similaridade)
                        map_id_para_similaridades[base['activity_id']].append({'id_similar': comparar['activity_id'], 'ratio': similaridade, 'cor': cor})
                        if comparar['activity_id'] not in map_id_para_similaridades: map_id_para_similaridades[comparar['activity_id']] = []
                        map_id_para_similaridades[comparar['activity_id']].append({'id_similar': base['activity_id'], 'ratio': similaridade, 'cor': cor})
            pastas_processadas_analise += 1
            if total_pastas_analise > 0: prog_bar.progress(pastas_processadas_analise / total_pastas_analise)
        prog_bar.empty(); prog_placeholder.text("Análise de similaridade concluída.")
        for act_id_sort_map in map_id_para_similaridades:
            map_id_para_similaridades[act_id_sort_map] = sorted(map_id_para_similaridades[act_id_sort_map], key=lambda x: x['ratio'], reverse=True)

    df_exibir = df_para_analise.copy() # Começa com os dados filtrados por pasta/status de análise
    if apenas_dup: df_exibir = df_exibir[df_exibir['activity_id'].isin(ids_com_duplicatas)]
    if apenas_multi: df_exibir = df_exibir[df_exibir['activity_folder'].isin(pastas_multi_user)]
    if usuarios_sel: df_exibir = df_exibir[df_exibir['user_profile_name'].isin(usuarios_sel)]

    st.sidebar.markdown("---")
    if st.sidebar.button("📥 Exportar para XLSX", key=f"export_btn{SUFFIX_DIALOG}"):
        if not df_exibir.empty:
            # ... (Lógica de exportação como antes, usando df_exibir e map_id_para_similaridades) ...
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_exibir.to_excel(writer, index=False, sheet_name='Atividades_Exibidas')
                lista_export_duplicatas = []
                if map_id_para_similaridades:
                    for id_base_export, lista_similares_export in map_id_para_similaridades.items():
                        if id_base_export in df_exibir['activity_id'].values: 
                            for sim_info_export in lista_similares_export:
                                detalhes_similar_export_rows = df_atividades_periodo[df_atividades_periodo['activity_id'] == sim_info_export['id_similar']]
                                if not detalhes_similar_export_rows.empty:
                                    detalhes_similar_export = detalhes_similar_export_rows.iloc[0]
                                    lista_export_duplicatas.append({
                                        'ID_Base': id_base_export,
                                        'ID_Duplicata_Potencial': sim_info_export['id_similar'],
                                        'Percentual_Similaridade': sim_info_export['ratio'],
                                        'Cor_Similaridade': sim_info_export['cor'], # Adicionado
                                        'Data_Duplicata': detalhes_similar_export['activity_date'].strftime('%Y-%m-%d %H:%M'),
                                        'Usuario_Duplicata': detalhes_similar_export['user_profile_name'],
                                        'Status_Duplicata': detalhes_similar_export['activity_status']
                                    })
                    if lista_export_duplicatas:
                        pd.DataFrame(lista_export_duplicatas).to_excel(writer, index=False, sheet_name='Detalhes_Duplicatas')
            st.sidebar.download_button("Baixar XLSX", output.getvalue(), f"duplicatas_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else: st.sidebar.warning("Nenhum dado para exportar.")


    st.header("Análise Detalhada por Pasta")
    if df_exibir.empty: st.info("Nenhuma atividade para os filtros selecionados.")
    
    pastas_ordenadas = sorted(df_exibir['activity_folder'].unique()) if not df_exibir.empty else []

    for nome_pasta in pastas_ordenadas:
        df_pasta_exibicao = df_exibir[df_exibir['activity_folder'] == nome_pasta]
        if df_pasta_exibicao.empty: continue
        total_analisado_pasta = len(df_para_analise[df_para_analise['activity_folder'] == nome_pasta])
        titulo = f"📁 Pasta: {nome_pasta} ({len(df_pasta_exibicao)} exibidas / {total_analisado_pasta} analisadas)"
        
        with st.expander(titulo, expanded=len(df_pasta_exibicao) < 10): # Expandir se poucas atividades
            for _, atividade_row in df_pasta_exibicao.iterrows():
                atividade = atividade_row.to_dict()
                st.markdown("---")
                col_info, col_sim_display = st.columns([0.6, 0.4])
                with col_info:
                    st.markdown(f"**ID:** `{atividade['activity_id']}` | **Data:** {atividade['activity_date'].strftime('%d/%m/%Y %H:%M')} | **Status:** `{atividade['activity_status']}`")
                    st.markdown(f"**Usuário:** {atividade['user_profile_name']}")
                    st.text_area("Texto:", str(atividade['Texto']), height=100, key=f"texto_exp{SUFFIX_DIALOG}_{nome_pasta}_{atividade['activity_id']}", disabled=True)
                    btn_cols = st.columns(3)
                    links = gerar_links_zflow(atividade['activity_id'])
                    if btn_cols[0].button("👁️ Ver Completo", key=f"ver_completo_btn{SUFFIX_DIALOG}_{atividade['activity_id']}", on_click=on_click_ver_texto_completo, args=(atividade,)): pass
                    btn_cols[1].link_button("🔗 ZFlow v1", links['antigo'])
                    btn_cols[2].link_button("🔗 ZFlow v2", links['novo'])

                with col_sim_display:
                    similares_para_esta_atividade = map_id_para_similaridades.get(atividade['activity_id'], [])
                    if similares_para_esta_atividade:
                        st.markdown(f"**<span style='color:red;'>Duplicatas (Intra-Pasta):</span>** ({len(similares_para_esta_atividade)})", unsafe_allow_html=True)
                        for sim_data in similares_para_esta_atividade:
                            # Usar df_atividades_periodo para buscar detalhes, pois contém todos os dados do período carregado para a UI
                            info_dupe_rows = df_atividades_periodo[df_atividades_periodo['activity_id'] == sim_data['id_similar']]
                            if not info_dupe_rows.empty:
                                info_dupe = info_dupe_rows.iloc[0].to_dict()
                                container_dup = st.container(border=True)
                                container_dup.markdown(f"""<small><div style='background-color:{sim_data['cor']}; padding: 3px 6px; border-radius: 5px; color: black; margin-bottom: 5px; font-weight: 500;'>
                                <b>ID: {info_dupe['activity_id']} ({sim_data['ratio']:.0%})</b><br>
                                Data: {info_dupe['activity_date'].strftime('%d/%m/%y %H:%M')} | Status: {info_dupe['activity_status']}<br>
                                Usuário: {info_dupe['user_profile_name']}
                                </div></small>""", unsafe_allow_html=True)
                                if container_dup.button("⚖️ Comparar (Detalhado)", key=f"comp_html_dialog_btn{SUFFIX_DIALOG}_{atividade['activity_id']}_{info_dupe['activity_id']}", on_click=on_click_comparar_textos_html_dialog, args=(atividade, info_dupe)): pass
                            else: st.caption(f"Detalhes da ID {sim_data['id_similar']} não disponíveis.")
                    else:
                        if not apenas_dup: st.markdown("**<span style='color:green;'>Sem duplicatas (nesta análise)</span>**", unsafe_allow_html=True)

    if st.session_state.get(f'show_texto_dialog{SUFFIX_DIALOG}', False):
        mostrar_texto_completo_dialog()
    if st.session_state.get(f'show_comparacao_dialog{SUFFIX_DIALOG}', False):
        mostrar_comparacao_html_diff_dialog()

# ==============================================================================
# LÓGICA DE LOGIN
# ==============================================================================
def check_credentials(username, password):
    try:
        user_creds = st.secrets["credentials"]["usernames"]
        if username in user_creds and str(user_creds[username]) == password: return True
    except KeyError: return False
    except Exception: return False
    return False

def login_form():
    st.header("Login - Verificador de Duplicidade")
    with st.form(f"login_form{SUFFIX_DIALOG}_main"): # Chave única
        username = st.text_input("Usuário", key=f"login_username{SUFFIX_DIALOG}_main")
        password = st.text_input("Senha", key=f"login_password{SUFFIX_DIALOG}_main", type="password")
        submitted = st.form_submit_button("Entrar")
        if submitted:
            if check_credentials(username, password):
                st.session_state["logged_in"] = True; st.session_state["username"] = username; st.rerun()
            else: st.error("Usuário ou senha inválidos.")
    st.info("Use as credenciais do secrets.toml.")

if __name__ == "__main__":
    # Renomeado de app_principal para app, conforme traceback do usuário
    # A lógica de login será chamada primeiro
    if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
    
    if st.session_state["logged_in"]:
        app() # Chamando a função principal do app renomeada
    else:
        login_form()
