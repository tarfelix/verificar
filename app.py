import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from difflib import SequenceMatcher, HtmlDiff # SequenceMatcher e HtmlDiff
import io # Para exportação XLSX

# ==============================================================================
# CONFIGURAÇÕES E FUNÇÕES AUXILIARES
# ==============================================================================
st.set_page_config(layout="wide", page_title="Verificador de Duplicidade (Estável)")

def calcular_similaridade(texto_a, texto_b):
    if texto_a is None or texto_b is None:
        return 0.0
    return SequenceMatcher(None, str(texto_a), str(texto_b)).ratio()

def obter_cor_similaridade(ratio):
    if ratio >= 0.91: return "red"
    elif ratio >= 0.71: return "orange"
    elif ratio >= 0.50: return "gold"
    return "grey"

@st.cache_resource
def get_db_engine():
    db_user = "tarcisio"
    db_pass = "123qwe"
    db_host = "40.88.40.110"
    db_name = "zion_flow"
    if not all([db_user, db_pass, db_host, db_name]):
        st.error("Credenciais do banco não definidas.")
        return None
    db_uri = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_name}"
    try:
        engine_instance = create_engine(db_uri)
        with engine_instance.connect() as conn: conn.execute(text("SELECT 1"))
        return engine_instance
    except Exception as e:
        st.error(f"Erro ao conectar ao banco: {e}")
        return None

@st.cache_data(ttl=300)
def buscar_atividades_db(_engine, data_inicio, data_fim):
    if _engine is None: return pd.DataFrame()
    query = text("""
        SELECT activity_id, activity_folder, activity_subject, user_id, user_profile_name,
               activity_date, activity_fatal, activity_status, activity_type,
               activity_publish_date, Texto, observacoes, tags,
               activity_created_at, activity_updated_at
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type = :tipo_atividade
          AND activity_date BETWEEN :data_inicio AND :data_fim
        ORDER BY activity_folder, activity_date DESC, activity_id DESC
    """)
    try:
        with _engine.connect() as connection:
            df = pd.read_sql(query, connection, params={
                "tipo_atividade": "Verificar",
                "data_inicio": data_inicio, "data_fim": data_fim
            })
        if 'activity_date' in df.columns:
            df['activity_date'] = pd.to_datetime(df['activity_date']).dt.date
        if 'Texto' in df.columns:
            df['Texto'] = df['Texto'].astype(str).fillna('')
        return df
    except Exception as e:
        st.error(f"Erro ao buscar atividades: {e}")
        return pd.DataFrame()

def gerar_links_zflow(activity_id):
    link_antigo = f"https://zflow.zionbyonset.com.br/activity/3/details/{activity_id}"
    link_novo = f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={activity_id}#/fixcol1"
    return {"antigo": link_antigo, "novo": link_novo}

# ==============================================================================
# Estado da Sessão
# ==============================================================================
# Para o dialog de "Ver Texto Completo"
if 'show_texto_dialog_s2' not in st.session_state:
    st.session_state.show_texto_dialog_s2 = False
if 'atividade_para_texto_dialog_s2' not in st.session_state:
    st.session_state.atividade_para_texto_dialog_s2 = None

# Para a comparação de texto direto na página
if 'comparacao_ativa_s2' not in st.session_state:
    st.session_state.comparacao_ativa_s2 = None # Guarda {'base_id': ID, 'comparar_id': ID}
if 'dados_comparacao_s2' not in st.session_state:
    st.session_state.dados_comparacao_s2 = None # Guarda {'base_obj': {}, 'comparar_obj': {}}

# ==============================================================================
# Funções para Dialog de Texto
# ==============================================================================
def abrir_dialog_texto_s2(atividade):
    st.session_state.atividade_para_texto_dialog_s2 = atividade
    st.session_state.show_texto_dialog_s2 = True

def fechar_dialog_texto_s2():
    st.session_state.show_texto_dialog_s2 = False
    st.session_state.atividade_para_texto_dialog_s2 = None
    if 'comparacao_ativa_s2' in st.session_state: # Limpa comparação se um dialog de texto for aberto/fechado
        st.session_state.comparacao_ativa_s2 = None
        st.session_state.dados_comparacao_s2 = None


# ==============================================================================
# INTERFACE DO USUÁRIO
# ==============================================================================
st.title("🔎 Verificador de Duplicidade (Estável com Filtros)")
st.markdown("Análise de atividades 'Verificar' para identificar potenciais duplicidades.")

engine = get_db_engine()

if engine:
    st.sidebar.header("⚙️ Filtros e Opções")
    hoje = datetime.today().date()
    
    periodo_selecionado = st.sidebar.radio("Período:", ("Hoje, Ontem e Amanhã", "Intervalo Personalizado"), key="periodo_s2")
    data_inicio_f, data_fim_f = (hoje - timedelta(days=1), hoje + timedelta(days=1)) if periodo_selecionado == "Hoje, Ontem e Amanhã" else \
                                (st.sidebar.date_input("Data Início", hoje - timedelta(days=1), key="di_s2"), 
                                 st.sidebar.date_input("Data Fim", hoje + timedelta(days=1), key="df_s2"))

    if data_inicio_f > data_fim_f:
        st.sidebar.error("Data de início > data de fim.")
        st.stop()

    df_raw = buscar_atividades_db(engine, data_inicio_f, data_fim_f)

    if df_raw.empty:
        st.info(f"Nenhuma atividade 'Verificar' no período de {data_inicio_f.strftime('%d/%m/%Y')} a {data_fim_f.strftime('%d/%m/%Y')}.")
    else:
        st.success(f"{len(df_raw)} atividades 'Verificar' carregadas (período inicial).")

        st.sidebar.markdown("---")
        st.sidebar.subheader("Filtros de Análise:")
        pastas_disp_analise = sorted(df_raw['activity_folder'].dropna().unique())
        pastas_sel_analise = st.sidebar.multiselect("Analisar Pasta(s):", pastas_disp_analise, default=[], key="pasta_analise_s2")

        status_disp_analise = sorted(df_raw['activity_status'].dropna().unique())
        status_sel_analise = st.sidebar.multiselect("Analisar Status:", status_disp_analise, default=[], key="status_analise_s2")
        
        df_para_analise = df_raw.copy()
        if pastas_sel_analise:
            df_para_analise = df_para_analise[df_para_analise['activity_folder'].isin(pastas_sel_analise)]
        if status_sel_analise:
            df_para_analise = df_para_analise[df_para_analise['activity_status'].isin(status_sel_analise)]

        st.sidebar.markdown("---")
        st.sidebar.subheader("Filtros de Exibição:")
        usuarios_disp_exib = sorted(df_para_analise['user_profile_name'].dropna().unique()) if not df_para_analise.empty else []
        usuarios_sel_exib = st.sidebar.multiselect("Exibir Usuário(s):", usuarios_disp_exib, default=[], key="user_exib_s2")
        
        min_sim = st.sidebar.slider("Similaridade mínima (%):", 0, 100, 50, 5, key="sim_s2") / 100.0
        
        apenas_dup_cb = st.sidebar.checkbox("Exibir apenas com duplicatas", False, key="dup_cb_s2")
        
        pastas_multi_user_set = set()
        if not df_para_analise.empty:
            for nome_p, df_gp in df_para_analise.groupby('activity_folder'):
                if df_gp['user_profile_name'].nunique() > 1:
                    pastas_multi_user_set.add(nome_p)
        
        apenas_multi_user_cb = st.sidebar.checkbox("Exibir pastas com múltiplos usuários (na análise)", False, key="multiuser_cb_s2")

        ordem_p = st.sidebar.selectbox("Ordenar pastas:", ("Nome (A-Z)", "Mais Atividades"), key="ordem_p_s2")
        st.sidebar.markdown("---")

        similaridades_calc = {} 
        ids_com_dup = set()

        if not df_para_analise.empty:
            # Feedback para o usuário sobre a análise
            # num_a_analisar = len(df_para_analise)
            # st.sidebar.text(f"Analisando {num_a_analisar} atividades para similaridade...")
            # progress_bar = st.sidebar.progress(0)
            # count_processed = 0

            for _, df_pasta_calc in df_para_analise.groupby('activity_folder'):
                atividades_lc = df_pasta_calc.to_dict('records')
                for i in range(len(atividades_lc)):
                    base = atividades_lc[i]
                    if base['activity_id'] not in similaridades_calc: similaridades_calc[base['activity_id']] = []
                    for j in range(i + 1, len(atividades_lc)):
                        comparar = atividades_lc[j]
                        if base['activity_id'] == comparar['activity_id']: continue
                        
                        sim_val = calcular_similaridade(base.get('Texto'), comparar.get('Texto'))
                        if sim_val >= min_sim:
                            ids_com_dup.add(base['activity_id'])
                            ids_com_dup.add(comparar['activity_id'])
                            cor_val = obter_cor_similaridade(sim_val)
                            similaridades_calc[base['activity_id']].append({
                                'id_similar': comparar['activity_id'], 'ratio': sim_val, 'cor': cor_val, 
                                'data_similar': comparar['activity_date'], 'usuario_similar': comparar['user_profile_name'],
                                'status_similar': comparar['activity_status']
                            })
                            if comparar['activity_id'] not in similaridades_calc: similaridades_calc[comparar['activity_id']] = []
                            similaridades_calc[comparar['activity_id']].append({
                                'id_similar': base['activity_id'], 'ratio': sim_val, 'cor': cor_val,
                                'data_similar': base['activity_date'], 'usuario_similar': base['user_profile_name'],
                                'status_similar': base['activity_status']
                            })
                    # count_processed += 1
                    # progress_bar.progress(count_processed / num_a_analisar if num_a_analisar > 0 else 0)
            # progress_bar.empty() # Limpa a barra de progresso
            
            for k_sim in similaridades_calc:
                similaridades_calc[k_sim] = sorted(similaridades_calc[k_sim], key=lambda x: x['ratio'], reverse=True)
        
        df_exibir = df_para_analise.copy()
        if usuarios_sel_exib:
            df_exibir = df_exibir[df_exibir['user_profile_name'].isin(usuarios_sel_exib)]
        if apenas_dup_cb:
            df_exibir = df_exibir[df_exibir['activity_id'].isin(ids_com_dup)]
        if apenas_multi_user_cb:
            df_exibir = df_exibir[df_exibir['activity_folder'].isin(pastas_multi_user_set)]

        if st.sidebar.button("Exportar para XLSX", key="export_s2"):
            if not df_exibir.empty:
                output_buffer = io.BytesIO()
                with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer_xls:
                    df_exibir.to_excel(writer_xls, index=False, sheet_name='Atividades_Filtradas')
                    dup_export_l = []
                    ids_export_dup = df_exibir['activity_id'].unique()
                    for aid_export in ids_export_dup:
                        if aid_export in similaridades_calc:
                            for d_info_export in similaridades_calc[aid_export]:
                                dup_export_l.append({
                                    'ID_Base': aid_export, 'ID_Duplicata': d_info_export['id_similar'],
                                    'Similaridade': d_info_export['ratio'], 'Data_Dup': d_info_export['data_similar'],
                                    'Usuario_Dup': d_info_export['usuario_similar'], 'Status_Dup': d_info_export['status_similar']
                                })
                    if dup_export_l:
                        pd.DataFrame(dup_export_l).to_excel(writer_xls, index=False, sheet_name='Potenciais_Duplicatas')
                st.sidebar.download_button(label="Baixar XLSX", data=output_buffer.getvalue(),
                                           file_name=f"atividades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else: st.sidebar.warning("Nada para exportar.")

        lista_pastas_render = []
        if not df_exibir.empty:
            pastas_g_render = df_exibir.groupby('activity_folder')
            if ordem_p == "Nome da Pasta (A-Z)": lista_pastas_render = sorted(pastas_g_render.groups.keys())
            elif ordem_p == "Mais Atividades Primeiro": lista_pastas_render = pastas_g_render.size().sort_values(ascending=False).index.tolist()
        
        st.header("Resultados da Análise")

        # --- Seção de Comparação Direta (se ativa) ---
        if st.session_state.comparacao_ativa_s2 and st.session_state.dados_comparacao_s2:
            dados_comp = st.session_state.dados_comparacao_s2
            base_c = dados_comp['base_obj']
            comparar_c = dados_comp['comparar_obj']
            
            st.markdown("---")
            st.subheader(f"🔎 Comparação: ID `{base_c['activity_id']}` vs ID `{comparar_c['activity_id']}`")
            
            txt_base_c = str(base_c['Texto'])
            txt_comp_c = str(comparar_c['Texto'])
            
            differ = HtmlDiff(wrapcolumn=70) # Renomeado para evitar conflito
            html_comp = differ.make_table(txt_base_c.splitlines(), txt_comp_c.splitlines(),
                                         fromdesc=f"ID: {base_c['activity_id']}", todesc=f"ID: {comparar_c['activity_id']}")
            st.components.v1.html(html_comp, height=600, scrolling=True)
            if st.button("Fechar Comparação", key=f"fechar_comp_s2_{base_c['activity_id']}_{comparar_c['activity_id']}"):
                st.session_state.comparacao_ativa_s2 = None
                st.session_state.dados_comparacao_s2 = None
                st.rerun()
            st.markdown("---")


        if not lista_pastas_render and not df_exibir.empty: st.info("Nenhuma pasta para os filtros de exibição.")
        elif df_exibir.empty: st.info("Nenhuma atividade para os filtros aplicados.")

        for nome_pasta_r in lista_pastas_render:
            df_pasta_r = df_exibir[df_exibir['activity_folder'] == nome_pasta_r]
            multi_user_disp = " (Múltiplos Usuários na Análise)" if nome_pasta_r in pastas_multi_user_set else ""
            
            with st.expander(f"📁 Pasta: {nome_pasta_r} ({len(df_pasta_r)} exibidas){multi_user_disp}", expanded=True):
                if nome_pasta_r in pastas_multi_user_set:
                     nomes_orig_analise = df_para_analise[df_para_analise['activity_folder'] == nome_pasta_r]['user_profile_name'].unique()
                     st.caption(f"👥 Usuários (análise): {', '.join(nomes_orig_analise)}")

                for _, atividade_r_row in df_pasta_r.iterrows():
                    at_dict = atividade_r_row.to_dict()
                    at_id = at_dict['activity_id']
                    at_links = gerar_links_zflow(at_id)
                    
                    st.markdown("---")
                    main_cols = st.columns([0.6, 0.4])  
                    
                    with main_cols[0]:
                        st.markdown(f"**ID:** `{at_id}` | **Data:** {at_dict['activity_date'].strftime('%d/%m/%Y')} | **Status:** `{at_dict['activity_status']}`")
                        st.markdown(f"**Usuário:** {at_dict['user_profile_name']}")
                        st.text_area("Texto:", str(at_dict['Texto']), height=100, key=f"txt_s2_{at_id}", disabled=True)
                        
                        act_btns = st.columns(3)
                        if act_btns[0].button("👁️ Ver Completo", key=f"ver_txt_btn_s2_{at_id}", on_click=abrir_dialog_texto_s2, args=(at_dict,)):
                            pass 
                        act_btns[1].link_button("🔗 ZFlow v1", at_links['antigo'])
                        act_btns[2].link_button("🔗 ZFlow v2", at_links['novo'])

                    with main_cols[1]:
                        dups_at = similaridades_calc.get(at_id, [])
                        if dups_at:
                            st.markdown(f"**<span style='color:red;'>Duplicatas:</span>** ({len(dups_at)})", unsafe_allow_html=True)
                            for d_info in dups_at:
                                d_container = st.container(border=True)
                                d_container.markdown(
                                    f"<small><span style='background-color:{d_info['cor']}; padding:1px 3px; border-radius:3px; color:black;'>"
                                    f"ID: {d_info['id_similar']} ({d_info['ratio']:.0%})</span><br>"
                                    f"Data: {d_info['data_similar'].strftime('%d/%m')} | Status: `{d_info['status_similar']}`<br>"
                                    f"Usuário: {d_info['usuario_similar']}</small>", unsafe_allow_html=True)
                                
                                if d_info['id_similar'] in df_raw['activity_id'].values: # Garante que o obj completo existe
                                    at_comp_obj = df_raw[df_raw['activity_id'] == d_info['id_similar']].iloc[0].to_dict()
                                    if d_container.button("⚖️ Comparar", key=f"comp_btn_s2_{at_id}_{d_info['id_similar']}"):
                                        st.session_state.comparacao_ativa_s2 = {'base_id': at_id, 'comparar_id': d_info['id_similar']}
                                        st.session_state.dados_comparacao_s2 = {'base_obj': at_dict, 'comparar_obj': at_comp_obj}
                                        st.rerun()
                                else:
                                    d_container.caption(f"ID {d_info['id_similar']} não nos dados carregados.")
                        elif apenas_dup_cb: pass 
                        else: st.markdown(f"<small style='color:green;'>Sem duplicatas ({min_sim:.0%})</small>", unsafe_allow_html=True)

        # --- Dialog para "Ver Texto Completo" ---
        if st.session_state.show_texto_dialog_s2 and st.session_state.atividade_para_texto_dialog_s2:
            with st.dialog("Texto Completo da Atividade"): 
                at_dialog_data = st.session_state.atividade_para_texto_dialog_s2
                st.markdown(f"### ID: `{at_dialog_data['activity_id']}`")
                st.markdown(f"**Pasta:** {at_dialog_data['activity_folder']} | **Data:** {at_dialog_data['activity_date'].strftime('%d/%m/%Y')} | **Usuário:** {at_dialog_data['user_profile_name']} | **Status:** {at_dialog_data['activity_status']}")
                st.text_area("Texto:", str(at_dialog_data['Texto']), height=400, disabled=True, key=f"dialog_txt_s2_{at_dialog_data['activity_id']}")
                if st.button("Fechar", key="fechar_dialog_txt_s2", on_click=fechar_dialog_texto_s2): pass 
else:
    st.error("Conexão com o banco falhou.")

st.sidebar.info("Verificador v5 - Estável")
