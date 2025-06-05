import streamlit as st
import pandas as pd
import re
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from datetime import datetime, timedelta, date
from unidecode import unidecode
from rapidfuzz import fuzz
import io
import html
import difflib
import os

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
SUFFIX_STATE = "_grifar_v1"
ITENS_POR_PAGINA = 20
st.set_page_config(layout="wide", page_title="Verificador de Duplicidade (Grifar Semelhanças)")

# ------------------------------------------------------------------------------
# CSS para o <mark>
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    mark { background-color:#ffeb3b; padding:0 2px; }
    pre  { white-space: pre-wrap; font-family: 'Courier New', monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================
def normalizar_texto(txt: str | None) -> str:
    if not txt or not isinstance(txt, str):
        return ""
    txt = unidecode(txt.lower())
    txt = re.sub(r"[^\w\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def calcular_similaridade(texto_a: str, texto_b: str) -> float:
    norm_a = normalizar_texto(texto_a)
    norm_b = normalizar_texto(texto_b)
    if not norm_a or not norm_b:
        return 0.0
    if abs(len(norm_a) - len(norm_b)) > 0.3 * max(len(norm_a), len(norm_b)):
        return 0.0
    return fuzz.token_set_ratio(norm_a, norm_b) / 100.0


def obter_cor_similaridade(ratio: float) -> str:
    LIMIAR_ALTA, LIMIAR_MEDIA = 0.90, 0.70
    CORES = {"alta": "#FF5252", "media": "#FFB74D", "baixa": "#FFD54F"}
    if ratio >= LIMIAR_ALTA:
        return CORES["alta"]
    if ratio >= LIMIAR_MEDIA:
        return CORES["media"]
    return CORES["baixa"]


def gerar_links_zflow(activity_id: int) -> dict:
    return {
        "antigo": f"https://zflow.zionbyonset.com.br/activity/3/details/{activity_id}",
        "novo": f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={activity_id}#/fixcol1",
    }


@st.cache_resource
def get_db_engine() -> Engine | None:
    """Obtém engine MySQL. Credenciais DEVEM estar em st.secrets ou em variáveis de ambiente."""
    db_host = (
        st.secrets.get("database", {}).get("host")
        or os.getenv("DB_HOST")
    )
    db_user = (
        st.secrets.get("database", {}).get("user")
        or os.getenv("DB_USER")
    )
    db_pass = (
        st.secrets.get("database", {}).get("password")
        or os.getenv("DB_PASS")
    )
    db_name = (
        st.secrets.get("database", {}).get("name")
        or os.getenv("DB_NAME")
    )

    if not all([db_host, db_user, db_pass, db_name]):
        st.error(
            "Credenciais do banco não encontradas. "
            "Configure em st.secrets ou nas variáveis de ambiente DB_HOST/DB_USER/DB_PASS/DB_NAME."
        )
        return None

    db_uri = f"mysql+mysqlconnector://{db_user}:{db_pass}@{db_host}/{db_name}"
    try:
        engine = create_engine(db_uri, pool_pre_ping=True, pool_recycle=3600)
        with engine.connect():
            pass
        return engine
    except exc.SQLAlchemyError as e:
        st.exception(e)
        return None


@st.cache_data(ttl=3600, hash_funcs={Engine: lambda _: None})
def carregar_dados_base(eng_param: Engine) -> tuple[pd.DataFrame | None, Exception | None]:
    hoje_dt = date.today()
    data_limite_historico = hoje_dt - timedelta(days=7)
    query_abertas = text(
        """
        SELECT activity_id, activity_folder, user_profile_name, activity_date,
               activity_status, Texto, activity_type
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type = 'Verificar' AND activity_status = 'Aberta'
        """
    )
    query_historico = text(
        """
        SELECT activity_id, activity_folder, user_profile_name, activity_date,
               activity_status, Texto, activity_type
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type = 'Verificar' AND DATE(activity_date) >= :data_limite
        """
    )
    try:
        with eng_param.connect() as connection:
            df_abertas = pd.read_sql(query_abertas, connection)
            df_historico = pd.read_sql(
                query_historico, connection, params={"data_limite": data_limite_historico}
            )
        df_combinado = pd.concat([df_abertas, df_historico], ignore_index=True)
        if df_combinado.empty:
            cols = [
                "activity_id",
                "activity_folder",
                "user_profile_name",
                "activity_date",
                "activity_status",
                "Texto",
                "activity_type",
            ]
            df_final = pd.DataFrame(columns=cols)
            df_final["activity_date"] = pd.Series(dtype="datetime64[ns]")
            df_final["Texto"] = pd.Series(dtype="object")
            return df_final, None

        df_combinado["activity_date"] = pd.to_datetime(
            df_combinado["activity_date"], errors="coerce"
        )
        df_combinado_sorted = df_combinado.sort_values(
            by=["activity_id", "activity_status"], ascending=[True, True]
        )
        df_final_temp = df_combinado_sorted.drop_duplicates(
            subset=["activity_id"], keep="first"
        )
        df_final = df_final_temp.sort_values(
            by=["activity_folder", "activity_date", "activity_id"],
            ascending=[True, False, False],
        ).copy()
        df_final["activity_date"] = pd.to_datetime(
            df_final["activity_date"], errors="coerce"
        )
        df_final["Texto"] = df_final["Texto"].astype(str).fillna("")
        return df_final, None
    except exc.SQLAlchemyError as e:
        return None, e


# ------------------------------------------------------------------------------
#  Similaridade – calcula APENAS quando necessário
# ------------------------------------------------------------------------------
def get_similarity_map(df: pd.DataFrame, min_sim: float):
    """Retorna (mapa_similaridades, ids_com_duplicatas) cacheado em session_state."""
    signature = (tuple(sorted(df["activity_id"])), float(min_sim))
    cache_key = "similarity_cache" + SUFFIX_STATE

    # Cache válido?
    if (
        cache_key in st.session_state
        and st.session_state[cache_key]["signature"] == signature
    ):
        return (
            st.session_state[cache_key]["map"],
            st.session_state[cache_key]["ids_dup"],
        )

    # ---------- Cálculo ----------
    ids_dup = set()
    similarity_map = {}

    for pasta, grp in df.groupby("activity_folder"):
        atividades = grp.to_dict("records")
        for i, base in enumerate(atividades):
            similarity_map.setdefault(base["activity_id"], [])
            for comparar in atividades[i + 1 :]:
                ratio = calcular_similaridade(base["Texto"], comparar["Texto"])
                if ratio >= min_sim:
                    cor = obter_cor_similaridade(ratio)
                    ids_dup |= {base["activity_id"], comparar["activity_id"]}
                    similarity_map[base["activity_id"]].append(
                        {
                            "id_similar": comparar["activity_id"],
                            "ratio": ratio,
                            "cor": cor,
                        }
                    )
                    similarity_map.setdefault(comparar["activity_id"], []).append(
                        {
                            "id_similar": base["activity_id"],
                            "ratio": ratio,
                            "cor": cor,
                        }
                    )

    for k in similarity_map:
        similarity_map[k].sort(key=lambda x: x["ratio"], reverse=True)

    st.session_state[cache_key] = {
        "signature": signature,
        "map": similarity_map,
        "ids_dup": ids_dup,
    }
    return similarity_map, ids_dup


# ------------------------------------------------------------------------------
#  Destaque das palavras em comum
# ------------------------------------------------------------------------------
def highlight_common_words(t1: str, t2: str, min_len: int = 3) -> tuple[str, str]:
    tokens1 = re.findall(r"\w+", unidecode(t1.lower()))
    tokens2 = re.findall(r"\w+", unidecode(t2.lower()))
    comuns = {w for w in tokens1 if w in tokens2 and len(w) >= min_len}

    def _wrap(txt):
        def repl(m):
            palavra = m.group(0)
            if unidecode(palavra.lower()) in comuns:
                return f"<mark>{html.escape(palavra)}</mark>"
            return html.escape(palavra)

        return "<pre>" + re.sub(r"\w+", repl, txt) + "</pre>"

    return _wrap(t1), _wrap(t2)


# ==============================================================================
# ESTADO DA SESSÃO (controles de diálogos e comparações)
# ==============================================================================
for key_base in [
    "show_texto_dialog",
    "atividade_para_texto_dialog",
    "comparacao_ativa",
    "pagina_atual",
]:
    full_key = f"{key_base}{SUFFIX_STATE}"
    if full_key not in st.session_state:
        st.session_state[full_key] = False if "show" in key_base else None

# ==============================================================================
# FUNÇÕES DE UI
# ==============================================================================
@st.dialog("Texto Completo da Atividade")
def mostrar_texto_completo_dialog():
    atividade_data = st.session_state[f"atividade_para_texto_dialog{SUFFIX_STATE}"]
    if atividade_data is None:
        return
    data_fmt = (
        atividade_data["activity_date"].strftime("%d/%m/%Y %H:%M")
        if pd.notna(atividade_data["activity_date"])
        else "N/A"
    )
    st.markdown(f"### ID: `{atividade_data['activity_id']}` | Data: {data_fmt}")
    st.text_area(
        "Texto:",
        value=str(atividade_data["Texto"]),
        height=400,
        disabled=True,
        key=f"txt_dialog{SUFFIX_STATE}_{atividade_data['activity_id']}",
    )
    if st.button("Fechar Texto", key=f"fechar_txt_dialog{SUFFIX_STATE}"):
        st.session_state[f"show_texto_dialog{SUFFIX_STATE}"] = False
        st.rerun()


def on_click_ver_texto_completo(atividade):
    st.session_state[f"atividade_para_texto_dialog{SUFFIX_STATE}"] = atividade
    st.session_state[f"show_texto_dialog{SUFFIX_STATE}"] = True


def on_click_comparar_textos(atividade_base, atividade_comparar):
    st.session_state[f"comparacao_ativa{SUFFIX_STATE}"] = {
        "base": atividade_base,
        "comparar": atividade_comparar,
    }


def fechar_comparacao_textos():
    st.session_state[f"comparacao_ativa{SUFFIX_STATE}"] = None


# ==============================================================================
# APLICAÇÃO PRINCIPAL
# ==============================================================================
def app():
    st.sidebar.success(f"Logado como: **{st.session_state['username']}**")
    if st.sidebar.button("Logout", key=f"logout_btn{SUFFIX_STATE}"):
        st.session_state.clear()
        st.rerun()

    st.title("🔎 Verificador de Duplicidade (Comparação Lado a Lado)")
    st.markdown("Análise de atividades 'Verificar' para identificar potenciais duplicidades.")

    eng = get_db_engine()
    if not eng:
        st.stop()

    # ------------------------------ Atualizar dados ---------------------------
    if st.sidebar.button("🔄 Atualizar Dados Base", key=f"buscar_btn_base{SUFFIX_STATE}"):
        carregar_dados_base.clear()
        st.session_state[f"comparacao_ativa{SUFFIX_STATE}"] = None

    df_raw_total, erro_db = carregar_dados_base(eng)
    if erro_db:
        st.error("Erro ao carregar dados.")
        st.exception(erro_db)
        st.stop()
    if df_raw_total is None or df_raw_total.empty:
        st.warning("Nenhuma atividade 'Verificar' retornada.")
        st.stop()

    # ------------------------------ Filtros de período ------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("1. Filtro de Período (Exibição)")
    hoje = date.today()
    data_inicio_padrao = hoje - timedelta(days=1)
    df_abertas_futuras = df_raw_total[
        (df_raw_total["activity_status"] == "Aberta")
        & (df_raw_total["activity_date"].notna())
        & (df_raw_total["activity_date"].dt.date > hoje)
    ]
    data_fim_padrao = (
        df_abertas_futuras["activity_date"].dt.date.max()
        if not df_abertas_futuras.empty
        else hoje + timedelta(days=14)
    )
    if data_inicio_padrao > data_fim_padrao:
        data_inicio_padrao = (
            data_fim_padrao - timedelta(days=1) if data_fim_padrao > hoje else hoje - timedelta(days=1)
        )

    data_inicio_selecionada = st.sidebar.date_input(
        "Data de Início (Exibição)",
        value=data_inicio_padrao,
        key=f"di_exib{SUFFIX_STATE}",
    )
    data_fim_selecionada = st.sidebar.date_input(
        "Data de Fim (Exibição)",
        value=data_fim_padrao,
        min_value=data_inicio_selecionada,
        key=f"df_exib{SUFFIX_STATE}",
    )
    if data_inicio_selecionada > data_fim_selecionada:
        st.sidebar.error("Data de início > data de fim.")
        st.stop()

    df_atividades_periodo_ui = df_raw_total[
        (df_raw_total["activity_date"].notna())
        & (df_raw_total["activity_date"].dt.date >= data_inicio_selecionada)
        & (df_raw_total["activity_date"].dt.date <= data_fim_selecionada)
    ]
    if df_atividades_periodo_ui.empty:
        st.info("Nenhuma atividade para o período de exibição.")
    else:
        st.success(f"**{len(df_atividades_periodo_ui)}** atividades no período de exibição.")

    # ------------------------------ Filtros de Análise ------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Filtros de Análise")
    pastas_disp = sorted(df_atividades_periodo_ui["activity_folder"].dropna().unique())
    pastas_sel = st.sidebar.multiselect(
        "Analisar Pasta(s):", pastas_disp, default=[], key=f"pasta_sel{SUFFIX_STATE}"
    )
    status_disp_analise = sorted(df_atividades_periodo_ui["activity_status"].dropna().unique())
    status_sel_analise = st.sidebar.multiselect(
        "Analisar Status:", status_disp_analise, default=[], key=f"status_sel{SUFFIX_STATE}"
    )
    df_para_analise = df_atividades_periodo_ui.copy()
    if pastas_sel:
        df_para_analise = df_para_analise[df_para_analise["activity_folder"].isin(pastas_sel)]
    if status_sel_analise:
        df_para_analise = df_para_analise[df_para_analise["activity_status"].isin(status_sel_analise)]

    # ------------------------------ Filtros de Exibição Final -----------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Filtros de Exibição Final")
    min_sim = (
        st.sidebar.slider("Similaridade ≥ que (%):", 0, 100, 70, 5, key=f"sim_slider{SUFFIX_STATE}") / 100.0
    )
    apenas_dup = st.sidebar.checkbox("Exibir apenas com duplicatas", value=True, key=f"dup_cb{SUFFIX_STATE}")
    pastas_multi_user = {
        nome for nome, grupo in df_para_analise.groupby("activity_folder") if grupo["user_profile_name"].nunique() > 1
    }
    apenas_multi = st.sidebar.checkbox(
        "Exibir pastas com múltiplos usuários", False, key=f"multi_cb{SUFFIX_STATE}"
    )
    usuarios_disp_ex = sorted(df_para_analise["user_profile_name"].dropna().unique())
    usuarios_sel = st.sidebar.multiselect(
        "Exibir Usuário(s):", usuarios_disp_ex, default=[], key=f"user_sel{SUFFIX_STATE}"
    )

    # ------------------------------ Similaridade (cache) ----------------------
    similarity_map, ids_com_duplicatas = get_similarity_map(df_para_analise, min_sim)
    map_id_para_similaridades = similarity_map

    # ------------------------------ Exibição -------------------------------
    df_exibir = df_para_analise.copy()
    if apenas_dup:
        df_exibir = df_exibir[df_exibir["activity_id"].isin(ids_com_duplicatas)]
    if apenas_multi:
        df_exibir = df_exibir[df_exibir["activity_folder"].isin(pastas_multi_user)]
    if usuarios_sel:
        df_exibir = df_exibir[df_exibir["user_profile_name"].isin(usuarios_sel)]

    # ------------------------------ Exportar --------------------------------
    st.sidebar.markdown("---")
    if st.sidebar.button("📥 Exportar para XLSX", key=f"export_btn{SUFFIX_STATE}"):
        if df_exibir.empty:
            st.sidebar.warning("Nenhum dado para exportar.")
        else:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_exibir.to_excel(writer, index=False, sheet_name="Atividades_Exibidas")
                lista_export_duplicatas = []
                for id_base_export, lista_similares_export in map_id_para_similaridades.items():
                    if id_base_export in df_exibir["activity_id"].values:
                        for sim_info_export in lista_similares_export:
                            detalhes_similar_export_rows = df_raw_total[
                                df_raw_total["activity_id"] == sim_info_export["id_similar"]
                            ]
                            if not detalhes_similar_export_rows.empty:
                                detalhes_similar_export = detalhes_similar_export_rows.iloc[0]
                                data_dup_exp_str = (
                                    detalhes_similar_export["activity_date"].strftime("%Y-%m-%d %H:%M")
                                    if pd.notna(detalhes_similar_export["activity_date"])
                                    else None
                                )
                                lista_export_duplicatas.append(
                                    {
                                        "ID_Base": id_base_export,
                                        "ID_Duplicata_Potencial": sim_info_export["id_similar"],
                                        "Percentual_Similaridade": sim_info_export["ratio"],
                                        "Cor_Similaridade": sim_info_export["cor"],
                                        "Data_Duplicata": data_dup_exp_str,
                                        "Usuario_Duplicata": detalhes_similar_export["user_profile_name"],
                                        "Status_Duplicata": detalhes_similar_export["activity_status"],
                                    }
                                )
                if lista_export_duplicatas:
                    pd.DataFrame(lista_export_duplicatas).to_excel(
                        writer, index=False, sheet_name="Detalhes_Duplicatas"
                    )
            st.sidebar.download_button(
                "Baixar XLSX",
                output.getvalue(),
                f"duplicatas_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # ------------------------------ Comparação Lado-a-Lado -------------------
    if st.session_state.get(f"comparacao_ativa{SUFFIX_STATE}"):
        dados_comp = st.session_state[f"comparacao_ativa{SUFFIX_STATE}"]
        base_c = dados_comp["base"]
        comparar_c = dados_comp["comparar"]

        with st.container(border=True):
            st.subheader(
                f"🔎 Comparação Detalhada: ID `{base_c['activity_id']}` vs `{comparar_c['activity_id']}`"
            )
            col1, col2 = st.columns(2)
            html_a, html_b = highlight_common_words(base_c["Texto"], comparar_c["Texto"])
            with col1:
                st.markdown(f"**ID {base_c['activity_id']}**", unsafe_allow_html=True)
                st.markdown(html_a, unsafe_allow_html=True)
            with col2:
                st.markdown(f"**ID {comparar_c['activity_id']}**", unsafe_allow_html=True)
                st.markdown(html_b, unsafe_allow_html=True)

            if st.button("Ocultar Comparação", key=f"fechar_comp{SUFFIX_STATE}"):
                fechar_comparacao_textos()
                st.rerun()
        st.markdown("---")

    # ------------------------------ Listagem por Pasta ------------------------
    st.header("Análise Detalhada por Pasta")
    if df_exibir.empty:
        st.info("Nenhuma atividade para os filtros selecionados.")

    pastas_ordenadas = sorted(df_exibir["activity_folder"].unique()) if not df_exibir.empty else []
    pagina_atual = st.session_state.get(f"pagina_atual{SUFFIX_STATE}", 0)

    if pastas_ordenadas:
        total_pastas_exibiveis = len(pastas_ordenadas)
        total_paginas = (total_pastas_exibiveis + ITENS_POR_PAGINA - 1) // ITENS_POR_PAGINA
        if total_paginas > 1:
            col_pag_1, col_pag_2, col_pag_3 = st.columns([1, 2, 1])
            with col_pag_1:
                if st.button(
                    "⬅️ Anterior",
                    key=f"prev_page{SUFFIX_STATE}",
                    disabled=(pagina_atual == 0),
                ):
                    st.session_state[f"pagina_atual{SUFFIX_STATE}"] -= 1
                    st.rerun()
            with col_pag_3:
                if st.button(
                    "Próxima ➡️",
                    key=f"next_page{SUFFIX_STATE}",
                    disabled=(pagina_atual >= total_paginas - 1),
                ):
                    st.session_state[f"pagina_atual{SUFFIX_STATE}"] += 1
                    st.rerun()
            with col_pag_2:
                st.markdown(
                    f"<p style='text-align: center;'>Página {pagina_atual + 1} de {total_paginas}</p>",
                    unsafe_allow_html=True,
                )

        inicio_idx = pagina_atual * ITENS_POR_PAGINA
        fim_idx = inicio_idx + ITENS_POR_PAGINA
        pastas_para_pagina_atual = pastas_ordenadas[inicio_idx:fim_idx]

        for nome_pasta in pastas_para_pagina_atual:
            df_pasta_exibicao = df_exibir[df_exibir["activity_folder"] == nome_pasta]
            if df_pasta_exibicao.empty:
                continue
            total_analisado_pasta = len(df_para_analise[df_para_analise["activity_folder"] == nome_pasta])
            titulo = f"📁 Pasta: {nome_pasta} ({len(df_pasta_exibicao)} exibidas / {total_analisado_pasta} analisadas)"

            with st.expander(titulo, expanded=len(df_pasta_exibicao) < 10):
                for _, atividade_row in df_pasta_exibicao.iterrows():
                    atividade = atividade_row.to_dict()
                    st.markdown("---")
                    col_info, col_sim_display = st.columns([0.6, 0.4])
                    with col_info:
                        data_at_str = (
                            atividade["activity_date"].strftime("%d/%m/%Y %H:%M")
                            if pd.notna(atividade["activity_date"])
                            else "N/A"
                        )
                        st.markdown(
                            f"**ID:** `{atividade['activity_id']}` | **Data:** {data_at_str} | **Status:** `{atividade['activity_status']}`"
                        )
                        st.markdown(f"**Usuário:** {atividade['user_profile_name']}")
                        st.text_area(
                            "Texto:",
                            str(atividade["Texto"]),
                            height=100,
                            key=f"texto_exp{SUFFIX_STATE}_{nome_pasta}_{atividade['activity_id']}",
                            disabled=True,
                        )
                        btn_cols = st.columns(3)
                        links = gerar_links_zflow(atividade["activity_id"])
                        btn_cols[0].button(
                            "👁️ Ver Completo",
                            key=f"ver_completo_btn{SUFFIX_STATE}_{atividade['activity_id']}",
                            on_click=on_click_ver_texto_completo,
                            args=(atividade,),
                        )
                        btn_cols[1].link_button("🔗 ZFlow v1", links["antigo"])
                        btn_cols[2].link_button("🔗 ZFlow v2", links["novo"])

                    with col_sim_display:
                        similares_para_esta_atividade = map_id_para_similaridades.get(
                            atividade["activity_id"], []
                        )
                        if similares_para_esta_atividade:
                            st.markdown(
                                f"**<span style='color:red;'>Duplicatas (Intra-Pasta):</span>** ({len(similares_para_esta_atividade)})",
                                unsafe_allow_html=True,
                            )
                            for sim_data in similares_para_esta_atividade:
                                info_dupe_rows = df_raw_total[
                                    df_raw_total["activity_id"] == sim_data["id_similar"]
                                ]
                                if not info_dupe_rows.empty:
                                    info_dupe = info_dupe_rows.iloc[0].to_dict()
                                    container_dup = st.container(border=True)
                                    data_dupe_str_disp = (
                                        info_dupe["activity_date"].strftime("%d/%m/%y %H:%M")
                                        if pd.notna(info_dupe["activity_date"])
                                        else "N/A"
                                    )
                                    container_dup.markdown(
                                        f"""<small><div style='background-color:{sim_data['cor']}; padding: 3px 6px; border-radius: 5px; color: black; margin-bottom: 5px; font-weight: 500;'>
                                        <b>ID: {info_dupe['activity_id']} ({sim_data['ratio']:.0%})</b><br>
                                        Data: {data_dupe_str_disp} | Status: {info_dupe['activity_status']}<br>
                                        Usuário: {info_dupe['user_profile_name']}
                                        </div></small>""",
                                        unsafe_allow_html=True,
                                    )
                                    container_dup.button(
                                        "⚖️ Comparar Textos",
                                        key=f"comp_direta_btn{SUFFIX_STATE}_{atividade['activity_id']}_{info_dupe['activity_id']}",
                                        on_click=on_click_comparar_textos,
                                        args=(atividade, info_dupe),
                                    )
                                else:
                                    st.caption(f"Detalhes da ID {sim_data['id_similar']} não disponíveis.")
                        else:
                            if not apenas_dup:
                                st.markdown(
                                    "**<span style='color:green;'>Sem duplicatas</span>**",
                                    unsafe_allow_html=True,
                                )

    # ------------------------------ Diálogos ----------------------------------
    if st.session_state.get(f"show_texto_dialog{SUFFIX_STATE}"):
        mostrar_texto_completo_dialog()


# ==============================================================================
# LOGIN
# ==============================================================================
def check_credentials(username, password):
    try:
        user_creds = st.secrets["credentials"]["usernames"]
        return username in user_creds and str(user_creds[username]) == password
    except KeyError:
        return False


def login_form():
    st.header("Login - Verificador de Duplicidade")
    with st.form(f"login_form{SUFFIX_STATE}_main"):
        username = st.text_input("Usuário", key=f"login_username{SUFFIX_STATE}_main")
        password = st.text_input(
            "Senha", key=f"login_password{SUFFIX_STATE}_main", type="password"
        )
        submitted = st.form_submit_button("Entrar")
        if submitted:
            if check_credentials(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Usuário ou senha inválidos.")
    st.info("Use as credenciais configuradas no secrets.toml.")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if st.session_state["logged_in"]:
        app()
    else:
        login_form()
