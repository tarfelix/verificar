import streamlit as st
import pandas as pd
import re
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from datetime import datetime, timedelta, date
from unidecode import unidecode
from rapidfuzz import fuzz
import io
from difflib import HtmlDiff

# ==============================================================================
# CONFIGURAÇÃO GERAL
# ==============================================================================
st.set_page_config(page_title="Verificador de Duplicidade", layout="wide")

# Dialogs ocupam 90 % da viewport para terem espaço
st.markdown("""
<style>
[data-testid="stDialogContainer"]  {
    width: 90vw !important;
    max-width: 90vw !important;
}
</style>
""", unsafe_allow_html=True)

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
    if abs(len(norm_a) - len(norm_b)) > 0.3 * max(len(norm_a), len(norm_b)):
        return 0.0
    return fuzz.token_set_ratio(norm_a, norm_b) / 100.0


def obter_cor_similaridade(ratio: float) -> str:
    LIMIAR_ALTA, LIMIAR_MEDIA = 0.90, 0.70
    cores = {'alta': '#FF5252', 'media': '#FFB74D', 'baixa': '#FFD54F'}
    if ratio >= LIMIAR_ALTA:
        return cores['alta']
    if ratio >= LIMIAR_MEDIA:
        return cores['media']
    return cores['baixa']


def gerar_links_zflow(activity_id: int) -> dict:
    return {
        "antigo": f"https://zflow.zionbyonset.com.br/activity/3/details/{activity_id}",
        "novo": f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={activity_id}#/fixcol1"
    }

# ------------------------------------------------------------------------------
# GERAÇÃO E CACHE DO DIFF HTML
# ------------------------------------------------------------------------------
def _gerar_diff_html(txt_a: str, txt_b: str, desc_a: str, desc_b: str) -> str:
    diff = HtmlDiff(wrapcolumn=80, tabsize=2)
    return diff.make_file(
        txt_a.splitlines(), txt_b.splitlines(),
        fromdesc=desc_a, todesc=desc_b,
        context=True, numlines=2
    )


@st.cache_data(show_spinner=False)
def diff_cache(id_a: int, id_b: int, txt_a: str, txt_b: str) -> str:
    # chave ordenada ⇒ evita duplicar cache A-B / B-A
    if id_a > id_b:
        id_a, id_b, txt_a, txt_b = id_b, id_a, txt_b, txt_a
    return _gerar_diff_html(txt_a, txt_b, f"ID {id_a}", f"ID {id_b}")

# ------------------------------------------------------------------------------
# CÁLCULO DE DUPLICIDADES (CACHE 15 min)
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=900)
def obter_duplicidades(df: pd.DataFrame, min_sim: float):
    """
    Devolve (map_id_para_similaridades, set_atividades_com_duplicata)
    Resultado é cacheado – nada de recalcular a cada clique.
    """
    mapa = {}
    duplicadas = set()

    for _, df_pasta in df.groupby('activity_folder'):
        if len(df_pasta) < 2:
            continue
        registros = df_pasta.to_dict('records')
        for i, base in enumerate(registros):
            lista_base = mapa.setdefault(base['activity_id'], [])
            for comparar in registros[i + 1:]:
                sim = calcular_similaridade(base['Texto'], comparar['Texto'])
                if sim < min_sim:
                    continue
                cor = obter_cor_similaridade(sim)
                duplicadas |= {base['activity_id'], comparar['activity_id']}
                lista_base.append({'id_similar': comparar['activity_id'], 'ratio': sim, 'cor': cor})
                lista_comp = mapa.setdefault(comparar['activity_id'], [])
                lista_comp.append({'id_similar': base['activity_id'], 'ratio': sim, 'cor': cor})

    for lst in mapa.values():
        lst.sort(key=lambda x: x['ratio'], reverse=True)

    return mapa, duplicadas

# ------------------------------------------------------------------------------
# CONEXÃO COM BANCO
# ------------------------------------------------------------------------------
@st.cache_resource
def get_db_engine() -> Engine | None:
    try:
        cfg = st.secrets["database"]
        db_uri = f"mysql+mysqlconnector://{cfg['user']}:{cfg['password']}@{cfg['host']}/{cfg['name']}"
    except KeyError:
        st.warning("Credenciais não encontradas em st.secrets – usando fallback local.")
        db_uri = "mysql+mysqlconnector://tarcisio:123qwe@40.88.40.110/zion_flow"
    try:
        engine = create_engine(db_uri, pool_pre_ping=True, pool_recycle=3600)
        with engine.connect():
            pass
        return engine
    except exc.SQLAlchemyError:
        return None


@st.cache_data(ttl=3600)
def carregar_dados(engine: Engine):
    hoje = date.today()
    limite_hist = hoje - timedelta(days=7)
    sql_abertas = """
        SELECT activity_id, activity_folder, user_profile_name,
               activity_date, activity_status, Texto, activity_type
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type='Verificar' AND activity_status='Aberta'
    """
    sql_hist = """
        SELECT activity_id, activity_folder, user_profile_name,
               activity_date, activity_status, Texto, activity_type
        FROM ViewGrdAtividadesTarcisio
        WHERE activity_type='Verificar' AND DATE(activity_date)>=:lim
    """
    try:
        with engine.connect() as con:
            df_ab = pd.read_sql(text(sql_abertas), con)
            df_hi = pd.read_sql(text(sql_hist), con, params={'lim': limite_hist})
        df = pd.concat([df_ab, df_hi], ignore_index=True).drop_duplicates('activity_id')
        df.sort_values(['activity_folder', 'activity_date', 'activity_id'],
                       ascending=[True, False, False], inplace=True)
        df['activity_date'] = pd.to_datetime(df['activity_date'])
        df['Texto'] = df['Texto'].astype(str).fillna('')
        return df, None
    except exc.SQLAlchemyError as e:
        return None, e

# ==============================================================================
# ESTADO DE DIÁLOGOS
# ==============================================================================
SUF = "_dlg"
for key in (
    f'show_txt{SUF}', f'show_cmp{SUF}',
    f'atividade_txt{SUF}', f'atividades_cmp{SUF}'
):
    st.session_state.setdefault(key, False if key.startswith('show') else None)

# ==============================================================================
# DIÁLOGOS
# ==============================================================================
@st.dialog("Texto Completo")
def dlg_texto():
    a = st.session_state[f'atividade_txt{SUF}']
    if not a:
        return
    st.markdown(f"### ID `{a['activity_id']}` – {a['activity_folder']}")
    st.markdown(f"*{a['activity_date'].strftime('%d/%m/%Y %H:%M')}* – {a['user_profile_name']} – **{a['activity_status']}**")
    st.text_area("Conteúdo", a['Texto'], height=400, disabled=True)
    if st.button("Fechar"):
        st.session_state[f'show_txt{SUF}'] = False
        st.rerun()


@st.dialog("Comparação Detalhada", width="large")
def dlg_cmp():
    data = st.session_state[f'atividades_cmp{SUF}']
    if not data:
        return
    a, b = data['base'], data['comp']
    st.markdown(f"### ID `{a['activity_id']}` × ID `{b['activity_id']}`")
    with st.spinner("Gerando diff…"):
        html = diff_cache(a['activity_id'], b['activity_id'], a['Texto'], b['Texto'])
    st.components.v1.html(html, height=600, width=1200, scrolling=True)
    if st.button("Fechar"):
        st.session_state[f'show_cmp{SUF}'] = False
        st.rerun()

# ==============================================================================
# CALLBACKS DE BOTÕES
# ==============================================================================
def ver_txt(a):  # on_click
    st.session_state[f'atividade_txt{SUF}'] = a
    st.session_state[f'show_txt{SUF}'] = True

def cmp_txt(a, b):  # on_click
    st.session_state[f'atividades_cmp{SUF}'] = {'base': a, 'comp': b}
    st.session_state[f'show_cmp{SUF}'] = True

# ==============================================================================
# APP PRINCIPAL
# ==============================================================================
def app():
    st.sidebar.success(f"Logado como **{st.session_state['user']}**")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    st.title("🔎 Verificador de Duplicidade")
    st.caption("Analisa atividades 'Verificar' para apontar textos potencialmente duplicados.")

    eng = get_db_engine()
    if not eng:
        st.error("Falha na conexão com banco."); st.stop()

    if st.sidebar.button("🔄 Atualizar dados"):
        carregar_dados.clear()
        st.toast("Recarregando dados…", icon="🔄")

    df, err = carregar_dados(eng)
    if err:
        st.error("Erro ao buscar dados."); st.exception(err); st.stop()
    if df.empty:
        st.info("Sem dados."); st.stop()

    # ----- FILTROS -----
    hoje = date.today()
    di = st.sidebar.date_input("Início", hoje - timedelta(days=1))
    dfim = st.sidebar.date_input("Fim", hoje + timedelta(days=14))
    if di > dfim:
        st.sidebar.error("Data inicial > final"); st.stop()

    mask = df['activity_date'].dt.date.between(di, dfim)
    df_f = df[mask]

    pasta_sel = st.sidebar.multiselect("Pasta(s)", sorted(df_f['activity_folder'].unique()))
    status_sel = st.sidebar.multiselect("Status", sorted(df_f['activity_status'].unique()))
    if pasta_sel:
        df_f = df_f[df_f['activity_folder'].isin(pasta_sel)]
    if status_sel:
        df_f = df_f[df_f['activity_status'].isin(status_sel)]

    min_sim = st.sidebar.slider("Similaridade mínima (%)", 0, 100, 70, 5) / 100
    apenas_dup = st.sidebar.checkbox("Somente com duplicata", True)

    # ----- CÁLCULO DE DUPLICIDADES (CACHEADO) -----
    with st.spinner("Calculando duplicidades…"):
        mapa_sim, set_dup = obter_duplicidades(df_f, min_sim)

    if apenas_dup:
        df_f = df_f[df_f['activity_id'].isin(set_dup)]

    st.subheader(f"Atividades exibidas: {len(df_f)}")

    for pasta, df_pasta in df_f.groupby('activity_folder'):
        with st.expander(f"📁 {pasta} ({len(df_pasta)})"):
            for _, row in df_pasta.iterrows():
                act = row.to_dict()
                st.markdown("---")
                c1, c2 = st.columns([0.6, 0.4])
                with c1:
                    st.markdown(f"**ID** `{act['activity_id']}` – *{act['activity_date']:%d/%m/%Y %H:%M}* – **{act['activity_status']}**")
                    st.markdown(f"Usuário: {act['user_profile_name']}")
                    st.text_area("Trecho", act['Texto'], 100, disabled=True,
                                 key=f"ta_{act['activity_id']}")
                    b1, b2, b3 = st.columns(3)
                    b1.button("👁️ Completo", on_click=ver_txt, args=(act,),
                              key=f"ver_{act['activity_id']}")
                    links = gerar_links_zflow(act['activity_id'])
                    b2.link_button("🔗 ZFlow v1", links['antigo'])
                    b3.link_button("🔗 ZFlow v2", links['novo'])

                with c2:
                    sims = mapa_sim.get(act['activity_id'], [])
                    if sims:
                        st.markdown(f"<span style='color:red;font-weight:600;'>Duplicatas:</span> {len(sims)}",
                                    unsafe_allow_html=True)
                        for s in sims:
                            info = df[df['activity_id'] == s['id_similar']].iloc[0].to_dict()
                            st.markdown(
                                f"""<div style='background:{s['cor']};padding:4px 6px;
                                       border-radius:6px;margin-bottom:4px;'>
                                    <b>ID {info['activity_id']} ({s['ratio']:.0%})</b><br>
                                    {info['activity_date']:%d/%m/%y %H:%M} – {info['user_profile_name']}
                                   </div>""", unsafe_allow_html=True
                            )
                            st.button("⚖️ Comparar", on_click=cmp_txt, args=(act, info),
                                      key=f"cmp_{act['activity_id']}_{info['activity_id']}")
                    elif not apenas_dup:
                        st.success("Sem duplicatas")

    # Exibe dialogs se sinalizados
    if st.session_state.get(f'show_txt{SUF}'):
        dlg_texto()
    if st.session_state.get(f'show_cmp{SUF}'):
        dlg_cmp()

# ==============================================================================
# LOGIN BÁSICO
# ==============================================================================
def cred_ok(user, pwd):
    try:
        return st.secrets["credentials"]["usernames"].get(user) == pwd
    except KeyError:
        return False

def login():
    st.header("Login")
    with st.form("f_login"):
        u = st.text_input("Usuário")
        p = st.text_input("Senha", type="password")
        if st.form_submit_button("Entrar"):
            if cred_ok(u, p):
                st.session_state['log'] = True
                st.session_state['user'] = u
                st.rerun()
            else:
                st.error("Credenciais inválidas.")

if __name__ == "__main__":
    if not st.session_state.get('log'):
        login()
    else:
        app()
