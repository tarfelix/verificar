import streamlit as st
import pandas as pd
import re, html, os, logging
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from unidecode import unidecode
from rapidfuzz import fuzz
from api_functions import HttpClient
from difflib import SequenceMatcher

# ═════════════════ CONFIG ═════════════════
SUFFIX = "_final_v8_corrigido"
ITENS_POR_PAGINA = 20
TZ_SP  = ZoneInfo("America/Sao_Paulo")
TZ_UTC = ZoneInfo("UTC")

st.set_page_config(layout="wide", page_title="Verificador de Duplicidade")

# CSS para a visualização de diferenças (diff) e outros elementos.
st.markdown("""
<style>
    pre.highlighted-text {
        white-space: pre-wrap;
        word-wrap: break-word;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
        font-size: .9em;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        background-color: #f9f9f9;
        height: 400px;
        overflow-y: auto;
    }
    .similarity-badge {
        padding: 3px 6px; border-radius: 5px; color: black; font-weight: 500;
        display: inline-block; margin-bottom: 4px;
    }
    .diff-del {
        background-color: #ffcdd2 !important;  /* vermelho claro para texto removido */
        text-decoration: none !important;
    }
    .diff-ins {
        background-color: #c8e6c9 !important;  /* verde claro para texto adicionado */
        text-decoration: none !important;
    }
</style>
""", unsafe_allow_html=True)


cfg=st.secrets.get("api",{})
url_api,entity_id,token=[cfg.get(k) for k in["url_api","entity_id","token"]]

api = HttpClient(
    base_url=url_api,
    entity_id=entity_id,
    token=token
)

# ═════════════ HELPERS E FUNÇÕES DE LÓGICA ═════════════

def get_cancelados():
    cancelados = set()
    for k, v in st.session_state.items():
        if k.startswith("cancel_") and v:
            try:
                act_id = k.split("_")[-1]
                cancelados.add(act_id)
            except (ValueError, IndexError):
                pass
    return sorted(list(cancelados))

def check_total(pasta):
    return sum(1 for k, v in st.session_state.items() if k.startswith(f"cancel_{pasta}_") and v is True)

def clean_cancelados():
    chaves_para_remover = [k for k in st.session_state.keys() if k.startswith("cancel_")]
    for k in chaves_para_remover:
        del st.session_state[k]

@st.dialog("Confirmação de Cancelamento")
def confirmar_cancelamento():
    cancelados = get_cancelados()

    if len(cancelados) == 0:
        st.write("Nenhuma atividade foi marcada para cancelamento.")
        if st.button("❌ Fechar"):
            st.session_state.pop("step_cancel", None)
            st.rerun()
    else:
        st.write("As seguintes atividades serão canceladas:")
        st.code("\n".join(str(c) for c in cancelados), language="text")

        if not st.session_state.get("process_cancel") and not st.session_state.get("step_cancel_processado"):
            col1, col2 = st.columns(2)
            with col1:
                st.button("✅ Confirmar",disabled=st.session_state.get("process_cancel", False), key='process_cancel')
            with col2:
                if st.button("❌ Cancelar"):
                    st.session_state.pop("step_cancel", None)
                    st.rerun()
        elif not st.session_state.get("step_cancel_processado"):
            progress = st.progress(0, text="Iniciando...")
            total = len(cancelados)
            for idx, act_id in enumerate(cancelados, start=1):
                with st.status(f"Cancelando {act_id}...", expanded=True):
                    try:
                        response = api.activity_canceled(act_id, st.session_state['username'])
                        if response is not None and response.get("code") == '200':
                            st.success(f"✅ {act_id} cancelada.")
                        else:
                            err_msg = f"❌ Erro ao cancelar {act_id}."
                            if response is not None:
                                err_msg += f" {response.get('message')}."
                            st.error(err_msg)
                    except Exception as e:
                        st.error(f"❌ Erro ao cancelar {act_id}: {e}")
                progress.progress(idx / total, text=f"{idx}/{total} concluídos")
            st.success("Todas as atividades foram processadas.")
            st.session_state["step_cancel_processado"] = True

        if st.session_state.get("step_cancel_processado"):
            if st.button("✅ Concluir"):
                atualizar_dados()
                clean_cancelados()
                st.session_state.pop("step_cancel", None)
                st.session_state.pop("step_cancel_processado", None)
                st.session_state.pop("process_cancel", None)
                st.rerun()

def atualizar_dados():
    carregar.clear()
    st.session_state.pop("simcache_cruzado"+SUFFIX, None)
    st.session_state[f"last_update{SUFFIX}"]=datetime.now(TZ_SP)

def as_sp(ts: pd.Timestamp | None):
    if pd.isna(ts): return None
    if ts.tzinfo is None: ts = ts.tz_localize(TZ_UTC)
    return ts.tz_convert(TZ_SP)

def norm(t:str|None)->str:
    if not isinstance(t,str): return ""
    return re.sub(r"\s+"," ",re.sub(r"[^\w\s]"," ",unidecode(t.lower()))).strip()

def calc_sim(a:str,b:str)->float:
    a,b = map(norm,(a,b))
    if not a or not b or abs(len(a)-len(b))>.3*max(len(a),len(b)): return 0.0
    return fuzz.token_set_ratio(a,b)/100

cor_sim = lambda r:"#FF5252" if r>=.9 else "#FFB74D" if r>=.7 else "#FFD54F"

def highlight_diffs(t1: str, t2: str) -> tuple[str, str]:
    t1, t2 = (t1 or ""), (t2 or "")
    t1_tokens = [token for token in re.split(r'(\W+)', t1) if token]
    t2_tokens = [token for token in re.split(r'(\W+)', t2) if token]
    sm = SequenceMatcher(None, t1_tokens, t2_tokens, autojunk=False)
    out1, out2 = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        slice1 = html.escape("".join(t1_tokens[i1:i2]))
        slice2 = html.escape("".join(t2_tokens[j1:j2]))
        if tag == 'equal':
            out1.append(slice1)
            out2.append(slice2)
        elif tag == 'replace':
            out1.append(f"<span class='diff-del'>{slice1}</span>")
            out2.append(f"<span class='diff-ins'>{slice2}</span>")
        elif tag == 'delete':
            out1.append(f"<span class='diff-del'>{slice1}</span>")
        elif tag == 'insert':
            out2.append(f"<span class='diff-ins'>{slice2}</span>")
    h1 = f"<pre class='highlighted-text'>{''.join(out1)}</pre>"
    h2 = f"<pre class='highlighted-text'>{''.join(out2)}</pre>"
    return h1, h2

# ═════════════ BANCO DE DADOS ═════════════
@st.cache_resource
def db_engine() -> Engine | None:
    cfg=st.secrets.get("database",{})
    host,user,pw,db=[cfg.get(k) or os.getenv(f"DB_{k.upper()}") for k in["host","user","password","name"]]
    if not all([host,user,pw,db]): st.error("Credenciais de banco ausentes."); return None
    try:
        eng=create_engine(f"mysql+mysqlconnector://{user}:{pw}@{host}/{db}",pool_pre_ping=True,pool_recycle=3600)
        with eng.connect(): pass
        return eng
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error("Erro ao conectar."); return None

@st.cache_data(ttl=3600,hash_funcs={Engine:lambda _:None})
def carregar(eng:Engine)->pd.DataFrame:
    # [ALTERAÇÃO] Aumentado o período de busca para 90 dias para criar uma base de comparação robusta.
    lim=date.today()-timedelta(days=15)
    q_open=text("""SELECT activity_id,activity_folder,user_profile_name,
                   activity_date,activity_status,Texto
                   FROM ViewGrdAtividadesTarcisio
                   WHERE activity_type='Verificar' AND activity_status='Aberta'""")
    q_hist=text("""SELECT activity_id,activity_folder,user_profile_name,
                   activity_date,activity_status,Texto
                   FROM ViewGrdAtividadesTarcisio
                   WHERE activity_type='Verificar' AND DATE(activity_date)>=:lim""")
    try:
        with eng.connect() as c:
            df=pd.concat([pd.read_sql(q_open,c),
                          pd.read_sql(q_hist,c,params={"lim":lim})],ignore_index=True)
        if df.empty: return df
        df["activity_id"] = df["activity_id"].astype(str)
        df["activity_date"]=pd.to_datetime(df["activity_date"],errors="coerce")
        df["Texto"]=df["Texto"].astype(str).fillna("")
        df["status_ord"]=df["activity_status"].map({"Aberta":0}).fillna(1)
        df=(df.sort_values(["activity_id","status_ord"])
                .drop(columns="status_ord")
                .drop_duplicates("activity_id"))
        return df.sort_values(["activity_folder","activity_date","activity_id"],
                              ascending=[True,False,False])
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error("Erro SQL"); return pd.DataFrame()

# [NOVA FUNÇÃO] Substitui sim_cache. Compara um dataframe (df_exibir) contra outro (df_comparar).
def gerar_mapa_similaridade(df_exibir: pd.DataFrame, df_comparar: pd.DataFrame, min_sim: float):
    if df_exibir.empty:
        return {}, set()
        
    sig = (tuple(sorted(df_exibir["activity_id"])), tuple(sorted(df_comparar["activity_id"])), min_sim)
    key = "simcache_cruzado" + SUFFIX
    
    cached = st.session_state.get(key)
    if cached and cached.get("sig") == sig:
        return cached["map"], cached["dup"]

    mapa_similaridade = {}
    ids_duplicados = set()
    
    atividades_para_exibir = df_exibir.to_dict("records")
    atividades_para_comparar = df_comparar.to_dict("records")

    bar = st.sidebar.progress(0, text="Calculando similaridades cruzadas…")
    total_a_exibir = len(atividades_para_exibir)

    for i, atividade_principal in enumerate(atividades_para_exibir):
        bar.progress((i + 1) / total_a_exibir, text=f"Verificando atividade {i+1}/{total_a_exibir}")
        id_principal = atividade_principal["activity_id"]
        mapa_similaridade.setdefault(id_principal, [])
        
        for atividade_historica in atividades_para_comparar:
            id_historico = atividade_historica["activity_id"]
            
            if id_principal == id_historico:
                continue

            ratio = calc_sim(atividade_principal["Texto"], atividade_historica["Texto"])

            if ratio >= min_sim:
                ids_duplicados.add(id_principal)
                ids_duplicados.add(id_historico)
                
                cor = cor_sim(ratio)
                mapa_similaridade[id_principal].append({
                    "id": id_historico, 
                    "ratio": ratio, 
                    "cor": cor
                })

    bar.empty()
    
    for k in mapa_similaridade:
        mapa_similaridade[k].sort(key=lambda z: z["ratio"], reverse=True)
        
    st.session_state[key] = {"sig": sig, "map": mapa_similaridade, "dup": ids_duplicados}
    return mapa_similaridade, ids_duplicados


# ═════════════ STATE DEFAULTS ═════════════
for k,v in {f"show_text{SUFFIX}":False,f"full_act{SUFFIX}":None,
            f"open_cmps{SUFFIX}":set(), f"page{SUFFIX}":0,
            f"last_update{SUFFIX}":None}.items():
    st.session_state.setdefault(k,v)

# ═════════════ LINK ZFLOW ═════════════
link_z=lambda i:{"antigo":f"https://zflow.zionbyonset.com.br/activity/3/details/{i}",
                 "novo":f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={i}#/fixcol1"}

# ═════════════ DIALOG TEXTO ═════════════
@st.dialog("Texto completo")
def dlg_full():
    d=st.session_state[f"full_act{SUFFIX}"]
    if d is None: return
    dt=as_sp(d["activity_date"])
    st.markdown(f"### ID {d['activity_id']} – {dt.strftime('%d/%m/%Y %H:%M') if dt else 'N/A'}")
    st.markdown(f"<pre style='max-height:400px;overflow:auto'>{html.escape(d['Texto'])}</pre>",unsafe_allow_html=True)
    st.button("Fechar", key="dialog_close_full_text", on_click=lambda:st.session_state.update({f"show_text{SUFFIX}":False}))

def toggle_cancel_state(pasta: str, act_id: str | int):
    state_key = f"cancel_{pasta}_{act_id}"
    current_value = st.session_state.get(state_key, False)
    st.session_state[state_key] = not current_value

# ═════════════ APP ═════════════
def app():
    st.sidebar.success(f"Logado como **{st.session_state['username']}**")
    if st.sidebar.button("Logout", key="logout_button"):
        st.session_state.clear()
        st.rerun()

    eng=db_engine(); st.session_state["eng"]=eng
    if not eng: st.stop()

    if st.sidebar.button("🔄 Atualizar dados", key="update_data_button"):
        atualizar_dados()

    df=carregar(eng)
    if df.empty: st.warning("Sem atividades."); st.stop()

    up=st.session_state[f"last_update{SUFFIX}"] or datetime.now(TZ_SP)
    st.sidebar.caption(f"Dados atualizados em: {up:%d/%m/%Y %H:%M:%S}")

    hoje=date.today()
    d_ini=st.sidebar.date_input("Início",hoje-timedelta(days=7))
    d_fim=st.sidebar.date_input("Fim",hoje+timedelta(days=14),min_value=d_ini)
    if d_ini>d_fim: st.sidebar.error("Início > fim."); st.stop()

    # [ALTERAÇÃO] Lógica de filtragem separada
    
    # 1. Filtro inicial de data para ambos os dataframes
    df_periodo = df[df["activity_date"].notna() & df["activity_date"].dt.date.between(d_ini, d_fim)]

    st.markdown("""
    <style>
    .vertical-align-bottom { display: flex; align-items: flex-end; height: 100%;}
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([6, 2])
    with col1:
        st.markdown(f"<div class=\"vertical-align-bottom\"><h3>🔎 Duplicidades ({len(df_periodo)})</h3></div>", unsafe_allow_html=True)
    with col2:
        if st.button("Cancelar selecionado(s)", key="process_cancel_button"):
            st.session_state["step_cancel"] = "confirmar"
            st.session_state["step_cancel_processado"] = False
            confirmar_cancelamento()

    # --- Filtros da Sidebar ---
    pastas_sel=st.sidebar.multiselect("Pastas p/ Análise",sorted(df["activity_folder"].dropna().unique()))
    status_sel=st.sidebar.multiselect("Status p/ Exibição",sorted(df_periodo["activity_status"].dropna().unique()))
    min_sim=st.sidebar.slider("Similaridade mínima (%)",0,100,90,5)/100
    only_dup=st.sidebar.checkbox("Somente duplicatas",True)
    only_multi=st.sidebar.checkbox("Pastas com múltiplos responsáveis")
    users_sel=st.sidebar.multiselect("Usuários",sorted(df_periodo["user_profile_name"].dropna().unique()))

    # [ALTERAÇÃO] Criação dos dois dataframes: base e de exibição
    
    # 2. Base de Comparação: todo o histórico (90 dias) das pastas selecionadas.
    df_base_comparacao = df if not pastas_sel else df[df["activity_folder"].isin(pastas_sel)]

    # 3. Itens para Exibir: começa com o filtro de data e aplica os outros filtros da tela.
    df_para_exibir = df_periodo.copy()
    if pastas_sel:
        df_para_exibir = df_para_exibir[df_para_exibir["activity_folder"].isin(pastas_sel)]
    if status_sel:
        df_para_exibir = df_para_exibir[df_para_exibir["activity_status"].isin(status_sel)]
    if only_multi:
        mult={p for p,g in df_base_comparacao.groupby("activity_folder") if g["user_profile_name"].nunique()>1}
        df_para_exibir = df_para_exibir[df_para_exibir["activity_folder"].isin(mult)]
    if users_sel:
        df_para_exibir = df_para_exibir[df_para_exibir["user_profile_name"].isin(users_sel)]

    # [ALTERAÇÃO] Chamada da nova função de similaridade
    sim_map, dup_ids = gerar_mapa_similaridade(df_para_exibir, df_base_comparacao, min_sim)
    
    df_view = df_para_exibir.copy()
    
    # O filtro "only_dup" é aplicado aqui, após o cálculo.
    if only_dup:
        df_view = df_view[df_view["activity_id"].isin(dup_ids)]

    # [ALTERAÇÃO] O idx_map usa a base de dados completa para poder buscar detalhes de qualquer atividade.
    idx_map = df_base_comparacao.set_index("activity_id").to_dict("index")

    pastas_ord=sorted(df_view["activity_folder"].dropna().unique())
    page=st.session_state[f"page{SUFFIX}"]
    total=max(1,(len(pastas_ord)+ITENS_POR_PAGINA-1)//ITENS_POR_PAGINA)
    page=max(0,min(page,total-1)); st.session_state[f"page{SUFFIX}"]=page

    if total>1:
        l,mid,r=st.columns([1,2,1])
        if l.button("⬅", key="prev_page_button", disabled=page==0):
            st.session_state[f"page{SUFFIX}"]-=1; st.rerun()
        mid.markdown(f"<p style='text-align:center'>Página {page+1}/{total}</p>",unsafe_allow_html=True)
        if r.button("➡", key="next_page_button", disabled=page==total-1):
            st.session_state[f"page{SUFFIX}"]+=1; st.rerun()

    for pasta in pastas_ord[page*ITENS_POR_PAGINA:(page+1)*ITENS_POR_PAGINA]:
        df_p=df_view[df_view["activity_folder"]==pasta]
        total_na_pasta_visivel = len(df_p)
        total_atividades_na_pasta = len(df_base_comparacao[df_base_comparacao["activity_folder"] == pasta])
        
        max_selecoes = max(0, total_atividades_na_pasta - 1)
        num_selecionados_atual = check_total(pasta)
        limite_atingido = num_selecionados_atual >= max_selecoes
        
        with st.expander(f"📁 {pasta} ({total_na_pasta_visivel} de {total_atividades_na_pasta}) - Selecionados: {num_selecionados_atual}/{max_selecoes}"):
            for row in df_p.itertuples():
                act=row.activity_id

                c1,c2=st.columns([.6,.4],gap="small")

                with c1:
                    dt=as_sp(row.activity_date)
                    st.markdown(f"**ID** `{act}` • {dt.strftime('%d/%m/%Y %H:%M') if dt else 'N/A'} • `{row.activity_status}`")
                    st.markdown(f"**Usuário:** {row.user_profile_name}")
                    st.text_area("Texto",row.Texto,height=100,disabled=True,key=f"txt_{pasta}_{act}_{page}")
                    b1,b2,b3=st.columns(3)
                    b1.button("👁 Completo",key=f"full_{pasta}_{act}_{page}",
                              on_click=lambda r=row: st.session_state.update({f"full_act{SUFFIX}":r._asdict(),f"show_text{SUFFIX}":True}))
                    links=link_z(act); b2.link_button("ZFlow v1",links["antigo"]); b3.link_button("ZFlow v2",links["novo"])

                with c2:
                    sims=sim_map.get(act,[])
                    if sims:
                        st.markdown(f"**Duplicatas Encontradas:** {len(sims)}")
                        for s in sims:
                            info=idx_map.get(s["id"])
                            if not info: continue
                            info_id = s["id"]
                            
                            d = as_sp(info["activity_date"])
                            d_fmt = d.strftime("%d/%m/%y %H:%M") if d else "N/A"
                            st.markdown(f"<div class='similarity-badge' style='background:{s['cor']};'>"
                                        f"<b>ID {info_id}</b> • {s['ratio']:.0%}<br>"
                                        f"{d_fmt} • {info['activity_status']}<br>{info['user_profile_name']}"
                                        "</div>",unsafe_allow_html=True)
                            
                            def add_comparison(id1, id2):
                                canonical_pair = tuple(sorted((id1, id2)))
                                st.session_state[f"open_cmps{SUFFIX}"].add(canonical_pair)

                            st.button("⚖ Comparar",
                                      key=f"cmp_{page}_{pasta}_{act}_{info_id}",
                                      on_click=add_comparison,
                                      args=(act, info_id))
                    elif not only_dup:
                        st.markdown("<span style='color:green;'>Sem duplicatas para os filtros atuais.</span>",unsafe_allow_html=True)
                
                open_comparisons = st.session_state.get(f"open_cmps{SUFFIX}", set())
                comparisons_to_show = [c for c in open_comparisons if act in c]

                if comparisons_to_show:
                    st.markdown("---")

                for base_id_tuple_part, comp_id_tuple_part in comparisons_to_show:
                    base_id = act
                    comp_id = comp_id_tuple_part if base_id_tuple_part == act else base_id_tuple_part

                    base_data = row
                    comp_data = idx_map.get(comp_id)
                    if not comp_data: continue

                    with st.container(border=True):
                        st.markdown("""
                        <div style="font-size: 0.85em; margin-bottom: 10px; padding: 5px; background-color: #f0f2f6; border-radius: 5px;">
                            <b>Legenda:</b>
                            <span style="padding: 0 3px; margin: 0 5px; background-color: #ffcdd2;">Texto removido</span> |
                            <span style="padding: 0 3px; margin: 0 5px; background-color: #c8e6c9;">Texto adicionado</span>
                        </div>
                        """, unsafe_allow_html=True)

                        hA,hB = highlight_diffs(base_data.Texto, comp_data["Texto"])
                        
                        colA, colB = st.columns(2)
                        with colA:
                            st.markdown(f"**Original: ID `{base_id}`**")
                            st.markdown(hA, unsafe_allow_html=True)
                        with colB:
                            sim_info = next((s for s in sim_map.get(base_id, []) if s['id'] == comp_id), None)
                            ratio_str = f"{sim_info['ratio']:.0%}" if sim_info else "N/A"
                            st.markdown(f"**Comparado: ID `{comp_id}` ({ratio_str})**")
                            st.markdown(hB, unsafe_allow_html=True)

                        st.markdown("##### ❎ Marcar para cancelamento")
                        col_chk1, col_chk2 = st.columns(2)
                        with col_chk1:
                            state_key1 = f"cancel_{pasta}_{base_id}"
                            is_checked1 = st.session_state.get(state_key1, False)
                            chk1_disabled = limite_atingido and not is_checked1
                            st.checkbox(f"Cancelar ID {base_id}", value=is_checked1, key=f"widget_chk_{pasta}_{base_id}_vs_{comp_id}", on_change=toggle_cancel_state, args=(pasta, base_id), disabled=chk1_disabled)
                        with col_chk2:
                            state_key2 = f"cancel_{pasta}_{comp_id}"
                            is_checked2 = st.session_state.get(state_key2, False)
                            chk2_disabled = limite_atingido and not is_checked2
                            st.checkbox(f"Cancelar ID {comp_id}", value=is_checked2, key=f"widget_chk_{pasta}_{comp_id}_vs_{base_id}", on_change=toggle_cancel_state, args=(pasta, comp_id), disabled=chk2_disabled)

                        def remove_comparison(b_id, c_id):
                            canonical_pair = tuple(sorted((b_id, c_id)))
                            st.session_state[f"open_cmps{SUFFIX}"].discard(canonical_pair)

                        st.button("❌ Fechar comparação", key=f"cls_{base_id}_{comp_id}", on_click=remove_comparison, args=(base_id, comp_id))
    
    st.markdown("---")

    if st.session_state.get("step_cancel") == "confirmar":
        confirmar_cancelamento()
        
    if st.session_state[f"show_text{SUFFIX}"]: dlg_full()

# ═════════════ LOGIN ═════════════
def cred_ok(u,p):
    user_data = st.secrets.get("credentials", {}).get("usernames", {}).get(u)
    if not user_data:
        return False
    return str(user_data) == p

def login():
    st.header("Login")
    with st.form("login_form_main"):
        u=st.text_input("Usuário"); p=st.text_input("Senha",type="password")
        if st.form_submit_button("Entrar"):
            if cred_ok(u,p):
                st.session_state.update({"logged_in":True,"username":u}); st.rerun()
            else: st.error("Credenciais inválidas.")

# ═════════════ MAIN ═════════════
if __name__=="__main__":
    if not st.session_state.get("logged_in"): st.session_state["logged_in"]=False
    (app() if st.session_state["logged_in"] else login())
