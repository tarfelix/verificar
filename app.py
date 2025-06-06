import streamlit as st
import pandas as pd
import re, html, os, logging
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from sqlalchemy import create_engine, text, exc
from sqlalchemy.engine import Engine
from unidecode import unidecode
from rapidfuzz import fuzz

# ╔═══════════════════════  CONFIG  ═══════════════════════╗
SUFFIX = "_final_v8_corrigido"
ITENS_POR_PAGINA = 20
HIGHLIGHT_COLOR  = "#a8d1ff"
TZ_SP  = ZoneInfo("America/Sao_Paulo")
TZ_UTC = ZoneInfo("UTC")   # assume timestamps em UTC

st.set_page_config(layout="wide", page_title="Verificador de Duplicidade")

st.markdown(f"""
<style>
 mark.common {{background:{HIGHLIGHT_COLOR};padding:0 2px;font-weight:bold}}
 pre.highlighted-text {{
   white-space:pre-wrap;word-wrap:break-word;
   font-family:'SFMono-Regular',Consolas,'Liberation Mono',Menlo,Courier,monospace;
   font-size:.9em;padding:10px;border:1px solid #ddd;border-radius:5px;background:#f9f9f9;
 }}
 .similarity-badge {{padding:3px 6px;border-radius:5px;color:black;font-weight:500;display:inline-block;margin-bottom:4px}}
</style>
""", unsafe_allow_html=True)

# ╔═══════════════  HELPERS  ═══════════════╗
def as_sp(ts: pd.Timestamp | None):
    if pd.isna(ts): return None
    if ts.tzinfo is None: ts = ts.tz_localize(TZ_UTC)
    return ts.tz_convert(TZ_SP)

def norm(txt: str | None) -> str:
    if not isinstance(txt,str): return ""
    return re.sub(r"\s+"," ", re.sub(r"[^\w\s]"," ", unidecode(txt.lower()))).strip()

def sim(a:str,b:str)->float:
    a,b = map(norm,(a,b))
    if not a or not b or abs(len(a)-len(b))>.3*max(len(a),len(b)): return 0.0
    return fuzz.token_set_ratio(a,b)/100

cor_sim = lambda r:"#FF5252" if r>=.9 else "#FFB74D" if r>=.7 else "#FFD54F"

def highlight_common(t1:str,t2:str,min_len:int=3):
    tk1,re_find = re.findall(r"\w+",norm(t1)), re.findall
    comuns = {w for w in tk1 if w in re_find(r"\w+",norm(t2)) and len(w)>=min_len}
    def wrap(txt):
        seg=[]
        for part in re.split(r"(\W+)",txt):
            if not part: continue
            if re.match(r"\w+",part) and norm(part) in comuns:
                seg.append(f"<mark class='common'>{html.escape(part)}</mark>")
            else: seg.append(html.escape(part))
        return "<pre class='highlighted-text'>"+"".join(seg)+"</pre>"
    return wrap(t1),wrap(t2)

# ╔═══════════════  DB  ═══════════════╗
@st.cache_resource
def db_engine()->Engine|None:
    cfg=st.secrets.get("database",{})
    host,user,pw,db=[cfg.get(k) or os.getenv(f"DB_{k.upper()}") for k in["host","user","password","name"]]
    if not all([host,user,pw,db]): st.error("Credenciais ausentes."); return None
    try:
        eng=create_engine(f"mysql+mysqlconnector://{user}:{pw}@{host}/{db}",pool_pre_ping=True,pool_recycle=3600)
        with eng.connect():pass
        return eng
    except exc.SQLAlchemyError as e:
        logging.exception(e); st.error("Erro de conexão."); return None

@st.cache_data(ttl=3600,hash_funcs={Engine:lambda _:None})
def carregar(eng:Engine)->pd.DataFrame:
    lim=date.today()-timedelta(days=7)
    q1=text("""SELECT activity_id,activity_folder,user_profile_name,
               activity_date,activity_status,Texto
               FROM ViewGrdAtividadesTarcisio
               WHERE activity_type='Verificar' AND activity_status='Aberta'""")
    q2=text("""SELECT activity_id,activity_folder,user_profile_name,
               activity_date,activity_status,Texto
               FROM ViewGrdAtividadesTarcisio
               WHERE activity_type='Verificar' AND DATE(activity_date)>=:lim""")
    try:
        with eng.connect() as c:
            df=pd.concat([pd.read_sql(q1,c),
                          pd.read_sql(q2,c,params={"lim":lim})],ignore_index=True)
        if df.empty:return df
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

# ╔══════════  SIM CACHE  ═══════════╗
def sim_cache(df:pd.DataFrame,min_sim:float):
    sig=(tuple(sorted(df["activity_id"])),min_sim)
    key="simcache"+SUFFIX
    cached=st.session_state.get(key)
    if cached and cached["sig"]==sig: return cached["map"],cached["dup"]
    mapa,dup=set(),{}
    dup=set(); mapa={}
    bar=st.sidebar.progress(0,text="Calculando similaridades…")
    groups=list(df.groupby("activity_folder"))
    for i,(_,g) in enumerate(groups,1):
        bar.progress(i/len(groups),text=f"{i}/{len(groups)} pastas")
        acts=g.to_dict("records")
        for a_i,a in enumerate(acts):
            mapa.setdefault(a["activity_id"],[])
            for b in acts[a_i+1:]:
                r=sim(a["Texto"],b["Texto"])
                if r>=min_sim:
                    c=cor_sim(r)
                    dup.update([a["activity_id"],b["activity_id"]])
                    mapa[a["activity_id"]].append(dict(id=b["activity_id"],ratio=r,cor=c))
                    mapa.setdefault(b["activity_id"],[]).append(dict(id=a["activity_id"],ratio=r,cor=c))
    bar.empty()
    for k in mapa: mapa[k].sort(key=lambda x:x["ratio"],reverse=True)
    st.session_state[key]={"sig":sig,"map":mapa,"dup":dup}
    return mapa,dup

# ╔══════════  STATE DEFAULTS  ═══════════╗
for k,v in {f"show_text{SUFFIX}":False,f"full_act{SUFFIX}":None,
            f"cmp{SUFFIX}":None,f"page{SUFFIX}":0,
            f"last_update{SUFFIX}":None}.items():
    st.session_state.setdefault(k,v)

link_z=lambda i:{"antigo":f"https://zflow.zionbyonset.com.br/activity/3/details/{i}",
                 "novo":f"https://zflowv2.zionbyonset.com.br/public/versatile_frame.php/?moduloid=2&activityid={i}#/fixcol1"}

# ╔══════════  DIALOG TEXTO  ═══════════╗
@st.dialog("Texto completo")
def dlg_full():
    d=st.session_state[f"full_act{SUFFIX}"]; 
    if d is None:return
    dt=as_sp(d["activity_date"])
    st.markdown(f"### ID {d['activity_id']} – {dt.strftime('%d/%m/%Y %H:%M') if dt else 'N/A'}")
    st.markdown(f"<pre style='max-height:400px;overflow:auto'>{html.escape(d['Texto'])}</pre>",unsafe_allow_html=True)
    st.button("Fechar",on_click=lambda:st.session_state.update({f"show_text{SUFFIX}":False}))

# ╔══════════  APP  ═══════════╗
def app():
    st.sidebar.success(f"Logado como **{st.session_state['username']}**")
    if st.sidebar.button("Logout"): st.session_state.clear(); st.rerun()

    eng=db_engine();  st.session_state["eng"]=eng
    if not eng: st.stop()

    if st.sidebar.button("🔄 Atualizar dados"):
        carregar.clear(); st.session_state.pop("simcache"+SUFFIX,None)
        st.session_state[f"last_update{SUFFIX}"]=datetime.now(TZ_SP)

    df=carregar(eng)
    if df.empty: st.warning("Sem atividades."); st.stop()

    up=st.session_state[f"last_update{SUFFIX}"] or datetime.now(TZ_SP)
    st.sidebar.caption(f"Dados atualizados em: {up:%d/%m/%Y %H:%M:%S}")

    hoje=date.today()
    d_ini=st.sidebar.date_input("Início",hoje-timedelta(days=1))
    d_fim=st.sidebar.date_input("Fim",hoje+timedelta(days=14),min_value=d_ini)
    if d_ini>d_fim: st.sidebar.error("Início > fim."); st.stop()

    df_per=df[df["activity_date"].notna() & df["activity_date"].dt.date.between(d_ini,d_fim)]
    st.title(f"🔎 Duplicidades ({len(df_per)})")

    pastas_sel=st.sidebar.multiselect("Pastas p/ Análise",sorted(df_per["activity_folder"].dropna().unique()))
    df_ana=df_per if not pastas_sel else df_per[df_per["activity_folder"].isin(pastas_sel)]

    status_sel=st.sidebar.multiselect("Status p/ Exibição",sorted(df_ana["activity_status"].dropna().unique()))
    min_sim=st.sidebar.slider("Similaridade mínima (%)",0,100,90,5)/100
    only_dup=st.sidebar.checkbox("Somente duplicatas",True)
    only_multi=st.sidebar.checkbox("Pastas com múltiplos responsáveis")
    users_sel=st.sidebar.multiselect("Usuários",sorted(df_ana["user_profile_name"].dropna().unique()))

    sim_map,dup_ids=sim_cache(df_ana,min_sim)

    df_view=df_ana.copy()
    if status_sel: df_view=df_view[df_view["activity_status"].isin(status_sel)]
    if only_dup:  df_view=df_view[df_view["activity_id"].isin(dup_ids)]
    if only_multi:
        mult={p for p,g in df_ana.groupby("activity_folder") if g["user_profile_name"].nunique()>1}
        df_view=df_view[df_view["activity_folder"].isin(mult)]
    if users_sel: df_view=df_view[df_view["user_profile_name"].isin(users_sel)]

    idx_map=df_ana.set_index("activity_id").to_dict("index")

    pastas_ord=sorted(df_view["activity_folder"].dropna().unique())
    page=st.session_state[f"page{SUFFIX}"]
    total=max(1,(len(pastas_ord)+ITENS_POR_PAGINA-1)//ITENS_POR_PAGINA)
    page=max(0,min(page,total-1)); st.session_state[f"page{SUFFIX}"]=page

    if total>1:
        l,mid,r=st.columns([1,2,1])
        if l.button("⬅",disabled=page==0): st.session_state[f"page{SUFFIX}"]-=1; st.rerun()
        mid.markdown(f"<p style='text-align:center'>Página {page+1}/{total}</p>",unsafe_allow_html=True)
        if r.button("➡",disabled=page==total-1): st.session_state[f"page{SUFFIX}"]+=1; st.rerun()

    cmp_state=st.session_state[f"cmp{SUFFIX}"]

    for pasta in pastas_ord[page*ITENS_POR_PAGINA:(page+1)*ITENS_POR_PAGINA]:
        df_p=df_view[df_view["activity_folder"]==pasta]
        tot=len(df_ana[df_ana["activity_folder"]==pasta])
        with st.expander(f"📁 {pasta} ({len(df_p)}/{tot})"):
            for row in df_p.itertuples():
                act=row.activity_id
                c1,c2=st.columns([.6,.4],gap="small")
                with c1:
                    dt=as_sp(row.activity_date)
                    st.markdown(f"**ID** `{act}` • {dt.strftime('%d/%m/%Y %H:%M') if dt else 'N/A'} • `{row.activity_status}`")
                    st.markdown(f"**Usuário:** {row.user_profile_name}")
                    st.text_area("Texto",row.Texto,height=100,disabled=True,
                                 key=f"txt_{pasta}_{act}_{page}")
                    b1,b2,b3=st.columns(3)
                    b1.button("👁 Completo",
                              key=f"full_{pasta}_{act}_{page}",
                              on_click=lambda r=row: st.session_state.update({
                                  f"full_act{SUFFIX}":r._asdict(),f"show_text{SUFFIX}":True}))
                    lnk=link_z(act); b2.link_button("ZFlow v1",lnk["antigo"]); b3.link_button("ZFlow v2",lnk["novo"])
                with c2:
                    sims=sim_map.get(act,[])
                    if sims:
                        st.markdown(f"**Duplicatas:** {len(sims)}")
                        for s in sims:
                            info=idx_map.get(s["id"]); 
                            if not info:continue
                            d=as_sp(info["activity_date"]); d_fmt=d.strftime("%d/%m/%y %H:%M") if d else "N/A"
                            st.markdown(f"<div class='similarity-badge' style='background:{s['cor']};'>"
                                        f"<b>{info['activity_id']}</b> • {s['ratio']:.0%}<br>"
                                        f"{d_fmt} • {info['activity_status']}<br>{info['user_profile_name']}"
                                        "</div>",unsafe_allow_html=True)
                            st.button("⚖ Comparar",
                                      key=f"cmp_{page}_{pasta}_{act}_{info['activity_id']}",
                                      on_click=lambda a=act,b=info['activity_id']:
                                         st.session_state.update({f"cmp{SUFFIX}":{"base_id":a,"comp_id":b}}))
                    elif not only_dup:
                        st.markdown("<span style='color:green;'>Sem duplicatas</span>",unsafe_allow_html=True)

                if cmp_state and cmp_state["base_id"]==act:
                    comp=idx_map.get(cmp_state["comp_id"])
                    if comp:
                        hA,hB=highlight_common(row.Texto,comp["Texto"])
                        st.markdown("---"); aCol,bCol=st.columns(2)
                        aCol.markdown(hA,unsafe_allow_html=True); bCol.markdown(hB,unsafe_allow_html=True)
                        if st.button("❌ Fechar comparação",key=f"cls_{act}_{page}"):
                            st.session_state[f"cmp{SUFFIX}"]=None; st.rerun()

    if st.session_state[f"show_text{SUFFIX}"]: dlg_full()

# ╔══════════  LOGIN  ═══════════╗
def cred_ok(u,p): return str(st.secrets["credentials"]["usernames"].get(u,""))==p
def login():
    st.header("Login")
    with st.form("login_form_main"):
        u=st.text_input("Usuário")
        p=st.text_input("Senha",type="password")
        if st.form_submit_button("Entrar"):
            if cred_ok(u,p):
                st.session_state.update({"logged_in":True,"username":u}); st.rerun()
            else: st.error("Credenciais inválidas.")

# ╔══════════  MAIN  ═══════════╗
if __name__=="__main__":
    if not st.session_state.get("logged_in"): st.session_state["logged_in"]=False
    (app() if st.session_state["logged_in"] else login())
