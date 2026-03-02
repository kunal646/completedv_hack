"""
Prism — SpecFlow · Linear-inspired redesign
Sidebar navigation · Clean dark UI · Same agents/logic
"""

import streamlit as st
import time
from queue_store import (
    add_message, get_messages, set_status, board_counts,
    get_items, get_item, update_item, delete_item,
    save_issue_card, save_tech_plan, save_lane_output, save_auto_process_result,
    get_cfg, set_cfg,
    add_context_file, get_context_files, delete_context_file, build_context_str,
    LANE_META, LANE_STATUSES,
)
from agent import (
    build_interpreter_graph, build_architect_graph, auto_process, check_relevance,
)
import discord_bot
import slack_bot

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prism",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* === BASE ============================================================ */
html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  color: #FFFFFF;
}
/* Global white text fallback for all Streamlit content */
.main p, .main span, .main div, .main label,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span { color: #FFFFFF !important; }
.main { background: #0D0D0D !important; }
.main .block-container {
  padding: 1.75rem 2rem 3rem 2rem !important;
  max-width: 100% !important;
}

/* === SIDEBAR ========================================================= */
section[data-testid="stSidebar"] {
  background: #141414 !important;
  border-right: 1px solid #222 !important;
  min-width: 220px !important;
  max-width: 220px !important;
}
section[data-testid="stSidebar"] > div:first-child {
  padding: 1.25rem 0.85rem 1rem !important;
}
section[data-testid="stSidebar"] hr {
  margin: 0.9rem 0 !important; border-color: #222 !important;
}
/* Sidebar default text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
  color: #CCCCCC !important;
}

/* Brand */
.brand {
  display: flex; align-items: center; gap: 0.55rem;
  padding: 0.1rem 0.3rem 1.1rem;
  border-bottom: 1px solid #222; margin-bottom: 1rem;
}
.brand-mark {
  width: 26px; height: 26px; border-radius: 6px;
  background: linear-gradient(135deg, #5E6AD2 0%, #8B94E8 100%);
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 700; color: #fff !important; flex-shrink: 0;
}
.brand-name { font-size: 15px; font-weight: 600; color: #EDEDED !important; letter-spacing: -0.3px; }

/* Sidebar radio → styled nav items */
section[data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none !important; }
section[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] {
  padding: 0.45rem 0.65rem !important;
  border-radius: 6px !important;
  margin-bottom: 2px !important;
  gap: 0 !important;
  transition: background 0.12s !important;
  cursor: pointer !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"]:hover {
  background: #1E1E1E !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
  display: none !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] p,
section[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] span {
  font-size: 13px !important; color: #BBBBBB !important; font-weight: 400 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] {
  background: #1E1E1E !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] p,
section[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] span {
  color: #F0F0F0 !important; font-weight: 500 !important;
}

/* Sidebar section labels */
.s-section {
  font-size: 10px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.9px; color: #777 !important;
  padding: 0 0.4rem; margin: 0.85rem 0 0.35rem;
}
/* Sidebar stat rows */
.s-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.3rem 0.5rem; border-radius: 5px; cursor: default;
  transition: background 0.1s;
}
.s-row:hover { background: #1A1A1A; }
.s-lbl { font-size: 12px; color: #DDDDDD !important; }
.s-badge {
  font-size: 10px; font-weight: 600; padding: 1px 6px;
  border-radius: 9px; background: #1E1E1E; color: #CCCCCC !important; border: 1px solid #2A2A2A;
}
.s-badge.r { color: #F07070 !important; border-color: #3A1818 !important; background: #1E0E0E !important; }
.s-badge.b { color: #6EB4F7 !important; border-color: #1A3060 !important; background: #0E1A2E !important; }
.s-badge.y { color: #E0AE40 !important; border-color: #3A2A10 !important; background: #1E160A !important; }
.s-badge.g { color: #5DC878 !important; border-color: #1A3A20 !important; background: #0E1E12 !important; }

/* Bot rows */
.bot-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.25rem 0.5rem; font-size: 12px; color: #BBBBBB;
}
.dot { width:6px; height:6px; border-radius:50%; display:inline-block; margin-right:5px; vertical-align:middle; }
.dot-on  { background:#5DC878; box-shadow:0 0 5px #5DC87855; }
.dot-off { background:#777; }
.dot-mid { background:#E0AE40; }
.c-on  { color:#5DC878 !important; }
.c-off { color:#AAAAAA !important; }
.c-mid { color:#E0AE40 !important; }

/* === PAGE HEADER ===================================================== */
.pg-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 1.25rem; padding-bottom: 0.9rem;
  border-bottom: 1px solid #222;
}
.pg-title { font-size: 17px; font-weight: 600; color: #F0F0F0; letter-spacing: -0.4px; margin: 0; }
.pg-sub   { font-size: 12px; color: #CCCCCC; margin: 3px 0 0; }

/* === PILLS =========================================================== */
.pill {
  display: inline-flex; align-items: center;
  border-radius: 4px; padding: 2px 7px;
  font-size: 11px; font-weight: 500; white-space: nowrap; vertical-align: middle;
}
.pi  { background:#2A1010; color:#F07070; border:1px solid #3A1818; }
.pf  { background:#0E1A2E; color:#6EB4F7; border:1px solid #1A3060; }
.pid { background:#1E1600; color:#E0AE40; border:1px solid #3A2A10; }
.pm  { background:#0E1E12; color:#5DC878; border:1px solid #1A3A20; }
.pd  { background:#141638; color:#7B93E0; border:1px solid #1E2468; }
.ps  { background:#1E1600; color:#ECB22E; border:1px solid #3A2A10; }
.pma { background:#1E1E1E; color:#888;    border:1px solid #2A2A2A; }
.pr-c { background:#200808; color:#FF5555; border:1px solid #440E0E; }
.pr-h { background:#1E1600; color:#E5A633; border:1px solid #3A2A10; }
.pr-m { background:#0E1A2E; color:#6EB4F7; border:1px solid #1A3060; }
.pr-l { background:#1A1A1A; color:#666;   border:1px solid #252525; }

/* Priority dot inline */
.dp { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:4px; vertical-align:middle; flex-shrink:0; }
.dp-c { background:#FF5555; }
.dp-h { background:#E5A633; }
.dp-m { background:#6EB4F7; }
.dp-l { background:#444; }

/* === CARDS =========================================================== */
.card {
  background: #161616; border: 1px solid #222;
  border-radius: 8px; padding: 0.8rem 1rem;
  margin-bottom: 0.45rem; transition: border-color 0.12s;
}
.card:hover { border-color: #333; }
.card-row   { display:flex; align-items:center; gap:0.4rem; flex-wrap:wrap; margin-bottom:5px; }
.card-title { font-size:13px; font-weight:500; color:#FFFFFF; line-height:1.4; margin:4px 0 3px; }
.card-body  { font-size:12px; color:#CCCCCC; line-height:1.6; margin-top:4px; }
.card-foot  { font-size:11px; color:#BBBBBB; display:flex; gap:0.75rem; margin-top:6px; }

/* === KANBAN ========================================================== */
.k-col {
  background: #111; border: 1px solid #1E1E1E;
  border-radius: 8px; padding: 0.9rem 0.8rem; min-height: 300px;
}
.k-header {
  font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.7px;
  padding-bottom: 0.6rem; margin-bottom: 0.7rem; border-bottom: 2px solid;
  display: flex; justify-content: space-between; align-items: center;
}
.k-count {
  font-size: 10px; font-weight: 500; opacity: 0.85;
  border: 1px solid currentColor; border-radius: 9px; padding: 0 5px; letter-spacing: 0;
}

/* === WORKFLOW STEPPER ================================================ */
.stepper {
  display: flex; align-items: center;
  background: #111; border: 1px solid #222;
  border-radius: 8px; padding: 0.9rem 1.3rem;
  margin-bottom: 1.5rem;
}
.step { display:flex; align-items:center; gap:0.5rem; flex:1; min-width:0; }
.step-num {
  width:22px; height:22px; border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  font-size:10px; font-weight:700; flex-shrink:0;
  border: 1.5px solid #444; color: #AAA; background: transparent;
}
.sn-done   { background:#4CAF6B; border-color:#4CAF6B; color:#fff; }
.sn-active { background:#5E6AD2; border-color:#5E6AD2; color:#fff; }
.step-lbl { font-size:12px; font-weight:500; }
.sl-done   { color:#4CAF6B; }
.sl-active { color:#F0F0F0; }
.sl-idle   { color:#BBBBBB; }
.step-line { width:36px; height:1px; background:#222; flex-shrink:0; margin:0 0.5rem; }
.sl-line-done { background:#4CAF6B; }
.sl-line-act  { background:#5E6AD2; }

/* === EMPTY STATE ===================================================== */
.empty {
  background: #111; border: 1px dashed #222;
  border-radius: 9px; padding: 3rem 2rem; text-align: center;
}
.empty-icon { font-size:1.75rem; opacity:0.4; margin-bottom:0.7rem; }
.empty-txt  { font-size:13px; color:#DDDDDD; line-height:1.7; }

/* === DETAIL PANEL ==================================================== */
.det-panel {
  background: #111; border: 1px solid #222;
  border-radius: 9px; padding: 1.2rem 1.4rem; margin-bottom: 1.25rem;
}
.det-title { font-size:15px; font-weight:600; color:#F0F0F0; margin:0 0 0.85rem; letter-spacing:-0.3px; }

/* Info banner */
.info-banner {
  background:#0E1C30; border:1px solid #1E3A60; border-radius:6px;
  padding:0.65rem 1rem; color:#6EB4F7; font-size:13px; margin-bottom:0.9rem;
}
.warn-banner {
  background:#1C1400; border:1px solid #3A2800; border-radius:6px;
  padding:0.65rem 1rem; color:#E0AE40; font-size:13px; margin-bottom:0.9rem;
}

/* === BUTTONS ========================================================= */
.stButton > button {
  background: #1C1C1C !important; border: 1px solid #2A2A2A !important;
  color: #DDDDDD !important; border-radius: 6px !important;
  font-size: 12px !important; font-weight: 500 !important;
  padding: 0.35rem 0.85rem !important; transition: all 0.12s !important;
  letter-spacing: 0.1px !important;
}
.stButton > button:hover {
  background: #242424 !important; border-color: #383838 !important; color: #F0F0F0 !important;
}
.btn-p .stButton > button {
  background: #5E6AD2 !important; border-color: #5E6AD2 !important; color: #fff !important;
}
.btn-p .stButton > button:hover { background: #6B77E0 !important; border-color:#6B77E0 !important; }
.btn-d .stButton > button {
  background: #1E0A0A !important; border-color: #3A1414 !important; color: #F07070 !important;
}
.btn-d .stButton > button:hover { background: #280E0E !important; }
.btn-s .stButton > button {
  background: #0E1E12 !important; border-color: #1A3A20 !important; color: #5DC878 !important;
}
.btn-s .stButton > button:hover { background: #122418 !important; }

/* === INPUTS ========================================================== */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
  background: #111 !important; border: 1px solid #252525 !important;
  border-radius: 6px !important; color: #E0E0E0 !important; font-size: 13px !important;
}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder { color: #777 !important; }
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: #5E6AD2 !important; box-shadow: 0 0 0 2px rgba(94,106,210,0.15) !important;
}
.stTextInput label, .stTextArea label, .stSelectbox label {
  font-size: 11px !important; color: #AAAAAA !important;
  font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.5px;
}
.stSelectbox > div > div {
  background: #111 !important; border: 1px solid #252525 !important;
  border-radius: 6px !important; color: #E0E0E0 !important;
}
.stMultiSelect > div > div {
  background: #111 !important; border: 1px solid #252525 !important;
}

/* === EXPANDERS ======================================================= */
details > summary, .streamlit-expanderHeader {
  background: #161616 !important; border: 1px solid #222 !important;
  border-radius: 6px !important; font-size: 13px !important; color: #DDDDDD !important;
  padding: 0.6rem 1rem !important; font-weight: 500 !important;
}
.streamlit-expanderContent {
  background: #111 !important; border: 1px solid #222 !important;
  border-top: none !important; border-radius: 0 0 6px 6px !important; padding: 1rem !important;
}

/* === MARKDOWN ======================================================== */
.stMarkdown h4 { color:#FFFFFF; font-size:14px; font-weight:600; letter-spacing:-0.2px; margin:0 0 0.65rem; }
.stMarkdown h5 { color:#FFFFFF; font-size:13px; font-weight:600; margin:0 0 0.5rem; }
.stMarkdown p  { color:#FFFFFF !important; font-size:13px; line-height:1.65; }
.stMarkdown li { color:#FFFFFF !important; font-size:13px; line-height:1.7; }
.stMarkdown a  { color:#8FA8F0 !important; }
.stMarkdown code { background:#1E1E1E; color:#FFFFFF; border-radius:3px; padding:1px 5px; font-size:12px; }
.stMarkdown table { font-size:12px; border-collapse:collapse; width:100%; }
.stMarkdown th { background:#1A1A1A; color:#DDDDDD; padding:7px 10px; border:1px solid #2A2A2A; font-weight:600; }
.stMarkdown td { color:#CCCCCC; padding:7px 10px; border:1px solid #222; }
.stMarkdown blockquote { border-left:3px solid #333; margin:0; padding-left:0.85rem; color:#BBBBBB; }
.stMarkdown strong { color:#FFFFFF; }

/* Alerts */
div[data-testid="stInfo"]    { background:#0E1C30 !important; border:1px solid #1E3A60 !important; border-radius:6px !important; }
div[data-testid="stSuccess"] { background:#0E1E12 !important; border:1px solid #1A3A20 !important; border-radius:6px !important; }
div[data-testid="stError"]   { background:#200A0A !important; border:1px solid #3A1414 !important; border-radius:6px !important; }
div[data-testid="stWarning"] { background:#1C1400 !important; border:1px solid #3A2800 !important; border-radius:6px !important; }
div[data-testid="stInfo"] p, div[data-testid="stInfo"] div,
div[data-testid="stInfo"] span    { color:#6EB4F7 !important; font-size:13px !important; }
div[data-testid="stSuccess"] p, div[data-testid="stSuccess"] div,
div[data-testid="stSuccess"] span { color:#5DC878 !important; font-size:13px !important; }
div[data-testid="stError"] p, div[data-testid="stError"] div,
div[data-testid="stError"] span   { color:#F07070 !important; font-size:13px !important; }
div[data-testid="stWarning"] p, div[data-testid="stWarning"] div,
div[data-testid="stWarning"] span { color:#E0AE40 !important; font-size:13px !important; }

/* Download button */
.stDownloadButton > button {
  background: #161616 !important; border: 1px solid #252525 !important;
  color: #DDDDDD !important; border-radius: 6px !important;
  font-size: 12px !important; font-weight: 500 !important;
  transition: all 0.12s !important;
}
.stDownloadButton > button:hover { background: #1E1E1E !important; color: #DDD !important; }

/* Spinner */
.stSpinner > div { border-top-color: #5E6AD2 !important; }

/* Scrollbar */
::-webkit-scrollbar { width:3px; height:3px; }
::-webkit-scrollbar-track { background:#0D0D0D; }
::-webkit-scrollbar-thumb { background:#2A2A2A; border-radius:2px; }
::-webkit-scrollbar-thumb:hover { background:#383838; }

/* Divider */
hr { border-color: #222 !important; margin: 0.9rem 0 !important; }

/* Toast */
div[data-testid="stToast"] { background:#161616 !important; border:1px solid #2A2A2A !important; color:#CCC !important; }

/* ── BUTTON ALIGNMENT FIX ─────────────────────────────────────────────
   Removes phantom height from orphaned HTML div wrappers and aligns
   all buttons in horizontal blocks to the same baseline.           */
[data-testid="stHorizontalBlock"] {
  align-items: center !important;
  gap: 0.5rem !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlock"] {
  padding-top: 0 !important;
  margin-top: 0 !important;
}
/* Collapse empty/orphan markdown containers (open/close div wrappers) */
[data-testid="stMarkdownContainer"] > div:empty { display: none !important; height: 0 !important; }
[data-testid="stVerticalBlock"] > [data-testid="stMarkdownContainer"]:empty { display: none !important; }
/* Remove automatic label-space padding above buttons in columns */
[data-testid="stHorizontalBlock"] .stButton { margin-top: 0 !important; padding-top: 0 !important; }
[data-testid="stHorizontalBlock"] .stButton > button { width: 100% !important; }

/* Primary button via Streamlit type="primary" — uses theme primaryColor */
button[kind="primary"] {
  background: #5E6AD2 !important; border-color: #5E6AD2 !important; color: #fff !important;
}
button[kind="primary"]:hover { background: #6B77E0 !important; border-color: #6B77E0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def ss(k, d=None):
    if k not in st.session_state:
        st.session_state[k] = d
    return st.session_state[k]


ss("wf_step", 1); ss("wf_raw", ""); ss("wf_msg_id", None)
ss("wf_issue_card", None); ss("wf_tech_plan", None)
ss("detail_id", None)


# ── Boot bots ─────────────────────────────────────────────────────────────────
def maybe_start_bots():
    dt = get_cfg("discord_token")
    if dt and not discord_bot.status()["running"]:
        discord_bot.start(dt)
    sbt = get_cfg("slack_bot_token"); sat = get_cfg("slack_app_token")
    if sbt and sat and not slack_bot.status()["running"]:
        slack_bot.start(sbt, sat)

maybe_start_bots()

d_st  = discord_bot.status()
sl_st = slack_bot.status()


# ── Helpers ───────────────────────────────────────────────────────────────────
def pri_pill(p):
    cls = {"critical":"pr-c","high":"pr-h","medium":"pr-m","low":"pr-l"}.get(p,"pr-l")
    return f'<span class="pill {cls}">{p}</span>'

def src_pill(s):
    cls = {"discord":"pd","slack":"ps","manual":"pma"}.get(s,"pma")
    return f'<span class="pill {cls}">{s}</span>'

def lane_pill(l):
    cls = {"issue":"pi","feature":"pf","idea":"pid","marketing":"pm"}.get(l,"pma")
    return f'<span class="pill {cls}">{l}</span>'

def dot_cls(p):
    return {"critical":"dp-c","high":"dp-h","medium":"dp-m","low":"dp-l"}.get(p,"dp-l")


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # Brand
    st.markdown("""
    <div class="brand">
      <div class="brand-mark">P</div>
      <span class="brand-name">Prism</span>
    </div>
    """, unsafe_allow_html=True)

    # API Key input
    api_key = st.text_input(
        "API KEY",
        type="password",
        placeholder="sk-... or gsk_...",
        value=get_cfg("ai_api_key", ""),
    )
    if api_key:
        set_cfg("ai_api_key", api_key)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Navigation
    nav = st.radio(
        "nav",
        ["Signals", "Workspace", "Issue Builder", "Settings"],
        label_visibility="collapsed",
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Stats
    c = board_counts()
    st.markdown('<div class="s-section">Overview</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="s-row">
      <span class="s-lbl">Inbox</span>
      <span class="s-badge">{c.get('pending', 0)}</span>
    </div>
    <div class="s-row">
      <span class="s-lbl">Issues</span>
      <span class="s-badge r">{c.get('issue', 0)}</span>
    </div>
    <div class="s-row">
      <span class="s-lbl">Features</span>
      <span class="s-badge b">{c.get('feature', 0)}</span>
    </div>
    <div class="s-row">
      <span class="s-lbl">Ideas</span>
      <span class="s-badge y">{c.get('idea', 0)}</span>
    </div>
    <div class="s-row">
      <span class="s-lbl">Marketing</span>
      <span class="s-badge g">{c.get('marketing', 0)}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Bot status
    d_on   = d_st["running"]
    sl_on  = sl_st.get("ready", False)
    sl_mid = sl_st["running"] and not sl_on

    d_dot_html  = f'<span class="dot {"dot-on" if d_on else "dot-off"}"></span>'
    sl_dot_html = f'<span class="dot {"dot-on" if sl_on else ("dot-mid" if sl_mid else "dot-off")}"></span>'
    d_cls  = "c-on" if d_on else "c-off"
    sl_cls = "c-on" if sl_on else ("c-mid" if sl_mid else "c-off")
    d_txt  = "Live" if d_on else "Offline"
    sl_txt = "Live" if sl_on else ("Connecting" if sl_mid else "Offline")

    st.markdown(f"""
    <div class="bot-row">
      <span>Discord</span>
      <span class="{d_cls}">{d_dot_html} {d_txt}</span>
    </div>
    <div class="bot-row">
      <span>Slack</span>
      <span class="{sl_cls}">{sl_dot_html} {sl_txt}</span>
    </div>
    """, unsafe_allow_html=True)

    if sl_mid and get_cfg("slack_bot_token"):
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Reconnect Slack", use_container_width=True):
            slack_bot.stop(); time.sleep(0.5)
            slack_bot.start(get_cfg("slack_bot_token"), get_cfg("slack_app_token"))
            st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)

    if st.button("Start Over", use_container_width=True):
        for k in ["wf_step","wf_raw","wf_msg_id","wf_issue_card","wf_tech_plan"]:
            st.session_state[k] = 1 if k=="wf_step" else ("" if k=="wf_raw" else None)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# VIEW — INBOX
# ══════════════════════════════════════════════════════════════════════════════
if nav == "Signals":
    c = board_counts()
    pending = get_messages("pending")

    h1, h2 = st.columns([6, 1])
    with h1:
        st.markdown(f"""
        <div class="pg-header">
          <div>
            <p class="pg-title">Signals</p>
            <p class="pg-sub">{c.get('pending', 0)} unanalyzed signals waiting</p>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Refresh"):
            st.rerun()

    # ── Analyze All ───────────────────────────────────────────────────────────
    if pending:
        has_context = bool(get_cfg("product_brief", "").strip() or get_context_files())

        if not has_context:
            st.markdown(
                '<div class="warn-banner">⚠️ No product context set — Analyze All will classify every signal without relevance filtering. '
                'Add a product brief in <b>Settings → Knowledge Base</b> to enable smart filtering.</div>',
                unsafe_allow_html=True,
            )

        aa1, aa2, _ = st.columns([3, 3, 4])
        with aa1:
            analyze_all_clicked = st.button(
                f"⚡ Analyze All ({len(pending)})",
                use_container_width=True,
                type="primary",
                key="analyze_all_btn",
            )
        with aa2:
            if has_context:
                filter_mode = st.checkbox("Filter irrelevant messages", value=True, key="aa_filter")
            else:
                filter_mode = False

        if analyze_all_clicked:
            if not api_key:
                st.error("Add API key in the sidebar first.")
            else:
                ctx = build_context_str()
                total    = len(pending)
                analyzed = 0
                skipped  = 0
                errors   = 0

                progress_bar = st.progress(0, text="Starting…")
                status_box   = st.empty()

                for i, msg in enumerate(pending):
                    progress_bar.progress((i) / total, text=f"Processing {i+1} of {total}…")
                    status_box.markdown(
                        f'<div class="info-banner" style="font-size:12px">'
                        f'<b>{i+1}/{total}</b> — analyzing signal from <b>{msg.get("author","unknown")}</b>…'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    try:
                        # ── Relevance check (only if context + filter enabled) ──
                        if filter_mode and ctx:
                            rel = check_relevance(msg["content"], api_key, ctx)
                            if not rel["relevant"]:
                                set_status(msg["id"], "irrelevant")
                                skipped += 1
                                continue

                        # ── Classify + process ────────────────────────────────
                        result = auto_process(msg["content"], api_key, ctx)
                        save_auto_process_result(
                            msg["id"], result["lane"], result["title"],
                            result["summary"], result["priority"],
                            result["output"] or "", result["needs_review"],
                        )
                        analyzed += 1

                    except Exception as e:
                        errors += 1

                progress_bar.progress(1.0, text="Done!")
                status_box.empty()

                # Summary
                parts = [f"✅ **{analyzed}** classified to board"]
                if skipped:
                    parts.append(f"🚫 **{skipped}** marked irrelevant")
                if errors:
                    parts.append(f"⚠️ **{errors}** failed")
                st.success(" · ".join(parts))
                st.rerun()

        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

    with st.expander("Add signal manually"):
        # ── Tabs: Paste vs Upload ──────────────────────────────────────────
        tab_paste, tab_upload = st.tabs(["Paste Text", "Upload File"])

        with tab_paste:
            manual = st.text_area(
                "CONTENT",
                height=85,
                placeholder="Paste a WhatsApp message, email snippet, call note, or any raw feedback…",
                key="manual_paste",
            )
            m1, m2, _ = st.columns([2, 2, 6])
            with m1:
                if st.button("Add Signal", use_container_width=True, type="primary"):
                    if manual.strip():
                        ok = add_message("manual", manual.strip(), channel="manual", author="You")
                        st.success("Added.") if ok else st.warning("Duplicate detected.")
                        st.rerun()

        with tab_upload:
            st.markdown(
                '<div style="font-size:12px;color:#AAAAAA;margin-bottom:0.6rem">'
                'Export your chat (WhatsApp, Slack, Telegram, CSV, plain text) and upload here. '
                'Prism will <b>parse each message as a separate signal</b> automatically.</div>',
                unsafe_allow_html=True,
            )

            def parse_chat_messages(filename: str, content: str) -> list[dict]:
                """
                Parse a chat export into individual messages.
                Supports (in detection order):
                  - DiscordChatExporter JSON  : messages[].author.name + content
                  - DiscordChatExporter HTML  : .chatlog__message-container DOM
                  - Telegram HTML             : from_name + .text div classes
                  - WhatsApp HTML             : data-pre-plain-text attribute
                  - WhatsApp .txt             : [DD/MM/YYYY, HH:MM:SS] Author: text
                  - CSV                       : auto-detect message/author columns
                  - Plain .txt/.md            : blank-line-separated blocks
                """
                import re, io, csv as _csv, json as _json
                from bs4 import BeautifulSoup

                messages = []
                fname_lower = filename.lower()

                # ── DiscordChatExporter JSON (-f Json) ───────────────────────
                if fname_lower.endswith(".json"):
                    try:
                        data = _json.loads(content)
                        msgs = data.get("messages", [])
                        if msgs:
                            for m in msgs:
                                text = (m.get("content") or "").strip()
                                author = (m.get("author") or {}).get("name", "unknown")
                                # skip empty, bot system messages, and pure attachment messages
                                if text and m.get("type", "Default") == "Default":
                                    messages.append({"author": author, "text": text, "source": "discord"})
                            return messages
                    except Exception:
                        pass

                # ── HTML-based exports ───────────────────────────────────────
                if fname_lower.endswith((".html", ".htm")):
                    soup = BeautifulSoup(content, "html.parser")

                    # ── DiscordChatExporter HTML (-f HtmlDark / HtmlLight) ────
                    # Structure: div.chatlog__message-container
                    #              span.chatlog__author  (may be absent for grouped msgs)
                    #              div.chatlog__content  (the message text)
                    discord_containers = soup.select(".chatlog__message-container")
                    if len(discord_containers) >= 3:
                        last_author = "unknown"
                        for container in discord_containers:
                            author_el = container.select_one(".chatlog__author")
                            if author_el:
                                last_author = author_el.get_text(strip=True)
                            content_el = container.select_one(".chatlog__content")
                            if not content_el:
                                # some versions use .chatlog__message
                                content_el = container.select_one(".chatlog__message")
                            if content_el:
                                # remove nested reply-previews, embeds, reactions
                                for tag in content_el.select(".chatlog__reply, .chatlog__embed, .chatlog__reactions"):
                                    tag.decompose()
                                text = content_el.get_text(separator=" ", strip=True)
                                if text and len(text) > 1:
                                    messages.append({"author": last_author, "text": text, "source": "discord"})
                        if len(messages) >= 3:
                            return messages

                    # ── Telegram HTML export ──────────────────────────────────
                    # Structure: div.message  >  div.from_name + div.text
                    tg_messages = soup.select("div.message")
                    if len(tg_messages) >= 3:
                        last_author = "unknown"
                        for msg in tg_messages:
                            name_el = msg.select_one(".from_name")
                            if name_el:
                                last_author = name_el.get_text(strip=True)
                            text_el = msg.select_one(".text")
                            if text_el:
                                text = text_el.get_text(separator=" ", strip=True)
                                if text:
                                    messages.append({"author": last_author, "text": text, "source": "telegram"})
                        if len(messages) >= 3:
                            return messages

                    # ── WhatsApp HTML — data-pre-plain-text attribute ─────────
                    wa_els = soup.select("[data-pre-plain-text]")
                    if len(wa_els) >= 3:
                        for el in wa_els:
                            meta = el.get("data-pre-plain-text", "")
                            # meta format: "[date, time] Author: "
                            author_match = re.search(r'\]\s*(.+?):\s*$', meta)
                            author = author_match.group(1).strip() if author_match else "unknown"
                            copyable = el.select_one(".copyable-text, .selectable-text")
                            text = (copyable or el).get_text(separator=" ", strip=True)
                            if text and not re.match(r'^(image|video|audio|sticker|document|Media) omitted$', text, re.I):
                                messages.append({"author": author, "text": text, "source": "whatsapp"})
                        if len(messages) >= 3:
                            return messages

                    # ── Generic HTML fallback — strip tags, dedupe by line ────
                    for tag in soup(["script", "style", "head"]):
                        tag.decompose()
                    lines = soup.get_text(separator="\n").split("\n")
                    seen = set()
                    for line in lines:
                        line = line.strip()
                        if len(line) < 12 or line in seen:
                            continue
                        if re.match(r'^[\d\s:\/\-,\.]+$', line):
                            continue
                        seen.add(line)
                        messages.append({"author": filename, "text": line, "source": "html-upload"})
                    return messages

                # ── WhatsApp .txt export ─────────────────────────────────────
                wa_pattern = re.compile(
                    r'[\[‎]?(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})[,\s]+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\]?\s*[-–]?\s*([^:]+?):\s(.+)',
                    re.IGNORECASE,
                )
                wa_matches = wa_pattern.findall(content)
                if len(wa_matches) >= 3:
                    for date, time_, author, text in wa_matches:
                        text = text.strip()
                        if text and not re.match(r'^<?(Media|image|video|audio|sticker|document) omitted>?$', text, re.I):
                            messages.append({"author": author.strip(), "text": text, "source": "whatsapp"})
                    return messages

                # ── CSV export ───────────────────────────────────────────────
                if fname_lower.endswith(".csv"):
                    try:
                        reader = _csv.DictReader(io.StringIO(content))
                        msg_col = next(
                            (c for c in (reader.fieldnames or [])
                             if c.lower() in ("message", "text", "content", "body", "msg")),
                            None,
                        )
                        author_col = next(
                            (c for c in (reader.fieldnames or [])
                             if c.lower() in ("author", "from", "sender", "user", "name", "username")),
                            None,
                        )
                        if msg_col:
                            for row in reader:
                                text = (row.get(msg_col) or "").strip()
                                author = (row.get(author_col) or "unknown").strip() if author_col else "unknown"
                                if text:
                                    messages.append({"author": author, "text": text, "source": "csv"})
                            return messages
                    except Exception:
                        pass

                # ── Plain text / Markdown: blank-line-separated blocks ───────
                blocks = re.split(r'\n{2,}', content)
                for block in blocks:
                    block = block.strip()
                    if block and len(block) > 10:
                        messages.append({"author": filename, "text": block, "source": "file-upload"})
                return messages

            uploaded_signals = st.file_uploader(
                "FILES",
                type=["txt", "md", "csv", "html", "htm", "json"],
                accept_multiple_files=True,
                key="signal_uploader",
            )

            if uploaded_signals:
                all_messages = []
                for f in uploaded_signals:
                    try:
                        content = f.read().decode("utf-8", errors="replace").strip()
                        msgs = parse_chat_messages(f.name, content)
                        for m in msgs:
                            m["file"] = f.name
                        all_messages.extend(msgs)
                    except Exception as e:
                        st.error(f"Could not read {f.name}: {e}")

                if all_messages:
                    st.markdown(
                        f'<div class="info-banner">Detected <b>{len(all_messages)} messages</b> across '
                        f'{len(uploaded_signals)} file(s). Preview below — click confirm to add all as signals.</div>',
                        unsafe_allow_html=True,
                    )

                    # Preview table — show first 10
                    preview_msgs = all_messages[:10]
                    rows_html = "".join(
                        f'<tr>'
                        f'<td style="color:#AAAAAA;font-size:11px;white-space:nowrap;padding:5px 8px">{m["author"]}</td>'
                        f'<td style="color:#DDDDDD;font-size:11px;padding:5px 8px">{m["text"][:120]}{"…" if len(m["text"])>120 else ""}</td>'
                        f'<td style="color:#888;font-size:10px;padding:5px 8px;white-space:nowrap">{m["file"]}</td>'
                        f'</tr>'
                        for m in preview_msgs
                    )
                    more = f'<tr><td colspan="3" style="color:#888;font-size:11px;padding:5px 8px">… and {len(all_messages)-10} more messages</td></tr>' if len(all_messages) > 10 else ""
                    st.markdown(
                        f'<table style="width:100%;border-collapse:collapse;background:#111;border:1px solid #222;border-radius:6px;overflow:hidden">'
                        f'<thead><tr>'
                        f'<th style="color:#888;font-size:10px;text-align:left;padding:6px 8px;border-bottom:1px solid #222;text-transform:uppercase">Author</th>'
                        f'<th style="color:#888;font-size:10px;text-align:left;padding:6px 8px;border-bottom:1px solid #222;text-transform:uppercase">Message</th>'
                        f'<th style="color:#888;font-size:10px;text-align:left;padding:6px 8px;border-bottom:1px solid #222;text-transform:uppercase">File</th>'
                        f'</tr></thead><tbody>{rows_html}{more}</tbody></table>',
                        unsafe_allow_html=True,
                    )

                    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                    u1, _, u2 = st.columns([3, 1, 6])
                    with u1:
                        if st.button(f"Add {len(all_messages)} Signal(s)", use_container_width=True, type="primary", key="upload_add"):
                            added_count = 0
                            for m in all_messages:
                                ok = add_message(m["source"], m["text"], channel=m["file"], author=m["author"])
                                if ok:
                                    added_count += 1
                            if added_count:
                                st.success(f"Added {added_count} signal(s) from chat export.")
                            else:
                                st.warning("All messages were duplicates — no new signals added.")
                            st.rerun()
                else:
                    st.warning("No messages could be parsed from the uploaded file(s). Check the format.")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    if not pending:
        st.markdown("""
        <div class="empty">
          <div class="empty-icon">📭</div>
          <div class="empty-txt">
            No signals yet.<br>Connect Discord or Slack in <b>Settings</b>, or add one manually above.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in pending:
            src = msg["source"]
            ch  = msg.get("channel", "—")
            au  = msg.get("author", "—")
            ts  = (msg.get("created_at", "") or "")[:16].replace("T", " ")
            preview = msg["content"][:200] + ("…" if len(msg["content"]) > 200 else "")

            st.markdown(f"""
            <div class="card">
              <div class="card-row">
                {src_pill(src)}
                <span style="font-size:11px;color:#AAAAAA">#{ch} · {au}</span>
              </div>
              <div class="card-body">{preview}</div>
              <div class="card-foot"><span>{ts}</span></div>
            </div>
            """, unsafe_allow_html=True)

            a1, a2, _ = st.columns([3, 2, 8])
            with a1:
                if st.button("Analyze Signal", key=f"ap_{msg['id']}", use_container_width=True, type="primary"):
                    if not api_key:
                        st.error("Add API key in the sidebar first.")
                    else:
                        with st.spinner("Classifying & processing…"):
                            try:
                                result = auto_process(msg["content"], api_key, build_context_str())
                                save_auto_process_result(
                                    msg["id"], result["lane"], result["title"],
                                    result["summary"], result["priority"],
                                    result["output"] or "", result["needs_review"],
                                )
                                if result["needs_review"]:
                                    st.session_state.wf_raw = msg["content"]
                                    st.session_state.wf_msg_id = msg["id"]
                                    st.session_state.wf_issue_card = result["output"]
                                    st.session_state.wf_step = 2
                                    st.toast("Issue detected → review in Issue Builder", icon="⚡")
                                else:
                                    meta = LANE_META.get(result["lane"], {})
                                    st.toast(f"→ {meta.get('label', result['lane'])} · {result['title']}", icon="✅")
                                st.rerun()
                            except Exception as e:
                                st.error(f"{e}")
            with a2:
                if st.button("Ignore", key=f"dis_{msg['id']}", use_container_width=True):
                    set_status(msg["id"], "dismissed")
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# VIEW — BOARD
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Workspace":
    counts = board_counts()
    total  = sum(counts.get(l, 0) for l in ["issue","feature","idea","marketing"])

    st.markdown(f"""
    <div class="pg-header">
      <div>
        <p class="pg-title">Workspace</p>
        <p class="pg-sub">{total} items across Issues, Features, Ideas, and Marketing</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Filters
    f1, f2, f3 = st.columns([2, 2, 6])
    with f1:
        lane_filter = st.selectbox(
            "LANE",
            ["all", "issue", "feature", "idea", "marketing"],
            format_func=lambda x: "All Lanes" if x == "all"
                else f"{LANE_META[x]['emoji']} {LANE_META[x]['label']}",
        )
    with f2:
        status_filter = st.selectbox(
            "STATUS",
            ["all","new","confirmed","evaluated","interesting","saved",
             "in_progress","backlog","parked","resolved","shipped","used","promoted"],
            format_func=lambda x: "All Statuses" if x == "all" else x.replace("_"," ").title(),
        )

    # Detail panel
    if st.session_state.detail_id:
        item = get_item(st.session_state.detail_id)
        if item:
            meta = LANE_META.get(item["lane"], LANE_META["unclassified"])
            st.markdown(f"""
            <div class="det-panel">
              <p class="det-title">{meta['emoji']} {item.get('title', 'Untitled')}</p>
            </div>
            """, unsafe_allow_html=True)

            d1, d2, d3 = st.columns(3)
            with d1:
                new_lane = st.selectbox(
                    "LANE", list(LANE_META.keys())[:4],
                    index=list(LANE_META.keys()).index(item["lane"])
                        if item["lane"] in LANE_META else 0,
                    format_func=lambda x: f"{LANE_META[x]['emoji']} {LANE_META[x]['label']}",
                    key="det_lane",
                )
            with d2:
                statuses = LANE_STATUSES.get(new_lane, ["new"])
                cur_st   = item["item_status"] if item["item_status"] in statuses else statuses[0]
                new_st   = st.selectbox(
                    "STATUS", statuses,
                    index=statuses.index(cur_st),
                    format_func=lambda x: x.replace("_"," ").title(),
                    key="det_status",
                )
            with d3:
                new_pri = st.selectbox(
                    "PRIORITY", ["critical","high","medium","low"],
                    index=["critical","high","medium","low"].index(item.get("priority","medium")),
                    key="det_priority",
                )

            notes = st.text_area("NOTES", value=item.get("notes",""), height=70, key="det_notes")

            b1, b2, b3, b4, _ = st.columns([2,2,2,2,4])
            with b1:
                if st.button("Save", key="det_save", use_container_width=True, type="primary"):
                    update_item(item["id"], lane=new_lane, item_status=new_st, priority=new_pri, notes=notes)
                    st.success("Saved."); st.rerun()
            with b2:
                if st.button("Run Pipeline", key="det_proc", use_container_width=True):
                    st.session_state.wf_raw = item["content"]
                    st.session_state.wf_msg_id = item["id"]
                    st.session_state.wf_step = 2 if item.get("issue_card") else 1
                    st.session_state.wf_issue_card = item.get("issue_card")
                    st.session_state.wf_tech_plan  = item.get("tech_plan")
                    st.toast("Loaded → switch to Workflow", icon="⚡"); st.rerun()
            with b3:
                if st.button("Close", key="det_close", use_container_width=True):
                    st.session_state.detail_id = None; st.rerun()
            with b4:
                if st.button("Delete", key="det_del", use_container_width=True):
                    delete_item(item["id"]); st.session_state.detail_id = None; st.rerun()

            # Output section
            lane_i = item.get("lane","")
            has_output = item.get("lane_output") or item.get("issue_card") or item.get("tech_plan")
            if has_output:
                output_labels = {
                    "issue":"Issue Card & Plan","feature":"Feature Brief",
                    "idea":"Idea Brief","marketing":"Marketing Signal",
                }
                with st.expander(f"View {output_labels.get(lane_i,'Output')}"):
                    if lane_i == "issue":
                        if item.get("issue_card"):
                            st.markdown("**Issue Card**"); st.markdown(item["issue_card"])
                        if item.get("tech_plan"):
                            st.markdown("---"); st.markdown("**Implementation Plan**"); st.markdown(item["tech_plan"])
                        if item.get("needs_review") and not item.get("tech_plan"):
                            st.markdown('<div class="info-banner">Awaiting human review — click Run Pipeline to continue.</div>', unsafe_allow_html=True)
                    else:
                        if item.get("lane_output"):
                            st.markdown(item["lane_output"])
            st.markdown("<hr/>", unsafe_allow_html=True)

    # Kanban columns
    lanes_to_show = ["issue","feature","idea","marketing"] if lane_filter == "all" else [lane_filter]
    cols = st.columns(len(lanes_to_show), gap="small")

    lane_colors = {"issue":"#E05252","feature":"#4B9EE8","idea":"#D29922","marketing":"#4CAF6B"}

    for i, lane in enumerate(lanes_to_show):
        meta  = LANE_META[lane]
        color = lane_colors.get(lane, "#555")
        items = get_items(
            lane=lane,
            item_status=None if status_filter == "all" else status_filter,
        )
        with cols[i]:
            st.markdown(f"""
            <div class="k-col">
              <div class="k-header" style="color:{color};border-bottom-color:{color}22">
                {meta['emoji']} {meta['label'].upper()}
                <span class="k-count">{len(items)}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if not items:
                st.markdown('<div style="text-align:center;padding:1.5rem 0;font-size:11px;color:#888">Empty</div>', unsafe_allow_html=True)
            else:
                for item in items:
                    pri   = item.get("priority","medium")
                    ist   = item.get("item_status","new")
                    src   = item.get("source","manual")
                    title = item.get("title") or item["content"][:50]
                    summ  = item.get("summary","")
                    ts    = (item.get("created_at","") or "")[:10]

                    st.markdown(f"""
                    <div class="card">
                      <div class="card-row">
                        <span class="dot-pri {dot_cls(pri)}" style="width:7px;height:7px;border-radius:50%;display:inline-block;background:{'#FF4444' if pri=='critical' else '#E5A633' if pri=='high' else '#4B9EE8' if pri=='medium' else '#333'}"></span>
                        {src_pill(src)}
                        <span style="font-size:10px;color:#AAAAAA;margin-left:auto">{ist.replace('_',' ')}</span>
                      </div>
                      <div class="card-title">{title}</div>
                      <div class="card-body" style="font-size:11px">{summ[:90]}</div>
                      <div class="card-foot"><span>{ts}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("Open", key=f"open_{item['id']}", use_container_width=True):
                        st.session_state.detail_id = item["id"]
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# VIEW — WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Issue Builder":
    step = st.session_state.wf_step

    st.markdown(f"""
    <div class="pg-header">
      <div>
        <p class="pg-title">Workflow</p>
        <p class="pg-sub">3-step pipeline — interpret · review · plan</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Stepper
    def s_num(n):
        if step > n:  return f'<div class="step-num sn-done">✓</div>'
        if step == n: return f'<div class="step-num sn-active">{n}</div>'
        return f'<div class="step-num">{n}</div>'

    def s_lbl(n, txt):
        cls = "sl-done" if step > n else ("sl-active" if step == n else "sl-idle")
        return f'<span class="step-lbl {cls}">{txt}</span>'

    def s_line(n):
        cls = "sl-line-done" if step > n else ("sl-line-act" if step == n else "")
        return f'<div class="step-line {cls}"></div>'

    st.markdown(f"""
    <div class="stepper">
      <div class="step">
        {s_num(1)}
        {s_lbl(1, "Raw Feedback")}
      </div>
      {s_line(1)}
      <div class="step">
        {s_num(2)}
        {s_lbl(2, "Issue Card")}
      </div>
      {s_line(2)}
      <div class="step">
        {s_num(3)}
        {s_lbl(3, "Implementation Plan")}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Context status
    ctx_files = get_context_files()
    ctx_brief = get_cfg("product_brief", "").strip()
    ctx_count = len(ctx_files) + (1 if ctx_brief else 0)
    if ctx_count:
        ctx_char  = sum(f["char_count"] for f in ctx_files) + len(ctx_brief)
        st.markdown(f'<div class="info-banner">Knowledge base active — {ctx_count} source(s) · ~{ctx_char:,} chars injected into every agent run.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn-banner">No knowledge base configured — agents have no product or codebase context. Add a brief or files in <b>Setup → Knowledge Base</b>.</div>', unsafe_allow_html=True)

    # ── Step 1 ────────────────────────────────────────────────────────────────
    if step == 1:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("#### Paste Feedback")

            if st.session_state.wf_raw:
                st.markdown('<div class="info-banner">Message loaded and ready to interpret.</div>', unsafe_allow_html=True)

            e1, e2 = st.columns(2)
            with e1:
                if st.button("Bug example", use_container_width=True):
                    st.session_state.wf_raw = "yo the app crashes every time i upload a PDF bigger than 5mb. happened 3 times today"
                    st.rerun()
            with e2:
                if st.button("Feature example", use_container_width=True):
                    st.session_state.wf_raw = "can we add dark mode? users keep asking in reviews and it'd help night usage"
                    st.rerun()

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            raw = st.text_area(
                "FEEDBACK",
                value=st.session_state.wf_raw,
                height=200,
                key="wf_raw_input",
                placeholder="Paste any raw feedback — Slack message, WhatsApp, email, call notes…",
            )

            if st.button("Run Issue Interpreter →", use_container_width=True, type="primary"):
                rv = st.session_state.get("wf_raw_input", "").strip()
                if not rv:
                    st.warning("Paste some feedback first.")
                elif not api_key:
                    st.error("Add API key in the sidebar.")
                else:
                    with st.spinner("Interpreting…"):
                        try:
                            g = build_interpreter_graph(api_key, build_context_str())
                            r = g.invoke({"raw_feedback":rv,"issue_card":None,"error":None,"debug":None})
                            if r["error"]:
                                st.error(r["error"])
                            else:
                                st.session_state.wf_raw = rv
                                st.session_state.wf_issue_card = r["issue_card"]
                                if st.session_state.wf_msg_id:
                                    save_issue_card(st.session_state.wf_msg_id, r["issue_card"])
                                st.session_state.wf_step = 2
                                st.rerun()
                        except Exception as e:
                            st.error(f"{e}")

        with col2:
            st.markdown("#### Pipeline Agents")
            st.markdown("""
            <div class="card" style="padding:1rem 1.1rem">
              <div class="card-title" style="margin-bottom:0.5rem">Issue Interpreter</div>
              <div class="card-body">Classifies, extracts acceptance criteria, edge cases, and priority from raw text.</div>
            </div>
            <div class="card" style="padding:1rem 1.1rem">
              <div class="card-title" style="margin-bottom:0.5rem">You</div>
              <div class="card-body">Review and edit the structured issue card before sending to the architect.</div>
            </div>
            <div class="card" style="padding:1rem 1.1rem">
              <div class="card-title" style="margin-bottom:0.5rem">Implementation Architect</div>
              <div class="card-body">Generates a build-ready plan: affected areas, implementation steps, test checklist, and risk table.</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Step 2 ────────────────────────────────────────────────────────────────
    elif step == 2:
        st.markdown('<div class="info-banner">Review and edit the issue card below — approve to generate the implementation plan.</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown("#### Edit Issue Card")
            edited = st.text_area(
                "CARD",
                value=st.session_state.wf_issue_card or "",
                height=440,
                key="wf_card_edit",
            )
            if st.button("Approve → Build Plan", use_container_width=True, type="primary"):
                card = st.session_state.get("wf_card_edit","").strip()
                if not card:
                    st.warning("Card is empty.")
                elif not api_key:
                    st.error("Add API key in the sidebar.")
                else:
                    st.session_state.wf_issue_card = card
                    with st.spinner("Building implementation plan…"):
                        try:
                            g = build_architect_graph(api_key, build_context_str())
                            r = g.invoke({"issue_card":card,"tech_plan":None,"error":None,"debug":None})
                            if r["error"]:
                                st.error(r["error"])
                            else:
                                st.session_state.wf_tech_plan = r["tech_plan"]
                                mid = st.session_state.wf_msg_id
                                if mid:
                                    save_tech_plan(mid, r["tech_plan"])
                                st.session_state.wf_step = 3
                                st.rerun()
                        except Exception as e:
                            st.error(f"{e}")

        with col2:
            st.markdown("#### Preview")
            st.markdown(st.session_state.wf_issue_card or "")

    # ── Step 3 ────────────────────────────────────────────────────────────────
    elif step == 3:
        st.markdown('<div class="info-banner">Pipeline complete — review your outputs and download.</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown("#### Issue Card")
            st.markdown(st.session_state.wf_issue_card or "")
            st.download_button(
                "Download Issue Card (.md)",
                data=st.session_state.wf_issue_card or "",
                file_name="issue-card.md", mime="text/markdown",
                use_container_width=True,
            )
        with col2:
            st.markdown("#### Implementation Plan")
            st.markdown(st.session_state.wf_tech_plan or "")
            st.download_button(
                "Download Plan (.md)",
                data=st.session_state.wf_tech_plan or "",
                file_name="implementation-plan.md", mime="text/markdown",
                use_container_width=True,
            )

        st.markdown("<hr/>", unsafe_allow_html=True)
        bundle = f"# Prism Bundle\n\n{st.session_state.wf_issue_card}\n\n---\n\n{st.session_state.wf_tech_plan}"
        st.download_button(
            "Download Full Bundle (.md)",
            data=bundle, file_name="prism-bundle.md", mime="text/markdown",
        )


# ══════════════════════════════════════════════════════════════════════════════
# VIEW — SETUP
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Settings":
    st.markdown("""
    <div class="pg-header">
      <div>
        <p class="pg-title">Setup</p>
        <p class="pg-sub">Configure Discord, Slack, and AI integrations</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Discord
    with st.expander("Discord Bot", expanded=not d_st["running"]):
        st.markdown("""
        1. [discord.com/developers/applications](https://discord.com/developers/applications) → New App → Bot → copy **Bot Token**
        2. Enable **Message Content Intent** under Privileged Gateway Intents
        3. Invite with `bot` + `Read Messages` scope
        4. Right-click channel → **Copy Channel ID** (enable Developer Mode in settings)
        """)
        dc_token    = st.text_input("BOT TOKEN", value=get_cfg("discord_token",""), type="password", key="dc_tok")
        dc_channels = st.text_input("WATCH CHANNELS (names or IDs, comma-separated)", value=get_cfg("discord_channels",""), key="dc_ch")

        ca, cb = st.columns(2)
        with ca:
            if st.button("Save & Connect Discord", use_container_width=True, type="primary"):
                set_cfg("discord_token", dc_token); set_cfg("discord_channels", dc_channels)
                discord_bot.stop(); time.sleep(0.5); discord_bot.start(dc_token)
                st.success("Discord connecting…"); st.rerun()
        with cb:
            if st.button("Disconnect", key="dc_dis", use_container_width=True):
                discord_bot.stop(); set_cfg("discord_token",""); st.rerun()

        if d_st["running"]:
            st.success(f"Connected · watching `{get_cfg('discord_channels','all')}`")
        elif d_st["error"]:
            st.error(d_st["error"])

    # Slack
    with st.expander("Slack Bot (Socket Mode)", expanded=not sl_st.get("ready")):
        st.markdown("""
        1. [api.slack.com/apps](https://api.slack.com/apps) → Create App → From Scratch
        2. **Socket Mode** → Enable → Generate **App-Level Token** (`xapp-`, scope: `connections:write`)
        3. **OAuth & Permissions** → scopes: `channels:history`, `groups:history`, `channels:read`, `groups:read` → Install → copy **Bot Token** (`xoxb-`)
        4. **Event Subscriptions** → Enable → Subscribe to `message.channels`, `message.groups`
        5. In your Slack channel type `/invite @your-bot-name`
        """)
        sl_bot_token = st.text_input("BOT TOKEN (xoxb-...)", value=get_cfg("slack_bot_token",""), type="password", key="sl_bt")
        sl_app_token = st.text_input("APP-LEVEL TOKEN (xapp-...)", value=get_cfg("slack_app_token",""), type="password", key="sl_at")

        if sl_bot_token and st.button("Fetch channels"):
            chs, err = slack_bot.list_channels(sl_bot_token)
            if chs:
                st.session_state["slack_ch_list"] = chs
                st.success(f"Found {len(chs)} channels.")
            else:
                st.error(f"{err}")

        if "slack_ch_list" in st.session_state:
            opts = [f"#{c['name']} ({c['id']})" for c in st.session_state["slack_ch_list"]]
            sel  = st.multiselect("SELECT CHANNELS TO WATCH", opts)
            if sel:
                ids = [s.split("(")[1].rstrip(")") for s in sel]
                set_cfg("slack_channels", ",".join(ids))

        sl_channels = st.text_input("OR ENTER CHANNEL IDS MANUALLY", value=get_cfg("slack_channels",""), key="sl_ch_m")

        sa, sb = st.columns(2)
        with sa:
            if st.button("Save & Connect Slack", use_container_width=True, type="primary"):
                set_cfg("slack_bot_token", sl_bot_token); set_cfg("slack_app_token", sl_app_token)
                set_cfg("slack_channels", sl_channels)
                slack_bot.stop(); time.sleep(0.5); slack_bot.start(sl_bot_token, sl_app_token)
                st.success("Slack connecting…"); st.rerun()
        with sb:
            if st.button("Disconnect", key="sl_dis", use_container_width=True):
                slack_bot.stop(); set_cfg("slack_bot_token",""); st.rerun()

        if sl_st.get("ready"):
            st.success(f"Connected · watching `{get_cfg('slack_channels','all')}`")
        elif sl_st["error"]:
            st.error(sl_st["error"])

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ── Knowledge Base ────────────────────────────────────────────────────────
    st.markdown("#### Knowledge Base")
    st.markdown('<div class="info-banner">Everything here is injected into every agent run — the agents will reference your product and codebase to produce grounded, accurate output.</div>', unsafe_allow_html=True)

    # Product brief
    with st.expander("Product / Project Brief", expanded=not bool(get_cfg("product_brief","").strip())):
        st.markdown("Describe your product: what it does, who the users are, tech stack, key architecture decisions, naming conventions. The more specific, the better.")
        brief_val = st.text_area(
            "BRIEF",
            value=get_cfg("product_brief", ""),
            height=180,
            placeholder="e.g. Prism is a team signal intake tool built with Streamlit + LangGraph. It classifies Slack/Discord messages into Issues, Features, Ideas, and Marketing lanes. Backend uses SQLite. The agents are in agent.py, data layer in queue_store.py...",
            key="product_brief_input",
        )
        kb1, kb2, _ = st.columns([2, 2, 6])
        with kb1:
            if st.button("Save Brief", use_container_width=True, type="primary"):
                set_cfg("product_brief", brief_val)
                st.success("Brief saved.")
        with kb2:
            if st.button("Clear Brief", use_container_width=True):
                set_cfg("product_brief", "")
                st.rerun()

    # File upload
    with st.expander("Upload Files (code, docs, architecture)", expanded=False):
        st.markdown("Upload `.py`, `.ts`, `.js`, `.md`, `.txt`, `.json`, `.yaml` files. Each file is stored and injected into agent prompts. Cap: 60k chars per file.")
        uploaded = st.file_uploader(
            "FILES",
            type=["py", "ts", "js", "tsx", "jsx", "md", "txt", "json", "yaml", "yml", "toml"],
            accept_multiple_files=True,
            key="ctx_uploader",
        )
        if uploaded:
            if st.button("Add to Knowledge Base", use_container_width=True, type="primary"):
                added = 0
                for f in uploaded:
                    try:
                        content = f.read().decode("utf-8", errors="replace")
                        ext = f.name.rsplit(".", 1)[-1] if "." in f.name else "text"
                        add_context_file(f.name, content, ext)
                        added += 1
                    except Exception as e:
                        st.error(f"Failed to add {f.name}: {e}")
                if added:
                    st.success(f"Added {added} file(s) to knowledge base.")
                    st.rerun()

    # List existing context files
    ctx_files = get_context_files()
    if ctx_files:
        st.markdown(f"**{len(ctx_files)} file(s) in knowledge base**")
        for cf in ctx_files:
            fa, fb = st.columns([7, 1])
            with fa:
                ts = (cf.get("created_at","") or "")[:10]
                st.markdown(
                    f'<div class="card" style="padding:0.5rem 0.9rem;margin-bottom:0.3rem">'
                    f'<span style="font-size:13px;color:#FFFFFF;font-weight:500">{cf["filename"]}</span>'
                    f'<span style="font-size:11px;color:#888;margin-left:0.75rem">{cf["char_count"]:,} chars · {cf["file_type"]} · {ts}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with fb:
                if st.button("✕", key=f"del_ctx_{cf['id']}", use_container_width=True):
                    delete_context_file(cf["id"])
                    st.rerun()
    else:
        st.markdown('<div style="font-size:12px;color:#888;padding:0.5rem 0">No files uploaded yet.</div>', unsafe_allow_html=True)

    # Context preview
    with st.expander("Preview injected context", expanded=False):
        ctx_preview = build_context_str()
        if ctx_preview:
            st.markdown(f"**~{len(ctx_preview):,} chars** will be injected into each agent.")
            st.code(ctx_preview[:3000] + ("\n... (truncated for preview)" if len(ctx_preview) > 3000 else ""), language="markdown")
        else:
            st.markdown('<div style="font-size:12px;color:#888">No context configured yet.</div>', unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Danger zone
    st.markdown("#### Danger Zone")
    if st.button("Clear all dismissed items"):
        with __import__("queue_store")._conn() as cx:
            cx.execute("DELETE FROM messages WHERE status='dismissed'")
        st.success("Cleared.")
        st.rerun()
