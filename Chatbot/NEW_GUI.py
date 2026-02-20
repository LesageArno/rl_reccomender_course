import os
import sys
import yaml
import streamlit as st
import pdfplumber

from collections import defaultdict

# Ensure project root is on sys.path so that `UIR` and `Chatbot` can be imported
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)  # parent of Chatbot/, i.e. rl_reccomender_course

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from UIR.Scripts.Dataset import Dataset
from Chatbot.chat_handler import ChatHandler
from Chatbot.state import PrefState
from Chatbot.Embeddings.skill_search import SkillSearcher
from Chatbot.data_loader import initialize_all_data


# ---------------------------
#  SPACE THEME (CSS)
# ---------------------------
SPACE_CSS = """
<style>
/* =========================
   BASE / LAYOUT
   ========================= */
.block-container { padding-top: 1.2rem; max-width: 1200px; }
html, body, [class*="css"] {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}

/* =========================
   STREAMLIT HEADER + SIDEBAR TOGGLE (VISIBLE)
   ========================= */

/* Header visibile (non nascosto) */
header[data-testid="stHeader"]{
  background: transparent !important;
  box-shadow: none !important;
}

/* Toolbar visibile (serve per il toggle sidebar) */
div[data-testid="stToolbar"]{
  visibility: visible !important;
}

/* Nascondi menu e footer */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Icona toggle sidebar (freccia) ben visibile */
header[data-testid="stHeader"] button,
header[data-testid="stHeader"] svg,
div[data-testid="stSidebarCollapsedControl"] svg{
  color: #ffffff !important;
  fill: #ffffff !important;
  opacity: 1 !important;
}

/* =========================
   BACKGROUND + GLOBAL TEXT
   ========================= */
.stApp{
  background:
    radial-gradient(1200px 700px at 15% 10%, rgba(59,130,246,0.20), transparent 60%),
    radial-gradient(900px 600px at 85% 20%, rgba(168,85,247,0.18), transparent 60%),
    radial-gradient(1200px 700px at 40% 90%, rgba(16,185,129,0.10), transparent 65%),
    linear-gradient(180deg, #050814 0%, #070b18 45%, #06081a 100%);
  color: #f9fafb !important;
}

/* Titles */
h1, h2, h3, h4, h5, h6 {
  color: #ffffff !important;
  opacity: 1 !important;
}

/* Only force text where WE render content (markdown + chat).
   Avoid forcing generic div everywhere (breaks widgets/menus). */
[data-testid="stMarkdownContainer"] *{
  color: #f9fafb !important;
  opacity: 1 !important;
}

/* Captions / small help */
.stCaption, .stHelp, small,
[data-testid="stWidgetLabel"] *{
  color: rgba(249,250,251,0.85) !important;
  opacity: 1 !important;
}

/* Placeholder visibility */
input::placeholder, textarea::placeholder{
  color: rgba(249,250,251,0.60) !important;
  opacity: 1 !important;
}

/* =========================
   SIDEBAR
   ========================= */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)) !important;
  border-right: 1px solid rgba(255,255,255,0.08) !important;
}
section[data-testid="stSidebar"] *{
  color: #f9fafb !important;
  opacity: 1 !important;
}

/* =========================
   INPUTS (text, textarea, number)
   ========================= */
.stTextInput input, .stNumberInput input, .stTextArea textarea{
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  color: #f9fafb !important;
  opacity: 1 !important;
  border-radius: 14px !important;
}

/* Select / Multiselect container */
.stSelectbox div, .stMultiSelect div{
  background: rgba(255,255,255,0.05) !important;
  border-radius: 14px !important;
  color: #f9fafb !important;
}

/* =========================
   DROPDOWNS (the popup menu)
   Fix white dropdown / unreadable options
   ========================= */
div[role="listbox"]{
  background: #0b1220 !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  border-radius: 14px !important;
}
div[role="option"]{
  background: transparent !important;
  color: #f9fafb !important;
}
div[role="option"]:hover{
  background: rgba(255,255,255,0.06) !important;
}
div[aria-selected="true"][role="option"]{
  background: rgba(59,130,246,0.18) !important;
}

/* Fallback (some Streamlit versions) */
[data-baseweb="menu"]{
  background: #0b1220 !important;
}
[data-baseweb="menu"] *{
  color: #f9fafb !important;
}

/* =========================
   BUTTONS (always visible)
   ========================= */
.stButton > button{
  background: linear-gradient(135deg, rgba(59,130,246,0.85), rgba(168,85,247,0.85)) !important;
  border: 1px solid rgba(255,255,255,0.22) !important;
  border-radius: 16px !important;
  min-height: 44px !important;
  font-size: 14px !important;
  color: #ffffff !important;
  padding: .60rem .95rem !important;
  transition: transform .06s ease, box-shadow .2s ease, border .2s ease, filter .2s ease;
}
.stButton > button:hover{
  transform: translateY(-1px);
  box-shadow: 0 12px 30px rgba(0,0,0,0.35);
  border: 1px solid rgba(255,255,255,0.35) !important;
  filter: brightness(1.05);
}
.stButton > button:active{ transform: translateY(0px); }

/* =========================
   FORM SUBMIT BUTTONS (BaseWeb)
   ========================= */
button[data-testid^="baseButton"]{
  background: linear-gradient(135deg, rgba(59,130,246,0.85), rgba(168,85,247,0.85)) !important;
  border: 1px solid rgba(255,255,255,0.22) !important;
  border-radius: 16px !important;
  min-height: 64px !important;

  color: #ffffff !important;
  font-size: 13px !important;
  line-height: 1.15 !important;

  white-space: normal !important;
  text-align: center !important;
}

/* hover */
button[data-testid^="baseButton"]:hover{
  filter: brightness(1.08);
  box-shadow: 0 12px 30px rgba(0,0,0,0.35);
}


/* =========================
   BUTTONS (uniform + readable text)
   ========================= */
.stButton > button{
  width: 100% !important;

  /* tutti uguali: scegliamo un’altezza che contiene anche 2 righe */
  height: 64px !important;
  min-height: 64px !important;

  padding: 0.55rem 0.5rem !important;
  font-size: 13px !important;
  line-height: 1.15 !important;

  display: flex !important;
  align-items: center !important;
  justify-content: center !important;

  /* testo visibile (no ellipsis) */
  white-space: normal !important;      /* permette a capo */
  overflow: visible !important;        /* niente tagli */
  text-overflow: clip !important;
  text-align: center !important;
}

/* anche per eventuali span/p interni */
.stButton > button *{
  white-space: normal !important;
  line-height: 1.15 !important;
  text-align: center !important;
}


/* =========================
   FILE UPLOADER (sidebar) visibility
   ========================= */
[data-testid="stFileUploader"] *{
  color: rgba(249,250,251,0.92) !important;
  opacity: 1 !important;
}

/* Dropzone (Drag & drop area) */
[data-testid="stFileUploaderDropzone"]{
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  border-radius: 14px !important;
}

/* Testi piccoli tipo "Limit 200MB..." o simili */
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] small{
  color: rgba(249,250,251,0.80) !important;
  opacity: 1 !important;
}

/* =========================
   CARDS / BADGES
   ========================= */
.card{
  padding: 16px 16px;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.10);
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  box-shadow: 0 18px 45px rgba(0,0,0,0.30);
}
.card-soft{
  padding: 12px 14px;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
}
.badge{
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.05);
  margin-right: 8px;
  color: #f9fafb !important;
}

/* =========================
   CHAT
   ========================= */
[data-testid="stChatMessage"]{
  border-radius: 18px !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  background: rgba(255,255,255,0.06) !important;
}
[data-testid="stChatMessage"] *{
  color: #f9fafb !important;
  opacity: 1 !important;
}

/* Slider text visibility */
[data-testid="stSlider"] *{
  color: #f9fafb !important;
  opacity: 1 !important;
}



/* kill any white background inside expanders (aggressive) */
div[data-testid="stExpander"] *{
  background-color: transparent !important;
}

/* ===== Search input inside select / multiselect ===== */
[data-baseweb="input"] input{
  color: #ffffff !important;
  caret-color: #ffffff !important;
}

/* Multiselect / Selectbox search field */
.stMultiSelect [data-baseweb="input"] input,
.stSelectbox [data-baseweb="input"] input{
  color: #ffffff !important;
  caret-color: #ffffff !important;
}


/* =========================
   CODE (inline + blocks) — fix white background
   ========================= */
[data-testid="stMarkdownContainer"] code,
[data-testid="stChatMessage"] code{
  background: rgba(255,255,255,0.08) !important;
  color: #f9fafb !important;
  padding: 0.15rem 0.35rem !important;
  border-radius: 8px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
}

[data-testid="stMarkdownContainer"] pre,
[data-testid="stChatMessage"] pre{
  background: rgba(11,18,32,0.75) !important;
  color: #f9fafb !important;
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  padding: 0.85rem 1rem !important;
  overflow-x: auto !important;
}

[data-testid="stMarkdownContainer"] pre code,
[data-testid="stChatMessage"] pre code{
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}
</style>
"""



def create_chat_handler() -> ChatHandler:
    """Initialize all data and create a ChatHandler instance."""
    data_maps = initialize_all_data(canonical_col="Type Level 3")

    canon2uid = data_maps["canon2uid"]
    uid2canon = data_maps["uid2canon"]
    jobs = data_maps["jobs"]
    levels = data_maps["levels"]
    courses_requirements = data_maps["courses_requirements"]
    courses_acquisitions = data_maps["courses_acquisitions"]
    skills_pool = data_maps["skills_pool"]
    skills2int_tl3 = data_maps["skills2int_tl3"]
    n_tl3 = data_maps["n_tl3"]
    int2skills_tl3 = data_maps["int2skills_tl3"]

    with open("UIR/config/run.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset = Dataset(config)

    state = PrefState()
    searcher = SkillSearcher(
        valid_uids=uid2canon.keys(),
        emb_path="./Chatbot/Embeddings/E_skills.npy",
        uids_path="./Chatbot/Embeddings/uids.npy",
    )

    handler = ChatHandler(
        state=state,
        canon2uid=canon2uid,
        uid2canon=uid2canon,
        levels=levels,
        skills_pool=skills_pool,
        jobs=jobs,
        courses_requirements=courses_requirements,
        courses_acquisitions=courses_acquisitions,
        searcher=searcher,
        dataset=dataset,
        skills2int_tl3=skills2int_tl3,
        n_tl3=n_tl3,
        int2skills_tl3=int2skills_tl3,
    )

    return handler


def _ensure_session(handler: ChatHandler):
    if "handler" not in st.session_state:
        st.session_state.handler = handler

    if "messages" not in st.session_state:
        st.session_state.messages = []  # list[{role, content}]

    if "show_skill_catalog" not in st.session_state:
        st.session_state.show_skill_catalog = False

    # quick solution without changing PrefState
    if "goal_levels" not in st.session_state:
        st.session_state.goal_levels = {}  # uid_str -> target_level
    
    if "cv_loaded" not in st.session_state:
        st.session_state.cv_loaded = False

    if "last_reco" not in st.session_state:
        st.session_state.last_reco = None


def _send_and_store(handler: ChatHandler, message_to_send: str, cv_text=None):
    # show compact user message in chat history (prettier than raw internal commands)
    pretty_user = message_to_send
    if message_to_send.startswith(":sem "):
        pretty_user = message_to_send.replace(":sem ", "", 1)

    st.session_state.messages.append({"role": "user", "content": pretty_user})

    with st.spinner("🛰️ Thinking..."):
        out = handler.handle(message=message_to_send, cv_text=cv_text)

    # out may be only text or (text, payload)
    if isinstance(out, tuple) and len(out) == 2:
        reply, payload = out
    else:
        reply, payload = out, None

    if message_to_send == ":rec":
        st.session_state.last_reco = payload

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Job-Oriented Course Recommender", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")
    st.markdown(SPACE_CSS, unsafe_allow_html=True)

    # header
    st.markdown(
        """
        <div class="card">
          <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
            <div>
              <div style="font-size:34px; font-weight:800; line-height:1.1;">🚀 Job-Oriented Course Recommendation Chatbot</div>
              <div style="opacity:1; margin-top:6px;">
                Upload CV → set preferences → get a course sequence aligned to your job goals.
              </div>
              <div style="margin-top:12px;">
                <span class="badge">Space theme</span>
                <span class="badge">Skills TL3</span>
                <span class="badge">CV → Profile</span>
                <span class="badge">Preferences</span>
              </div>
            </div>
            <div style="text-align:right; opacity:1;">
              <div style="font-size:12px;">Tip</div>
              <div style="font-size:14px;">Use the toolbar buttons or type normally.</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    # init handler once
    if "handler" not in st.session_state:
        st.session_state.handler = create_chat_handler()
    handler: ChatHandler = st.session_state.handler

    _ensure_session(handler)

    # ---------------------------
    # Sidebar (dashboard)
    # ---------------------------
    with st.sidebar:
        st.markdown("### ⚙️ Controls")

        k_courses = st.slider(
            "Recommendation sequence length",
            min_value=2,
            max_value=5,
            value=int(st.session_state.get("k_courses", 2)),
            key="k_courses",
            help="How long do you want the recommendation sequence to be?",
        )
        handler.state.set_k(k_courses)
        handler.k_changed = True

        st.markdown("---")
        st.markdown("### 📄 Resume / CV")

        uploaded_file = st.file_uploader(
            "Upload your resume (PDF only)",
            type=["pdf"],
            key="cv_uploader",
            disabled=st.session_state.cv_loaded,
        )
        load_resume_clicked = st.button("📥 Load resume", use_container_width=True)

        st.markdown("---")
        st.markdown("### 🧠 Skills snapshot")

        acquired = handler.state.get_acquired() or set()
        include = handler.state.get_include() or set()
        avoid = handler.state.get_avoid() or set()

        a = len(acquired)
        i = len(include)
        v = len(avoid)

        c1, c2, c3 = st.columns(3)
        c1.metric("Acquired", a)
        c2.metric("Include", i)
        c3.metric("Avoid", v)

        denom = max(1, a + i + v)
        st.progress(min(1.0, (a + i) / denom))
        st.caption("Profile signal strength (rough)")

    # ---------------------------
    # Main layout columns
    # ---------------------------
    left, right = st.columns([0.62, 0.38], gap="large")

    # ---------------------------
    # Chat area (left)
    # ---------------------------
    with left:
        st.markdown('<div class="card-soft">💬 <b>Chat</b> — Your conversation history is stored in-session.</div>', unsafe_allow_html=True)
        st.write("")

        # render chat history
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        # toolbar
        st.write("")
        t1, t2, t3, t4, t5 = st.columns([1, 1, 1, 1, 1], gap="small")

        with t1:
            rec_clicked = st.button("✨ Recommend", use_container_width=True)
        with t2:
            myskills_clicked = st.button("🧾 My skills", use_container_width=True)
        with t3:
            filter_clicked = st.button("🧩 Filter jobs", use_container_width=True)
        with t4:
            show_skills_clicked = st.button("🎛️ Preferences", use_container_width=True)
        with t5:
            clear_clicked = st.button("🧼 Clear", use_container_width=True)

        # input
        user_text = st.chat_input("Type a message… (or just use the buttons)")

        # handle sidebar CV load
        if load_resume_clicked:
            if uploaded_file is None:
                st.warning("Please upload a PDF first.")
            else:
                with st.spinner("📄 Reading your PDF..."):
                    cv_text = ""
                    with pdfplumber.open(uploaded_file) as pdf:
                        for page in pdf.pages:
                            text = page.extract_text()
                            if text:
                                cv_text += text + "\n"
                _send_and_store(handler, "load resume", cv_text=cv_text)
                st.session_state.cv_loaded = True
        
        if st.session_state.cv_loaded:
            if st.button("🗑️ Remove CV", use_container_width=True):
                st.session_state.cv_loaded = False
                st.session_state["cv_uploader"] = None  # File uploader reset
                st.success("CV removed. You can upload a new one.")
                st.rerun()

        # handle clear
        if clear_clicked:
            handler.handle("clear")
            st.session_state.messages = []
            st.session_state.last_reco = None
            st.rerun()
            #_send_and_store(handler, "clear", cv_text=None)
            #st.session_state.messages = []
            #st.rerun()

        # handle buttons
        if rec_clicked:
            _send_and_store(handler, ":rec", cv_text=None)
        if myskills_clicked:
            _send_and_store(handler, ":myskills", cv_text=None)
        if filter_clicked:
            _send_and_store(handler, ":filter", cv_text=None)
        if show_skills_clicked:
            _send_and_store(handler, ":show", cv_text=None)

        # handle typed message
        if user_text and user_text.strip():
            raw = user_text.strip()
            _send_and_store(handler, f":sem {raw}", cv_text=None)

    # ---------------------------
    # Right panel (skills management + catalog)
    # ---------------------------
    with right:
        st.markdown('<div class="card"><b>🧠 Skills Management</b><div style="opacity:1; margin-top:6px;">Remove items quickly without digging into state.</div></div>', unsafe_allow_html=True)
        st.write("")

        tabs = st.tabs(["🧩 Skill catalog", "🗑️ Remove acquired", "🟢 Remove include", "🔴 Remove avoid", "📊 Insights", "ℹ️ Help"])

        # --- Remove acquired ---
        with tabs[1]:
            acquired = handler.state.get_acquired() or set()  # {(name, uid, level), ...}

            if acquired:
                acq_label2name = {}
                acq_labels = []
                names_seen = []

                for name, uid, level in acquired:
                    if name not in names_seen:
                        names_seen.append(name)
                        label = f"{name} (level {level}) [id: {uid}]"
                        acq_labels.append(label)
                        acq_label2name[label] = name

                to_remove_acq = st.multiselect(
                    "Select acquired skills to remove",
                    options=acq_labels,
                    key="remove_acquired",
                )

                if st.button("🗑️ Remove selected acquired", use_container_width=True):
                    names_to_remove = [acq_label2name[label] for label in to_remove_acq]
                    removed = handler.state.remove_by_names(names_to_remove, "acquired")

                    if removed > 0:
                        st.success(f"Removed {removed} skill(s).")
                    else:
                        st.info("No skills removed.")
                    st.rerun()
            else:
                st.caption("No acquired skills yet. Load a CV or add skills in the catalog.")

        # --- Remove include ---
        with tabs[2]:
            include_prefs = handler.state.get_include() or set()
            if include_prefs:
                inc_label2name = {}
                inc_labels = []
                names = []
                for name, uid in include_prefs:
                    if name not in names:
                        names.append(name)
                        label = f"{name} [id: {uid}]"
                        inc_labels.append(label)
                        inc_label2name[label] = name

                to_remove_inc = st.multiselect("Remove learning goals (Include)", options=inc_labels, key="remove_include")

                if st.button("🗑️ Remove selected include", use_container_width=True):
                    #uids_to_remove = {inc_label2uid[label] for label in to_remove_inc}
                    #removed = handler.state.remove_by_uids(uids_to_remove, "include")
                    names_to_remove = [inc_label2name[label] for label in to_remove_inc]
                    removed = handler.state.remove_by_names(names_to_remove, "include") 
                    
                    #for uid in uids_to_remove:
                    #    st.session_state.goal_levels.pop(uid, None)

                    if removed > 0:
                        st.success(f"Removed {removed} learning goal(s).")
                    else:
                        st.info("No learning goals removed.")
                    st.rerun()
            else:
                st.caption("No include preferences yet.")

        # --- Remove avoid ---
        with tabs[3]:
            avoid_prefs = handler.state.get_avoid() or set()
            if avoid_prefs:
                avo_label2name = {}
                avo_labels = []
                names = []
                for name, uid in avoid_prefs:
                    if name not in names:
                        names.append(name)
                        label = f"{name} [id: {uid}]"
                        avo_labels.append(label)
                        avo_label2name[label] = name

                to_remove_avo = st.multiselect("Remove avoided skills", options=avo_labels, key="remove_avoid")

                if st.button("🗑️ Remove selected avoid", use_container_width=True):
                    #uids_to_remove = {avo_label2uid[label] for label in to_remove_avo}
                    #removed = handler.state.remove_by_uids(uids_to_remove, "avoid")
                    names_to_remove = [avo_label2name[label] for label in to_remove_avo]
                    removed = handler.state.remove_by_names(names_to_remove, "avoid")

                    if removed > 0:
                        st.success(f"Removed {removed} avoided skill(s).")
                    else:
                        st.info("No avoided skills removed.")
                    st.rerun()
            else:
                st.caption("No avoided skills yet.")

        # --- Help ---
        with tabs[0]:
            st.write("")
            st.markdown('<div class="card"><b>🧩 Skill catalog</b><div style="opacity:1; margin-top:6px;">Add skills as Acquired / Include / Avoid.</div></div>', unsafe_allow_html=True)
            st.write("")

            with st.expander("ℹ️ Skill levels legend", expanded=False):
                st.markdown(
                    """
                    - **Acquired**: current mastery level (1..3)  
                    - **Include**: no level needed  
                    - **Avoid**: no level needed
                    """
                )

            skill_options = sorted(handler.canon2uid.keys())

            with st.form("skill_catalog_form", clear_on_submit=True):
                add_as = st.radio(
                    "Add selected skills as",
                    ["Acquired", "Learning goal (Include)", "Avoid"],
                    index=0,
                    horizontal=True,
                )

                selected_skills = st.multiselect(
                    "Search and select skills",
                    options=skill_options,
                    help="Start typing to search in the catalog.",
                    key="catalog_multiselect",
                )

                if add_as == "Acquired":
                    level = st.slider("Current mastery level", 1, 3, 1)
                elif add_as == "Learning goal (Include)":
                    level = st.slider("Target mastery level", 1, 3, 2)
                else:
                    level = None

                cA, cB = st.columns(2)
                confirm = cA.form_submit_button("✅ Confirm")
                cancel = cB.form_submit_button("❌ Cancel")

                if cancel:
                    st.session_state.show_skill_catalog = False
                    st.rerun()

                if confirm:
                    if not selected_skills:
                        st.warning("Please select at least one skill.")
                        st.stop()

                    current_acq = set(handler.state.get_acquired() or set())
                    current_inc = set(handler.state.get_include() or set())
                    current_avo = set(handler.state.get_avoid() or set())

                    # de-duplicate by name (keep one tuple per name)
                    if handler.state.get_acquired():
                        current_acq = {next(v for v in handler.state.get_acquired() if v[0] == n)
                                       for n in {v[0] for v in handler.state.get_acquired()}}
                    #if handler.state.get_acquired():
                    #    current_acq = {
                    #        v[0]: v
                    #        for v in handler.state.get_acquired()
                    #    }.values()
                    if handler.state.get_include():
                        current_inc = {next(v for v in handler.state.get_include() if v[0] == n)
                                       for n in {v[0] for v in handler.state.get_include()}}
                    #if handler.state.get_include():
                    #    current_inc = {
                    #        v[0]: v
                    #        for v in handler.state.get_include()
                    #    }.values()
                    if handler.state.get_avoid():
                        current_avo = {next(v for v in handler.state.get_avoid() if v[0] == n)
                                       for n in {v[0] for v in handler.state.get_avoid()}}
                    #if handler.state.get_avoid():
                    #    current_avo = {
                    #        v[0]: v
                    #        for v in handler.state.get_avoid()
                    #    }.values()
                    added = 0

                    for skill_name in selected_skills:
                        uids = handler.canon2uid.get(skill_name)
                        if uids is None:
                            continue
                        uids = uids if isinstance(uids, list) else [uids]

                        for uid_str in uids:
                            uid_str = str(uid_str)

                            if add_as == "Acquired":
                                current_acq.add((skill_name, uid_str, int(level)))
                                added += 1

                            elif add_as == "Learning goal (Include)":
                                current_inc.add((skill_name, uid_str))
                                #st.session_state.goal_levels[uid_str] = int(level)

                                # prevent include+avoid simultaneously
                                if (skill_name, uid_str) in current_avo:
                                    current_avo.remove((skill_name, uid_str))
                                added += 1

                            else:  # Avoid
                                current_avo.add((skill_name, uid_str))

                                # prevent include+avoid simultaneously
                                if (skill_name, uid_str) in current_inc:
                                    current_inc.remove((skill_name, uid_str))
                                #st.session_state.goal_levels.pop(uid_str, None)
                                added += 1

                    handler.state.set_acquired(current_acq)
                    handler.state.set_include(current_inc)
                    handler.state.set_avoid(current_avo)

                    st.success(f"Updated: {added} skill(s) → {add_as}.")
                    st.session_state.show_skill_catalog = False
                    st.rerun()
        with tabs[4]:
            lr = st.session_state.get("last_reco")

            if not lr:
                st.caption("No recommendation yet. Press ✨ Recommend.")
            else:

                st.markdown("### 📌 Last recommendation")

                reco_courses = lr["recommended_courses"]   
                # Mostra corsi con expander e skill insegnate
                for course_id, taught in reco_courses.items():

                    with st.expander(f"{course_id}", expanded=False):
                        # esempio: skills insegnate dal corso
                        
                        if not taught:
                            st.caption("No skill found for this course")
                        else:
                            levels = {"beginner": 1, "intermediate": 2, "advanced": 3, "unknown": 2}
                            st.markdown("<br>".join(f"{name}, (level {levels[level]})" for name, level in taught), unsafe_allow_html=True)
                            #st.text("\n".join(taught))

                st.markdown("---")
                st.markdown("### 📊 Insights (Last Recommendation)")

                c1, c2, c3 = st.columns(3)
                c1.metric("required skills", lr["skills_required_unique"])
                c2.metric("Covered skills", lr["skills_fully_covered_unique"])
                c3.metric("Uncovered skills", lr["skills_not_fully_covered_unique"])

                c4, c5, c6 = st.columns(3)
                c4.metric("required levels", lr["levels_required_total"])
                c5.metric("covered levels", lr["levels_covered_total"])
                c6.metric("missing levels", lr["levels_missing_total"])
                
                c7, _c8, _c9 = st.columns(3)
                c7.metric("Applicable jobs", lr["nb_applicable_jobs"])

                st.markdown("#### 🔝 Most relevant skills to jobs_goal")
                st.dataframe(lr["ranking"])  # lista di dict -> Streamlit la mostra bene

                st.markdown("#### ➕ Acquired skills not required by jobs_goal")
                st.dataframe(lr["extra_ranking"])
            
        with tabs[5]:
            st.markdown(
                """
        ### ℹ️ Help

        #### ✅ What you can do by typing in the chat
        Typing a message is for **conversation and profile/preferences updates**, for example:
        - Ask questions and get explanations (about skills, courses, jobs, or the system)
        - Update your preferences (skills to learn / avoid)
        - Add or refine information about your background (to improve your profile)

        > Note: typing is **not** meant to trigger the main “action commands”.

        ---

        #### 🎛️ Actions that must be done via the UI (buttons / sidebar / slider)
        Some actions are **UI-only** and can be triggered only using buttons, the sidebar, or sliders:

        - **✨ Recommend**  
        Generates a course sequence (length depends on the slider).

        - **🧩 Filter jobs**  
        Shows the jobs that match your current profile and constraints.

        - **🧾 My skills**  
        Displays your current skill snapshot.

        - **🎛️ Preferences (button)**  
        Opens the preference-related view/flow in the chat interface.

        - **🧼 Clear**  
        Resets the conversation and the last recommendation.

        - **Recommendation sequence length (slider)**  
        Controls how many courses are proposed in the recommendation sequence.

        - **📥 Load resume (sidebar)**  
        Reads your uploaded CV (PDF) and uses it to build/update your profile.

        - **🗑️ Remove CV (sidebar)**  
        Removes the current CV and enables uploading a new one.

        ---

        #### 🧠 Skills Management (Right panel)
        You can always edit your skills manually:
        - **🧩 Skill catalog**: add skills as **Acquired**, **Learning goal**, or **Avoid**
        - Remove skills quickly from the dedicated tabs

        ---

        #### 🧭 Skill Levels
        - **Level 1 – Beginner**: basic knowledge  
        - **Level 2 – Intermediate**: able to work independently  
        - **Level 3 – Advanced**: can mentor or teach others
        """
            )

        # ---------------------------
        # Skill catalog (toggle)
        # ---------------------------

            


if __name__ == "__main__":
    main()
