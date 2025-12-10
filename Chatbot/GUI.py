import yaml
import streamlit as st
import pdfplumber

import os
import sys

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


def create_chat_handler() -> ChatHandler:
    """Initialize all data and create a ChatHandler instance."""
    data_maps = initialize_all_data(canonical_col="Type Level 4")

    canon2uid = data_maps["canon2uid"]
    uid2canon = data_maps["uid2canon"]
    jobs = data_maps["jobs"]
    levels = data_maps["levels"]
    courses_requirements = data_maps["courses_requirements"]
    courses_acquisitions = data_maps["courses_acquisitions"]
    skills_pool = data_maps["skills_pool"]
    # df_taxonomy = data_maps["df_taxonomy"]  # kept in case it is needed elsewhere

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
    )

    return handler


def main() -> None:
    st.title("Job Oriented Course Recommendation Chatbot")
    st.caption("Interact with the system to update your preferences and get course recommendation")
    st.write(
        "Type your message below or use the buttons to trigger specific commands. "
    )

    # Initialize ChatHandler once per session
    if "handler" not in st.session_state:
        st.session_state.handler = create_chat_handler()

    if "show_skill_catalog" not in st.session_state:
        st.session_state.show_skill_catalog = False

    if "last_user" not in st.session_state:
        st.session_state.last_user = ""
    if "last_bot" not in st.session_state:
        st.session_state.last_bot = ""

    handler: ChatHandler = st.session_state.handler

    with st.sidebar:
        k_courses = st.slider(
            "Recommendation sequence length",
            min_value=2,
            max_value=5,
            value=2,
            key="k_courses",
            help="How long do you want the recommendation sequence will be?"
        )

        handler.state.set_k(k_courses)
        handler.k_changed = True  # inform the handler that k has changed

        st.markdown("### Resume / CV")
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF only)",
            type=["pdf"],
        )
        load_resume_clicked = st.button("Load resume from file")
        
        st.markdown("---")
        st.markdown("### 🧠 Manage skills")

        profile_skills = handler.state.profile.skills_explicit

        if profile_skills:
            # Costruisco etichette leggibili e una mappa label -> uid
            label2uid = {}
            labels = []
            for uid, level in profile_skills.items():
                try:
                    name = handler.uid2canon.get(int(uid), "Unknown")
                except ValueError:
                    name = "Unknown"
                label = f"{name} (level {level}) [id: {uid}]"
                labels.append(label)
                label2uid[label] = uid

            skills_to_remove = st.multiselect(
                "Select skills to remove",
                options=labels,
                key="skills_to_remove",
                help="Uncheck skills you no longer want in your profile."
            )

            if st.button("🗑️ Remove selected skills"):
                removed = 0
                for label in skills_to_remove:
                    uid = label2uid[label]

                    # 1) togli dallo user profile esplicito
                    if uid in handler.state.profile.skills_explicit:
                        del handler.state.profile.skills_explicit[uid]
                        removed += 1

                    # 2) togli anche dalle preferenze (include/avoid/acquired) se presente
                    if uid in handler.state.skills:
                        del handler.state.skills[uid]

                if removed > 0:
                    st.success(f"Removed {removed} skill(s) from profile.")
                    st.rerun()
                else:
                    st.info("No skills removed.")
                    st.rerun()
        else:
            st.caption("No explicit skills in profile yet. Load a CV or add skills during the chat.")

        if st.button("Show skill catalog"):
            st.session_state.show_skill_catalog = True



    if st.session_state.show_skill_catalog:
        st.markdown("## 🧩 Skill catalog")
        st.info("Select the skills you already have and set your mastery level. These will be added to your profile.")

        st.markdown(
        """
        **Skill levels**

        - **1 – Beginner**: You have basic notion about this skill.
        - **2 – Intermediate**: You can work in autonomy with task regarding this skill.
        - **3 – Advanced**: You mastered this skill and you are able to teach it to others.
                """
            )

        # let's get all skill names from canon2uid
        skill_options = sorted(handler.canon2uid.keys())

        with st.form("skill_catalog_form"):
            selected_skills = st.multiselect(
                "Search and select your skills",
                options=skill_options,
                help="Start typing to search in the catalog.",
                key="catalog_multiselect",
            )

            level = st.slider(
                "Mastery level for selected skills",
                min_value=1,
                max_value=3,
                value=1,
                help="You can adjust level later if needed.",
            )

            col1, col2 = st.columns(2)
            confirm = col1.form_submit_button("✅ Confirm")
            cancel = col2.form_submit_button("❌ Cancel")

            if confirm:
                entries = set()
                for skill_name in selected_skills:
                    uid_int = handler.canon2uid.get(skill_name)
                    if uid_int is None:
                        continue
                    uid_str = str(uid_int)
                    entries.add((skill_name, uid_str, level))

                # aggiorna il profilo con set_acquired
                handler.state.set_acquired(entries)
                st.success(f"Profile updated with {len(entries)} skills.")
                st.session_state.show_skill_catalog = False  # Close the "window"
                st.rerun()

            if cancel:
                st.session_state.show_skill_catalog = False  # Close the "window"
                st.rerun()

            return


    # User input
    user_message = st.text_input("You", value="", key="user_input")

    cols = st.columns(5)
    send_clicked = st.button("Send")
    with cols[0]:
        rec_clicked = st.button("Recommend courses (:rec)")
    with cols[1]:
        myskills_clicked = st.button("Show my skills (:myskills)")
    with cols[3]:
        filter_clicked = st.button("Filter jobs (:filter)")
    with cols[4]:
        show_skills_clicked = st.button("Show skills preferences (:show)")

    st.markdown("##### Clear profile state and preferences")
    clear_clicked = st.button("Clear (clear)")

    cv_text = None
    message_to_send = None

    if send_clicked and user_message.strip():
        raw = user_message.strip()
        message_to_send = f":sem {raw}"

    if rec_clicked:
        message_to_send = ":rec"

    if myskills_clicked:
        message_to_send = ":myskills"

    if load_resume_clicked:
        if uploaded_file is None:
            st.warning("Please upload a resume file first.")
        else:
            with pdfplumber.open(uploaded_file) as pdf:
                cv_text = ""
                for page in pdf.pages:
                    cv_text += page.extract_text() + "\n"
            message_to_send = "load resume"


    if filter_clicked:
        message_to_send = ":filter"

    if show_skills_clicked:
        message_to_send = ":show"

    if clear_clicked:
        message_to_send = "clear"

        # Reset PDF uploader state (streamlit stores uploaded files under a predictable key)
        if "uploaded_file" in st.session_state:
            st.session_state.uploaded_file = None

    if message_to_send is not None:
        reply = handler.handle(message=message_to_send, cv_text=cv_text)
        st.session_state.last_user = message_to_send
        st.session_state.last_bot = reply

    if st.session_state.last_user:
        st.markdown(f"**You:** {st.session_state.last_user}")
        st.markdown(f"**Bot:** {st.session_state.last_bot}")


if __name__ == "__main__":
    main()
