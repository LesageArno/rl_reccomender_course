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

    if "last_user" not in st.session_state:
        st.session_state.last_user = ""
    if "last_bot" not in st.session_state:
        st.session_state.last_bot = ""

    handler: ChatHandler = st.session_state.handler

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

    st.markdown("### Resume / CV")
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF only)",
        type=["pdf"],
    )
    load_resume_clicked = st.button("Load resume from file")

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
