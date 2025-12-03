import yaml
import streamlit as st

from UIR.Scripts.Dataset import Dataset
from .chat_handler import ChatHandler
from .state import PrefState
from .Embeddings.skill_search import SkillSearcher
from .data_loader import initialize_all_data


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
    st.title("Course Recommendation Chatbot")
    st.write(
        "Type your message below or use the buttons to trigger specific commands. "
        "Commands supported include `:rec`, `:myskills`, `:filter`, `load resume`, `:sem ...`, etc."
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

    cols = st.columns(4)
    send_clicked = st.button("Send")
    with cols[0]:
        rec_clicked = st.button("Recommend courses (:rec)")
    with cols[1]:
        myskills_clicked = st.button("Show my skills (:myskills)")
    with cols[2]:
        load_resume_clicked = st.button("Load resume (placeholder)")
    with cols[3]:
        filter_clicked = st.button("Filter jobs (:filter)")

    message_to_send = None

    if send_clicked and user_message.strip():
        message_to_send = user_message.strip()

    if rec_clicked:
        message_to_send = ":rec"

    if myskills_clicked:
        message_to_send = ":myskills"

    if load_resume_clicked:
        message_to_send = "load resume"

    if filter_clicked:
        message_to_send = ":filter"

    if message_to_send is not None:
        reply = handler.handle(message_to_send)
        st.session_state.last_user = message_to_send
        st.session_state.last_bot = reply

    if st.session_state.last_user:
        st.markdown(f"**You:** {st.session_state.last_user}")
        st.markdown(f"**Bot:** {st.session_state.last_bot}")


if __name__ == "__main__":
    main()
