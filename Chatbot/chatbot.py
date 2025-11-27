import yaml

from UIR.Scripts.Dataset import Dataset
from .chat_handler import ChatHandler
from .state import PrefState
from .Embeddings.skill_search import SkillSearcher
from .data_loader import initialize_all_data


def main() -> None:
    """Command-line entry point for the course recommendation chatbot."""
    data_maps = initialize_all_data(canonical_col="Type Level 4")

    canon2uid = data_maps["canon2uid"]
    uid2canon = data_maps["uid2canon"]
    jobs = data_maps["jobs"]
    levels = data_maps["levels"]
    courses_requirements = data_maps["courses_requirements"]
    courses_acquisitions = data_maps["courses_acquisitions"]
    skills_pool = data_maps["skills_pool"]
    df_taxonomy = data_maps["df_taxonomy"]  # kept in case it is needed elsewhere

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

    print("Chat started. Example: 'I like programming, but I don't like java'.")
    print("Commands: ':show', ':filter', ':rec', ':myskills', 'load resume', 'clear', ':quit'.")
    print("Type ':help' to see a short description of each command.\n")

    while True:
        msg = input("You: ").strip()
        reply = handler.handle(msg)
        print("Bot:", reply)

        if msg in {":quit", "quit"}:
            break

        print()
        print("Type ':help' to see the commands list.\n")


if __name__ == "__main__":
    main()
