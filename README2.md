## Project Structure

The repository is organized into three main components:
- a **Chatbot module** with GUI and conversational logic
- a **Reinforcement Learning backend** (UIR)
- supporting datasets and experimental components


```text
rl_recommender_course/
├── Chatbot/                     # Conversational interface and GUI
│   ├── GUI.py                   # Streamlit GUI (main demo entry point)
│   ├── chat_handler.py          # Conversation handling and state updates
│   ├── LLMDialogManager.py      # LLM wrapper (Mistral v2)
│   ├── chatbot.py               # Terminal-based chatbot (legacy)
│   ├── state.py                 # User preference state
│   ├── learnerProfile.py        # Learner skill profile representation
│   ├── taxonomy_index.py        # ESCO taxonomy indexing
│   ├── data_loader.py           # Dataset loading utilities
│   ├── utils.py                 # Helper functions
│   │
│   ├── CV_pdf/                  # Example resumes (PDF)
│   │
│   ├── Embeddings/              # Skill embedding and semantic search
│   │   ├── build_skill_embeddings.py
│   │   ├── skill_search.py
│   │   ├── E_skills.npy
│   │   └── uids.npy
│   │
│   └── NER/                     # Named Entity Recognition (skills)
│       ├── BIO.ipynb            # NER experimentation notebook
│       ├── training_data_*.json
│       ├── dataset*.json
│       └── pretrained checkpoints
│
├── UIR/                         # Reinforcement Learning backend (main)
│   ├── Scripts/
│   │   ├── CourseRecEnv.py      # RL environment
│   │   ├── Dataset.py           # Dataset handling
│   │   ├── Reinforce.py         # RL agent logic
│   │   ├── pipeline.py          # Training pipeline entry point
│   │   ├── matchings.py         # Skill-job matching utilities
│   │   └── tuning.py            # Hyperparameter optimization (optional)
│   │
│   ├── config/
│   │   └── run.yaml             # Training and inference configuration
│   │
│   ├── models_weights/          # Pretrained RL models (not versioned)
│   └── results/                 # Training outputs and plots
│
├── CLASS/                       # Clustering-based approach (secondary)
│
├── Data-Collection/
│   └── Final/                   # Datasets
│       ├── courses.json
│       ├── jobs.json
│       ├── resumes.json
│       ├── taxonomy.csv
│       └── mastery_levels.json
│
├── requirements.txt             # Project dependencies
├── README.md
└── LICENSE
```

## Installation (Windows)

We recommend using a Python virtual environment.

### 1. Create and activate a virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2. Install PyTorch

CUDA (recommended for LLM inference):
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
CPU-only (Not Recommended for LLM Inference)
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Project Dependencies
```
pip install -r requirements.txt
```


## Running the Demo (Streamlit GUI)

The Streamlit GUI is the **main entry point** for the project demo.

### Start the GUI

From the project root directory (`rl_recommender_course`):

```bash
streamlit run Chatbot/GUI.py
```

Demo Capabilities

Through the GUI, users can:

- interact with the chatbot in natural language

- specify skills they want to acquire or avoid

- load a resume (CV) in PDF format

- receive personalized course recommendations

- obtain explanations for both preferences and recommendations



---


## Chatbot Commands

The chatbot supports the following commands:

- `:sem <text>`  
  Extracts skill preferences from natural language input and updates the user profile.

- `:rec`  
  Generates a personalized sequence of recommended courses.

- `:myskills`  
  Displays the skills currently associated with the user profile.

- `:show`  
  Shows current include and avoid skill preferences.

- `load resume`  
  Extracts skills from a resume (PDF) and updates the user profile.

- `clear`  
  Resets the user profile and clears all preferences.

which are automatically computed when using streamlit with specific buttons


## Reinforcement Learning Backend (UIR)

The **UIR (Usefulness-based Information Reward)** approach is the primary reinforcement learning
method used in this project.

### Key Characteristics

- **Algorithm**: PPO / MaskablePPO  
- **State**: learner skill vector  
- **Action space**: available courses  
- **Reward**: usefulness-based (skill acquisition and job applicability)

Pretrained models are included in the repository and are used by default in the demo.



## Training a Reinforcement Learning Model (Optional)

Training a new reinforcement learning model is optional and **not required** to run the demo.

### Configure Training

Edit the configuration file:

text
UIR/config/run.yaml


This file controls:

TODO

### Run Training
```
python UIR/Scripts/pipeline.py --config UIR/config/run.yaml
```

Training outputs, logs, and models are saved according to the configuration.

#### Hyperparameter Optimization

Hyperparameter optimization is supported via tuning.py.
This feature is experimental and not required for standard usage


---



## Models

### Named Entity Recognition (NER)

- A pretrained NER model is included in the repository.
- The model is used to extract skill mentions from user input.
- The model path is configured directly in `chat_handler.py`.

### Large Language Model (LLM)

- The chatbot uses an LLM downloaded automatically from HuggingFace.
- Default model:
  - `mistralai/Mistral-7B-Instruct-v0.2`
- The first run may take several minutes due to model download.

### Reinforcement Learning Models

- Pretrained RL models are stored in:
  text
  UIR/models_weights/


These models are loaded automatically during the demo.


---

## Hardware Notes

- A GPU is **strongly recommended** for running the chatbot with the LLM.
- CPU-only execution is supported but will be significantly slower.
- Reinforcement learning training primarily runs on CPU.


## Notes

- This project is intended for **research and demonstration purposes**.
- Code clarity and modularity are prioritized over production-level optimization.
- Detailed methodological explanations are provided in the associated publication.


## Acknowledgements

This project extends and builds upon the **JCRec** framework by Jibril Frej:

https://github.com/Jibril-Frej/JCRec
