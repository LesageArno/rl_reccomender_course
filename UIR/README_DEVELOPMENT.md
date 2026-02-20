# A Usefulness of Information Approach to Goal-Directed Course Recommendation

This repository implements an advanced **Reinforcement Learning (RL)** framework for sequential course recommendation. Moving beyond simple employability metrics, the system optimizes learning pathways based on the **Usefulness of Information (UIR)**, aligning recommendations with specific learner goals, job market demands, and personal learning preferences.

## 🌟 Key Innovations
* **Information Usefulness Reward (UIR):** A novel reward function based on Cholvy et al.’s framework, measuring knowledge gap reduction rather than just job counts.
* **Action Masking (MaskablePPO):** Eliminates structurally invalid actions (courses with unmet prerequisites), leading to **10x faster convergence**.
* **Mastery-Aware Modeling:** Explicitly handles skill progression across discrete levels (*Beginner, Intermediate, Advanced*) using the **ESCO** taxonomy.
* **Preference-Driven Guidance:** Integrates user-defined interests directly into the reinforcement learning signal.

---

## 🧠 Methodology: Usefulness of Information (UIR)

The system evaluates the informational value of a course using three core dimensions:
1.  **Knowledge Gap Reduction ($N_r$):** Measures how many missing skills required by goals are covered.
2.  **Residual Informational Needs ($N_m$):** Measures the skills still missing after the course.
3.  **Useless Content ($N_{nr}$):** Quantifies redundant or irrelevant information provided by the course.

### Reward Variants
The framework supports multiple reward formulations:
* **UIR-Threshold-Based:** A strict interpretation where a skill is "acquired" only when the mastery level fully meets the job requirement.
* **UIR-Gap-Based:** A continuous approach that rewards incremental progress (reducing the distance between current and target mastery).
* **EUIR (Hybrid):** Combines informational usefulness ($U$) with a normalized employability signal ($E$) for maximum stability at longer horizons.

---

## 🎭 User Preferences Integration

The system incorporates personal learning objectives through a dual-vector preference model. For each skill in the taxonomy, the user profile includes two binary vectors of dimension $N$ (where $N$ is the total number of skills):

* **Want Vector ($V_{want}$):** A binary vector where $1$ indicates a skill the user is specifically interested in acquiring.
* **Avoid Vector ($V_{avoid}$):** A binary vector where $1$ indicates a skill the user wishes to exclude or deprioritize.

These vectors act as a **shaping signal** during the RL training process. The reward is adjusted to favor courses providing "wanted" skills while penalizing those containing "avoid" skills, ensuring the recommended sequence is not only effective for the market but also aligned with the learner's personal journey.

---

## 🛠️ Technical Architecture

### Core Components
* **Agent:** Proximal Policy Optimization (**PPO**) with invalid action masking via `MaskablePPO`.
* **Environment:** A custom Gymnasium-based environment representing learners, courses, and job requirements.
* **Taxonomy:** Skills are mapped to the **ESCO** (European Skills, Competences, Qualifications and Occupations) standard.

### Project Structure
```text
UIR-Recommendation/
├── Scripts/            # Core RL logic and MaskablePPO implementation
├── config/             # YAML configurations for rewards, horizons, and preference vectors
├── results/            # Training logs and performance heatmaps
└── README.md