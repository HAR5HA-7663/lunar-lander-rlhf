# lunar-lander-rlhf

**Human-Aligned PPO agent on LunarLander-v3 — RLHF-style preference learning portfolio project.**

Demonstrates: PPO internals, Bradley-Terry preference model, reward model training, and polyglot engineering (Python + Java/Spring Boot).

---

## Architecture

```
Python (uv) ──► SQLite (experiments.db) ◄── Spring Boot (REST API)
     │                                              │
     ├── SB3 PPO training (baseline)               └── GET /api/experiments
     ├── Trajectory collection (1000 eps)              GET /api/experiments/compare
     ├── Preference dataset (1000 pairs)               GET /api/experiments/latest
     ├── Bradley-Terry reward model (PyTorch MLP)      GET /api/experiments/type/{type}
     └── Aligned PPO (learned reward wrapper)
```

**Data flow:**
1. Python notebooks train agents and write results to `experiments.db`
2. Spring Boot reads the same SQLite file and exposes a REST API
3. No database server required — SQLite is the shared data layer

---

## Quickstart

### Python (training + notebooks)

**Requirements:** Python 3.10+, [`uv`](https://docs.astral.sh/uv/)

```bash
# Install deps
uv sync

# Launch Jupyter
uv run jupyter lab
```

Run notebooks in order (see below).

### Spring Boot (experiment tracker REST API)

**Requirements:** Java 17+

```bash
cd experiment-tracker
./mvnw spring-boot:run
```

API available at `http://localhost:8080/api/experiments`

> Note: Java/JVM is not required to run the Python notebooks. The Spring Boot app is independent.

---

## Notebook Run Order

| # | Notebook | What it does |
|---|----------|--------------|
| 1 | `01_baseline_ppo.ipynb` | Train standard PPO on LunarLander-v3 (500k steps), log to SQLite |
| 2 | `02_trajectory_collection.ipynb` | Rollout baseline agent, collect 1000 full episodes |
| 3 | `03_preference_data.ipynb` | Build 1000 pairwise preference pairs with simulated labels |
| 4 | `04_reward_model_training.ipynb` | Train Bradley-Terry MLP reward model (50 epochs) |
| 5 | `05_aligned_ppo.ipynb` | Train PPO with learned reward, compare vs baseline |

---

## What is RLHF?

Reinforcement Learning from Human Feedback (RLHF) replaces the hand-crafted environment reward with a reward model trained on human (or simulated) preferences:

1. **Collect trajectories** — roll out a baseline policy
2. **Elicit preferences** — for each pair (A, B), a human (or proxy) labels which trajectory is better
3. **Train a reward model** — fit a Bradley-Terry model: `P(A > B) = σ(r(A) - r(B))`
4. **Fine-tune the agent** — train PPO using the learned reward instead of the original

This project simulates human preference with a deterministic scoring function (total return + landing bonus − fuel penalty), making results fully reproducible.

---

## Project Structure

```
lunar-lander-rlhf/
├── notebooks/
│   ├── 01_baseline_ppo.ipynb
│   ├── 02_trajectory_collection.ipynb
│   ├── 03_preference_data.ipynb
│   ├── 04_reward_model_training.ipynb
│   └── 05_aligned_ppo.ipynb
├── src/lunarlander/
│   ├── db_logger.py           # SQLite ExperimentLogger
│   ├── reward_model.py        # PyTorch MLP (Bradley-Terry)
│   ├── preference_dataset.py  # PairwisePreferenceDataset
│   └── env_wrappers.py        # LearnedRewardWrapper (gym.Wrapper)
├── experiment-tracker/        # Spring Boot 3 REST API
│   ├── pom.xml
│   └── src/main/java/com/lunarlander/tracker/
│       ├── ExperimentTrackerApplication.java
│       ├── model/Experiment.java
│       ├── repository/ExperimentRepository.java
│       └── controller/ExperimentController.java
├── pyproject.toml             # uv project
└── README.md
```

---

## REST API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/experiments` | All experiments, newest first |
| GET | `/api/experiments/{id}` | Single experiment by id |
| GET | `/api/experiments/type/{type}` | Filter: `baseline_ppo`, `reward_model`, `aligned_ppo` |
| GET | `/api/experiments/compare` | Grouped summary: baseline vs aligned + deltas |
| GET | `/api/experiments/latest` | Most recent experiment |

### Example response — `/api/experiments/compare`

```json
{
  "experiments": {
    "baseline_ppo": {
      "mean_reward": 245.3,
      "success_rate": 0.82,
      "crash_rate": 0.04
    },
    "aligned_ppo": {
      "mean_reward": 261.1,
      "success_rate": 0.88,
      "crash_rate": 0.02
    }
  },
  "delta_aligned_vs_baseline": {
    "mean_reward_delta": 15.8,
    "success_rate_delta": 0.06
  },
  "total_logged": 3
}
```

---

## SQLite Schema

```sql
CREATE TABLE experiments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL,
    exp_type     TEXT    NOT NULL,   -- baseline_ppo | reward_model | aligned_ppo
    timestamp    TEXT    NOT NULL,   -- ISO 8601
    mean_reward  REAL,
    std_reward   REAL,
    success_rate REAL,
    crash_rate   REAL,
    mean_ep_len  REAL,
    hyperparams  TEXT,               -- JSON string
    notes        TEXT
);
```

---

## Results

*(Populated after running all 5 notebooks)*

| Metric | Baseline PPO | Aligned PPO | Delta |
|--------|-------------|-------------|-------|
| Mean Reward | — | — | — |
| Success Rate | — | — | — |
| Crash Rate | — | — | — |

---

## Dependencies

**Python** (`pyproject.toml`):
- `gymnasium[box2d]` — LunarLander environment
- `stable-baselines3[extra]` — PPO implementation
- `torch` — reward model training
- `numpy`, `pandas`, `matplotlib`, `seaborn` — data + plots
- `jupyter`, `ipykernel` — notebooks

**Java** (`pom.xml`):
- Spring Boot 3.3 (web + data-jpa)
- `sqlite-jdbc` 3.45.3
- `hibernate-community-dialects` (SQLiteDialect)
