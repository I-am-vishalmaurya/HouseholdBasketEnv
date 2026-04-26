---
title: HouseholdBasketEnv
emoji: 🛒
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
short_description: Multi-member grocery basket planning environment for OpenEnv
---

# HouseholdBasketEnv

HouseholdBasketEnv is an OpenEnv hackathon environment for personalized grocery
planning. An agent builds a packaged-food basket for a small Indian household
while balancing member-specific nutrition caps, minimum intake floors, dietary
restrictions, meal variety, and a hard INR budget.

The core challenge is not picking one "healthy" item. It is reasoning over a
coupled basket: products that look acceptable alone can jointly violate sugar,
sodium, fat, or calorie limits for a specific member.

## Why It Matters

Most food-label demos stop at single-item classification. Real grocery planning
is multi-person and cumulative: a diabetic adult, a hypertensive senior, and a
child do not need the same basket. HouseholdBasketEnv turns that into a
repeatable RL-style benchmark with deterministic seeds, dense rewards, terminal
grading, and adversarial products.

## Hackathon Readiness

Current status: **environment on track, full RL demo still at risk until proof is
committed**.

What is already solid:

- Step-by-step environment with objective verification.
- OpenEnv/FastAPI package, manifest, Dockerfile, and client.
- Curriculum tiers that avoid starting with a zero-reward hard task.
- Multiple reward components instead of one fragile scalar.
- Reward-hack tests for obvious degenerate policies.
- Local test suite: `58 passed`.

What still needs proof before judging:

- Baseline-vs-trained evaluation results under `docs/results/`.
- Evidence that GRPO/Unsloth training completed and improved behavior.
- Deployed Hugging Face Space URL with `/health`, `/tasks`, `/reset`, and
  `/step` smoke-test output.
- A scripted demo showing baseline attempt, reward breakdown, trained attempt,
  and measurable improvement.
- Model save/export validation for the LoRA/QLoRA adapter path.

See `HACKATHON_GUIDELINES_REPORT.md` for the guideline-by-guideline review and
`docs/superpowers/plans/2026-04-26-hackathon-gap-closure.md` for the execution
plan.

## What Judges Should Inspect

- `household_basket_env/openenv.yaml` defines the OpenEnv manifest and task
  tiers.
- `household_basket_env/server/environment.py` implements the episode lifecycle
  and action validation pipeline.
- `household_basket_env/server/basket_grader.py` performs terminal basket-level
  grading.
- `household_basket_env/server/rewards.py` contains dense reward shaping.
- `household_basket_env/tests/` covers the environment contract, rewards,
  grader behavior, seed verifier, and reward-hack policies.
- `BLOG.md` gives the judge-facing product narrative.
- `HACKATHON_GUIDELINES_REPORT.md` shows where the project is on/off track
  against the hackathon guide.
- `docs/plan_for_finale.md` contains the full design rationale.

## Quickstart

```bash
cd household_basket_env
uv sync --extra dev
export HOUSEHOLD_BASKET_PRODUCTS_PATH="$PWD/data/products.json"
uv run uvicorn household_basket_env.server.app:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the FastAPI surface. The health endpoint
is `/health`; task definitions are exposed at `/tasks`.

## Run Tests

```bash
cd household_basket_env
HOUSEHOLD_BASKET_PRODUCTS_PATH="$PWD/data/products.json" \
  uv run --extra dev python -m pytest tests/
```

Expected result: `58 passed`.

## Project Layout

```text
.
├── BLOG.md
├── README.md
├── docs/
│   ├── plan_for_finale.md
│   └── results/
└── household_basket_env/
    ├── client.py
    ├── models.py
    ├── openenv.yaml
    ├── Dockerfile
    ├── data/products.json
    ├── notebooks/
    ├── scripts/push_to_hf.sh
    ├── seeds/
    ├── server/
    └── tests/
```

## Demo Path

1. Start the API server.
2. Reset Task 3 with a fixed seed.
3. Show the household constraints and candidate products.
4. Run a baseline or prompted action sequence.
5. Show dense reward components and terminal grading.
6. Run the trained model sequence once training artifacts are available.
7. Compare baseline vs trained reward and terminal outcome.

That path demonstrates the actual contribution: personalized, cumulative
constraint satisfaction under budget, not a thin grocery chatbot.
