# HouseholdBasketEnv

Multi-stakeholder, constraint-satisfaction grocery-basket environment for OpenEnv.
The agent assembles a basket of `N` packaged-food items for a small Indian household,
balancing per-member nutritional caps and minimum-intake floors, allergen / dietary
restrictions, meal-type variety, and a hard INR budget.

This is **Module 2** of the OpenEnv exercise. See
[`docs/plan_for_finale.md`](../docs/plan_for_finale.md) for the full design.

---

## TL;DR

- **Tasks:** 3 difficulty tiers (1 member / 3 steps, 2 members / 5 steps, 3 members /
  7 steps + adversarial set).
- **Action:** `{"product_id": str, "member_id": str, "rationale": str}` — one JSON
  per step.
- **Reward:** dense per-step shaping (`R_format`, `R_threshold`, `R_budget`,
  `R_meal_type_coverage`) + sparse terminal grader in `{-0.5, 0.0, +0.3, +1.0}`.
- **Policy:** Qwen2.5-3B-Instruct fine-tuned with **Unsloth + GRPO** (per-step
  bandit view).
- **Base data:** reuses `module1/food_label_auditor/data/products.json` verbatim,
  augmented in-memory with `meal_type`, `price_inr`, `protein_g`, `fiber_g`.

---

## Repo layout

```
module2/
├── docs/
│   ├── plan_for_finale.md        # Source of truth design doc
│   └── results/                  # baseline / eval / curves outputs
└── household_basket_env/
    ├── client.py                 # HouseholdBasketEnv client
    ├── models.py                 # BasketAction / BasketObservation / BasketState
    ├── openenv.yaml              # OpenEnv manifest
    ├── pyproject.toml            # Build + deps
    ├── Dockerfile                # openenv-base multi-stage
    ├── data/products.json        # Mirror of module1 catalog
    ├── seeds/                    # verified_task1.json / 2.json / 3.json
    ├── notebooks/
    │   ├── baseline_eval.ipynb   # Phase 4.5 / Phase 5
    │   ├── training.ipynb        # Phases 6 + 7 + 8 (main + ablations)
    │   └── eval_and_plots.ipynb  # Phases 9 + 10
    ├── scripts/
    │   └── push_to_hf.sh         # Phase 11 deploy
    ├── server/
    │   ├── app.py                # FastAPI bootstrap
    │   ├── environment.py        # 5-check action pipeline + episode lifecycle
    │   ├── catalog.py            # In-memory augmentation of products.json
    │   ├── household_fixtures.py # Members + ICMR-NIN caps/floors
    │   ├── curriculum.py         # Tier configs + seed registry
    │   ├── basket_grader.py      # Terminal grader
    │   ├── rewards.py            # Dense reward components
    │   └── seed_verifier.py      # Offline solvability check
    └── tests/                    # 58 tests, all green
```

---

## Quickstart (local)

### 1. Setup

```bash
cd module2/household_basket_env
uv sync                               # or: pip install -e .[dev]
export HOUSEHOLD_BASKET_PRODUCTS_PATH=$PWD/data/products.json
```

### 2. Run the env server

```bash
uvicorn household_basket_env.server.app:app --reload --port 7860
```

Browse `http://localhost:7860/docs` for the auto-generated API surface.
The health endpoint lives at `/health`; task definitions at `/tasks`.

### 3. Use the client

```python
from household_basket_env import HouseholdBasketEnv

env = HouseholdBasketEnv(base_url="http://localhost:7860")
obs = env.reset(seed=42, task_id=2)

action = {"product_id": obs.candidates[0].product_id,
          "member_id": obs.household[0].member_id,
          "rationale": "high-fiber breakfast staple"}
result = env.step(action)
print(result.reward, result.done, result.reward_breakdown)
```

### 4. Run the test suite

```bash
HOUSEHOLD_BASKET_PRODUCTS_PATH=$PWD/data/products.json \
  python -m pytest tests/
```

Expected: **58 / 58 passed** (env contract, dense rewards, terminal grader,
reward-hack suite, seed verifier).

### 5. Generate verified seeds

```bash
python -m household_basket_env.server.seed_verifier --task 1
python -m household_basket_env.server.seed_verifier --task 2
python -m household_basket_env.server.seed_verifier --task 3
```

Output lands in `seeds/verified_task{1,2,3}.json` and is consumed by the
training + eval notebooks.

---

## Training (Google Colab T4)

Three notebooks — open them in Colab in order:

| # | Notebook | Phase | Output |
|---|----------|-------|--------|
| 1 | `notebooks/baseline_eval.ipynb` | §4.5 / §5 | `docs/results/baseline.json` |
| 2 | `notebooks/training.ipynb` | §6 + §7 + §8 | `runs/<RUN_NAME>/adapter_final/`, `training_log.json` |
| 3 | `notebooks/eval_and_plots.ipynb` | §9 + §10 | `docs/results/eval_<RUN_NAME>.json`, `curves_<RUN_NAME>.png`, `qualitative_<RUN_NAME>.json` |

The training notebook has a single config cell at the top:

```python
RUN_NAME = "main"             # "main" | "ablation_a" | "ablation_b"
```

- **`main`** — full reward + 70/30 Task 2/3 curriculum (plan §6)
- **`ablation_a`** — `enable_meal_type_coverage=False` (plan §7.A)
- **`ablation_b`** — Task 3 only, no curriculum (plan §7.B)

For each run, restart the notebook and change `RUN_NAME`.

---

## Reward at a glance

| Signal | Value | Fires on |
|--------|-------|----------|
| `R_format` | `+0.20` | Action parses + member exists + product exists + not duplicate + within budget |
| `R_threshold` | `+0.10` peak / `+0.02` under / `-0.30` per nutrient violation, summed and clipped to `[-0.6, +0.4]` | Each valid step, per watched nutrient |
| `R_budget` | `+0.10` | Pick price ≤ `budget_remaining / steps_remaining` |
| `R_meal_type_coverage` | `+0.15` | First time `(member_id, meal_type)` appears in this episode |
| `R_terminal` | `+1.0` / `+0.3` / `0.0` / `-0.5` | At max_steps: full pass / partial / over-budget / hard violation |
| `P_parse` | `-0.25` | JSON parse / schema fail (does NOT advance step) |
| `P_duplicate` | `-0.30` | Already-tagged product (does NOT advance step) |
| `P_unknown_member` | `-0.40` | Bad `member_id` (does NOT advance step) |
| `P_over_budget` | terminates | Pick exceeds remaining budget |

See `server/rewards.py` and `server/basket_grader.py` for the implementations
and `tests/test_reward_hacks.py` for the 10 adversarial-policy bounds that
guarantee robustness against trivial gaming.

---

## Deploy to HuggingFace Space

```bash
HF_USERNAME=<your-handle> ./scripts/push_to_hf.sh
```

The script:

1. Builds the Docker image off `openenv-base`.
2. Verifies `/health` passes locally.
3. Calls `openenv push` to upload to `hf://<HF_USERNAME>/household-basket-env`.

Set `HF_TOKEN` for non-interactive pushes.

---

## Status

- [x] Phase 1–4 — env, client, server, manifest, Dockerfile, tests (58/58 green)
- [x] Phase 1.5 — verified seeds for all 3 tiers
- [x] Phase 4.5 — baseline eval notebook
- [x] Phase 6/7/8 — training notebook (main + Ablation A + Ablation B)
- [x] Phase 9/10 — eval + plots notebook
- [x] Phase 11 — deploy script + this README
- [ ] Run on Colab T4 (user-driven)
