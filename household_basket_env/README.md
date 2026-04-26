---
title: HouseholdBasketEnv
emoji: 🛒
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
short_description: Multi-member grocery basket planning environment for OpenEnv
---

# HouseholdBasketEnv

HouseholdBasketEnv is a multi-member grocery basket environment for OpenEnv.
The agent builds a basket of packaged-food items for a small Indian household
while balancing nutrition caps, minimum intake floors, dietary restrictions,
meal variety, and a fixed INR budget.

The environment is designed for sequential decision making. Each product choice
changes the remaining budget and the cumulative nutrition profile for a specific
household member, so the agent has to plan across the whole basket instead of
scoring one item at a time.

See [`docs/plan_for_finale.md`](../docs/plan_for_finale.md) for the full design.

## Links

- [Project blog](BLOG.md)
- [Evaluation results](results/)
- [Training run artifacts](notebooks/runs/)
- [Live API docs](https://5h4dy-household-basket-env.hf.space/docs)

## At a Glance

- **Tasks:** three tiers, from one member and three steps to a three-member
  household with seven steps and adversarial products.
- **Action:** `{"product_id": str, "member_id": str, "rationale": str}`.
- **Reward:** dense per-step shaping plus a terminal basket grader.
- **Training path:** Qwen2.5-3B-Instruct fine-tuning with Unsloth and GRPO.
- **Data:** packaged-food catalog in `data/products.json`, augmented with
  `meal_type`, `price_inr`, `protein_g`, and `fiber_g`.

## Repo Layout

```text
.
├── docs/
│   ├── plan_for_finale.md
│   └── results/
└── household_basket_env/
    ├── client.py
    ├── models.py
    ├── openenv.yaml
    ├── pyproject.toml
    ├── Dockerfile
    ├── data/products.json
    ├── seeds/
    ├── notebooks/
    │   ├── baseline_eval.ipynb
    │   ├── training.ipynb
    │   └── eval_and_plots.ipynb
    ├── scripts/push_to_hf.sh
    ├── server/
    └── tests/
```

## Quickstart

```bash
cd household_basket_env
uv sync --extra dev
export HOUSEHOLD_BASKET_PRODUCTS_PATH=$PWD/data/products.json
uv run uvicorn household_basket_env.server.app:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the API docs. The health endpoint is
`/health`; task definitions are exposed at `/tasks`.

## Use The Client

```python
from household_basket_env import BasketAction, HouseholdBasketEnv

with HouseholdBasketEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(seed=42, task_id=2)
    obs = result.observation

    action = BasketAction(
        product_id=obs.candidates[0].product_id,
        member_id=obs.household[0].member_id,
        rationale="high-fiber breakfast staple",
    )
    result = env.step(action)
    print(result.reward, result.done, result.observation.reward_breakdown)
```

## Tests

```bash
HOUSEHOLD_BASKET_PRODUCTS_PATH=$PWD/data/products.json \
  uv run --extra dev python -m pytest tests/
```

Expected result: **58 / 58 passed**.

The tests cover the environment contract, dense rewards, terminal grading, seed
verification, and adversarial reward-hack policies.

## Training Notebooks

Open the notebooks in this order:

| # | Notebook | Purpose | Main output |
| --- | --- | --- | --- |
| 1 | `notebooks/baseline_eval.ipynb` | Prompted/random baseline evaluation | `results/baseline.json` |
| 2 | `notebooks/training.ipynb` | GRPO training and ablations | `runs/RUN_NAME/adapter_final/` |
| 3 | `notebooks/eval_and_plots.ipynb` | Trained-model evaluation and plots | `results/eval_RUN_NAME.json` |

The training notebook uses a single run-name setting:

```python
RUN_NAME = "main"  # "main" | "ablation_a" | "ablation_b"
```

- `main`: full reward with a 70/30 Task 2/3 curriculum.
- `ablation_a`: meal-type coverage reward disabled.
- `ablation_b`: Task 3 only, without curriculum.

## Results

Evaluation artifacts are stored in `results/`.

| Run | Eval episodes | Mean of task mean rewards | Parse failure rate | Terminal success or partial rate | Result file |
| --- | ---: | ---: | ---: | ---: | --- |
| Prompted baseline | 90 | `-1.4995` | `65.56%` | `22.22%` | [`results/baseline.json`](results/baseline.json) |
| GRPO main | 15 | `-1.8876` | `0.00%` | `0.00%` | [`results/eval_main.json`](results/eval_main.json) |
| Ablation A | 15 | `-1.4911` | `0.00%` | `0.00%` | [`results/eval_ablation_a.json`](results/eval_ablation_a.json) |
| Ablation B | 15 | `-1.6529` | `0.00%` | `0.00%` | [`results/eval_ablation_b.json`](results/eval_ablation_b.json) |

Task 3 results are the hardest to satisfy: baseline `-3.3827`, GRPO main
`-3.0713`, Ablation A `-2.9791`, and Ablation B `-2.8262` mean reward. None of
the available runs received positive terminal reward on Task 3.

Published result artifacts:

- [`results/baseline.json`](results/baseline.json)
- [`results/eval_main.json`](results/eval_main.json)
- [`results/eval_ablation_a.json`](results/eval_ablation_a.json)
- [`results/eval_ablation_b.json`](results/eval_ablation_b.json)
- [`results/curves_main.png`](results/curves_main.png)
- [`results/curves_ablation_a.png`](results/curves_ablation_a.png)
- [`results/curves_ablation_b.png`](results/curves_ablation_b.png)
- [`results/qualitative_main.json`](results/qualitative_main.json)
- [`results/qualitative_ablation_a.json`](results/qualitative_ablation_a.json)
- [`results/qualitative_ablation_b.json`](results/qualitative_ablation_b.json)

Training artifacts are available for all three GRPO runs:

| Run | Training mix | Final train reward | Adapter / log |
| --- | --- | ---: | --- |
| GRPO main | 70% Task 2, 30% Task 3 | `-0.2500` | [`notebooks/runs/main`](notebooks/runs/main/) |
| Ablation A | Main mix, meal-type coverage disabled | `-0.2500` | [`notebooks/runs/ablation_a`](notebooks/runs/ablation_a/) |
| Ablation B | 100% Task 3 | `-0.2500` | [`notebooks/runs/ablation_b`](notebooks/runs/ablation_b/) |

## Reward Summary

| Signal | Value | Fires on |
| --- | --- | --- |
| `R_format` | `+0.20` | Valid JSON, known member, known product, no duplicate, within budget |
| `R_threshold` | `+0.10` peak / `+0.02` under / `-0.30` per violation, clipped to `[-0.6, +0.4]` | Watched nutrients on each valid step |
| `R_budget` | `+0.10` | Price stays within the budget pace |
| `R_meal_type_coverage` | `+0.15` | First time a member receives a meal type |
| `R_terminal` | `+1.0` / `+0.3` / `0.0` / `-0.5` | Final basket grading |
| `P_parse` | `-0.25` | JSON parse or schema failure |
| `P_duplicate` | `-0.30` | Product already selected |
| `P_unknown_member` | `-0.40` | Unknown member id |
| `P_over_budget` | terminates | Product exceeds remaining budget |

See `server/rewards.py`, `server/basket_grader.py`, and
`tests/test_reward_hacks.py` for the implementation and reward-hack checks.

## Deploy to Hugging Face Space

```bash
HF_USERNAME=your-handle ./scripts/push_to_hf.sh
```

The script builds the Docker image, checks `/health` locally, and pushes the
environment to `hf://HF_USERNAME/household-basket-env`. Set `HF_TOKEN` for
non-interactive pushes.
