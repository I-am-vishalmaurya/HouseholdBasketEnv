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

HouseholdBasketEnv is an OpenEnv environment for a simple but surprisingly hard
question: can an agent build one grocery basket that works for an entire
household?

The agent selects packaged-food items for a small Indian household while staying
inside a fixed INR budget. Each choice has to respect member-specific nutrition
caps, minimum intake floors, dietary restrictions, and meal variety. A product
that looks healthy in isolation can still be a bad choice once it is combined
with the rest of the basket.

This is not a grocery chatbot. It is a reproducible environment for testing
whether an agent can plan across a sequence of choices under real constraints.

## Links

- [Project blog](BLOG.md)
- [Evaluation results](results/)
- [Training run artifacts](household_basket_env/notebooks/runs/)
- [Hugging Face Space](https://huggingface.co/spaces/5h4dy/household-basket-env)

## What the Environment Tests

HouseholdBasketEnv turns personalized grocery planning into a step-by-step RL
task:

- The household profile defines the personalization target.
- The product catalog includes prices, nutrition labels, meal types, and
  adversarial items.
- The agent chooses one product and one household member at each step.
- Dense rewards give feedback during the episode.
- A terminal grader checks whether the final basket actually works.

The hardest task includes a diabetic adult, a hypertensive senior, and a child.
The same basket has to satisfy all three, not just optimize for an average user.

## Results

Evaluation artifacts are in `results/`.

| Run | Eval episodes | Mean of task mean rewards | Parse failure rate | Terminal success or partial rate | Result file |
| --- | ---: | ---: | ---: | ---: | --- |
| Prompted baseline | 90 | `-1.4995` | `65.56%` | `22.22%` | [`baseline.json`](results/baseline.json) |
| GRPO main | 15 | `-1.8876` | `0.00%` | `0.00%` | [`eval_main.json`](results/eval_main.json) |
| Ablation A | 15 | `-1.4911` | `0.00%` | `0.00%` | [`eval_ablation_a.json`](results/eval_ablation_a.json) |
| Ablation B | 15 | `-1.6529` | `0.00%` | `0.00%` | [`eval_ablation_b.json`](results/eval_ablation_b.json) |

Task 3 is the hardest household: diabetic adult, hypertensive senior, and child.
On Task 3, the prompted baseline scored `-3.3827` mean reward across 30 seeds;
the GRPO main run scored `-3.0713` across 5 seeds; Ablation A scored `-2.9791`;
Ablation B scored `-2.8262`. None of the available runs reached a positive
terminal outcome on Task 3.

Available artifacts:

- `results/baseline.json`
- `results/eval_main.json`
- `results/eval_ablation_a.json`
- `results/eval_ablation_b.json`
- `results/curves_main.png`
- `results/curves_ablation_a.png`
- `results/curves_ablation_b.png`
- `results/qualitative_main.json`
- `results/qualitative_ablation_a.json`
- `results/qualitative_ablation_b.json`

Training run artifacts:

| Run | Training mix | Final train reward | Adapter / log |
| --- | --- | ---: | --- |
| GRPO main | 70% Task 2, 30% Task 3 | `-0.2500` | [`runs/main`](household_basket_env/notebooks/runs/main/) |
| Ablation A | Main mix, meal-type coverage disabled | `-0.2500` | [`runs/ablation_a`](household_basket_env/notebooks/runs/ablation_a/) |
| Ablation B | 100% Task 3 | `-0.2500` | [`runs/ablation_b`](household_basket_env/notebooks/runs/ablation_b/) |

## Quickstart

```bash
cd household_basket_env
uv sync --extra dev
export HOUSEHOLD_BASKET_PRODUCTS_PATH="$PWD/data/products.json"
uv run uvicorn household_basket_env.server.app:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the API docs. The health endpoint is
`/health`; task definitions are exposed at `/tasks`.

## Run Tests

```bash
cd household_basket_env
HOUSEHOLD_BASKET_PRODUCTS_PATH="$PWD/data/products.json" \
  uv run --extra dev python -m pytest tests/
```

Expected result: `58 passed`.

