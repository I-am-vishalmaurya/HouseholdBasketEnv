# HouseholdBasketEnv: Grocery Planning For A Whole Household

Most grocery assistants solve the easy version of the problem. They recommend a
single "healthy" item, summarize a nutrition label, or generate a generic
shopping list. Real households need something more careful: one basket that fits
different people with different health needs, food preferences, and budgets.

HouseholdBasketEnv turns that into an OpenEnv task. An agent builds a packaged
food basket for a small Indian household. Every choice has to respect the
household budget and the needs of individual members: sugar caps for a diabetic
adult, sodium caps for a hypertensive senior, protein and fiber floors for a
child, dietary restrictions, and enough meal variety to make the basket useful.

The core idea is simple: personalization is not just a sentence in a prompt. It
is a set of constraints that interact across people and across items.

## The Problem

Food looks different when you evaluate it as part of a basket. A snack can look
fine on its own and still push a diabetic member over their sugar limit after
three other choices. A low-sugar product can still be a bad fit for someone
watching sodium. A cheap basket can satisfy the budget and fail nutrition. A
diverse basket can still ignore one member entirely.

That makes grocery planning a good sequential decision-making problem. The agent
cannot just classify one product as good or bad. It has to track what has already
been selected, what each member still needs, and how much budget is left.

At each step, the agent returns one action:

```json
{
  "product_id": "item-from-candidate-list",
  "member_id": "household-member",
  "rationale": "short justification"
}
```

The environment updates the basket, remaining budget, cumulative nutrition, and
reward. At the end, a terminal grader checks whether the final basket satisfies
the household.

## Why This Is RL-Worthy

A prompt can read a label. The harder part is planning across the whole episode.

Two individually reasonable products can become unreasonable together. A model
that spends too much budget early may have no good choices left later. A model
that optimizes for one member may quietly fail the rest of the household. Those
are exactly the kinds of mistakes a sequential environment can expose.

HouseholdBasketEnv includes three task tiers:

- **Task 1:** one healthy adult, three valid steps, curated product set.
- **Task 2:** healthy adult plus diabetic adult, five valid steps.
- **Task 3:** diabetic adult, hypertensive senior, and child, seven valid steps
  over the full catalog with adversarial products.

Task 3 is the main challenge. It includes products that look attractive on the
surface but become risky when combined with the rest of the basket.

## What We Built

HouseholdBasketEnv is a FastAPI/OpenEnv package with:

- A deterministic environment server with `POST /reset`, `POST /step`,
  `GET /state`, and `GET /tasks`.
- Pydantic action and observation schemas for agent integration.
- A basket-level terminal grader for nutrient caps, intake floors, product
  compatibility, and budget.
- Dense reward shaping for valid format, nutrient thresholds, budget pacing, and
  meal-type coverage.
- Verified seed files for reproducible tasks.
- A 58-test suite covering environment contracts, rewards, grading, seed
  verification, and reward-hack policies.
- Colab notebooks for baseline evaluation, GRPO training, ablations, and plots.
- Docker/OpenEnv deployment support for Hugging Face Spaces.

The result is an evaluation environment, not a thin grocery chatbot. The agent is
judged by whether the final basket works.

## Reward Design

The reward combines step-level guidance with final basket grading.

Valid actions receive format reward. Nutrient thresholds reward choices near a
healthy band and penalize cap violations. Budget reward encourages the agent to
pace spending instead of exhausting the budget early. Meal-type coverage rewards
useful variety. Invalid JSON, duplicate products, unknown members, and
over-budget choices are penalized.

The terminal grader then scores the final basket:

- Full pass when every member satisfies all hard checks.
- Partial credit when individual choices are reasonable but cumulative basket
  constraints are not fully satisfied.
- Negative reward for hard cap violations.
- Zero terminal reward for exits where final grading is intentionally skipped.

This split keeps the training signal dense while preserving the real objective:
the final basket has to work for the household.

## Reward-Hack Resistance

Reward shaping is easy to game, so the project includes adversarial policy
tests. The suite checks obvious shortcuts such as buying the cheapest item,
repeating one category, assigning everything to one member, picking adversarial
products, or ignoring health entirely.

These tests do not claim the reward is perfect. They show that the most obvious
failure modes were considered and bounded.

## Training Results

The available evaluation artifacts are in `results/`.

| Run | Eval episodes | Mean of task mean rewards | Parse failure rate | Terminal success or partial rate | Result artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| Prompted baseline | 90 | `-1.4995` | `65.56%` | `22.22%` | `baseline.json` |
| GRPO trained model | 15 | `-1.8876` | `0.00%` | `0.00%` | `eval_main.json` |
| Ablation A | 15 | `-1.4911` | `0.00%` | `0.00%` | `eval_ablation_a.json` |
| Ablation B | 15 | `-1.6529` | `0.00%` | `0.00%` | `eval_ablation_b.json` |

Task 3 is the real stress test. The prompted baseline scored `-3.3827` mean
reward across 30 Task 3 seeds. The GRPO main run scored `-3.0713` across 5 Task
3 seeds. Ablation A scored `-2.9791`, and Ablation B scored `-2.8262`. None of
the available runs reached a positive terminal outcome on Task 3.

The honest read is that the environment is doing its job: it exposes where the
model fails instead of hiding those failures behind a polished shopping-list
demo. The newer evaluated runs fixed the JSON/action stability problem, but they
still do not solve the terminal basket objective. The next training iteration
should focus on terminal basket quality.

The new training artifacts also include saved adapters and logs for all three
GRPO runs:

| Run | Training mix | Final train reward | Artifact directory |
| --- | --- | ---: | --- |
| Main | 70% Task 2, 30% Task 3 | `-0.2500` | `household_basket_env/notebooks/runs/main/` |
| Ablation A | Main mix, meal-type coverage disabled | `-0.2500` | `household_basket_env/notebooks/runs/ablation_a/` |
| Ablation B | 100% Task 3 | `-0.2500` | `household_basket_env/notebooks/runs/ablation_b/` |

Ablation B is now evaluated. It has the best Task 3 mean reward among the
available trained runs, but still no positive terminal outcomes.

## Why It Fits Personalized Tasks

The household profile is the personalization signal. The same product can be
good for one member and bad for another. The same shopping strategy can succeed
for a healthy adult and fail for a household with diabetic, hypertensive, and
child nutrition constraints.

HouseholdBasketEnv makes personalization operational. The agent must satisfy the
specific household in front of it, under budget, across a full sequence of
choices.
