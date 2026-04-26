# HouseholdBasketEnv: Grocery Planning Is a Multi-Person Constraint Problem

Most grocery assistants answer the wrong question. They recommend a healthy item,
or summarize a food label, or produce a generic shopping list. Real households
need something harder: a basket that works for everyone at once.

HouseholdBasketEnv is an OpenEnv hackathon environment where an agent assembles
a packaged-food basket for a small Indian household under realistic constraints.
The same basket may need to respect a diabetic adult's sugar cap, a hypertensive
senior's sodium cap, a child's protein and fiber floors, dietary restrictions,
meal variety, and a hard INR budget.

That is the core idea: personalization is not a user preference string. It is a
set of constraints that interact across people and across items.

## Where We Stand

The environment is the strong part. It has a real OpenEnv/FastAPI interface,
deterministic task tiers, multiple reward components, verified seeds, and tests
for obvious reward hacks. That puts the project on track for the "build an RL
environment" part of the hackathon guide.

The incomplete part is proof. A judge should not have to infer that RL helped.
Before final judging, the project still needs committed evidence for:

- baseline-vs-trained evaluation results,
- a completed GRPO/Unsloth run,
- a deployed Space smoke test,
- a model reload / adapter export smoke test,
- and a scripted demo that compares baseline and trained behavior.

That is not cosmetic. The hackathon target is not just "build an environment";
it is "build an environment, train an LLM, ship a demo." Right now the public
story should be honest: the environment is credible, the final proof artifacts
are the remaining work.

## The Problem

Single-item food scoring is easy to demo and weak in practice. A snack can look
fine by itself but become a bad choice after three other products have already
been added. A low-sugar product may still be poor for a hypertensive member. A
cheap basket can satisfy the budget and fail nutrition. A diverse basket can
still overrun sodium.

HouseholdBasketEnv turns this into a repeatable decision-making task. At each
step, the agent chooses:

```json
{
  "product_id": "item-from-candidate-list",
  "member_id": "household-member",
  "rationale": "short justification"
}
```

The environment then updates the basket, budget, cumulative per-member nutrition,
and reward. At the end of the episode, a terminal grader evaluates whether the
full basket satisfies the household.

## Why This Is RL-Worthy

A prompt can read a label. The harder part is tracking cumulative effects.

Two products that are individually acceptable can jointly violate a member's cap.
The agent has to plan across a sequence, not classify one row of data. Dense
reward components provide learning signal during the episode, while the terminal
grader preserves the real objective: the final basket must work.

The environment includes three difficulty tiers:

- Task 1: one healthy adult, three valid steps, curated product set.
- Task 2: healthy adult plus diabetic adult, five valid steps.
- Task 3: diabetic adult, hypertensive senior, and child, seven valid steps over
  the full catalog with adversarial products.

Task 3 is the interesting one. It contains products that look attractive on the
surface but are risky in combination, forcing the agent to reason beyond
marketing claims.

## What We Built

HouseholdBasketEnv is a FastAPI/OpenEnv package with:

- A deterministic environment server with `POST /reset`, `POST /step`,
  `GET /state`, and `GET /tasks`.
- Pydantic action and observation schemas for agent integration.
- A basket-level terminal grader for caps, floors, per-item compatibility, and
  budget.
- Dense reward shaping for valid format, nutrient thresholds, budget pacing, and
  meal-type coverage.
- Verified seed files for reproducible tasks.
- A 58-test suite covering environment contracts, rewards, grading, seed
  verification, and reward-hack policies.
- Colab notebooks for baseline evaluation, GRPO training, ablations, and plots.
- Docker/OpenEnv deployment support.

The important distinction: this is not a grocery chatbot. It is an evaluation
environment for personalized, multi-stakeholder constraint satisfaction.

## The Reward Design

The reward combines local feedback with final grading.

Valid actions receive format reward. Nutrient thresholds use a smooth reward
curve that peaks near a healthy band and penalizes cap violations. Budget reward
encourages pacing instead of spending everything early. Meal-type coverage
encourages basket diversity. Invalid JSON, duplicate products, unknown members,
and over-budget choices are penalized.

The terminal grader then scores the final basket:

- Full pass when every member satisfies all hard checks.
- Partial credit when individual items are sane but cumulative constraints fail.
- Negative reward for hard cap violations.
- Zero terminal reward for attempt-cap and over-budget exits where the grader is
  intentionally skipped.

This split matters. Dense rewards help training, but the terminal grader keeps
the environment honest.

## Reward-Hack Resistance

The project includes adversarial policy tests because reward shaping is easy to
game. The suite checks strategies like buying the cheapest item, repeating one
category, assigning everything to one member, picking adversarial products, or
ignoring health entirely.

These tests do not prove the reward is perfect. They do prove the obvious hacks
were considered and bounded. That is the difference between a toy environment
and one worth evaluating.

## What To Demo

Start the server:

```bash
cd household_basket_env
uv sync --extra dev
export HOUSEHOLD_BASKET_PRODUCTS_PATH="$PWD/data/products.json"
uv run uvicorn household_basket_env.server.app:app --reload --port 8000
```

Then open `http://localhost:8000/docs`.

For judges, the strongest demo path is not a free-form live walkthrough. Script
it:

1. Reset Task 3 with a fixed seed.
2. Show the household: diabetic adult, hypertensive senior, child.
3. Show candidate products and remaining budget.
4. Run a baseline or prompted model action sequence.
5. Show the reward breakdown and terminal grader result.
6. Run the trained model action sequence.
7. Show the improved reward, terminal grade, or reduced invalid-action rate.
8. Point to the reward-hack tests and verified seeds.

That tells the real story in under five minutes.

## Why It Fits Personalized Tasks

The household profile is the personalization signal. The same product can be
good for one member and bad for another. The same shopping strategy can succeed
for a healthy adult and fail badly for a diabetic-hypertensive-child household.

HouseholdBasketEnv makes personalization operational: the agent must satisfy the
specific household in front of it, not optimize for a generic average user.

## What Is Next

The strongest next step is to show measured model improvement over prompted and
random baselines on held-out Task 3 seeds. The environment already has the
pieces for that: deterministic seeds, training notebooks, ablations, and an eval
notebook. The judge-facing proof becomes much stronger once the result artifacts
are committed under `docs/results/`.

The concrete gap-closure plan lives in
`docs/superpowers/plans/2026-04-26-hackathon-gap-closure.md`. The order matters:
prove deployment, run a tiny training/eval loop, validate model save/reload,
then polish the demo. More architecture will not save the submission if those
proof artifacts are missing.
