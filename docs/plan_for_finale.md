# HouseholdBasketEnv — Design Document v3.1

**Status:** v3.1 — polish pass on top of v3 (judge-perspective red-team review).
**Theme:** OpenEnv Hackathon #3.2 — Personalized Tasks
**Target:** OpenEnv hackathon finale, Bangalore, Apr 26 2026
**Changes from v3:** `BasketAction` schema added (§2), attempt-cap terminal behavior clarified (R_terminal skipped, dense rewards retained — §4.7), naming-binding table added (§1), explicit repo-deliverable paths enumerated (§10.5), README skeleton added (§10.6), HF Space slimness risk added (§12). Reward weights, GRPO config, task tiers, seed verifier, and demo script unchanged from v3.
**Changes from v2 (carried forward):** training distribution restructured, prompted-baseline gate added, two ablations, demo rewritten for storytelling, R_threshold smoothed, KL/entropy monitoring tightened, framing rewritten to lead with novelty.

---

## 0. The pitch (read this first)

**HouseholdBasketEnv is a multi-stakeholder constraint-satisfaction environment for personalized grocery planning.** A single agent must compose a basket that simultaneously satisfies the conflicting nutritional constraints of multiple household members — e.g., a diabetic adult's sugar cap and a growing child's protein floor and a hypertensive grandparent's sodium cap — under a hard budget, drawing from a real catalog of 492 Indian packaged-food products with FSSAI-derived nutrition data.

What makes this RL-worthy and not prompt-solvable: **the constraints couple across items**. Two products that look healthy in isolation can jointly violate a member's cap. The agent has to learn the coupling structure, not just read labels. We demonstrate this with a held-out hard tier (Task 3) the model never sees during training and an adversarial product set designed to fool surface-level pattern matching.

This fits Theme #3.2 (Personalized Tasks) because the household profile *is* the personalization signal — the same prompt produces different optimal baskets for different households, and the agent has to internalize "what does this specific family need."

---

## 1. Architectural principle

`household_basket_env/` is an OpenEnv-compliant package. Per-item nutritional sanity uses a vendored grading utility (`food_label_auditor`); **basket-level scoring, multi-member constraint reasoning, and all RL-relevant logic are new code in this package**. The basket grader is the contribution.

File layout:

```
household_basket_env/
├── __init__.py
├── models.py
├── client.py
├── openenv.yaml
├── pyproject.toml
├── server/
│   ├── environment.py
│   ├── app.py
│   ├── rewards.py
│   ├── basket_grader.py        ← cumulative basket-level check (ours)
│   ├── curriculum.py
│   ├── household_fixtures.py
│   ├── seed_verifier.py        ← offline solver, proves seeds are achievable
│   ├── Dockerfile
│   └── requirements.txt
└── tests/
    ├── test_env_contract.py
    ├── test_rewards.py
    ├── test_basket_grader.py
    ├── test_reward_hacks.py    ← 10 adversarial policies
    └── test_seed_verifier.py
```

### 1.1 Naming-binding table

Every reference to the environment must agree with this table. Anything that drifts from it is a bug. Mirrors the `FoodLabelAuditorEnv` / `food_label_auditor` pattern from module1.

| Surface | Value |
|---|---|
| `openenv.yaml` `name` | `household_basket_env` |
| `openenv.yaml` `client.class_name` | `HouseholdBasketEnv` |
| `openenv.yaml` `client.module` | `household_basket_env.client` |
| `openenv.yaml` `action.class_name` | `BasketAction` |
| `openenv.yaml` `action.module` | `household_basket_env.models` |
| `openenv.yaml` `observation.class_name` | `BasketObservation` |
| `openenv.yaml` `observation.module` | `household_basket_env.models` |
| `openenv.yaml` `default_image` | `household-basket-env:latest` |
| Package directory | `module2/household_basket_env/` |
| Python import path | `from household_basket_env.client import HouseholdBasketEnv` |
| Docker image tag | `household-basket-env:latest` |
| HF Space slug | `household-basket-env` |
| README H1 | `HouseholdBasketEnv: Multi-Stakeholder Nutrition Reasoning` |

---

## 2. Schemas

### 2.1 `BasketAction`

The agent emits exactly one of these per step. JSON shape, Pydantic-validated server-side.

```python
class BasketAction(BaseModel):
    product_id: str = Field(
        ...,
        description="ID of the product to add to the basket. Must be present in the current observation's `candidates` list."
    )
    member_id: str = Field(
        ...,
        description="ID of the household member this item is tagged to. Must be present in the current observation's `household` list."
    )
    rationale: str | None = Field(
        None,
        description="Optional one-line natural-language justification. Logged for inspection, never used in reward."
    )
```

**Validation pipeline (server-side, in order):**

1. **JSON parse** — payload must deserialize to a JSON object. Failure → `parse_error="invalid_json"`, `R_format = -0.25` (P_parse), `attempt_index += 1`, `step_index` unchanged.
2. **Pydantic validation** — fields and types must match. Failure → `parse_error="schema_mismatch: <pydantic msg>"`, P_parse fires.
3. **Membership checks:**
   - `product_id` must be in current `candidates` → else `parse_error="unknown_product"`, P_parse fires.
   - `member_id` must be in current `household` → else `parse_error="unknown_member"`, **P_unknown_member = -0.4** fires (not P_parse — this is the one validation failure with its own penalty per §4.6).
4. **Duplicate check** — if `product_id` already in `basket_so_far`, action is rejected with **P_duplicate = -0.3** (§4.6). `step_index` does not advance.
5. **Budget check** — if `cumulative_spend + product.price > budget`, episode terminates immediately per §4.6 P_over_budget.

Only an action that passes all 5 checks advances `step_index` and accrues the positive R_format / R_threshold / R_budget / R_meal_type_coverage signals.

### 2.2 `BasketObservation`

| Field | Type | Purpose |
|---|---|---|
| `prompt` | str | task instructions + current situation |
| `household` | list[MemberSummary] | per-member conditions, cumulative intake so far, thresholds |
| `basket_so_far` | list[TaggedItem] | items + member tags |
| `budget_remaining` | float | INR |
| `candidates` | list[ProductSummary] | tier-dependent catalog |
| `step_index` | int | counts valid steps only |
| `attempt_index` | int | counts every step including parse errors; for logging |
| `max_steps` | int | 3 / 5 / 7 |
| `seed` | int | echoed for reproducibility debugging |
| `reward` | float | last step's dense reward |
| `done` | bool | |
| `parse_error` | str \| None | populated on invalid action |

`MemberSummary` exposes `cumulative_intake` (dict: nutrient → quantity consumed across items tagged to this member) and `thresholds_cap` (hard caps from profile). The agent can reason about margin to cap explicitly.

### 2.3 `BasketState`

Internal-only, never exposed to the agent: `seed: int`, `rng` (seeded `random.Random`), `verified: bool` (checked against `seed_verifier` at reset), `cumulative_spend: float`, `attempt_index: int`.

### 2.4 Reset determinism

**Reset is deterministic from seed.** Household composition, budget jitter (±10%), and candidate ordering all derive from the seed via the internal RNG. Two resets with the same seed produce byte-identical observations. This invariant is asserted in `test_env_contract.py`.

---

## 3. Episode lifecycle and task tiers

| Tier | Members | Valid steps | Catalog | Budget (INR) | Role |
|---|---|---|---|---|---|
| Task 1 | 1 healthy adult | 3 | 20 curated | 500 | **Held-out easy eval** (regression check) |
| Task 2 | 2 (healthy + diabetic) | 5 | 50 | 1000 | **Training (70%)** |
| Task 3 | 3 (diabetic + hypertensive + child) | 7 | 492 incl. 32 adversarial | 1500 | **Training (30%) + held-out hard eval seeds** |

**Adversarial products in Task 3:** 32 items deliberately constructed (or curated) so each looks healthy in isolation — "low fat," "no added sugar," "high protein" marketing — but combinations of them violate caps. A trained model should learn to deprioritize this set; a prompted model picks them readily because each label looks fine.

**Train/eval split for Task 3:** 100 verified seeds total, 70 used during training, 30 held out. Held-out seeds are never seen during GRPO and form the headline eval number.

---

## 4. Reward decomposition

Four positive signals plus terminal, all dense except R_terminal.

### 4.1 R_format

+0.2 dense when action JSON validates. **−0.25 on parse error** (matched to format-reward scale).

### 4.2 R_threshold — smooth triangular

For each watched nutrient on the member tagged to the picked item:

```
let m = (post-pick cumulative intake) / (member's cap)
R_per_nutrient(m) =
    -0.30                       if m > 1.00            (violation)
    +0.10 * (1 - |m - 0.60|/0.40)  if 0.20 ≤ m ≤ 1.00   (triangular peak at 60% of cap)
    +0.02                       if m < 0.20            (under-consumption, mildly positive)
```

Triangular kernel peaks at +0.10 when post-pick intake is at 60% of cap (the "healthy band center"), decays linearly to 0 at 20% and 100%, and crashes to −0.30 above cap. Sum across the member's watched nutrients, clip to [−0.6, +0.4] per step.

This closes the "buy water forever" loophole (water parks the agent at m≈0, earning only +0.02 per nutrient) and creates a smooth gradient toward genuinely healthy choices.

### 4.3 R_budget

+0.1 when item's price ≤ per-step allowance (`budget_remaining / steps_remaining`).

### 4.4 R_meal_type_coverage

Each catalog product has a `meal_type` tag: `staple`, `protein`, `vegetable`, `dairy`, `snack`, `beverage`. +0.15 dense when the picked item's meal_type is new to the member's tagged subset. Hard cap: one bonus per meal_type per member.

**Honest framing:** this is a dense proxy for the terminal-floor compliance check, not an orthogonal signal. It rewards meal-type diversity early so the agent doesn't fill the basket with one category and fail the floor at terminal. The ablation (§7) measures whether this proxy actually helps vs. relying on terminal alone.

### 4.5 R_terminal

Delegates to `basket_grader.py`:

1. For each member, sum cumulative nutrients from products in their tagged subset.
2. Check each sum against the member's threshold caps.
3. Check each member hits minimum-intake floors for calorie, protein, fiber (ICMR-NIN derived).
4. For each item individually, run per-item sanity (allergen check, profile compatibility).

Terminal reward:
- **+1.0** if every member passes all four checks
- **+0.3** partial credit if every item passes per-item sanity but cumulative caps violated
- **−0.5** if any member exceeds a cap (hard violation)

### 4.6 Penalties

| Penalty | Weight | Fires when |
|---|---|---|
| P_parse | −0.25 | invalid JSON |
| P_duplicate | −0.3 | product_id already in basket |
| P_unknown_member | −0.4 | member_id not in household |
| P_over_budget | terminates | cumulative spend > budget |

### 4.7 Step-advance semantics and attempt-cap exit

**Counters:**
- `step_index` advances only on valid actions (passed all 5 checks in §2.1).
- `attempt_index` advances every step including parse errors, schema mismatches, unknown-product, unknown-member, duplicate.

**Per-attempt cost:** Parse / schema / membership failures cost `P_parse = -0.25` (or `P_unknown_member = -0.4` for the member-id case). Duplicates cost `P_duplicate = -0.3`. The agent gets another attempt within the same episode.

**Two terminal paths:**

1. **Normal terminal** — `step_index == max_steps`. `basket_grader.py` runs on the final basket, returns `R_terminal ∈ {+1.0, +0.3, -0.5}` per §4.5. `done=True`.
2. **Attempt-cap exit** — `attempt_index >= max_steps × 2` triggers *before* `step_index` reaches `max_steps`. Episode terminates with:
   - `done = True`
   - `terminal_reward = 0.0` — **`basket_grader.py` is NOT invoked.**
   - `final_episode_reward = sum(dense_rewards_so_far)` — typically deeply negative because parse penalties have accumulated.
   - Observation surfaces a new field `terminated_reason: "attempt_cap"` for logging.

**Why skip the grader on attempt-cap exit:** the partial basket is artificially short, which mechanically makes some terminal checks (cumulative caps) easier to pass — running the grader would perversely reward thrashing-into-shorter-baskets. Accumulated parse penalties already encode the cost proportionally: 14 parse errors = 14 × −0.25 = −3.5, which is 7× worse than the worst possible terminal (−0.5). The signal is loud enough.

**`test_env_contract.py` invariants (must pass before training):**

```python
def test_attempt_cap_skips_terminal():
    # Force max_steps × 2 parse errors, assert grader never runs
    env.reset(seed=42)
    for _ in range(env.max_steps * 2):
        result = env.step(BasketAction(product_id="garbage", member_id="garbage"))
    assert result.done is True
    assert result.observation.terminated_reason == "attempt_cap"
    assert result.terminal_reward == 0.0
    assert grader_call_count == 0  # via spy/mock

def test_normal_terminal_runs_grader():
    # Take exactly max_steps valid actions, assert grader runs once
    env.reset(seed=42)
    for action in valid_actions_for_seed_42:
        result = env.step(action)
    assert result.done is True
    assert result.observation.terminated_reason == "max_steps"
    assert result.terminal_reward in {-0.5, 0.3, 1.0}
    assert grader_call_count == 1
```

---

## 5. Reward-hack suite — 10 policies

| Policy | Expected max reward | Test asserts |
|---|---|---|
| AlwaysBuyRice | < 1.0 | terminal floor violation for non-rice-tolerant members |
| AlwaysCheapestItem | < 1.5 | threshold violations dominate |
| AlwaysSameCategory | < 2.0 | meal-type coverage never fires past 1 |
| MinCostIgnoreHealth | < 1.0 | terminal fails for diabetic/hypertensive |
| RandomValidJSON | measured p(reward>0) ≥ tier target | baseline reproducibility |
| OneMemberGetsEverything | < half of max | other members fail terminal |
| BuySameItemNTimes | < 1.5 | P_duplicate dominates |
| EmptyBasketPolicy | exactly 0 | bounded-below check |
| AlwaysPickFromAdversarialSet | < 1.5 on Task 3 | adversarial items fail per-item sanity at terminal |
| TerminalOnlyPolicy | bounded by dense reward floor | proves terminal alone can't be gamed without dense accrual |

All 10 must pass before training starts.

---

## 6. Training plan

**Model:** Qwen2.5-3B-Instruct, Unsloth 4-bit QLoRA, rank 16, alpha 32, dropout 0.05.

**Training distribution:** Task 2 (70%) + Task 3 training seeds (30%). Task 1 is held-out eval. Task 3 held-out seeds (30 of 100) are never seen during training.

**GRPO config:**
- 4 prompts × 8 generations = 32 rollouts/step
- temperature 0.8, top_p 0.9
- max_new_tokens = 128
- learning_rate 5e-6, cosine schedule
- **beta = 0.1** (KL penalty)
- reference model = base Qwen2.5-3B-Instruct, LoRA off
- **80 GRPO steps** main run, checkpoint every 10 steps

**Monitoring (per step):**
- Mean reward, broken down by reward component
- Valid-JSON rate
- **Per-token KL divergence** — inspect at 2.0, abort at 4.0
- **Action-distribution entropy** — guards against mode collapse
- Sample 4 generations per step into a log file for spot inspection

**Success criteria:**
- Mean reward on Task 2 climbs from prompted baseline to ≥ +30% delta
- Mean reward on Task 3 **held-out seeds** shows positive delta vs prompted baseline
- Mean reward on Task 1 (held-out, never trained on) within 10% of prompted baseline (no regression)
- Valid-JSON rate ≥ 99% by step 40
- Per-token KL stays under 4.0 throughout
- Entropy doesn't collapse (defined: action-distribution entropy stays above 50% of initial value)

---

## 7. Ablations — two runs

Both at 40 GRPO steps each, evaluated on the same Task 2 + Task 3 held-out seeds as the main run.

**Ablation A — no R_meal_type_coverage.** Tests whether the dense meal-type proxy is doing real work or merely shadowing the terminal floor. Hypothesis: removing it slows convergence and causes more terminal-floor failures. If results show no difference, R_meal_type_coverage was redundant and we say so honestly in the README.

**Ablation B — no curriculum, Task 3 only from step 0.** Tests whether the 70/30 mix helps. Hypothesis: pure Task 3 has sparser reward, slower convergence, possibly higher final ceiling on Task 3 itself. Useful either way — confirms the curriculum decision or tells us we left performance on the table.

Results from both ablations go in a single table in the README. This is the kind of evidence judges almost never see at hackathons.

---

## 8. Seed verification

Before any curriculum seed ships, it must be proven achievable. `seed_verifier.py`:

1. For each seed and each tier, reconstruct the household, catalog, budget.
2. Greedy pick: at each step, choose the product maximizing R_threshold + R_meal_type_coverage, breaking budget ties toward cheaper.
3. Run all dense + terminal rewards on the greedy basket.
4. Seed verified iff greedy achieves reward ≥ 0.6 × theoretical max.

Only verified seeds enter `curriculum.py`. Target: 100 verified seeds per tier. Run as offline script, results checked into repo.

---

## 9. Demo script — 115 seconds

| Time | Beat | Content |
|---|---|---|
| 0–15s | **Hook** | "Indian household, three members, conflicting health needs, 1500 rupees, 7 picks. Diabetic grandfather, hypertensive mother, growing child. What goes in the cart?" Show the household profile slide. |
| 15–30s | **Why this is hard** | Show two products both labeled "low fat" — explain that picking both pushes the diabetic over his sugar cap. "Constraints couple across items. You can't solve this by reading labels." |
| 30–45s | **The environment** | Quick repo tour, `reset(seed=42)` → observation. Show seed verifier output: "every training seed provably solvable." |
| 45–65s | **Baseline fails** | Prompted Qwen2.5-3B on a held-out Task 3 seed. It picks the two low-fat-but-high-sugar items. Terminal reward: −0.5. Show reasoning trace. |
| 65–85s | **Trained model succeeds** | Same seed, trained model. It avoids the coupling, picks paneer + dal + atta + vegetables. Terminal: +1.0. Show reward curve: prompted baseline → trained, on Task 3 held-out. |
| 85–100s | **Qualitative before/after** | Side-by-side basket comparison on a second hard seed. One-line caption per basket explaining what the model learned. **This is the storytelling slide.** |
| 100–110s | **Honesty slide** | Ablation table (A and B), reward-hack suite green, KL stayed under 2.5 throughout. One failure case shown: "trained model still struggles when budget < 1200 — here's why." |
| 110–115s | **Close** | Repo URL, Space URL, README link. |

**Recording:** locally via Docker against a stable container, not against live Colab. Pre-record once, review, re-record if needed.

---

## 10. Execution order

1. **Phase 1** — scaffolding, schemas, Dockerfile, `openenv validate` ✅
2. **Phase 1.5** — seed verifier implementation + offline run, produce verified seed lists for all three tiers
3. **Gate A — Baseline sanity** — 50 random-valid rollouts on Task 2 verified seeds, confirm p(reward>0) ≈ 0.20
4. **Phase 2** — core env logic, `test_env_contract.py` green
5. **Phase 3** — rewards module + `basket_grader.py`, unit tests green
6. **Phase 4** — reward-hack suite (10 policies), all bounded as expected
7. **Phase 4.5 — Prompted baseline gate (NEW, critical)** — run prompted Qwen2.5-3B on 30 seeds each of Task 2 and Task 3 held-out. **If prompted Qwen scores within 15% of theoretical max on Task 2, the task is too easy and we escalate difficulty before training.** Specific escalation levers: tighten budget, add adversarial products to Task 2, increase member count to 3.
8. **Phase 5** — full baseline eval notebook, numbers recorded for Task 1, Task 2, Task 3 held-out
9. **Phase 6** — main GRPO run, 80 steps, mixed Task 2 + Task 3
10. **Phase 7** — Ablation A (no R_meal_type_coverage), 40 steps
11. **Phase 8** — Ablation B (Task 3 only, no curriculum), 40 steps
12. **Phase 9** — trained eval on all three tiers' held-out seeds, deltas recorded, KL/entropy curves saved
13. **Phase 10** — qualitative before/after generation: pick 3 hard seeds, generate baskets from prompted vs. trained, write captions
14. **Phase 11** — `openenv push`, README with embedded plots and ablation table, demo recording

---

## 10.5 Repo deliverables — explicit artifact paths

Every artifact below must exist in the repo by Phase 11. Anything missing is a submission gap. Paths are relative to repo root.

| Artifact | Path | Produced by | Phase |
|---|---|---|---|
| Verified seed list, Task 1 | `module2/household_basket_env/seeds/verified_task1.json` | `seed_verifier.py` | 1.5 |
| Verified seed list, Task 2 | `module2/household_basket_env/seeds/verified_task2.json` | `seed_verifier.py` | 1.5 |
| Verified seed list, Task 3 | `module2/household_basket_env/seeds/verified_task3.json` | `seed_verifier.py` | 1.5 |
| Reward-hack suite results | `module2/docs/results/reward_hack_suite.json` | `tests/test_reward_hacks.py` (export mode) | 4 |
| Prompted baseline numbers | `module2/docs/results/baseline_eval.json` (Task 1 / 2 / 3 held-out, mean reward + component breakdown) | Phase 5 notebook cell | 5 |
| Reward curves | `module2/docs/results/reward_curves.png` (baseline vs trained on same axes, both axes labeled, component-breakdown subplot) | Phase 9 notebook | 9 |
| KL & entropy curves | `module2/docs/results/kl_entropy_curves.png` | Phase 9 notebook | 9 |
| Ablation table | `module2/docs/results/ablation_table.md` (main vs Ablation A vs Ablation B, three rows) | Phase 7 + 8 + 9 | 9 |
| Qualitative before/after | `module2/docs/results/qualitative_examples.md` (3 hard seeds, baseline basket vs trained basket, one-line caption per row) | Phase 10 | 10 |
| Trained LoRA checkpoint | `module2/checkpoints/main_run_step80/` (or HF Hub link if size > 1 GB) | Phase 6 | 6 |
| Training notebook | `module2/notebooks/training.ipynb` (Colab-runnable: baseline eval → main run → ablations → held-out eval → plot generation) | Phase 5 onward, finalized Phase 11 | 11 |
| Demo video | external (YouTube unlisted or HF Spaces video tab); URL in README | Phase 11 | 11 |
| Blog post (optional) | external; URL in README if written | Phase 11 | 11 |

**Sizing rule:** anything under `module2/docs/results/` should be < 5 MB total. Plots are PNG at reasonable DPI, not vector. Notebook outputs cleared before commit (use `nbstripout` or `--ClearOutputPreprocessor.enabled=True`).

---

## 10.6 README skeleton

The README at `module2/household_basket_env/README.md` (the file judges open first) follows this section order. Total target length: 800–1200 words plus tables and 2 embedded images.

1. **H1:** `HouseholdBasketEnv: Multi-Stakeholder Nutrition Reasoning` (per §1.1 binding table)
2. **§0 pitch** — lifted verbatim from this plan's §0, three short paragraphs
3. **Quickstart** — `pip install`, `docker run`, `from household_basket_env.client import HouseholdBasketEnv` minimal example, 6 lines max
4. **Why it's RL-worthy** — 1 paragraph on coupled constraints + adversarial set; link to §0 plan and seed-verifier output
5. **Reward curves** — embed `reward_curves.png`, 1 caption sentence: "prompted baseline vs trained, Task 3 held-out, both axes labeled"
6. **Ablation table** — embed `ablation_table.md` content directly (not link); 1 paragraph interpreting whether each ablation supported the design
7. **Qualitative before/after** — 2–3 examples from `qualitative_examples.md`; one caption per example explaining what the model learned
8. **Honesty section** — heading: "Where it still fails." One named failure mode (e.g., "budget < 1200 forces tradeoffs the model hasn't learned"), one example, one sentence on what would fix it
9. **Reward-hack suite** — table with 10 policies, expected max, measured max, pass/fail — lifted from §5
10. **Reproducing the results** — bullet list: link to `notebooks/training.ipynb`, mention checkpoint path, verified seed list paths, expected runtime (Colab T4)
11. **Links** — Colab notebook URL, HF Space URL, demo video URL, blog post URL (if exists), upstream OpenEnv repo
12. **Citation / license / acknowledgements** — BSD-style header per [CLAUDE.md](CLAUDE.md), FSSAI/ICMR-NIN attribution, upstream OpenEnv attribution

**Storytelling note:** sections 5–8 (curves, ablation, qualitative, honesty) are the 30% storytelling weight. They must be visible above the fold without scrolling past dense reward math. Keep §4 (Why RL-worthy) and §5 (curves) tight so a judge skimming the first screen sees: pitch → why hard → trained model wins → here's where it still loses.

---

## 11. Out of scope for v1

Pantry decay, recipe graphs, 14-day horizons, conflict subgames, LLM-as-judge, images, tool use, multi-agent (one shopper per episode).

---

## 12. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Prompted baseline matches trained model** | **Medium-high** | **Phase 4.5 gate catches this before training. If hit, escalate Task 2 difficulty.** |
| Task 3 adversarial set is gameable through reward hacking | Medium | `AlwaysPickFromAdversarialSet` test exists; if it fails, fix grader not test |
| GRPO mode collapse | Medium | Entropy monitoring at every step; if entropy halves, raise temperature mid-run or abort |
| KL climbs past 4.0 during training | Medium | Inspect at 2.0, raise beta to 0.2 if pattern persists |
| Seed verifier marks too few Task 3 seeds verifiable | Low-medium | Relax greedy threshold to 0.5 × max; if still bad, Task 3 catalog is too adversarial — tune |
| Colab disconnect mid-training | High | Checkpoint every 10 steps to Drive, notebook auto-resumes |
| Demo recording fails on live Colab | High | Record locally against Docker container, not live Colab |
| `openenv push` fails on dependency layout | Low | Vendor dependencies as copied sub-package if needed |
| HF Space bundle bloat | Medium | Space contains code + Dockerfile + small seed JSONs only. Plots, video, training notebook outputs, and LoRA checkpoint > 1 GB hosted externally (HF Hub model repo, YouTube unlisted) and linked from README. Target Space size < 50 MB. Verify with `du -sh` before `openenv push`. |
| Ablation results contradict main hypothesis | Low (and fine) | Report honestly. Negative results in ablations build credibility, not destroy it. |

---

## 13. Mapping to judging criteria

| Criterion | Weight | How v3 addresses it |
|---|---|---|
| Environment Innovation | 40% | Multi-stakeholder constraint-satisfaction framing; coupled-constraint structure that defeats prompted baselines; 492-product catalog with adversarial subset; held-out hard eval; seed verifier proves seeds are non-trivially achievable. |
| Storytelling | 30% | Demo opens with concrete Indian household scenario; qualitative before/after slide on a hard seed; honesty slide showing one failure case + ablation table; clear theme fit (#3.2 Personalized Tasks). |
| Reward Improvement | 20% | Prompted baseline (not random-valid) as the reference; held-out Task 3 seeds as headline number; Task 1 regression check; reward curves embedded in README with both axes labeled. |
| Reward & Pipeline | 10% | 10 reward-hack policies, all green; smooth differentiable R_threshold; KL + entropy monitoring; two ablations validating reward design choices. |