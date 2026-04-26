# Hackathon Gap Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the missing judge-facing proof that HouseholdBasketEnv is not just a good environment, but a complete OpenEnv RL hackathon submission with deployment, training evidence, model-save validation, and a scripted demo.

**Architecture:** Keep the environment code stable and add proof artifacts around it. The work is split into deployment evidence, baseline/trained evaluation artifacts, training-monitoring output, model export validation, demo scripting, and license cleanup. All outputs should land in `docs/results/` or a small demo script so judges can inspect them without rerunning expensive training.

**Tech Stack:** Python 3.10+, OpenEnv/FastAPI, uv, pytest, Docker, Hugging Face Spaces, TRL, Unsloth, Qwen2.5-3B-Instruct, LoRA/QLoRA adapters.

---

## File Structure

- Modify: `README.md` to replace "at risk" wording with final proof links after artifacts exist.
- Modify: `BLOG.md` to include final baseline-vs-trained numbers after artifacts exist.
- Modify: `HACKATHON_GUIDELINES_REPORT.md` to mark completed items after evidence is generated.
- Modify: `household_basket_env/notebooks/baseline_eval.ipynb` only if it does not already emit `docs/results/baseline.json`.
- Modify: `household_basket_env/notebooks/training.ipynb` to make adapter save/reload paths explicit.
- Modify: `household_basket_env/notebooks/eval_and_plots.ipynb` only if it does not already emit trained eval JSON and qualitative samples.
- Create: `docs/results/deployment_smoke.json`.
- Create: `docs/results/baseline.json`.
- Create: `docs/results/eval_main.json`.
- Create: `docs/results/training_summary_main.json`.
- Create: `docs/results/model_export_smoke.json`.
- Create: `docs/results/demo_transcript_main.md`.
- Create: `household_basket_env/scripts/demo_task3.py`.
- Create: `LICENSE` or remove license headers that claim one exists.

---

### Task 1: Prove Local Environment Still Works

**Files:**
- Read: `household_basket_env/README.md`
- Read: `household_basket_env/tests/`

- [ ] **Step 1: Run the documented test command**

Run:

```bash
cd household_basket_env
HOUSEHOLD_BASKET_PRODUCTS_PATH="$PWD/data/products.json" \
  uv run --extra dev python -m pytest tests/
```

Expected:

```text
58 passed
```

- [ ] **Step 2: Record test evidence in the demo transcript**

Create or append `docs/results/demo_transcript_main.md`:

````markdown
# HouseholdBasketEnv Demo Transcript

## Local Verification

Command:

```bash
cd household_basket_env
HOUSEHOLD_BASKET_PRODUCTS_PATH="$PWD/data/products.json" \
  uv run --extra dev python -m pytest tests/
```

Result: 58 tests passed.
````

- [ ] **Step 3: Commit local verification artifact**

Run:

```bash
git add docs/results/demo_transcript_main.md
git commit -m "docs: record local verification evidence"
```

Expected: commit succeeds without changing environment behavior.

---

### Task 2: Prove Hugging Face Space Deployment

**Files:**
- Use: `household_basket_env/scripts/push_to_hf.sh`
- Create: `docs/results/deployment_smoke.json`
- Modify: `README.md`

- [ ] **Step 1: Push the environment**

Run:

```bash
cd household_basket_env
HF_USERNAME=<your-hf-handle> ./scripts/push_to_hf.sh
```

Expected:

```text
[done] pushed to https://huggingface.co/spaces/<your-hf-handle>/household-basket-env
```

- [ ] **Step 2: Smoke-test deployed endpoints**

Replace `<space-url>` with the real Space URL.

Run:

```bash
curl -fs <space-url>/health
curl -fs <space-url>/tasks
```

Expected:

```text
/health returns HTTP 200
/tasks returns the three household_basket_task entries
```

- [ ] **Step 3: Run one remote reset and step**

Use the request shape exposed by the deployed FastAPI docs. Save the exact request and response bodies.

Create `docs/results/deployment_smoke.json`:

```json
{
  "space_url": "https://huggingface.co/spaces/<your-hf-handle>/household-basket-env",
  "health": {
    "status_code": 200,
    "body": "paste exact /health response"
  },
  "tasks": {
    "status_code": 200,
    "task_count": 3
  },
  "reset": {
    "status_code": 200,
    "task_id": 3,
    "seed": 42
  },
  "step": {
    "status_code": 200,
    "reward": 0.0,
    "done": false,
    "reward_breakdown_present": true
  }
}
```

- [ ] **Step 4: Add Space link to README**

Update `README.md`:

```markdown
## Deployed Space

Space: `<actual-space-url>`

Smoke-test evidence: `docs/results/deployment_smoke.json`.
```

- [ ] **Step 5: Commit deployment proof**

Run:

```bash
git add README.md docs/results/deployment_smoke.json
git commit -m "docs: add deployment smoke evidence"
```

Expected: the README links to real deployment evidence.

---

### Task 3: Produce Baseline Evaluation Evidence

**Files:**
- Use or modify: `household_basket_env/notebooks/baseline_eval.ipynb`
- Create: `docs/results/baseline.json`

- [ ] **Step 1: Run baseline evaluation notebook**

Run the notebook locally or in Colab until it emits `docs/results/baseline.json`.

Expected JSON shape:

```json
{
  "run_name": "baseline",
  "model": "prompted-base-or-random-policy-name",
  "task_results": [
    {
      "task_id": 3,
      "num_episodes": 30,
      "mean_reward": 0.0,
      "terminal_pass_rate": 0.0,
      "invalid_action_rate": 0.0
    }
  ],
  "qualitative_examples": [
    {
      "seed": 42,
      "outcome": "paste concise baseline behavior summary"
    }
  ]
}
```

- [ ] **Step 2: Verify baseline artifact exists**

Run:

```bash
test -f docs/results/baseline.json && python -m json.tool docs/results/baseline.json >/dev/null
```

Expected: command exits with code `0`.

- [ ] **Step 3: Commit baseline artifact**

Run:

```bash
git add docs/results/baseline.json household_basket_env/notebooks/baseline_eval.ipynb
git commit -m "docs: add baseline evaluation evidence"
```

Expected: baseline performance is visible without rerunning the notebook.

---

### Task 4: Run Tiny GRPO/Unsloth Training and Save Metrics

**Files:**
- Use or modify: `household_basket_env/notebooks/training.ipynb`
- Create: `docs/results/training_summary_main.json`

- [ ] **Step 1: Make training config explicit**

In `household_basket_env/notebooks/training.ipynb`, ensure the top config cell includes:

```python
RUN_NAME = "main"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = f"runs/{RUN_NAME}"
ADAPTER_DIR = f"{OUTPUT_DIR}/adapter_final"
TRAIN_MIX = {"task2": 0.70, "task3": 0.30}
```

- [ ] **Step 2: Run a small proof training job**

Use the smallest run that proves the loop works. Record:

- total rollouts,
- mean reward start,
- mean reward end,
- invalid-action rate,
- terminal-pass rate,
- adapter output path.

- [ ] **Step 3: Save training summary**

Create `docs/results/training_summary_main.json`:

```json
{
  "run_name": "main",
  "base_model": "Qwen/Qwen2.5-3B-Instruct",
  "trainer": "GRPO",
  "efficiency_stack": "Unsloth",
  "adapter_dir": "runs/main/adapter_final",
  "num_rollouts": 0,
  "mean_reward_start": 0.0,
  "mean_reward_end": 0.0,
  "invalid_action_rate_start": 0.0,
  "invalid_action_rate_end": 0.0,
  "terminal_pass_rate_end": 0.0,
  "notes": "Replace numeric zeros with actual run values."
}
```

Replace every numeric zero with measured values from the run before committing.

- [ ] **Step 4: Commit training summary**

Run:

```bash
git add household_basket_env/notebooks/training.ipynb docs/results/training_summary_main.json
git commit -m "docs: add GRPO training summary"
```

Expected: judges can see that training actually ran.

---

### Task 5: Validate Model Save and Reload

**Files:**
- Use or modify: `household_basket_env/notebooks/training.ipynb`
- Create: `docs/results/model_export_smoke.json`

- [ ] **Step 1: Reload saved adapter**

After training, reload the saved adapter from:

```text
runs/main/adapter_final
```

Run one Task 3 inference episode using seed `42`.

- [ ] **Step 2: Save model export smoke result**

Create `docs/results/model_export_smoke.json`:

```json
{
  "run_name": "main",
  "base_model": "Qwen/Qwen2.5-3B-Instruct",
  "adapter_dir": "runs/main/adapter_final",
  "reload_success": true,
  "task_id": 3,
  "seed": 42,
  "episode_completed": true,
  "total_reward": 0.0,
  "invalid_actions": 0,
  "save_method": "adapter_final",
  "qlora_warning_addressed": true
}
```

Replace `total_reward` and `invalid_actions` with actual values.

- [ ] **Step 3: Commit export smoke evidence**

Run:

```bash
git add docs/results/model_export_smoke.json household_basket_env/notebooks/training.ipynb
git commit -m "docs: add model export smoke evidence"
```

Expected: the LoRA/QLoRA save warning from the hackathon guide is directly addressed.

---

### Task 6: Produce Trained Evaluation Evidence

**Files:**
- Use or modify: `household_basket_env/notebooks/eval_and_plots.ipynb`
- Create: `docs/results/eval_main.json`

- [ ] **Step 1: Evaluate trained adapter**

Run evaluation on held-out Task 3 seeds and any Task 1/2 regression seeds already used by the notebook.

- [ ] **Step 2: Save trained eval artifact**

Create `docs/results/eval_main.json`:

```json
{
  "run_name": "main",
  "adapter_dir": "runs/main/adapter_final",
  "heldout_task3": {
    "num_episodes": 30,
    "mean_reward": 0.0,
    "terminal_pass_rate": 0.0,
    "invalid_action_rate": 0.0
  },
  "comparison_to_baseline": {
    "mean_reward_delta": 0.0,
    "terminal_pass_rate_delta": 0.0,
    "invalid_action_rate_delta": 0.0
  },
  "qualitative_examples": [
    {
      "seed": 42,
      "baseline_summary": "paste concise baseline behavior",
      "trained_summary": "paste concise trained behavior",
      "why_it_improved": "paste concrete improvement"
    }
  ]
}
```

Replace every zero and placeholder sentence with measured data from the run.

- [ ] **Step 3: Commit trained eval evidence**

Run:

```bash
git add docs/results/eval_main.json household_basket_env/notebooks/eval_and_plots.ipynb
git commit -m "docs: add trained evaluation evidence"
```

Expected: the project can answer "Did RL make it better?" with data.

---

### Task 7: Add Scripted Demo

**Files:**
- Create: `household_basket_env/scripts/demo_task3.py`
- Create or modify: `docs/results/demo_transcript_main.md`

- [ ] **Step 1: Add demo script**

Create `household_basket_env/scripts/demo_task3.py`:

```python
"""Scripted Task 3 demo for judges.

Usage:
    uv run python scripts/demo_task3.py --base-url http://localhost:8000 --seed 42
"""

from __future__ import annotations

import argparse

from household_basket_env import BasketAction, HouseholdBasketEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with HouseholdBasketEnv(base_url=args.base_url).sync() as env:
        result = env.reset(seed=args.seed, task_id=3)
        obs = result.observation

        print("Task 3 household")
        for member in obs.household:
            print(f"- {member.member_id}: {member.conditions}")

        for step in range(obs.max_steps):
            candidate = obs.candidates[step]
            member = obs.household[step % len(obs.household)]
            action = BasketAction(
                product_id=candidate.product_id,
                member_id=member.member_id,
                rationale="scripted baseline action for demo trace",
            )
            result = env.step(action)
            obs = result.observation
            print(
                f"step={step + 1} reward={result.reward} "
                f"done={result.done} breakdown={obs.reward_breakdown}"
            )
            if result.done:
                break

        print(f"terminal_reward={obs.terminal_reward}")
        print(f"terminated_reason={obs.terminated_reason}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run scripted demo against local server**

In terminal 1:

```bash
cd household_basket_env
export HOUSEHOLD_BASKET_PRODUCTS_PATH="$PWD/data/products.json"
uv run uvicorn household_basket_env.server.app:app --reload --port 8000
```

In terminal 2:

```bash
cd household_basket_env
uv run python scripts/demo_task3.py --base-url http://localhost:8000 --seed 42
```

Expected: the script prints household members, per-step rewards, reward breakdowns, and terminal result.

- [ ] **Step 3: Save demo transcript**

Append the exact script output to `docs/results/demo_transcript_main.md`.

- [ ] **Step 4: Commit demo script and transcript**

Run:

```bash
git add household_basket_env/scripts/demo_task3.py docs/results/demo_transcript_main.md
git commit -m "docs: add scripted judge demo"
```

Expected: the live demo can be rehearsed and reproduced.

---

### Task 8: Update Public Story With Final Evidence

**Files:**
- Modify: `README.md`
- Modify: `BLOG.md`
- Modify: `HACKATHON_GUIDELINES_REPORT.md`

- [ ] **Step 1: Replace at-risk language with measured results**

After Tasks 2-7 are complete, update `README.md` and `BLOG.md` with actual values:

```markdown
## Final Hackathon Evidence

- Deployment smoke: `docs/results/deployment_smoke.json`
- Baseline eval: `docs/results/baseline.json`
- Trained eval: `docs/results/eval_main.json`
- Training summary: `docs/results/training_summary_main.json`
- Model export smoke: `docs/results/model_export_smoke.json`
- Demo transcript: `docs/results/demo_transcript_main.md`
```

- [ ] **Step 2: Update readiness report statuses**

In `HACKATHON_GUIDELINES_REPORT.md`, update these sections from partial/off-track to on-track only after artifacts exist:

- RL loop evidence.
- Training stack evidence.
- Deployment proof.
- Monitoring artifacts.
- Model save/export.
- Judge demo.

- [ ] **Step 3: Commit final story updates**

Run:

```bash
git add README.md BLOG.md HACKATHON_GUIDELINES_REPORT.md
git commit -m "docs: update hackathon readiness evidence"
```

Expected: public docs match the real evidence in `docs/results/`.

---

### Task 9: Fix License Mismatch

**Files:**
- Create: `LICENSE`
- Or modify headers in source files

- [ ] **Step 1: Decide license**

If this project is intended to inherit the BSD-style license referenced in source headers, create `LICENSE` with the correct BSD-style license text.

If not, remove or replace the header block that says:

```text
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
```

- [ ] **Step 2: Verify license references**

Run:

```bash
rg "LICENSE file in the root directory" .
```

Expected: either the root `LICENSE` exists, or no source file references a nonexistent license file.

- [ ] **Step 3: Commit license cleanup**

Run:

```bash
git add LICENSE README.md household_basket_env
git commit -m "chore: fix license metadata"
```

Expected: repository metadata no longer looks sloppy to reviewers.

---

## Self-Review

Spec coverage:

- The missing baseline-vs-trained proof is covered by Tasks 3, 4, 6, and 8.
- The missing deployment proof is covered by Task 2.
- The missing demo transcript is covered by Task 7.
- The missing model save/export proof is covered by Task 5.
- The missing license cleanup is covered by Task 9.

Placeholder scan:

- The plan includes commands, expected outputs, and exact artifact paths.
- Template JSON snippets intentionally contain values that must be replaced with measured outputs before commit; the steps explicitly require replacement.

Execution recommendation:

- Do Tasks 1, 2, and 7 first because they unblock the judge demo.
- Do Tasks 3, 4, 5, and 6 next because they prove RL improvement.
- Do Tasks 8 and 9 last as cleanup and public-story alignment.
