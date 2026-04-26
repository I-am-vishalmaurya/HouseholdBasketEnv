# Hackathon Guidelines Readiness Report

Source reviewed: **Hackathon Self-Serve Guide: Build an RL Environment, Train an LLM, Ship a Demo**.

Verdict: **mostly on track for the environment/reward part, still off track for proof-of-training and final judge demo evidence.**

This repo has a credible OpenEnv environment. It does not yet have enough committed evidence that a model improved through RL, that the deployed Space works end-to-end, or that post-training inference was saved and tested. For a hackathon, that gap matters because judges will reward measured improvement and a sharp demo, not just a well-designed environment.

## Executive Status

Overall status: **yellow / at risk, but recoverable**.

Strong:

- The project idea is step-based, verifiable, and personalized.
- OpenEnv-style server, client, schemas, Dockerfile, and manifest exist.
- Rewards are decomposed into multiple components.
- Reward-hack tests exist and currently pass.
- Curriculum tiers and verified seeds exist.
- README and BLOG now explain the contribution clearly.

Weak:

- No committed baseline-vs-trained result artifact is visible yet.
- No committed proof that GRPO/Unsloth training completed successfully.
- No committed remote deployment URL or Space health proof.
- No demo transcript showing baseline attempt, reward output, trained attempt, and measurable improvement.
- No model save/export validation artifact.
- Root license issue remains: source headers reference a root `LICENSE`, but no tracked `LICENSE` file is present.

## Guideline-by-Guideline Check

### 1. Pick the right project idea

Status: **on track**.

The guide says the task should allow step-by-step action, programmatic verification, and non-zero success probability. HouseholdBasketEnv fits this well:

- Step action: `{"product_id": str, "member_id": str, "rationale": str}`.
- Programmatic verification: basket grader and reward functions.
- Difficulty control: Task 1, Task 2, Task 3 curriculum.
- Non-zero success path: verified seeds and greedy verifier support.

Risk: keep the pitch focused. This is not "automate grocery shopping in real time." It is a verifiable personalized grocery-basket environment.

### 2. Understand and implement the RL loop

Status: **partially on track**.

The environment side of the loop is present: prompts, actions, step execution, rewards, and terminal grading. The training notebooks claim the TRL/Unsloth/GRPO path.

Missing evidence:

- No committed training logs proving rollouts ran.
- No committed before/after model behavior.
- No committed eval result showing reward improved after RL.

Judging impact: high. A nice environment without improvement evidence is only half the hackathon target.

### 3. Decide whether SFT is needed before RL

Status: **unclear / under-documented**.

The guide recommends using a capable instruct model, light formatting scaffolding if needed, then RL. The README says Qwen2.5-3B-Instruct plus GRPO notebooks, but the repo does not clearly show whether formatting priming, SFT, or prompt-only warm start was used.

Fix:

- Add a short section to the training notebook or report explaining why you skipped or used SFT.
- Show one baseline prompt output to prove the model can produce valid JSON before RL.

### 4. Design environment before trainer

Status: **on track**.

Evidence:

- `household_basket_env/openenv.yaml`
- `household_basket_env/models.py`
- `household_basket_env/server/environment.py`
- `household_basket_env/server/app.py`
- `household_basket_env/client.py`

The environment defines observation, action, state, reward, task tiers, and episode termination. This follows the guide.

### 5. Build using OpenEnv

Status: **mostly on track**.

The project is a Python package exposed through FastAPI and OpenEnv-style client/server files. The manifest declares `runtime: fastapi`, API endpoints, action model, observation model, tasks, and grading components.

Risk:

- Confirm `openenv push` succeeds after the port fix in `scripts/push_to_hf.sh`.
- Commit the deployed Space URL or a screenshot/log of `/health` and `/tasks`.

### 6. Keep task simple first / use curriculum

Status: **on track**.

The three-tier progression matches the guide:

- Task 1: one member, short horizon.
- Task 2: two members, medium horizon.
- Task 3: three members, full catalog, adversarial products.

This is exactly the right structure for avoiding zero-reward training.

### 7. Design rewards carefully

Status: **on track**.

The guide recommends multiple independent reward functions. This repo has:

- `R_format`
- `R_threshold`
- `R_budget`
- `R_meal_type_coverage`
- `R_terminal`
- `P_parse`
- `P_duplicate`
- `P_unknown_member`
- `P_over_budget`

That is substantially better than a single scalar reward. The split between dense rewards and terminal grading is the right call.

### 8. Protect against reward hacking

Status: **on track, with one caveat**.

Evidence:

- `household_basket_env/tests/test_reward_hacks.py`
- `docs/results/reward_hack_suite.json`

The project tests obvious bad policies: cheapest item, repeated category, duplicate products, adversarial set, terminal-only behavior, and assigning everything to one member.

Caveat:

- The report should not say reward hacking is "solved." Say obvious hacks are bounded. The current `BLOG.md` wording is appropriately cautious.

### 9. Use process-aware feedback

Status: **on track**.

Step-level reward breakdowns and per-action validation provide process feedback. This is a good fit for the guideline's recommendation: outcome-based verification plus lightweight process checks.

Missing improvement:

- For the demo, show reward breakdown after each step. Do not just show final score.

### 10. Pick the intended training stack: TRL + Unsloth + OpenEnv

Status: **partially on track**.

The docs and notebooks point to TRL, Unsloth, and GRPO. The package keeps training dependencies separate under the `training` optional extra.

Missing evidence:

- No committed training run directory.
- No committed adapter output.
- No committed training metric plot.
- No committed eval JSON for trained model.

Judging impact: high.

### 11. Prefer GRPO / RLVR for verifiable tasks

Status: **conceptually on track**.

This task is verifier-driven and suitable for GRPO/RLVR. The design doc and README align with that.

Missing evidence:

- Show the actual reward function plugged into GRPO.
- Show one tiny successful run, even if small.

### 12. Keep inference fast

Status: **unclear**.

The environment itself is lightweight and tests run quickly. But there is no throughput or rollout-time evidence for training/inference.

Fix:

- Add one simple benchmark: average reset/step time over a small number of episodes.
- If using Colab T4, record approximate rollout speed and model inference speed.

### 13. Deploy environment early

Status: **partially on track**.

Evidence:

- `Dockerfile`
- `openenv.yaml`
- `scripts/push_to_hf.sh`
- README deployment instructions

Problem:

- There is no committed deployment proof.
- The deploy script previously had a port mismatch. It has now been fixed to port `8000`, but it still needs a real deploy smoke test.

Fix:

- Run `HF_USERNAME=<handle> ./scripts/push_to_hf.sh`.
- Capture the Space URL.
- Verify `/health`, `/tasks`, `/reset`, and one `/step`.

### 14. Scale only after environment is stable

Status: **on track**.

The repo has environment tests, seed verifier tests, and reward tests before claiming training. That sequencing matches the guide.

Remaining risk:

- Do not spend hackathon time scaling training until deploy and demo proof are stable.

### 15. Monitor the right things during training

Status: **partially on track**.

The reward components exist, and the notebooks appear intended to emit logs. But the repo does not yet show committed monitoring artifacts.

Need:

- Overall reward curve.
- Per-component reward columns.
- Success rate or terminal-pass rate.
- Timeout / invalid-action rate.
- A few sampled generations before and after training.

### 16. Save models correctly

Status: **unknown / off track until proven**.

The guideline explicitly warns about LoRA/QLoRA saving. This repo currently does not show:

- final adapter path,
- merged-save path,
- post-training inference smoke test,
- or model card / artifact location.

Fix:

- In `training.ipynb`, make the save path explicit.
- Immediately reload the saved adapter and run one inference episode.
- Commit a small `docs/results/model_export_smoke.json` or equivalent.

### 17. Team split

Status: **not directly applicable, but workstreams are clear**.

The repo maps naturally to:

- Environment: server, models, client, OpenEnv manifest.
- Rewards/verifier: rewards, basket grader, seed verifier, reward-hack tests.
- Training: notebooks.
- Demo/product: README, BLOG, Space deployment.

If multiple people are working, assign ownership exactly this way.

### 18. One-day execution plan

Status: **environment phases complete; final phases incomplete**.

Done:

- Narrow task selected.
- Environment built.
- Rewards built.
- Curriculum added.
- Tests pass.
- Demo documentation improved.

Not done enough:

- Deploy proof.
- Tiny training proof.
- Reward-hack inspection during training.
- Bigger training / final eval.
- Saved model validation.
- Before/after demo.

### 19. What judges will find compelling

Status: **partially ready**.

Ready:

- Clear environment design.
- Objective reward functions.
- Reward-hack prevention.
- Reproducible local execution.
- Sharp written explanation.

Missing:

- Evidence that the model improved.
- Reproducible deployed Space story.
- Tight baseline-vs-trained demo.

This is the most important gap. Judges will likely ask: "Did RL actually make the model better?" Right now the repo does not visibly answer that.

### 20. Common mistakes to avoid

Status against mistakes:

- Too hard task: **avoided** with curriculum.
- One reward function: **avoided**.
- No reward-hacking checks: **avoided**.
- Training before stable env: **mostly avoided**.
- Only average reward: **risk remains** unless per-component training logs are shown.
- Forgetting timeouts/safety: **partially covered** by attempt caps and penalties.
- Incorrect LoRA save: **unknown**.

## Highest-Priority Fixes Before Judging

### P0: Produce evidence that RL helped

Minimum acceptable artifact:

- `docs/results/baseline.json`
- `docs/results/eval_main.json`
- one small reward curve image or JSON log
- three qualitative examples: baseline fail, trained improvement, adversarial case

No model-improvement evidence means the project looks like an environment-only submission.

### P0: Prove deployment

Minimum acceptable artifact:

- Hugging Face Space URL
- `/health` output
- `/tasks` output
- one reset/step transcript

The docs now describe deployment, but judges need proof it runs.

### P1: Add a demo script or notebook cell

Minimum demo flow:

1. Reset Task 3.
2. Print household constraints.
3. Run baseline action sequence.
4. Show reward breakdown and terminal grade.
5. Run trained model action sequence.
6. Show improved terminal grade.

Do not improvise this live. Script it.

### P1: Document model save/export

Add a small section covering:

- base model,
- adapter path,
- save method,
- reload test,
- inference smoke result.

This directly addresses the guideline's LoRA/QLoRA warning.

### P2: Add license file

Several files reference a root `LICENSE`. Add the license or remove the headers. This is not the biggest judging issue, but it is sloppy and easy to fix.

## Final Call

HouseholdBasketEnv is **on track as an OpenEnv environment** and **off track as a complete hackathon RL demo until training/deployment evidence is committed**.

The fastest winning move is not more architecture. It is proof:

- run the tiny training path,
- save before/after results,
- deploy the Space,
- script the demo,
- and keep the story focused on verifiable personalized constraint satisfaction.
