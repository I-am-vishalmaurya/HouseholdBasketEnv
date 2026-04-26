# Generated from: eval_and_plots-albation_b.ipynb
# Converted at: 2026-04-26T10:30:40.357Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # HouseholdBasketEnv — Held-out Eval + Plots (Phases 9 + 10)
# 
# After GRPO training, we:
# 
# 1. **§9** Run held-out eval per task tier with the trained adapters and compare against `docs/results/baseline.json`.
# 2. **§10** Plot reward / KL / entropy curves from `runs/<RUN_NAME>/training_log.json`.
# 3. **§10.3** Surface 5 qualitative episode samples for the demo / report.
# 
# Set `RUN_NAME` to choose which run to evaluate (`main`, `ablation_a`, `ablation_b`).
# 
# > **Hardware:** Local GPU or Colab T4. The first cells clone the repo and install deps automatically.


RUN_NAME = "main"   # "main" | "ablation_a" | "ablation_b"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEEDS_PER_TIER = 5
MAX_NEW_TOKENS = 96
TEMPERATURE = 0.2
LOAD_IN_4BIT = True
MAX_SEQ_LEN = 2048

import os, sys, pathlib, subprocess


CLONE_DIR = pathlib.Path("/app/hackathon/HouseholdBasketEnv")

REPO_ROOT = CLONE_DIR.resolve()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ENV_ROOT  = REPO_ROOT / "household_basket_env"
RESULTS_DIR = ENV_ROOT / "notebooks" / "results"
RUN_DIR = ENV_ROOT / "notebooks" / "runs" / RUN_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR = RUN_DIR / "adapter_final"

os.environ["HOUSEHOLD_BASKET_PRODUCTS_PATH"] = str(ENV_ROOT / "data" / "products.json")
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))

print("REPO_ROOT =", REPO_ROOT)
print("RUN_DIR   =", RUN_DIR, "exists =", RUN_DIR.exists())
print("ADAPTER   =", ADAPTER_DIR, "exists =", ADAPTER_DIR.exists())

# ## 1. Load model + adapters


model, tokenizer = None, None
model_path = str(ADAPTER_DIR) if ADAPTER_DIR.exists() else MODEL_NAME
os.environ["HTTP_PROXY"] = "http://azureproxy.jio.com:8080" 
os.environ["HTTPS_PROXY"] = "http://azureproxy.jio.com:8080"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

try:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=LOAD_IN_4BIT,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    print("Loaded via Unsloth.")
except Exception as e:
    print("Unsloth unavailable ->", e)

print("Model ready. Using:", model_path)

# ## 2. Held-out eval (3 tiers)


import json, re, statistics
from textwrap import dedent
from collections import Counter

from household_basket_env.server.environment import HouseholdBasketEnvironment
from household_basket_env.server.curriculum import load_verified_seeds

SYSTEM_PROMPT = """You are a careful nutrition-aware shopping assistant for an Indian household. \
At each step you pick exactly ONE product from the catalog and assign it to ONE household member. \
You MUST output ONLY a JSON object with three keys: \"product_id\" (string), \"member_id\" (string), \"rationale\" (1 short sentence). \
Do NOT add any prose, code fences, or extra keys. Respect each member's caps and allergies; spread meal types across breakfast/lunch/dinner/snack/beverage; stay under the budget."""
 
JSON_RE = re.compile(r"\{[^{}]*\}", re.S)
 
MAX_CANDS = 20  # 50 blows past 2048 tokens; 20 keeps prompt ~1200 tokens
 
def render_observation(obs) -> str:
    members = []
    for m in obs.household:
        caps = ", ".join(f"{k}={v:.0f}" for k, v in m.thresholds_cap.items())
        members.append(f"{m.member_id} | {m.conditions} | {caps}")
    cands = []
    for c in obs.candidates[:MAX_CANDS]:
        n = c.nutrition_per_100g
        cands.append(
            f"{c.product_id} | {c.meal_type} | {c.price_inr:.0f}r | "
            f"sug={n.get('sugars_g',0):.1f} sod={n.get('sodium_mg',0):.0f} fat={n.get('fat_g',0):.1f}"
        )
    basket = [f"{t.product_id}->{t.member_id}" for t in obs.basket_so_far]
    shown = min(len(obs.candidates), MAX_CANDS)
    return dedent(f"""
    Members:
    {chr(10).join(members)}
 
    Products ({shown}/{len(obs.candidates)}):
    {chr(10).join(cands)}
 
    Basket ({len(basket)}): {', '.join(basket) if basket else 'empty'}
    Budget: {obs.budget_remaining:.0f} INR | Step {obs.step_index}/{obs.max_steps}
 
    Pick ONE product for ONE member. JSON only.
    """).strip()

def chat_generate(system, user, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    inputs = tokenizer.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
    out = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0.0,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)

def extract_json(t):
    m = JSON_RE.search(t)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

env = HouseholdBasketEnvironment()
print("Eval helpers ready.")

def run_episode(seed, task_id):
    obs = env.reset(seed=seed, task_id=task_id)
    total = 0.0
    parse_fails = 0
    last = obs
    iter_cap = obs.max_steps * 4 + 2
    transcript = []
    for _ in range(iter_cap):
        if obs.done:
            break
        prompt = render_observation(obs)
        raw = chat_generate(SYSTEM_PROMPT, prompt)
        payload = extract_json(raw)
        if payload is None:
            parse_fails += 1
            payload = raw
        out = env.apply_raw_action(payload)
        total += out.reward or 0.0
        transcript.append({
            "step": obs.step_index,
            "raw": raw[:200],
            "parsed": payload if isinstance(payload, dict) else None,
            "reward": out.reward,
            "breakdown": out.reward_breakdown,
        })
        obs = out
        last = out
    return {
        "seed": seed,
        "task_id": task_id,
        "total_reward": round(total, 4),
        "parse_fails": parse_fails,
        "step_index": last.step_index,
        "attempt_index": last.attempt_index,
        "terminal_reward": last.terminal_reward,
        "basket_size": len(last.basket_so_far),
        "transcript": transcript,
    }

records, summary = {}, {}
for task_id in (1, 2, 3):
    payload = load_verified_seeds(task_id)
    seeds = payload["held_out_seeds"][:MAX_SEEDS_PER_TIER]
    print(f"\n=== Task {task_id}: {len(seeds)} held-out seeds ===")
    rows = [run_episode(s, task_id) for s in seeds]
    records[str(task_id)] = rows
    rewards = [r["total_reward"] for r in rows]
    term_dist = Counter(round(r["terminal_reward"], 1) for r in rows)
    summary[str(task_id)] = {
        "n_seeds": len(rows),
        "mean_reward": round(statistics.fmean(rewards), 4),
        "median_reward": round(statistics.median(rewards), 4),
        "stdev_reward": round(statistics.pstdev(rewards), 4) if len(rewards) > 1 else 0.0,
        "parse_failure_rate": round(sum(1 for r in rows if r["parse_fails"] > 0) / max(1, len(rows)), 4),
        "terminal_reward_distribution": dict(term_dist),
    }

eval_path = RESULTS_DIR / f"eval_{RUN_NAME}.json"
with open(eval_path, "w", encoding="utf-8") as f:
    json.dump({"run_name": RUN_NAME, "summary": summary, "rows": records}, f, indent=2)
print("\nSaved ->", eval_path)

# ## 3. Compare against baseline


baseline_path = REPO_ROOT / "docs" / "results" / "baseline.json"
if baseline_path.exists():
    with open(baseline_path) as f:
        baseline = json.load(f)
    print(f"{'Task':<6}{'Baseline':<14}{'Trained':<14}{'Δ':<10}")
    print("-" * 44)
    for task_id in ("1", "2", "3"):
        b = baseline["summary"].get(task_id, {}).get("mean_reward", float("nan"))
        t = summary[task_id]["mean_reward"]
        print(f"{task_id:<6}{b:<14.4f}{t:<14.4f}{(t-b):+.4f}")
else:
    print("No baseline.json found — run baseline_eval.ipynb first.")

# ## 4. Training curves (reward / KL / entropy)


log_path = RUN_DIR / "training_log.json"
if log_path.exists():
    with open(log_path) as f:
        log = json.load(f)
    import matplotlib.pyplot as plt
    steps = [r["step"] for r in log if "step" in r]
    rewards = [r.get("reward", r.get("rewards/reward_basket_step", None)) for r in log]
    kl = [r.get("kl", None) for r in log]
    entropy = [r.get("entropy", r.get("completions/mean_entropy", None)) for r in log]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, ys, title in zip(axes, [rewards, kl, entropy], ["Reward", "KL", "Entropy"]):
        ys2 = [y for y in ys if y is not None]
        xs2 = steps[:len(ys2)]
        ax.plot(xs2, ys2)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(True)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"curves_{RUN_NAME}.png", dpi=120)
    plt.show()
    print("Saved ->", RESULTS_DIR / f"curves_{RUN_NAME}.png")
else:
    print("No training_log.json found — train first.")

# ## 5. Qualitative samples (5 episodes)
# 
# For the demo / report we surface 5 representative episodes: best, worst, median, an over-budget terminal, and a clean +1.0 terminal.


all_rows = [r for rows in records.values() for r in rows]
sorted_rows = sorted(all_rows, key=lambda r: r["total_reward"])
samples = []
if sorted_rows:
    samples = [sorted_rows[0], sorted_rows[len(sorted_rows)//2], sorted_rows[-1]]
    full_pass = [r for r in all_rows if round(r["terminal_reward"], 1) == 1.0]
    if full_pass:
        samples.append(full_pass[0])
    over_b = [r for r in all_rows if round(r["terminal_reward"], 1) == -0.5]
    if over_b:
        samples.append(over_b[0])

samples_path = RESULTS_DIR / f"qualitative_{RUN_NAME}.json"
with open(samples_path, "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2)
print(f"Saved {len(samples)} qualitative samples -> {samples_path}")
for s in samples:
    print(f"  task={s['task_id']} seed={s['seed']} total={s['total_reward']:.3f} terminal={s['terminal_reward']} basket={s['basket_size']}")

# ### Outputs
# 
# - `docs/results/eval_<RUN_NAME>.json` — full per-seed eval
# - `docs/results/curves_<RUN_NAME>.png` — reward / KL / entropy
# - `docs/results/qualitative_<RUN_NAME>.json` — 5 representative episodes
# 
# Re-run with `RUN_NAME = "ablation_a"` and `"ablation_b"` to populate the comparison table referenced in the README.