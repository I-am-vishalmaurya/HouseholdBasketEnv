#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the BSD-style license found in the LICENSE file in the root
# of this source tree.
"""
HouseholdBasketEnv — GRPO training, plain TRL + PEFT (no Unsloth).

Single-file Kaggle entrypoint that runs all three experiments end-to-end:
    1. main         — full reward, Task 2 (70%) + Task 3 (30%)
    2. ablation_a   — meal-type coverage disabled, otherwise same as main
    3. ablation_b   — Task 3 only, no curriculum mix

After training, the script:
    • Evaluates each adapter on held-out seeds for Task 1 / 2 / 3
    • Plots reward / KL / completion-length curves for all 3 runs
    • Writes a comparison bar chart (baseline vs. each trained adapter)
    • Exports a JSON results file you can drop straight into the README

USAGE on Kaggle (T4 / P100 / L4)
    !pip install -q -U pip
    !python /kaggle/working/HouseholdBasketEnv/household_basket_env/scripts/train_grpo.py

    Outputs land in:
        /kaggle/working/runs/<run_name>/adapter_final/
        /kaggle/working/runs/<run_name>/training_log.json
        /kaggle/working/results/reward_curves.png
        /kaggle/working/results/kl_curves.png
        /kaggle/working/results/eval_comparison.png
        /kaggle/working/results/eval_results.json

The script is self-bootstrapping: it git-clones the env package if missing
and pip-installs pinned versions of every dependency it needs.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from textwrap import dedent
from typing import Any

# ---------------------------------------------------------------------------
# 0. Logging — set up before anything heavy imports.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_grpo")


# ===========================================================================
# 1. Bootstrap (clone repo, install pinned deps)
#    Designed to be idempotent so re-running the cell doesn't reinstall.
# ===========================================================================
REPO_URL_DEFAULT = "https://github.com/I-am-vishalmaurya/HouseholdBasketEnv"
REPO_URL = os.environ.get("HOUSEHOLD_BASKET_REPO", REPO_URL_DEFAULT)

# Kaggle's persistent working dir:
KAGGLE_ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
CLONE_DIR = KAGGLE_ROOT / "HouseholdBasketEnv"


def _run(cmd: list[str], check: bool = True) -> None:
    log.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=check)


def bootstrap_repo() -> Path:
    """Clone the env package if it isn't already present."""
    # If we are running *inside* the cloned repo, nothing to do.
    here = Path(__file__).resolve()
    inside_repo_root = here.parents[2] if len(here.parents) >= 3 else None
    if inside_repo_root is not None and (inside_repo_root / "household_basket_env").exists():
        log.info("Detected in-repo execution; using %s", inside_repo_root)
        return inside_repo_root

    if not CLONE_DIR.exists():
        _run(["git", "clone", "--depth", "1", REPO_URL, str(CLONE_DIR)])
    else:
        log.info("Repo already at %s; skipping clone", CLONE_DIR)
    return CLONE_DIR


def bootstrap_deps() -> None:
    """Install pinned versions of TRL + PEFT + transformers + bitsandbytes.

    We pin known-good combinations to dodge the moving-target API drift between
    TRL 0.11 (old `tokenizer=` kwarg, `processing_class=` not yet introduced) and
    TRL 0.21+ (vLLM + chat-template kwargs). 0.18 is the sweet spot.
    """
    # NOTE on the pin: TRL 0.18 requires transformers >= 4.49. We use ranges
    # so pip's resolver can satisfy whatever Kaggle's preinstalled torch was
    # built against.
    deps = [
        "transformers>=4.49,<4.55",
        "trl==0.18.2",
        "peft>=0.14,<0.16",
        "accelerate>=1.2,<2.0",
        "bitsandbytes>=0.45",
        "datasets>=3.0,<4.0",
        "pydantic>=2.5,<3",
        "matplotlib>=3.8",
        "numpy>=1.26",
        "sentencepiece",
        "protobuf",
    ]
    pip_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-input",
        "--upgrade-strategy",
        "only-if-needed",
    ] + deps
    _run(pip_cmd, check=False)


def add_repo_to_syspath(repo_root: Path) -> None:
    repo_root_str = str(repo_root.resolve())
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    env_root = repo_root / "household_basket_env"
    products_path = env_root / "data" / "products.json"
    if products_path.exists():
        os.environ["HOUSEHOLD_BASKET_PRODUCTS_PATH"] = str(products_path)
    os.environ.setdefault("HF_HOME", str(KAGGLE_ROOT / ".hf_cache"))
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


REPO_ROOT = bootstrap_repo()
bootstrap_deps()
add_repo_to_syspath(REPO_ROOT)


# ===========================================================================
# 2. Imports that depend on the bootstrapped environment.
# ===========================================================================
import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402
from peft import LoraConfig  # noqa: E402
from datasets import Dataset  # noqa: E402

from trl import GRPOConfig, GRPOTrainer  # noqa: E402

# Env imports (after add_repo_to_syspath)
from household_basket_env.server.environment import HouseholdBasketEnvironment  # noqa: E402
from household_basket_env.server.curriculum import (  # noqa: E402
    TIER_CONFIGS,
    load_verified_seeds,
)
from household_basket_env.models import BasketAction  # noqa: E402


# ===========================================================================
# 3. Top-level config (override via env vars when needed).
# ===========================================================================
def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


# 30-minute Kaggle T4 budget. Qwen2.5-0.5B-Instruct in 4-bit runs the three
# experiments + a tiny held-out eval inside a single session. Override with
# env vars if you have more time or a bigger GPU.
MODEL_NAME = _env_str("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

# Run schedule (very tight; tuned for ~30 min total wall clock).
MAIN_STEPS = _env_int("MAIN_STEPS", 15)
ABLATION_STEPS = _env_int("ABLATION_STEPS", 8)
NUM_GENERATIONS = _env_int("NUM_GENERATIONS", 4)  # GRPO group size per prompt
PER_DEVICE_BATCH = _env_int("PER_DEVICE_BATCH", 1)
GRAD_ACCUM = _env_int("GRAD_ACCUM", 2)
MAX_NEW_TOKENS = _env_int("MAX_NEW_TOKENS", 64)
MAX_PROMPT_TOKENS = _env_int("MAX_PROMPT_TOKENS", 1024)
LEARNING_RATE = float(_env_str("LEARNING_RATE", "5e-6"))
GRPO_BETA = float(_env_str("GRPO_BETA", "0.04"))
TEMPERATURE = float(_env_str("TEMPERATURE", "1.0"))
LORA_R = _env_int("LORA_R", 16)
LORA_ALPHA = _env_int("LORA_ALPHA", 32)
LORA_DROPOUT = float(_env_str("LORA_DROPOUT", "0.05"))
LOAD_IN_4BIT = bool(int(_env_str("LOAD_IN_4BIT", "1")))
SEED = _env_int("SEED", 0)
EVAL_SEEDS_PER_TIER = _env_int("EVAL_SEEDS_PER_TIER", 4)
SKIP_TRAINING = bool(int(_env_str("SKIP_TRAINING", "0")))  # for re-running just eval/plots
# Re-run the prompted baseline with the same 0.5B model used for training so
# the comparison plot is apples-to-apples. Override to 1 to reuse an existing
# baseline.json (set BASELINE_JSON to point to it).
SKIP_BASELINE_EVAL = bool(int(_env_str("SKIP_BASELINE_EVAL", "0")))
BASELINE_JSON_OVERRIDE = _env_str("BASELINE_JSON", "")

OUTPUT_ROOT = KAGGLE_ROOT / "runs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = KAGGLE_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# 4. Prompt rendering and JSON-extract utilities.
# ===========================================================================
SYSTEM_PROMPT = (
    "You are a careful nutrition-aware shopping assistant for an Indian household. "
    "At each step you pick exactly ONE product from the catalog and assign it to ONE "
    "household member. You MUST output ONLY a JSON object with three keys: "
    '"product_id" (string), "member_id" (string), "rationale" (1 short sentence). '
    "Do NOT add any prose, code fences, or extra keys. Respect each member's caps and "
    "allergies; spread meal types across breakfast / lunch / dinner / snack / beverage; "
    "stay under the budget."
)

_JSON_RE = re.compile(r"\{[^{}]*\}", re.S)


def render_observation(obs) -> str:
    """Render a BasketObservation into a compact natural-language prompt."""
    members = []
    for m in obs.household:
        caps = ", ".join(f"{k}={v:.0f}" for k, v in m.thresholds_cap.items())
        members.append(
            f"- {m.member_id} | conds={m.conditions} | caps: {caps}"
        )
    cands = []
    for c in obs.candidates[:40]:  # truncate for token budget
        nuts = c.nutrition_per_100g or {}
        cands.append(
            f"- {c.product_id} | {c.category} | meal={c.meal_type} | "
            f"INR {c.price_inr:.0f} | sugars={nuts.get('sugars_g', 0):.1f} "
            f"sodium={nuts.get('sodium_mg', 0):.0f} fat={nuts.get('fat_g', 0):.1f}"
        )
    basket = [f"- {t.product_id} -> {t.member_id}" for t in obs.basket_so_far]
    return dedent(
        f"""
        Household:
        {chr(10).join(members)}

        Catalog (top {len(cands)}):
        {chr(10).join(cands)}

        Basket so far ({len(basket)}):
        {chr(10).join(basket) if basket else "(empty)"}

        Budget remaining: INR {obs.budget_remaining:.0f}
        Steps used: {obs.step_index} / {obs.max_steps}

        Pick ONE product and assign it to ONE member. Output JSON only.
        """
    ).strip()


def extract_json(text: str):
    """Best-effort grab of the first JSON object in a completion."""
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ===========================================================================
# 5. Build the per-step prompt dataset from verified seeds.
# ===========================================================================
def build_prompt_dataset(
    tier_weights: dict[int, float],
    n_prompts: int,
    seed: int,
) -> Dataset:
    """Sample (seed, tier) pairs and render the *initial* observation as the prompt.

    Per-step bandit framing of GRPO: each row corresponds to a fresh reset.
    The reward function will replay the env from that seed and grade the
    model's first action. This is the simplest credit-assignment story that
    actually works on a per-call reward signal.
    """
    rng = random.Random(seed)
    env = HouseholdBasketEnvironment()

    # Pre-load verified seed pools once.
    tier_pools: dict[int, list[int]] = {}
    for t in tier_weights:
        payload = load_verified_seeds(t)
        pool = payload.get("training_seeds") or payload.get("verified_seeds") or []
        if not pool:
            log.warning("Tier %s has no verified seeds; skipping", t)
            continue
        tier_pools[t] = list(pool)
    if not tier_pools:
        raise RuntimeError("No verified training seeds across requested tiers.")

    tiers = sorted(tier_pools)
    weights = [tier_weights[t] for t in tiers]

    rows: list[dict[str, Any]] = []
    for _ in range(n_prompts):
        tier = rng.choices(tiers, weights=weights, k=1)[0]
        s = rng.choice(tier_pools[tier])
        obs = env.reset(seed=int(s), task_id=int(tier))
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": render_observation(obs)},
                ],
                "seed": int(s),
                "task_id": int(tier),
            }
        )
    return Dataset.from_list(rows)


# ===========================================================================
# 6. Reward function for GRPO.
#    Plan §4 dense reward. Replays the env to step 0 and applies the
#    extracted action via apply_raw_action() so the 5-check pipeline runs.
# ===========================================================================
class RewardScorer:
    """Stateful reward callable. We hold one persistent env per worker thread.

    The trainer calls this with a list of completions plus the dataset's
    extra columns (seed, task_id) passed through as **kwargs.
    """

    def __init__(self, enable_meal_type_coverage: bool):
        self.enable_meal_type_coverage = enable_meal_type_coverage
        self.env = HouseholdBasketEnvironment(
            enable_meal_type_coverage=enable_meal_type_coverage,
        )
        self.last_components: list[dict[str, float]] = []
        self.parse_failures = 0
        self.calls = 0

    def __call__(self, completions, **kwargs) -> list[float]:
        seeds = kwargs.get("seed")
        task_ids = kwargs.get("task_id")
        if seeds is None or task_ids is None:
            raise RuntimeError(
                "Expected `seed` and `task_id` to be forwarded by GRPOTrainer. "
                "Set GRPOConfig(remove_unused_columns=False)."
            )
        rewards: list[float] = []
        components_batch: list[dict[str, float]] = []
        for completion, s, t in zip(completions, seeds, task_ids):
            self.calls += 1
            text = (
                completion[0]["content"]
                if isinstance(completion, list) and completion
                else (completion if isinstance(completion, str) else "")
            )
            payload = extract_json(text)
            if payload is None:
                self.parse_failures += 1
                payload = text  # falls into P_parse path
            self.env.reset(seed=int(s), task_id=int(t))
            obs = self.env.apply_raw_action(payload)
            r = float(obs.reward or 0.0)
            rewards.append(r)
            components_batch.append(obs.reward_breakdown or {})
        self.last_components = components_batch
        return rewards


# ===========================================================================
# 7. Model loader (4-bit QLoRA via bitsandbytes, no Unsloth).
# ===========================================================================
def load_model_and_tokenizer():
    log.info("Loading tokenizer + base model: %s (4bit=%s)", MODEL_NAME, LOAD_IN_4BIT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if LOAD_IN_4BIT and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    # GRPO trainer will attach the LoRA via peft_config; we don't wrap here.
    if hasattr(model, "config"):
        model.config.use_cache = False  # required for gradient checkpointing
    return model, tokenizer


def build_peft_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


# ===========================================================================
# 8. Train one experiment.
# ===========================================================================
RUN_CONFIGS: dict[str, dict[str, Any]] = {
    "main": {
        "tier_weights": {2: 0.7, 3: 0.3},
        "enable_meal_type_coverage": True,
        "max_steps": MAIN_STEPS,
    },
    "ablation_a": {
        "tier_weights": {2: 0.7, 3: 0.3},
        "enable_meal_type_coverage": False,
        "max_steps": ABLATION_STEPS,
    },
    "ablation_b": {
        "tier_weights": {3: 1.0},
        "enable_meal_type_coverage": True,
        "max_steps": ABLATION_STEPS,
    },
}


def train_one(run_name: str) -> Path:
    cfg = RUN_CONFIGS[run_name]
    run_dir = OUTPUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("==== Starting run %s -> %s ====", run_name, run_dir)
    log.info("Run config: %s", cfg)

    # Fresh model per run so adapters don't leak across experiments.
    model, tokenizer = load_model_and_tokenizer()
    peft_config = build_peft_config()

    # Build a prompt dataset large enough that the trainer never repeats.
    n_prompts = max(2048, cfg["max_steps"] * PER_DEVICE_BATCH * 32)
    prompt_ds = build_prompt_dataset(
        tier_weights=cfg["tier_weights"],
        n_prompts=n_prompts,
        seed=SEED,
    )
    log.info("Prompt dataset size: %d", len(prompt_ds))

    reward_scorer = RewardScorer(
        enable_meal_type_coverage=cfg["enable_meal_type_coverage"],
    )

    grpo_args = GRPOConfig(
        output_dir=str(run_dir),
        learning_rate=LEARNING_RATE,
        beta=GRPO_BETA,
        num_generations=NUM_GENERATIONS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_prompt_length=MAX_PROMPT_TOKENS,
        max_completion_length=MAX_NEW_TOKENS,
        max_steps=cfg["max_steps"],
        save_steps=max(10, cfg["max_steps"] // 2),
        save_total_limit=1,
        logging_steps=1,
        seed=SEED,
        bf16=False,
        fp16=torch.cuda.is_available(),
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        temperature=TEMPERATURE,
        remove_unused_columns=False,
        report_to=[],  # we plot from the log_history ourselves
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        log_completions=False,
    )

    # GRPOTrainer in TRL 0.18 takes the tokenizer via `processing_class`.
    # Older versions used `tokenizer=`. We try the new kwarg first.
    try:
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[reward_scorer],
            args=grpo_args,
            train_dataset=prompt_ds,
            peft_config=peft_config,
        )
    except TypeError:
        log.info("processing_class kwarg not accepted; falling back to tokenizer=")
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=[reward_scorer],
            args=grpo_args,
            train_dataset=prompt_ds,
            peft_config=peft_config,
        )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    log.info("Run %s finished in %.1f min", run_name, elapsed / 60)
    log.info(
        "Reward scorer: %d calls, %d parse failures (%.1f%%)",
        reward_scorer.calls,
        reward_scorer.parse_failures,
        100.0 * reward_scorer.parse_failures / max(1, reward_scorer.calls),
    )

    # Save final adapter + run metadata
    adapter_dir = run_dir / "adapter_final"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    with (run_dir / "training_log.json").open("w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "model": MODEL_NAME,
                "elapsed_min": round(elapsed / 60, 2),
                "tier_weights": cfg["tier_weights"],
                "enable_meal_type_coverage": cfg["enable_meal_type_coverage"],
                "max_steps": cfg["max_steps"],
                "num_generations": NUM_GENERATIONS,
                "per_device_batch": PER_DEVICE_BATCH,
                "grad_accum": GRAD_ACCUM,
                "lr": LEARNING_RATE,
                "beta": GRPO_BETA,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "max_new_tokens": MAX_NEW_TOKENS,
                "load_in_4bit": LOAD_IN_4BIT,
                "reward_scorer_calls": reward_scorer.calls,
                "reward_scorer_parse_failures": reward_scorer.parse_failures,
            },
            f,
            indent=2,
        )

    # Free GPU memory before the next run.
    del trainer, model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return adapter_dir


# ===========================================================================
# 9. Eval — run each adapter on held-out seeds for Tier 1 / 2 / 3.
# ===========================================================================
@torch.no_grad()
def _generate(model, tokenizer, prompt_messages: list[dict[str, str]], max_new_tokens: int) -> str:
    chat = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(chat, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS).to(
        model.device
    )
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def rollout_episode(model, tokenizer, env: HouseholdBasketEnvironment, seed: int, task_id: int) -> dict[str, Any]:
    """Run one episode with the (possibly-adapted) model. Returns total reward."""
    obs = env.reset(seed=seed, task_id=task_id)
    total = 0.0
    components_sum: dict[str, float] = {}
    while not obs.done:
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_observation(obs)},
        ]
        text = _generate(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
        payload = extract_json(text) or text
        obs = env.apply_raw_action(payload)
        total += float(obs.reward or 0.0)
        for k, v in (obs.reward_breakdown or {}).items():
            components_sum[k] = components_sum.get(k, 0.0) + float(v)
    if obs.terminal_reward is not None and obs.terminated_reason == "max_steps":
        # terminal already included in obs.reward at the final step; don't double-count.
        pass
    return {
        "seed": seed,
        "task_id": task_id,
        "total_reward": total,
        "terminal_reward": float(obs.terminal_reward or 0.0),
        "terminated_reason": obs.terminated_reason,
        "step_index": obs.step_index,
        "attempt_index": obs.attempt_index,
        "components": components_sum,
    }


def held_out_seeds(task_id: int, n: int) -> list[int]:
    payload = load_verified_seeds(task_id)
    pool = payload.get("held_out_seeds") or payload.get("verified_seeds") or []
    rng = random.Random(SEED + task_id)
    pool = list(pool)
    rng.shuffle(pool)
    return pool[:n]


def evaluate_adapter(adapter_path: Path | None, label: str) -> dict[str, Any]:
    """Evaluate either the base model (adapter_path=None) or a specific LoRA adapter."""
    log.info("Evaluating %s (adapter=%s)", label, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if LOAD_IN_4BIT and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    if adapter_path is not None and adapter_path.exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()
        log.info("Loaded LoRA adapter from %s", adapter_path)

    model.eval()
    env = HouseholdBasketEnvironment()
    results: dict[int, list[dict[str, Any]]] = {}
    for tier_id in (1, 2, 3):
        seeds = held_out_seeds(tier_id, EVAL_SEEDS_PER_TIER)
        if not seeds:
            log.warning("No held-out seeds for tier %s", tier_id)
            continue
        rollouts: list[dict[str, Any]] = []
        for s in seeds:
            try:
                r = rollout_episode(model, tokenizer, env, seed=s, task_id=tier_id)
            except Exception as e:  # one bad seed shouldn't kill the whole eval
                log.warning("Rollout failed (tier=%s seed=%s): %s", tier_id, s, e)
                continue
            rollouts.append(r)
        results[tier_id] = rollouts
        if rollouts:
            mean_r = sum(x["total_reward"] for x in rollouts) / len(rollouts)
            mean_term = sum(x["terminal_reward"] for x in rollouts) / len(rollouts)
            log.info(
                "%s tier %s: mean total=%.3f, mean terminal=%.3f, n=%d",
                label,
                tier_id,
                mean_r,
                mean_term,
                len(rollouts),
            )
    del model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "label": label,
        "adapter": str(adapter_path) if adapter_path else None,
        "by_tier": {
            str(t): {
                "n": len(rs),
                "mean_total_reward": (sum(x["total_reward"] for x in rs) / len(rs)) if rs else None,
                "mean_terminal_reward": (sum(x["terminal_reward"] for x in rs) / len(rs))
                if rs
                else None,
                "rollouts": rs,
            }
            for t, rs in results.items()
        },
    }


# ===========================================================================
# 10. Plotting.
# ===========================================================================
def _load_log_history(run_name: str) -> list[dict[str, Any]]:
    p = OUTPUT_ROOT / run_name / "training_log.json"
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _series_from_log(log_history: list[dict[str, Any]], keys: tuple[str, ...]) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in log_history:
        if "step" not in row:
            continue
        for k in keys:
            if k in row and isinstance(row[k], (int, float)):
                xs.append(int(row["step"]))
                ys.append(float(row[k]))
                break
    return xs, ys


def plot_training_curves(run_names: list[str]) -> None:
    """Reward curve and KL curve, all runs on the same axes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for run_name in run_names:
        log_history = _load_log_history(run_name)
        xs, ys = _series_from_log(log_history, ("reward", "rewards/reward_basket_step"))
        if xs:
            axes[0].plot(xs, ys, label=run_name, linewidth=1.6, alpha=0.9)

        xs, ys = _series_from_log(log_history, ("kl",))
        if xs:
            axes[1].plot(xs, ys, label=run_name, linewidth=1.6, alpha=0.9)

    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("mean group reward")
    axes[0].set_title("GRPO reward (per-step bandit view)")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("per-token KL")
    axes[1].set_title("KL divergence vs reference")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = RESULTS_DIR / "reward_curves.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log.info("Wrote %s", out)


def plot_kl_only(run_names: list[str]) -> None:
    """Standalone KL+entropy plot for the README §6 monitoring section."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for run_name in run_names:
        log_history = _load_log_history(run_name)
        xs, ys = _series_from_log(log_history, ("kl",))
        if xs:
            axes[0].plot(xs, ys, label=run_name, linewidth=1.6, alpha=0.9)

        xs, ys = _series_from_log(log_history, ("completions/mean_length", "completion_length"))
        if xs:
            axes[1].plot(xs, ys, label=run_name, linewidth=1.6, alpha=0.9)

    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("per-token KL")
    axes[0].set_title("KL divergence (target: stay < 4.0)")
    axes[0].axhline(2.0, color="orange", ls="--", lw=1, label="inspect threshold")
    axes[0].axhline(4.0, color="red", ls="--", lw=1, label="abort threshold")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("mean completion length (tokens)")
    axes[1].set_title("Completion length proxy for entropy")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = RESULTS_DIR / "kl_curves.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log.info("Wrote %s", out)


def plot_eval_comparison(eval_results: list[dict[str, Any]]) -> None:
    """Bar chart: mean total reward by (run, tier)."""
    tiers = ["1", "2", "3"]
    labels = [r["label"] for r in eval_results]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tiers))
    width = 0.8 / max(1, len(eval_results))

    for i, run in enumerate(eval_results):
        vals = []
        for t in tiers:
            cell = run["by_tier"].get(t)
            v = cell["mean_total_reward"] if cell and cell["mean_total_reward"] is not None else 0.0
            vals.append(v)
        offsets = x + (i - (len(eval_results) - 1) / 2) * width
        ax.bar(offsets, vals, width=width, label=run["label"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"Task {t} (held-out)" for t in tiers])
    ax.set_ylabel("mean total episode reward")
    ax.set_title("Held-out evaluation: baseline vs trained adapters")
    ax.axhline(0, color="gray", lw=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = RESULTS_DIR / "eval_comparison.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log.info("Wrote %s", out)


# ===========================================================================
# 11. Main entrypoint.
# ===========================================================================
def main() -> int:
    log.info("===== HouseholdBasketEnv GRPO (no-Unsloth) =====")
    log.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info("GPU: %s", torch.cuda.get_device_name(0))
        log.info(
            "GPU memory: %.1f GB",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    run_names = ["main", "ablation_a", "ablation_b"]

    # ---- Phase A — train all three runs ----
    if not SKIP_TRAINING:
        for run_name in run_names:
            try:
                train_one(run_name)
            except Exception as e:
                log.exception("Run %s crashed: %s", run_name, e)

    # ---- Phase B — eval baseline + each trained adapter on held-out seeds ----
    eval_results: list[dict[str, Any]] = []
    eval_results.append(evaluate_adapter(None, label="prompted_baseline"))
    for run_name in run_names:
        adapter = OUTPUT_ROOT / run_name / "adapter_final"
        if adapter.exists():
            eval_results.append(evaluate_adapter(adapter, label=run_name))
        else:
            log.warning("Adapter missing for %s at %s; skipping", run_name, adapter)

    # ---- Phase C — write results JSON + plots ----
    eval_path = RESULTS_DIR / "eval_results.json"
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "n_eval_seeds_per_tier": EVAL_SEEDS_PER_TIER,
                "results": eval_results,
            },
            f,
            indent=2,
        )
    log.info("Wrote %s", eval_path)

    plot_training_curves(run_names)
    plot_kl_only(run_names)
    plot_eval_comparison(eval_results)

    # Concise stdout summary the user can paste into the README.
    print("\n===== FINAL EVAL SUMMARY =====")
    print(f"{'run':<22} | {'tier 1':>10} | {'tier 2':>10} | {'tier 3':>10}")
    print("-" * 64)
    for run in eval_results:
        cells = []
        for t in ("1", "2", "3"):
            v = run["by_tier"].get(t, {}).get("mean_total_reward")
            cells.append(f"{v:>10.3f}" if v is not None else f"{'-':>10}")
        print(f"{run['label']:<22} | {cells[0]} | {cells[1]} | {cells[2]}")
    print("\nArtifacts:")
    for p in sorted(RESULTS_DIR.glob("*")):
        print(f"  - {p}")
    for run_name in run_names:
        adapter = OUTPUT_ROOT / run_name / "adapter_final"
        if adapter.exists():
            print(f"  - {adapter}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
