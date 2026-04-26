# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GRPO training driver for HouseholdBasketEnv (plan §6, §7).

CLI replacement for ``notebooks/training.ipynb``. Same hyperparameters, same
MODE presets, same reward function — designed for batch jobs / nohup runs / CI.

Usage
-----
    # Smoke test the full loop (~5 min on A100)
    python scripts/train_grpo.py --run-name main --mode smoke

    # Default iteration config (20 GRPO steps, gens=8)
    python scripts/train_grpo.py --run-name main --mode fast

    # Plan §6 headline config (40 steps, gens=16)
    python scripts/train_grpo.py --run-name main --mode full

    # Ablation A — disable R_meal_type_coverage
    python scripts/train_grpo.py --run-name ablation_a --mode fast

    # Ablation B — Task 3 only, no curriculum mix
    python scripts/train_grpo.py --run-name ablation_b --mode fast

    # Override individual knobs from the preset
    python scripts/train_grpo.py --run-name main --mode fast \\
        --max-train-steps 30 --num-generations 6

Outputs land in ``household_basket_env/notebooks/runs/<run-name>/``:
  - ``adapter_final/``    LoRA adapter + tokenizer
  - ``training_log.json`` trainer.state.log_history
  - ``run_config.json``   resolved hyperparameters for this run
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import sys
import time
from textwrap import dedent

# -----------------------------------------------------------------------------
# Path bootstrap — make ``household_basket_env.*`` importable when invoked from
# anywhere (repo root, scripts dir, cron, etc.).
# -----------------------------------------------------------------------------
SCRIPT_PATH = pathlib.Path(__file__).resolve()
ENV_ROOT = SCRIPT_PATH.parent.parent          # household_basket_env/
REPO_ROOT = ENV_ROOT.parent                   # repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# MODE presets — mirror notebook cell `23073d56`.
# =============================================================================
MODE_PRESETS: dict[str, dict] = {
    "smoke": dict(MAX_TRAIN_STEPS=5,  NUM_GENERATIONS=4,  GRAD_ACCUM=2,
                  MAX_NEW_TOKENS=96,  MAX_SEQ_LEN=1792, SAVE_EVERY=5),
    "fast":  dict(MAX_TRAIN_STEPS=20, NUM_GENERATIONS=8,  GRAD_ACCUM=4,
                  MAX_NEW_TOKENS=128, MAX_SEQ_LEN=1792, SAVE_EVERY=10),
    "full":  dict(MAX_TRAIN_STEPS=40, NUM_GENERATIONS=16, GRAD_ACCUM=8,
                  MAX_NEW_TOKENS=192, MAX_SEQ_LEN=3072, SAVE_EVERY=20),
}


# =============================================================================
# Argument parsing
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for full usage examples.",
    )
    p.add_argument("--run-name", choices=["main", "ablation_a", "ablation_b"],
                   default="main",
                   help="Plan §6/§7 run identifier; controls TIER_WEIGHTS and "
                        "ENABLE_MEAL_TYPE_COVERAGE.")
    p.add_argument("--mode", choices=list(MODE_PRESETS.keys()), default="fast",
                   help="Hyperparameter preset (default: fast).")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cuda-visible-devices", default=None,
                   help="If set, override CUDA_VISIBLE_DEVICES before torch import.")
    p.add_argument("--no-fast-inference", action="store_true",
                   help="Disable Unsloth fast_inference (vLLM). Use if vLLM "
                        "fails to load on the host.")
    p.add_argument("--no-unsloth", action="store_true",
                   help="Skip Unsloth entirely; use Transformers + PEFT path.")

    # Per-knob overrides (None = take from MODE preset)
    p.add_argument("--max-train-steps", type=int, default=None)
    p.add_argument("--num-generations", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None)

    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--grpo-beta", type=float, default=0.04)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    p.add_argument("--gpu-mem-util", type=float, default=0.55,
                   help="vLLM gpu_memory_utilization when fast_inference=True.")
    p.add_argument("--n-prompts", type=int, default=None,
                   help="Override prompt-dataset size; default scales with demand.")

    p.add_argument("--dry-run", action="store_true",
                   help="Resolve config and exit without loading the model.")
    return p.parse_args()


# =============================================================================
# Resolved config
# =============================================================================
class Config:
    """Frozen view of the resolved hyperparameters for one run."""

    def __init__(self, args: argparse.Namespace):
        preset = MODE_PRESETS[args.mode]

        self.RUN_NAME = args.run_name
        self.MODE = args.mode
        self.MODEL_NAME = args.model_name
        self.SEED = args.seed

        self.MAX_TRAIN_STEPS = args.max_train_steps or preset["MAX_TRAIN_STEPS"]
        self.NUM_GENERATIONS = args.num_generations or preset["NUM_GENERATIONS"]
        self.GRAD_ACCUM      = args.grad_accum      or preset["GRAD_ACCUM"]
        self.MAX_NEW_TOKENS  = args.max_new_tokens  or preset["MAX_NEW_TOKENS"]
        self.MAX_SEQ_LEN     = args.max_seq_len     or preset["MAX_SEQ_LEN"]
        self.SAVE_EVERY      = args.save_every      or preset["SAVE_EVERY"]

        self.PER_DEVICE_BATCH = 1
        self.GRPO_BETA = args.grpo_beta
        self.LEARNING_RATE = args.learning_rate
        self.TEMPERATURE = args.temperature
        self.TOP_P = args.top_p
        self.TOP_K = args.top_k

        self.LORA_R = args.lora_r
        self.LORA_ALPHA = args.lora_alpha
        self.LORA_DROPOUT = args.lora_dropout
        self.LOAD_IN_4BIT = True

        # Run-specific knobs (plan §7).
        if args.run_name == "ablation_b":
            self.TIER_WEIGHTS = {3: 1.0}
        else:
            self.TIER_WEIGHTS = {2: 0.7, 3: 0.3}
        self.ENABLE_MEAL_TYPE_COVERAGE = (args.run_name != "ablation_a")

        self.USE_UNSLOTH = not args.no_unsloth
        self.FAST_INFERENCE = not args.no_fast_inference
        self.GPU_MEM_UTIL = args.gpu_mem_util

        # Prompt dataset sizing — derive from demand, mirror notebook.
        demand = self.MAX_TRAIN_STEPS * self.GRAD_ACCUM * self.NUM_GENERATIONS
        self.N_PROMPTS = args.n_prompts or max(256, min(2 * demand, 2048))

        # Output paths
        self.RUN_DIR = ENV_ROOT / "notebooks" / "runs" / self.RUN_NAME
        self.RESULTS_DIR = ENV_ROOT / "notebooks" / "results"

    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items() if k.isupper() or k in {"RUN_DIR", "RESULTS_DIR"}}

    def pretty(self) -> str:
        rows = []
        for k in ("RUN_NAME", "MODE", "MODEL_NAME", "TIER_WEIGHTS",
                  "ENABLE_MEAL_TYPE_COVERAGE", "MAX_TRAIN_STEPS", "NUM_GENERATIONS",
                  "GRAD_ACCUM", "MAX_NEW_TOKENS", "MAX_SEQ_LEN", "SAVE_EVERY",
                  "LEARNING_RATE", "GRPO_BETA", "TEMPERATURE", "LORA_R",
                  "LORA_ALPHA", "USE_UNSLOTH", "FAST_INFERENCE", "N_PROMPTS"):
            rows.append(f"  {k:30s} = {getattr(self, k)}")
        return "\n".join(rows)


# =============================================================================
# Model loading — Unsloth path with Transformers fallback (mirrors cell 7)
# =============================================================================
def load_model(cfg: Config):
    import torch  # local import: respect CUDA_VISIBLE_DEVICES set above

    print("=" * 60)
    print("PRE-LOAD STATE")
    print("=" * 60)
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        used = (total - free) / 1e9
        print(f"  GPU {i}: {used:.2f} / {total/1e9:.1f} GB")
    print("=" * 60)

    t_total = time.time()
    use_unsloth = cfg.USE_UNSLOTH

    if use_unsloth:
        try:
            print("\n[1/4] Importing Unsloth ...")
            t = time.time()
            from unsloth import FastLanguageModel, PatchFastRL
            print(f"  unsloth imported in {time.time()-t:.1f}s")

            print("[2/4] Patching GRPO ...")
            t = time.time()
            PatchFastRL("GRPO", FastLanguageModel)
            print(f"  GRPO patched in {time.time()-t:.1f}s")

            print("[3/4] Loading base model ...")
            t = time.time()
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=cfg.MODEL_NAME,
                max_seq_length=cfg.MAX_SEQ_LEN,
                load_in_4bit=cfg.LOAD_IN_4BIT,
                fast_inference=cfg.FAST_INFERENCE,
                max_lora_rank=cfg.LORA_R,
                gpu_memory_utilization=cfg.GPU_MEM_UTIL,
            )
            print(f"  base loaded in {time.time()-t:.1f}s")

            print("[4/4] Attaching LoRA ...")
            t = time.time()
            model = FastLanguageModel.get_peft_model(
                model,
                r=cfg.LORA_R,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_alpha=cfg.LORA_ALPHA,
                lora_dropout=cfg.LORA_DROPOUT,
                use_gradient_checkpointing="unsloth",
                random_state=cfg.SEED,
            )
            print(f"  LoRA attached in {time.time()-t:.1f}s")

            total_params = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  trainable: {trainable/1e6:.1f}M / {total_params/1e6:.1f}M "
                  f"({100*trainable/total_params:.2f}%)")
            print(f"\nUnsloth model ready (total {time.time()-t_total:.1f}s)")
            return model, tokenizer, True

        except Exception as e:
            print(f"\n[warn] Unsloth path failed: {e}")
            print("       Falling back to Transformers + PEFT.\n")
            use_unsloth = False

    # Fallback: vanilla Transformers + PEFT
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    print("[1/3] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)

    print("[2/3] Loading base model ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    ) if cfg.LOAD_IN_4BIT else None
    base = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print("[3/3] Attaching LoRA ...")
    lora_config = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_config)
    model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable: {trainable/1e6:.1f}M / {total_params/1e6:.1f}M")
    print(f"\nTransformers model ready (total {time.time()-t_total:.1f}s)")
    return model, tokenizer, False


# =============================================================================
# Prompt rendering + JSON extraction (mirrors notebook cell `b2ae1e94`)
# =============================================================================
SYSTEM_PROMPT = (
    "You are a careful nutrition-aware shopping assistant for an Indian household. "
    "At each step you pick exactly ONE product from the catalog and assign it to "
    "ONE household member. You MUST output ONLY a JSON object with three keys: "
    '"product_id" (string), "member_id" (string), "rationale" (1 short sentence). '
    "Do NOT add any prose, code fences, or extra keys. Respect each member's caps "
    "and allergies; spread meal types across breakfast/lunch/dinner/snack/beverage; "
    "stay under the budget."
)

MAX_CANDS = 20  # 20 keeps prompt ~1.2k tokens

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


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
            f"sug={n.get('sugars_g',0):.1f} sod={n.get('sodium_mg',0):.0f} "
            f"fat={n.get('fat_g',0):.1f}"
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


def extract_json(text: str):
    """Robust JSON extractor: code-fence strip + brace-balance scan + json.loads."""
    if not text:
        return None
    s = text.strip()
    m = _FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
        start = s.find("{", start + 1)
    return None


# =============================================================================
# Reward function + prompt dataset
# =============================================================================
def build_prompt_dataset(cfg: Config):
    import datasets
    from household_basket_env.server.environment import HouseholdBasketEnvironment
    from household_basket_env.server.curriculum import load_verified_seeds

    train_env = HouseholdBasketEnvironment()

    train_pool = []
    for tier_id in cfg.TIER_WEIGHTS:
        payload = load_verified_seeds(tier_id)
        for s in payload["training_seeds"]:
            train_pool.append((s, tier_id))
    if not train_pool:
        raise RuntimeError("Empty training pool. Re-run seed_verifier.")
    print(f"Training pool: {len(train_pool)} (seed, tier) pairs across "
          f"tiers {sorted(cfg.TIER_WEIGHTS)}.")

    rng = random.Random(cfg.SEED)
    tiers = list(cfg.TIER_WEIGHTS.keys())
    weights = [cfg.TIER_WEIGHTS[t] for t in tiers]

    rows = []
    for _ in range(cfg.N_PROMPTS):
        tier_id = rng.choices(tiers, weights=weights, k=1)[0]
        payload = load_verified_seeds(tier_id)
        seed = rng.choice(payload["training_seeds"])
        obs = train_env.reset(seed=seed, task_id=tier_id)
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": render_observation(obs)},
            ],
            "seed": seed,
            "task_id": tier_id,
        })
    ds = datasets.Dataset.from_list(rows)
    print(f"Prompt dataset: {ds}")
    return ds


def make_reward_fn(cfg: Config):
    from household_basket_env.server.environment import HouseholdBasketEnvironment

    # Hoisted to closure scope: one env instance reused across all reward calls.
    try:
        reward_env = HouseholdBasketEnvironment(
            enable_meal_type_coverage=cfg.ENABLE_MEAL_TYPE_COVERAGE
        )
    except TypeError:
        print("[warn] enable_meal_type_coverage kwarg missing; ablation_a will no-op.")
        reward_env = HouseholdBasketEnvironment()

    def reward_basket_step(prompts, completions, **kwargs):
        del prompts  # unused; TRL passes it positionally
        seeds = kwargs["seed"]
        task_ids = kwargs["task_id"]
        rewards = []
        for completion, seed, task_id in zip(completions, seeds, task_ids):
            reward_env.reset(seed=int(seed), task_id=int(task_id))
            text = completion[0]["content"] if isinstance(completion, list) else completion
            payload = extract_json(text)
            if payload is None:
                payload = text  # let env fire P_parse
            out = reward_env.apply_raw_action(payload)
            rewards.append(float(out.reward or 0.0))
        return rewards

    return reward_basket_step


# =============================================================================
# Trainer build + main
# =============================================================================
def build_trainer(cfg: Config, model, tokenizer, prompt_ds, reward_fn):
    from trl import GRPOConfig, GRPOTrainer

    config = GRPOConfig(
        output_dir=str(cfg.RUN_DIR),
        learning_rate=cfg.LEARNING_RATE,
        beta=cfg.GRPO_BETA,
        num_generations=cfg.NUM_GENERATIONS,
        per_device_train_batch_size=cfg.PER_DEVICE_BATCH,
        gradient_accumulation_steps=cfg.GRAD_ACCUM,
        max_prompt_length=cfg.MAX_SEQ_LEN - cfg.MAX_NEW_TOKENS,
        max_completion_length=cfg.MAX_NEW_TOKENS,
        max_steps=cfg.MAX_TRAIN_STEPS,
        save_steps=cfg.SAVE_EVERY,
        logging_steps=5,
        seed=cfg.SEED,
        bf16=True,
        fp16=False,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        temperature=cfg.TEMPERATURE,
        remove_unused_columns=False,
        report_to="none",
        top_p=cfg.TOP_P,
        top_k=cfg.TOP_K,
    )
    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=config,
        train_dataset=prompt_ds,
    )


def save_artifacts(cfg: Config, trainer, tokenizer):
    adapter_dir = cfg.RUN_DIR / "adapter_final"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved adapters -> {adapter_dir}")

    log_path = cfg.RUN_DIR / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"Saved log -> {log_path}")

    config_path = cfg.RUN_DIR / "run_config.json"
    serializable = {
        k: (str(v) if isinstance(v, pathlib.Path) else v)
        for k, v in cfg.to_dict().items()
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved config -> {config_path}")


def main():
    args = parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cfg = Config(args)

    # Side-effects the package needs (path to products, HF cache).
    os.environ["HOUSEHOLD_BASKET_PRODUCTS_PATH"] = str(ENV_ROOT / "data" / "products.json")
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))

    cfg.RUN_DIR.mkdir(parents=True, exist_ok=True)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RESOLVED CONFIG")
    print("=" * 60)
    print(cfg.pretty())
    print("=" * 60)

    if args.dry_run:
        print("[dry-run] exiting before model load")
        return

    # Sanity import — fail fast if env package isn't on PYTHONPATH.
    import household_basket_env.server.environment  # noqa: F401

    model, tokenizer, used_unsloth = load_model(cfg)
    print(f"USE_UNSLOTH = {used_unsloth}")

    prompt_ds = build_prompt_dataset(cfg)
    reward_fn = make_reward_fn(cfg)

    trainer = build_trainer(cfg, model, tokenizer, prompt_ds, reward_fn)
    print("Trainer constructed; beginning training ...")
    trainer.train()
    print("Training finished.")

    save_artifacts(cfg, trainer, tokenizer)


if __name__ == "__main__":
    main()
