# HouseholdBasketEnv — GRPO Training (Phases 6 + 7 + 8)
# QLoRA + GRPO fine-tuning of Qwen2.5-3B-Instruct on HouseholdBasketEnv
# Unsloth removed — using plain transformers + peft + trl for reliability

# =============================================================================
# Pick exactly ONE run.
# =============================================================================
RUN_NAME = "ablation_b"            # "main" | "ablation_a" | "ablation_b"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Trainer hyperparameters -----------------------------------------------------
MAX_TRAIN_STEPS = 40
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8
NUM_GENERATIONS = 8       # GRPO group size
GRPO_BETA = 0.0           # 0.0 = no KL penalty (recommended, saves memory)
LEARNING_RATE = 5e-6
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 192
MAX_PROMPT_LENGTH = 1856  # MAX_SEQ_LEN - MAX_NEW_TOKENS
SAVE_EVERY = 10
SEED = 0

# LoRA / 4-bit ----------------------------------------------------------------
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0        # 0.0 required for gradient checkpointing efficiency
LOAD_IN_4BIT = True
MAX_SEQ_LEN = 2048

# Curriculum mix --------------------------------------------------------------
TIER_WEIGHTS = {2: 0.7, 3: 0.3}
if RUN_NAME == "ablation_b":
    TIER_WEIGHTS = {3: 1.0}

# Reward ablation -------------------------------------------------------------
ENABLE_MEAL_TYPE_COVERAGE = (RUN_NAME != "ablation_a")

print({
    "RUN_NAME": RUN_NAME,
    "TIER_WEIGHTS": TIER_WEIGHTS,
    "ENABLE_MEAL_TYPE_COVERAGE": ENABLE_MEAL_TYPE_COVERAGE,
    "MAX_TRAIN_STEPS": MAX_TRAIN_STEPS,
    "GRPO_BETA": GRPO_BETA,
})

# =============================================================================
# 2. Paths & env setup
# =============================================================================
import os, sys, pathlib

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

CLONE_DIR  = pathlib.Path("/app/hackathon/HouseholdBasketEnv")
REPO_ROOT  = CLONE_DIR.resolve()
ENV_ROOT   = REPO_ROOT / "household_basket_env"
RESULTS_DIR = ENV_ROOT / "notebooks" / "results"
RUN_DIR    = ENV_ROOT / "notebooks" / "runs" / RUN_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HOUSEHOLD_BASKET_PRODUCTS_PATH"] = str(ENV_ROOT / "data" / "products.json")
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))
sys.path.insert(0, "/app/hackathon/HouseholdBasketEnv")

os.environ["HTTP_PROXY"]  = "http://azureproxy.jio.com:8080"
os.environ["HTTPS_PROXY"] = "http://azureproxy.jio.com:8080"

print("REPO_ROOT   =", REPO_ROOT)
print("ENV_ROOT    =", ENV_ROOT)
print("RESULTS_DIR =", RESULTS_DIR)

from household_basket_env.server.environment import HouseholdBasketEnvironment
from household_basket_env.models import BasketAction
from household_basket_env.server.curriculum import (
    TIER_CONFIGS,
    load_verified_seeds,
    TRAIN_TIER_WEIGHTS,
)
print("Env package imports OK.")

# =============================================================================
# 3. Load Qwen2.5-3B-Instruct + QLoRA  (plain transformers + peft, no unsloth)
# =============================================================================
import time
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

print("=" * 60)
print("🔍 PRE-LOAD STATE")
print("=" * 60)
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    used = (total - free) / 1e9
    print(f"  GPU {i}: {used:.2f} GB used / {total/1e9:.1f} GB total")
print("=" * 60)

t_total = time.time()

# ── 4-bit quantisation config ─────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("\n⏳ [Step 1/3] Loading base model weights...")
t = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",   # sdpa/flash not needed on A100 with QLoRA
)
model.config.use_cache = False     # required for gradient checkpointing
print(f"  ✅ Base model loaded in {time.time()-t:.1f}s")

for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    used = (total - free) / 1e9
    print(f"  📊 GPU {i} after base load: {used:.2f} GB / {total/1e9:.1f} GB")

# ── Tokeniser ─────────────────────────────────────────────────────────────────
print("\n⏳ [Step 2/3] Loading tokeniser...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
print("  ✅ Tokeniser ready.")

# ── LoRA adapters ─────────────────────────────────────────────────────────────
print("\n⏳ [Step 3/3] Attaching LoRA adapters...")
t = time.time()
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()   # needed for gradient checkpointing with PEFT
print(f"  ✅ LoRA attached in {time.time()-t:.1f}s")
model.print_trainable_parameters()

print(f"\n✅ Model + LoRA ready (total: {time.time()-t_total:.1f}s)")
print("\n" + "=" * 60)
print("🚀 Model ready!")
print("=" * 60)

# =============================================================================
# 4. Prompt utilities & reward function
# =============================================================================
import json, re, random
from textwrap import dedent

SYSTEM_PROMPT = (
    "You are a careful nutrition-aware shopping assistant for an Indian household. "
    "At each step you pick exactly ONE product from the catalog and assign it to ONE household member. "
    'You MUST output ONLY a JSON object with three keys: "product_id" (string), "member_id" (string), '
    '"rationale" (1 short sentence). '
    "Do NOT add any prose, code fences, or extra keys. "
    "Respect each member's caps and allergies; spread meal types across breakfast/lunch/dinner/snack/beverage; "
    "stay under the budget."
)

JSON_RE  = re.compile(r"\{[^{}]*\}", re.S)
MAX_CANDS = 20

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
    shown  = min(len(obs.candidates), MAX_CANDS)
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
    m = JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

print("Prompt utilities ready.")

# ── Training pool ─────────────────────────────────────────────────────────────
train_pool = []
for tier_id, w in TIER_WEIGHTS.items():
    payload = load_verified_seeds(tier_id)
    for s in payload["training_seeds"]:
        train_pool.append((s, tier_id))

if not train_pool:
    raise RuntimeError("Empty training pool. Re-run seed_verifier.")

print(f"Training pool: {len(train_pool)} (seed, tier) pairs across tiers {sorted(TIER_WEIGHTS.keys())}.")

TRAIN_ENV = HouseholdBasketEnvironment()

def sample_starting_observation(rng: random.Random):
    tiers   = list(TIER_WEIGHTS.keys())
    weights = [TIER_WEIGHTS[t] for t in tiers]
    tier_id = rng.choices(tiers, weights=weights, k=1)[0]
    payload = load_verified_seeds(tier_id)
    seed    = rng.choice(payload["training_seeds"])
    obs     = TRAIN_ENV.reset(seed=seed, task_id=tier_id)
    return seed, tier_id, obs

# ── Prompt dataset ────────────────────────────────────────────────────────────
import datasets

def make_prompt_record(rng: random.Random):
    seed, tier_id, obs = sample_starting_observation(rng)
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": render_observation(obs)},
        ],
        "seed":    seed,
        "task_id": tier_id,
    }

PROMPT_RNG = random.Random(SEED)
prompt_rows = [make_prompt_record(PROMPT_RNG) for _ in range(4096)]
prompt_ds   = datasets.Dataset.from_list(prompt_rows)
print("Prompt dataset:", prompt_ds)

# ── Reward function ───────────────────────────────────────────────────────────
import inspect

def reward_basket_step(prompts, completions, **kwargs):
    """Returns one float per (prompt, completion)."""
    seeds    = kwargs["seed"]
    task_ids = kwargs["task_id"]
    rewards  = []
    env = HouseholdBasketEnvironment(enable_meal_type_coverage=ENABLE_MEAL_TYPE_COVERAGE)
    for completion, seed, task_id in zip(completions, seeds, task_ids):
        env.reset(seed=int(seed), task_id=int(task_id))
        text    = completion[0]["content"] if isinstance(completion, list) else completion
        payload = extract_json(text)
        if payload is None:
            payload = text
        out = env.apply_raw_action(payload)
        rewards.append(float(out.reward or 0.0))
    return rewards

sig = inspect.signature(HouseholdBasketEnvironment.__init__)
if "enable_meal_type_coverage" not in sig.parameters:
    print("[warn] enable_meal_type_coverage kwarg not in env; ablation_a will silently no-op.")
print("Reward fn ready.")

# =============================================================================
# 5. GRPOTrainer  (plain TRL — no unsloth patches)
# =============================================================================
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    output_dir=str(RUN_DIR),
    learning_rate=LEARNING_RATE,
    beta=GRPO_BETA,                          # 0.0 = no KL term, no ref model needed
    num_generations=NUM_GENERATIONS,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    max_completion_length=MAX_NEW_TOKENS,
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_steps=MAX_TRAIN_STEPS,
    save_steps=SAVE_EVERY,
    logging_steps=10,
    seed=SEED,
    bf16=True,
    fp16=False,
    optim="adamw_8bit",                      # bitsandbytes 8-bit Adam
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    temperature=TEMPERATURE,
    remove_unused_columns=False,
    report_to="none",
    gradient_checkpointing=True,             # saves VRAM
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_basket_step],
    args=config,
    train_dataset=prompt_ds,
)

print("Trainer constructed. Starting training...")
history = trainer.train()
print("Training finished.")

# =============================================================================
# 6. Save checkpoints + training log
# =============================================================================
ADAPTER_DIR = RUN_DIR / "adapter_final"
trainer.save_model(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))
print("Saved final adapters ->", ADAPTER_DIR)

log_path = RUN_DIR / "training_log.json"
with open(log_path, "w", encoding="utf-8") as f:
    json.dump(trainer.state.log_history, f, indent=2)
print("Saved log ->", log_path)

config_path = RUN_DIR / "run_config.json"
with open(config_path, "w", encoding="utf-8") as f:
    json.dump({
        "RUN_NAME":                 RUN_NAME,
        "MODEL_NAME":               MODEL_NAME,
        "TIER_WEIGHTS":             TIER_WEIGHTS,
        "ENABLE_MEAL_TYPE_COVERAGE": ENABLE_MEAL_TYPE_COVERAGE,
        "MAX_TRAIN_STEPS":          MAX_TRAIN_STEPS,
        "PER_DEVICE_BATCH":         PER_DEVICE_BATCH,
        "GRAD_ACCUM":               GRAD_ACCUM,
        "NUM_GENERATIONS":          NUM_GENERATIONS,
        "GRPO_BETA":                GRPO_BETA,
        "LEARNING_RATE":            LEARNING_RATE,
        "TEMPERATURE":              TEMPERATURE,
        "MAX_NEW_TOKENS":           MAX_NEW_TOKENS,
        "LORA_R":                   LORA_R,
        "LORA_ALPHA":               LORA_ALPHA,
        "LORA_DROPOUT":             LORA_DROPOUT,
        "LOAD_IN_4BIT":             LOAD_IN_4BIT,
    }, f, indent=2)
print("Saved config ->", config_path)