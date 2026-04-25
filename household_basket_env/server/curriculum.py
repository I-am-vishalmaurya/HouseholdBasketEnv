# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Curriculum: tier configs, candidate-pool construction, train/eval seed split.

Plan §3:
| Tier | Members | Valid steps | Catalog            | Budget (INR) | Role                                |
| 1    | 1       | 3           | 20 curated         | 500          | Held-out easy eval                  |
| 2    | 2       | 5           | 50                 | 1000         | Training (70%)                      |
| 3    | 3       | 7           | 492 incl. 32 adv   | 1500         | Training (30%) + held-out hard eval |

Plan §3 train/eval split: 100 verified seeds total per tier; for Task 3, 70 training
+ 30 held-out. Task 1 is held-out eval entirely. Task 2 is mostly training.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .catalog import filter_catalog


_THIS_DIR = Path(__file__).resolve().parent
SEEDS_DIR = _THIS_DIR.parent / "seeds"


@dataclass(frozen=True)
class TierConfig:
    task_id: int
    n_members: int
    max_steps: int
    catalog_size: int
    include_adversarial: bool
    base_budget_inr: float

    # Curriculum bookkeeping
    n_verified_seeds_target: int   # how many verified seeds to materialise offline
    n_held_out_seeds: int          # how many of those are held-out (never seen during train)


TIER_CONFIGS: dict[int, TierConfig] = {
    1: TierConfig(
        task_id=1,
        n_members=1,
        max_steps=3,
        catalog_size=20,
        include_adversarial=False,
        base_budget_inr=500.0,
        n_verified_seeds_target=40,
        n_held_out_seeds=40,    # entirely held-out per plan §3
    ),
    2: TierConfig(
        task_id=2,
        n_members=2,
        max_steps=5,
        catalog_size=50,
        include_adversarial=False,
        base_budget_inr=1000.0,
        n_verified_seeds_target=100,
        n_held_out_seeds=30,
    ),
    3: TierConfig(
        task_id=3,
        n_members=3,
        max_steps=7,
        catalog_size=492,        # plan: full catalog + adversarial
        include_adversarial=True,
        base_budget_inr=1500.0,
        n_verified_seeds_target=100,
        n_held_out_seeds=30,
    ),
}


# Curriculum mixing weights for the main GRPO run (plan §6).
# Used by training.ipynb to sample (task_id, seed) pairs.
TRAIN_TIER_WEIGHTS: dict[int, float] = {
    2: 0.70,
    3: 0.30,
}


def select_candidates_for_tier(
    augmented_catalog: list[dict],
    tier: TierConfig,
    rng: random.Random,
) -> list[dict]:
    """Deterministically pick `tier.catalog_size` candidate products for an episode."""
    pool = filter_catalog(augmented_catalog, include_adversarial=tier.include_adversarial)

    if tier.task_id == 3 and tier.include_adversarial:
        adversarial = [p for p in pool if p.get("is_adversarial")]
        normal = [p for p in pool if not p.get("is_adversarial")]
        rng.shuffle(adversarial)
        rng.shuffle(normal)
        n_adv = min(len(adversarial), 32)
        n_normal = max(0, tier.catalog_size - n_adv)
        chosen = adversarial[:n_adv] + normal[:n_normal]
        rng.shuffle(chosen)
        return chosen

    if tier.task_id == 1:
        # Curated: 20 mostly-healthy, no adversarial, varied meal_types
        meal_types = ("staple", "protein", "dairy", "snack", "beverage")
        bucket: dict[str, list[dict]] = {m: [] for m in meal_types}
        for p in pool:
            mt = p.get("meal_type")
            if mt in bucket:
                bucket[mt].append(p)
        for mt in bucket:
            rng.shuffle(bucket[mt])
        chosen: list[dict] = []
        per_bucket = tier.catalog_size // len(bucket)
        for mt in bucket:
            chosen.extend(bucket[mt][:per_bucket])
        if len(chosen) < tier.catalog_size:
            extras = [p for p in pool if p not in chosen]
            rng.shuffle(extras)
            chosen.extend(extras[: tier.catalog_size - len(chosen)])
        rng.shuffle(chosen)
        return chosen[: tier.catalog_size]

    rng.shuffle(pool)
    return pool[: tier.catalog_size]


def jittered_budget(base_budget: float, rng: random.Random) -> float:
    """±10% deterministic budget jitter (plan §2.4)."""
    factor = rng.uniform(0.9, 1.1)
    return round(base_budget * factor, 2)


# ---------------------------------------------------------------------------
# Seed registry I/O
# ---------------------------------------------------------------------------

def verified_seeds_path(task_id: int) -> Path:
    return SEEDS_DIR / f"verified_task{task_id}.json"


def load_verified_seeds(task_id: int) -> dict:
    path = verified_seeds_path(task_id)
    if not path.exists():
        return {"task_id": task_id, "verified_seeds": [], "held_out_seeds": [], "training_seeds": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_seed_verified(task_id: int, seed: int) -> bool:
    data = load_verified_seeds(task_id)
    return seed in set(data.get("verified_seeds", []))


def split_train_eval(seeds: list[int], n_held_out: int, rng: random.Random) -> tuple[list[int], list[int]]:
    """Return (training_seeds, held_out_seeds). Order is deterministic given rng."""
    seeds = list(seeds)
    rng.shuffle(seeds)
    held_out = seeds[:n_held_out]
    training = seeds[n_held_out:]
    return training, held_out
