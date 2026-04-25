# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Offline seed verifier (plan §8).

For each candidate seed, reconstruct the household / catalog / budget, run a
greedy solver that maximises (R_threshold + R_meal_type_coverage with budget
ties broken by cheaper), and compute the resulting episode reward. Mark the
seed as verified iff greedy_reward >= 0.6 * theoretical_max.

`theoretical_max` is conservatively defined per plan as the maximum *plausible*
episode reward for the tier:

    R_format       = +0.20 * max_steps
    R_threshold    = +0.40 * max_steps   (clip ceiling per step)
    R_budget       = +0.10 * max_steps
    R_meal_cov     = +0.15 * min(max_steps, n_meal_types_used_per_member)
                              ~= +0.15 * max_steps (worst case)
    R_terminal     = +1.0
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .basket_grader import grade_basket
from .catalog import absolute_nutrients_for_pick, load_augmented_catalog
from .curriculum import (
    SEEDS_DIR,
    TIER_CONFIGS,
    TierConfig,
    jittered_budget,
    select_candidates_for_tier,
    split_train_eval,
    verified_seeds_path,
)
from .household_fixtures import materialise_household
from .rewards import (
    R_FORMAT_VALID,
    R_MEAL_TYPE_COVERAGE,
    StepRewardBreakdown,
    compose_valid_step_reward,
)


def theoretical_max_reward(tier: TierConfig) -> float:
    return (
        R_FORMAT_VALID * tier.max_steps
        + 0.40 * tier.max_steps           # R_threshold clip ceiling
        + 0.10 * tier.max_steps           # R_budget bonus
        + R_MEAL_TYPE_COVERAGE * tier.max_steps  # very loose upper bound
        + 1.0                              # R_terminal ceiling
    )


@dataclass
class GreedyResult:
    seed: int
    task_id: int
    total_dense_reward: float
    terminal_reward: float
    total_reward: float
    theoretical_max: float
    ratio: float
    verified: bool
    basket: list[tuple[str, str]]


def _greedy_pick(
    candidates: list[dict],
    pre_pick_intake: dict[str, float],
    member_caps: dict[str, float],
    watched_nutrients: list[str],
    budget_remaining: float,
    steps_remaining: int,
    member_id: str,
    coverage_so_far: dict[str, set[str]],
    already_picked: set[str],
) -> tuple[dict, StepRewardBreakdown] | tuple[None, None]:
    best: tuple[float, dict, StepRewardBreakdown] | None = None
    for product in candidates:
        if product["product_id"] in already_picked:
            continue
        price = float(product.get("price_inr", 0.0))
        if price > budget_remaining:
            continue
        absolute = absolute_nutrients_for_pick(product)
        # Use the same composition as the env -- but with a non-mutating coverage view
        coverage_copy = {k: set(v) for k, v in coverage_so_far.items()}
        bd = compose_valid_step_reward(
            pre_pick_intake=pre_pick_intake,
            pick_absolute_nutrients=absolute,
            member_caps=member_caps,
            watched_nutrients=watched_nutrients,
            pick_price=price,
            budget_remaining_before=budget_remaining,
            steps_remaining_before=steps_remaining,
            member_id=member_id,
            pick_meal_type=product.get("meal_type", "snack"),
            coverage_so_far=coverage_copy,
            enable_meal_type_coverage=True,
        )
        score = bd.r_threshold + bd.r_meal_type_coverage - 1e-6 * price
        if best is None or score > best[0]:
            best = (score, product, bd)
    if best is None:
        return None, None
    return best[1], best[2]


def run_greedy_episode(seed: int, tier: TierConfig, catalog: list[dict]) -> GreedyResult:
    rng = random.Random(seed)
    household = materialise_household(tier.task_id)
    candidates = select_candidates_for_tier(catalog, tier, rng)
    candidates_by_id = {p["product_id"]: p for p in candidates}
    budget_total = jittered_budget(tier.base_budget_inr, rng)
    budget_remaining = budget_total

    cumulative_intake: dict[str, dict[str, float]] = {m.member_id: {} for m in household}
    coverage: dict[str, set[str]] = {m.member_id: set() for m in household}
    basket: list[tuple[str, str]] = []
    picked: set[str] = set()
    total_dense = 0.0

    # Round-robin members so each gets coverage; greedy chooses the best product
    # for the current member at each step.
    for step in range(tier.max_steps):
        member = household[step % len(household)]
        pre = dict(cumulative_intake[member.member_id])
        product, bd = _greedy_pick(
            candidates=candidates,
            pre_pick_intake=pre,
            member_caps=member.thresholds_cap,
            watched_nutrients=member.watched_nutrients,
            budget_remaining=budget_remaining,
            steps_remaining=tier.max_steps - step,
            member_id=member.member_id,
            coverage_so_far=coverage,
            already_picked=picked,
        )
        if product is None:
            break
        absolute = absolute_nutrients_for_pick(product)
        for k, v in absolute.items():
            pre[k] = round(pre.get(k, 0.0) + v, 4)
        cumulative_intake[member.member_id] = pre
        coverage.setdefault(member.member_id, set()).add(product.get("meal_type", "snack"))
        basket.append((product["product_id"], member.member_id))
        picked.add(product["product_id"])
        budget_remaining = round(budget_remaining - float(product.get("price_inr", 0.0)), 2)
        total_dense += bd.total

    grader = grade_basket(household, basket, candidates_by_id)
    terminal = grader.terminal_reward
    total_reward = total_dense + terminal
    max_r = theoretical_max_reward(tier)
    ratio = total_reward / max_r if max_r > 0 else 0.0

    return GreedyResult(
        seed=seed,
        task_id=tier.task_id,
        total_dense_reward=round(total_dense, 4),
        terminal_reward=terminal,
        total_reward=round(total_reward, 4),
        theoretical_max=round(max_r, 4),
        ratio=round(ratio, 4),
        verified=ratio >= 0.6,
        basket=list(basket),
    )


def verify_seeds_for_tier(
    task_id: int,
    *,
    n_target: int | None = None,
    seed_range: tuple[int, int] = (0, 10_000),
    rng_seed: int = 0,
) -> dict[str, Any]:
    tier = TIER_CONFIGS[task_id]
    if n_target is None:
        n_target = tier.n_verified_seeds_target
    catalog = load_augmented_catalog()

    rng = random.Random(rng_seed)
    candidate_seeds = list(range(seed_range[0], seed_range[1]))
    rng.shuffle(candidate_seeds)

    verified: list[GreedyResult] = []
    inspected = 0
    for s in candidate_seeds:
        if len(verified) >= n_target:
            break
        result = run_greedy_episode(s, tier, catalog)
        inspected += 1
        if result.verified:
            verified.append(result)

    seeds = [r.seed for r in verified]
    split_rng = random.Random(rng_seed + 1)
    training, held_out = split_train_eval(seeds, tier.n_held_out_seeds, split_rng)

    return {
        "task_id": task_id,
        "n_verified": len(verified),
        "n_inspected": inspected,
        "verified_seeds": seeds,
        "training_seeds": training,
        "held_out_seeds": held_out,
        "theoretical_max": theoretical_max_reward(tier),
        "tier": {
            "n_members": tier.n_members,
            "max_steps": tier.max_steps,
            "catalog_size": tier.catalog_size,
            "base_budget_inr": tier.base_budget_inr,
        },
        "details": [asdict(r) for r in verified],
    }


def write_seed_registry(payload: dict[str, Any]) -> Path:
    SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    path = verified_seeds_path(payload["task_id"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=None, help="Task id (1, 2, or 3). Default: all.")
    parser.add_argument("--max-seeds", type=int, default=None, help="Override n_verified_seeds_target.")
    parser.add_argument("--seed-range-end", type=int, default=10_000)
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args()

    task_ids = [args.task] if args.task is not None else [1, 2, 3]
    for tid in task_ids:
        print(f"[seed_verifier] verifying tier {tid} ...", flush=True)
        payload = verify_seeds_for_tier(
            tid,
            n_target=args.max_seeds,
            seed_range=(0, args.seed_range_end),
            rng_seed=args.rng_seed,
        )
        path = write_seed_registry(payload)
        print(
            f"[seed_verifier] tier={tid} verified={payload['n_verified']} "
            f"inspected={payload['n_inspected']} -> {path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
