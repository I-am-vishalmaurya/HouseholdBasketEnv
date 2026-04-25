# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for seed_verifier.py (plan §8)."""

from __future__ import annotations

import pytest

from household_basket_env.server.seed_verifier import (
    run_greedy_episode,
    theoretical_max_reward,
    verify_seeds_for_tier,
)
from household_basket_env.server.curriculum import TIER_CONFIGS
from household_basket_env.server.catalog import load_augmented_catalog


def test_theoretical_max_reward_is_positive():
    for tid in (1, 2, 3):
        tier = TIER_CONFIGS[tid]
        assert theoretical_max_reward(tier) > 0


def test_run_greedy_episode_produces_reasonable_basket():
    catalog = load_augmented_catalog()
    tier = TIER_CONFIGS[2]
    result = run_greedy_episode(seed=42, tier=tier, catalog=catalog)
    assert result.task_id == 2
    assert isinstance(result.verified, bool)
    assert -3.0 < result.total_reward < 6.0  # sanity envelope
    assert 0 < len(result.basket) <= tier.max_steps


def test_greedy_results_deterministic():
    catalog = load_augmented_catalog()
    tier = TIER_CONFIGS[2]
    a = run_greedy_episode(seed=42, tier=tier, catalog=catalog)
    b = run_greedy_episode(seed=42, tier=tier, catalog=catalog)
    assert a.basket == b.basket
    assert a.total_reward == pytest.approx(b.total_reward)


@pytest.mark.parametrize("task_id", [1, 2])  # skip 3 for speed (large catalog)
def test_verify_seeds_smoke(task_id):
    """Verify a small batch of seeds for the tier and assert structural shape."""
    payload = verify_seeds_for_tier(task_id, n_target=3, seed_range=(0, 50))
    assert payload["task_id"] == task_id
    assert payload["n_inspected"] >= payload["n_verified"]
    assert isinstance(payload["verified_seeds"], list)
    assert isinstance(payload["training_seeds"], list)
    assert isinstance(payload["held_out_seeds"], list)
    # Training + held_out must reconcile
    assert sorted(payload["training_seeds"] + payload["held_out_seeds"]) == sorted(payload["verified_seeds"])
