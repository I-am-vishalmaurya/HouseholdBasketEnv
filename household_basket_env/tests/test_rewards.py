# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the dense reward components (plan §4.1-§4.4, §4.6)."""

from __future__ import annotations

import pytest

from household_basket_env.server.rewards import (
    R_BUDGET_BONUS,
    R_FORMAT_VALID,
    R_MEAL_TYPE_COVERAGE,
    R_THRESHOLD_CLIP_MAX,
    R_THRESHOLD_CLIP_MIN,
    R_THRESHOLD_PEAK,
    R_THRESHOLD_UNDER,
    R_THRESHOLD_VIOLATION,
    StepRewardBreakdown,
    compose_valid_step_reward,
    r_budget_for_pick,
    r_meal_type_coverage,
    r_threshold_for_pick,
    r_threshold_per_nutrient,
)


# ---------------------------------------------------------------------------
# R_threshold smooth triangular (plan §4.2)
# ---------------------------------------------------------------------------

def test_r_threshold_violation_above_cap():
    assert r_threshold_per_nutrient(post_pick_intake=110, cap=100) == pytest.approx(R_THRESHOLD_VIOLATION)


def test_r_threshold_peaks_at_60_percent_of_cap():
    val = r_threshold_per_nutrient(post_pick_intake=60, cap=100)
    assert val == pytest.approx(R_THRESHOLD_PEAK)


def test_r_threshold_under_consumption_band():
    assert r_threshold_per_nutrient(post_pick_intake=10, cap=100) == pytest.approx(R_THRESHOLD_UNDER)


def test_r_threshold_decays_linearly_to_low_band():
    # At m=0.20 -> +0.10 * (1 - |0.20-0.60|/0.40) = +0.10 * 0 = 0
    val = r_threshold_per_nutrient(post_pick_intake=20, cap=100)
    assert val == pytest.approx(0.0, abs=1e-9)


def test_r_threshold_decays_linearly_to_high_band():
    # At m=1.00 -> +0.10 * (1 - 0.40/0.40) = 0
    val = r_threshold_per_nutrient(post_pick_intake=100, cap=100)
    assert val == pytest.approx(0.0, abs=1e-9)


def test_r_threshold_clipped_to_range():
    # Force several big violations -> total should clip to CLIP_MIN
    pre = {"sugars_g": 100, "sodium_mg": 1500, "fat_g": 100, "energy_kcal": 2500}
    pick = {"sugars_g": 100, "sodium_mg": 1500, "fat_g": 100, "energy_kcal": 2500}
    caps = {
        "sugars_g_max": 25,
        "sodium_mg_max": 1500,
        "fat_g_max": 65,
        "energy_kcal_max": 1900,
    }
    total, _ = r_threshold_for_pick(pre, pick, caps, ["sugars_g", "sodium_mg", "fat_g", "energy_kcal"])
    assert total == pytest.approx(R_THRESHOLD_CLIP_MIN)


def test_r_threshold_zero_cap_returns_zero():
    assert r_threshold_per_nutrient(50, 0) == pytest.approx(0.0)
    assert r_threshold_per_nutrient(0, 0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# R_budget (plan §4.3)
# ---------------------------------------------------------------------------

def test_r_budget_bonus_when_within_allowance():
    # allowance = 100 / 5 = 20; pick 18 -> bonus
    assert r_budget_for_pick(pick_price=18, budget_remaining_before=100, steps_remaining_before=5) == pytest.approx(R_BUDGET_BONUS)


def test_r_budget_no_bonus_when_overshooting_allowance():
    assert r_budget_for_pick(pick_price=22, budget_remaining_before=100, steps_remaining_before=5) == pytest.approx(0.0)


def test_r_budget_zero_steps_remaining_returns_zero():
    assert r_budget_for_pick(pick_price=10, budget_remaining_before=100, steps_remaining_before=0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# R_meal_type_coverage (plan §4.4)
# ---------------------------------------------------------------------------

def test_r_meal_type_coverage_first_pick_fires():
    coverage: dict[str, set[str]] = {}
    val = r_meal_type_coverage(member_id="m1", pick_meal_type="dairy", coverage_so_far=coverage)
    assert val == pytest.approx(R_MEAL_TYPE_COVERAGE)
    assert coverage == {"m1": {"dairy"}}


def test_r_meal_type_coverage_repeat_pick_zero():
    coverage: dict[str, set[str]] = {"m1": {"dairy"}}
    val = r_meal_type_coverage(member_id="m1", pick_meal_type="dairy", coverage_so_far=coverage)
    assert val == pytest.approx(0.0)


def test_r_meal_type_coverage_separate_members_independent():
    coverage: dict[str, set[str]] = {"m1": {"dairy"}}
    val = r_meal_type_coverage(member_id="m2", pick_meal_type="dairy", coverage_so_far=coverage)
    assert val == pytest.approx(R_MEAL_TYPE_COVERAGE)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def test_compose_valid_step_includes_format_and_threshold_and_budget_and_meal():
    coverage: dict[str, set[str]] = {}
    bd = compose_valid_step_reward(
        pre_pick_intake={"sugars_g": 0},
        pick_absolute_nutrients={"sugars_g": 15},
        member_caps={"sugars_g_max": 25},
        watched_nutrients=["sugars_g"],
        pick_price=10,
        budget_remaining_before=100,
        steps_remaining_before=5,
        member_id="m1",
        pick_meal_type="dairy",
        coverage_so_far=coverage,
        enable_meal_type_coverage=True,
    )
    assert bd.r_format == pytest.approx(R_FORMAT_VALID)
    assert bd.r_threshold > 0  # 15/25=0.6 -> peak
    assert bd.r_budget == pytest.approx(R_BUDGET_BONUS)
    assert bd.r_meal_type_coverage == pytest.approx(R_MEAL_TYPE_COVERAGE)
    assert bd.total > 0


def test_ablation_a_disables_meal_type_coverage():
    coverage: dict[str, set[str]] = {}
    bd = compose_valid_step_reward(
        pre_pick_intake={"sugars_g": 0},
        pick_absolute_nutrients={"sugars_g": 15},
        member_caps={"sugars_g_max": 25},
        watched_nutrients=["sugars_g"],
        pick_price=10,
        budget_remaining_before=100,
        steps_remaining_before=5,
        member_id="m1",
        pick_meal_type="dairy",
        coverage_so_far=coverage,
        enable_meal_type_coverage=False,
    )
    assert bd.r_meal_type_coverage == pytest.approx(0.0)
    # Coverage dict should NOT have been updated when disabled
    assert "m1" not in coverage


def test_step_reward_breakdown_total_sums_components():
    bd = StepRewardBreakdown(r_format=0.20, r_threshold=0.10, r_budget=0.10, r_meal_type_coverage=0.15, terminal=1.0)
    assert bd.total == pytest.approx(0.20 + 0.10 + 0.10 + 0.15 + 1.0)


def test_step_reward_breakdown_serializable():
    bd = StepRewardBreakdown(r_format=0.20)
    d = bd.as_dict()
    assert "r_format" in d and "total" in d


# ---------------------------------------------------------------------------
# Anti-water-park check (plan §4.2 commentary)
# ---------------------------------------------------------------------------

def test_water_strategy_only_earns_under_consumption_bonus():
    """If every pick adds essentially zero to every nutrient, R_threshold returns
    only the +0.02 under-consumption bonus per watched nutrient. This is by
    design — water strategy must NOT dominate."""
    pre = {"sugars_g": 0, "sodium_mg": 0, "fat_g": 0, "energy_kcal": 0}
    pick = {"sugars_g": 0, "sodium_mg": 0, "fat_g": 0, "energy_kcal": 0}
    caps = {"sugars_g_max": 25, "sodium_mg_max": 1500, "fat_g_max": 65, "energy_kcal_max": 1900}
    watched = ["sugars_g", "sodium_mg", "fat_g", "energy_kcal"]
    total, breakdown = r_threshold_for_pick(pre, pick, caps, watched)
    expected = R_THRESHOLD_UNDER * 4  # one per watched nutrient
    expected_clipped = max(R_THRESHOLD_CLIP_MIN, min(R_THRESHOLD_CLIP_MAX, expected))
    assert total == pytest.approx(expected_clipped)
    assert all(v == pytest.approx(R_THRESHOLD_UNDER) for v in breakdown.values())
