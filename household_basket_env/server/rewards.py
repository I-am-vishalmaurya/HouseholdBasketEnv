# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Dense reward components for HouseholdBasketEnv.

Implements every signal in plan §4 except R_terminal (which lives in
basket_grader.py). All functions here are pure, side-effect free, and operate
on dict-shaped state for trivial unit-testability.

Key invariants the agent must NOT be able to game:
- R_threshold uses a smooth triangular peak at 60% of cap. Stuffing the basket
  with water-style zero-everything items earns only the +0.02 under-consumption
  bonus; not enough to dominate.
- R_meal_type_coverage is hard-capped at one bonus per (member, meal_type) so
  spamming the same category for the same member yields zero after the first.
- All penalties (P_*) are negative constants matched in scale to the positive
  signals (per plan §4.6).
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Reward weights (plan §4)
# ---------------------------------------------------------------------------

R_FORMAT_VALID: float = 0.20
R_FORMAT_PARSE: float = -0.25  # P_parse

R_THRESHOLD_VIOLATION: float = -0.30
R_THRESHOLD_PEAK: float = 0.10
R_THRESHOLD_UNDER: float = 0.02
R_THRESHOLD_LOW_BAND: float = 0.20
R_THRESHOLD_PEAK_BAND: float = 0.60
R_THRESHOLD_HIGH_BAND: float = 1.00
R_THRESHOLD_CLIP_MIN: float = -0.6
R_THRESHOLD_CLIP_MAX: float = 0.4

R_BUDGET_BONUS: float = 0.10

R_MEAL_TYPE_COVERAGE: float = 0.15

P_DUPLICATE: float = -0.30
P_UNKNOWN_MEMBER: float = -0.40
# P_over_budget: terminates the episode; numerical value 0.0 returned for the
# step (the negative signal comes from cutting off future reward + the dense
# penalties already accumulated).
P_OVER_BUDGET: float = 0.0


@dataclass
class StepRewardBreakdown:
    """Per-step reward broken into component scalars (plan §6 monitoring)."""

    r_format: float = 0.0
    r_threshold: float = 0.0
    r_budget: float = 0.0
    r_meal_type_coverage: float = 0.0
    p_duplicate: float = 0.0
    p_unknown_member: float = 0.0
    p_parse: float = 0.0
    p_over_budget: float = 0.0
    terminal: float = 0.0  # populated only at terminal step

    @property
    def total(self) -> float:
        return (
            self.r_format
            + self.r_threshold
            + self.r_budget
            + self.r_meal_type_coverage
            + self.p_duplicate
            + self.p_unknown_member
            + self.p_parse
            + self.p_over_budget
            + self.terminal
        )

    def as_dict(self) -> dict[str, float]:
        d = {
            "r_format": self.r_format,
            "r_threshold": self.r_threshold,
            "r_budget": self.r_budget,
            "r_meal_type_coverage": self.r_meal_type_coverage,
            "p_duplicate": self.p_duplicate,
            "p_unknown_member": self.p_unknown_member,
            "p_parse": self.p_parse,
            "p_over_budget": self.p_over_budget,
            "terminal": self.terminal,
            "total": self.total,
        }
        return d


# ---------------------------------------------------------------------------
# R_threshold smooth triangular (plan §4.2)
# ---------------------------------------------------------------------------

def r_threshold_per_nutrient(post_pick_intake: float, cap: float) -> float:
    """Smooth triangular reward for a single nutrient on a single member.

    Returns:
        -0.30 if post_pick_intake / cap > 1.00 (violation)
        triangular peak +0.10 at m=0.60 in band [0.20, 1.00]
        +0.02 if m < 0.20 (mild under-consumption)
    """
    if cap <= 0:
        return 0.0
    m = post_pick_intake / cap
    if m > R_THRESHOLD_HIGH_BAND:
        return R_THRESHOLD_VIOLATION
    if m < R_THRESHOLD_LOW_BAND:
        return R_THRESHOLD_UNDER
    half_width = R_THRESHOLD_HIGH_BAND - R_THRESHOLD_PEAK_BAND  # 0.40 in standard config
    return R_THRESHOLD_PEAK * (1.0 - abs(m - R_THRESHOLD_PEAK_BAND) / half_width)


def r_threshold_for_pick(
    pre_pick_intake: dict[str, float],
    pick_absolute_nutrients: dict[str, float],
    member_caps: dict[str, float],
    watched_nutrients: list[str],
) -> tuple[float, dict[str, float]]:
    """Sum-then-clip the per-nutrient triangular signal across watched nutrients.

    Returns (clipped_total, per_nutrient_breakdown).
    """
    breakdown: dict[str, float] = {}
    total = 0.0
    for nut in watched_nutrients:
        cap_key = f"{nut}_max"
        cap = member_caps.get(cap_key)
        if cap is None or cap <= 0:
            continue
        post = pre_pick_intake.get(nut, 0.0) + pick_absolute_nutrients.get(nut, 0.0)
        signal = r_threshold_per_nutrient(post, cap)
        breakdown[nut] = round(signal, 6)
        total += signal
    clipped = max(R_THRESHOLD_CLIP_MIN, min(R_THRESHOLD_CLIP_MAX, total))
    return round(clipped, 6), breakdown


# ---------------------------------------------------------------------------
# R_budget (plan §4.3)
# ---------------------------------------------------------------------------

def r_budget_for_pick(
    pick_price: float,
    budget_remaining_before: float,
    steps_remaining_before: int,
) -> float:
    """+0.10 when pick price <= per-step allowance.

    `steps_remaining_before` includes the current step. So allowance = budget /
    steps_remaining. Returns 0 when steps_remaining_before <= 0.
    """
    if steps_remaining_before <= 0:
        return 0.0
    allowance = budget_remaining_before / steps_remaining_before
    return R_BUDGET_BONUS if pick_price <= allowance else 0.0


# ---------------------------------------------------------------------------
# R_meal_type_coverage (plan §4.4)
# ---------------------------------------------------------------------------

def r_meal_type_coverage(
    member_id: str,
    pick_meal_type: str,
    coverage_so_far: dict[str, set[str]],
) -> float:
    """+0.15 when (member_id, meal_type) is new for this episode.

    Mutates `coverage_so_far` to record the new (member, meal_type) pair iff the
    bonus fires. The caller is responsible for passing in a per-episode dict.
    """
    seen = coverage_so_far.setdefault(member_id, set())
    if pick_meal_type in seen:
        return 0.0
    seen.add(pick_meal_type)
    return R_MEAL_TYPE_COVERAGE


# ---------------------------------------------------------------------------
# Composition helper used by environment.py
# ---------------------------------------------------------------------------

def compose_valid_step_reward(
    *,
    pre_pick_intake: dict[str, float],
    pick_absolute_nutrients: dict[str, float],
    member_caps: dict[str, float],
    watched_nutrients: list[str],
    pick_price: float,
    budget_remaining_before: float,
    steps_remaining_before: int,
    member_id: str,
    pick_meal_type: str,
    coverage_so_far: dict[str, set[str]],
    enable_meal_type_coverage: bool = True,
) -> StepRewardBreakdown:
    """Aggregate every dense signal that fires on a fully-valid step.

    `enable_meal_type_coverage` toggles R_meal_type_coverage off for Ablation A
    (plan §7).
    """
    bd = StepRewardBreakdown()
    bd.r_format = R_FORMAT_VALID
    r_thr, _ = r_threshold_for_pick(
        pre_pick_intake=pre_pick_intake,
        pick_absolute_nutrients=pick_absolute_nutrients,
        member_caps=member_caps,
        watched_nutrients=watched_nutrients,
    )
    bd.r_threshold = r_thr
    bd.r_budget = r_budget_for_pick(
        pick_price=pick_price,
        budget_remaining_before=budget_remaining_before,
        steps_remaining_before=steps_remaining_before,
    )
    if enable_meal_type_coverage:
        bd.r_meal_type_coverage = r_meal_type_coverage(
            member_id=member_id,
            pick_meal_type=pick_meal_type,
            coverage_so_far=coverage_so_far,
        )
    return bd
