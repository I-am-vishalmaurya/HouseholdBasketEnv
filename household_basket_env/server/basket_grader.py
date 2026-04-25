# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Basket grader — terminal scoring for HouseholdBasketEnv (plan §4.5).

Pure function. Zero side effects. Zero network. References only the augmented
catalog dict and member-state dicts.

Returns a structured GraderResult so:
- environment.py can populate `terminal_reward` deterministically;
- tests can assert per-component facts without re-implementing the grader.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import HouseholdMember
from .catalog import absolute_nutrients_for_pick
from .household_fixtures import per_item_compatible


# Plan §4.5 reward levels.
TERMINAL_FULL_PASS: float = 1.0
TERMINAL_PARTIAL_CREDIT: float = 0.3
TERMINAL_HARD_VIOLATION: float = -0.5


@dataclass
class MemberGrade:
    member_id: str
    cap_violations: list[str] = field(default_factory=list)
    floor_failures: list[str] = field(default_factory=list)
    incompatible_picks: list[str] = field(default_factory=list)
    cumulative_intake: dict[str, float] = field(default_factory=dict)

    @property
    def passes(self) -> bool:
        return not self.cap_violations and not self.floor_failures and not self.incompatible_picks


@dataclass
class GraderResult:
    terminal_reward: float
    members: list[MemberGrade]
    any_cap_violation: bool
    any_floor_failure: bool
    any_per_item_failure: bool

    def as_dict(self) -> dict:
        return {
            "terminal_reward": self.terminal_reward,
            "any_cap_violation": self.any_cap_violation,
            "any_floor_failure": self.any_floor_failure,
            "any_per_item_failure": self.any_per_item_failure,
            "members": [
                {
                    "member_id": m.member_id,
                    "passes": m.passes,
                    "cap_violations": m.cap_violations,
                    "floor_failures": m.floor_failures,
                    "incompatible_picks": m.incompatible_picks,
                    "cumulative_intake": m.cumulative_intake,
                }
                for m in self.members
            ],
        }


def _compute_member_intake(
    basket: list[tuple[str, str]],
    products_by_id: dict[str, dict],
    member_id: str,
) -> dict[str, float]:
    """Sum absolute nutrients across all items tagged to `member_id`."""
    totals: dict[str, float] = {}
    for product_id, tagged_member in basket:
        if tagged_member != member_id:
            continue
        product = products_by_id.get(product_id)
        if product is None:
            continue
        absolute = absolute_nutrients_for_pick(product)
        for k, v in absolute.items():
            totals[k] = round(totals.get(k, 0.0) + v, 4)
    return totals


def grade_member(
    member: HouseholdMember,
    basket: list[tuple[str, str]],
    products_by_id: dict[str, dict],
) -> MemberGrade:
    intake = _compute_member_intake(basket, products_by_id, member.member_id)

    cap_violations: list[str] = []
    for nut, cap in member.thresholds_cap.items():
        # cap key shape: "<nutrient>_max"
        if not nut.endswith("_max"):
            continue
        nutrient = nut[: -len("_max")]
        if intake.get(nutrient, 0.0) > cap:
            cap_violations.append(f"{nutrient}={intake.get(nutrient, 0.0):.2f} > {cap}")

    floor_failures: list[str] = []
    for nut, floor in member.thresholds_floor.items():
        if not nut.endswith("_min"):
            continue
        nutrient = nut[: -len("_min")]
        if intake.get(nutrient, 0.0) < floor:
            floor_failures.append(f"{nutrient}={intake.get(nutrient, 0.0):.2f} < {floor}")

    incompatible_picks: list[str] = []
    for product_id, tagged_member in basket:
        if tagged_member != member.member_id:
            continue
        product = products_by_id.get(product_id)
        if product is None:
            continue
        if not per_item_compatible(product, member):
            incompatible_picks.append(product_id)

    return MemberGrade(
        member_id=member.member_id,
        cap_violations=cap_violations,
        floor_failures=floor_failures,
        incompatible_picks=incompatible_picks,
        cumulative_intake=intake,
    )


def grade_basket(
    household: list[HouseholdMember],
    basket: list[tuple[str, str]],
    products_by_id: dict[str, dict],
) -> GraderResult:
    """Compute terminal reward and per-member breakdown.

    `basket` is a list of (product_id, member_id) tuples — the flat sequence of
    valid picks the agent made. `products_by_id` is the augmented catalog as a
    mapping. `household` is the list of HouseholdMember for this episode.
    """
    member_grades = [grade_member(m, basket, products_by_id) for m in household]

    any_cap = any(mg.cap_violations for mg in member_grades)
    any_floor = any(mg.floor_failures for mg in member_grades)
    any_item = any(mg.incompatible_picks for mg in member_grades)

    if any_cap:
        # Plan §4.5: -0.5 if any member exceeds a cap (hard violation).
        reward = TERMINAL_HARD_VIOLATION
    elif not any_item and not any_floor:
        reward = TERMINAL_FULL_PASS
    else:
        # Per-item sanity passes but cumulative caps NOT violated AND
        # (item-incompatibility or floor failure remains) -> partial credit.
        reward = TERMINAL_PARTIAL_CREDIT

    return GraderResult(
        terminal_reward=reward,
        members=member_grades,
        any_cap_violation=any_cap,
        any_floor_failure=any_floor,
        any_per_item_failure=any_item,
    )
