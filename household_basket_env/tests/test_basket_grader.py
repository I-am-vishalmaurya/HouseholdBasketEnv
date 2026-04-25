# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for basket_grader.py (plan §4.5)."""

from __future__ import annotations

import pytest

from household_basket_env.models import HouseholdMember
from household_basket_env.server.basket_grader import (
    TERMINAL_FULL_PASS,
    TERMINAL_HARD_VIOLATION,
    TERMINAL_PARTIAL_CREDIT,
    grade_basket,
    grade_member,
)
from household_basket_env.server.household_fixtures import materialise_household


def _toy_product(pid: str, category: str, **kwargs):
    nut = {
        "energy_kcal": 100.0,
        "sugars_g": 5.0,
        "sodium_mg": 100.0,
        "fat_g": 5.0,
        "protein_g": 5.0,
        "fiber_g": 1.0,
    }
    nut.update(kwargs.get("nutrition_per_100g", {}))
    return {
        "product_id": pid,
        "product_name": pid,
        "brand": "Test",
        "category": category,
        "nutrition_per_100g": nut,
        "marketing_claims": [],
        "nutri_score": "C",
        "nova_group": 2,
        "is_adversarial": False,
        "meal_type": kwargs.get("meal_type", "staple"),
        "price_inr": kwargs.get("price_inr", 50.0),
        "grams_per_pick": kwargs.get("grams_per_pick", 100.0),
    }


def test_full_pass_returns_plus_one():
    """Healthy adult, picks within caps, meets floors -> terminal +1.0."""
    household = materialise_household(task_id=1)
    member = household[0]
    products_by_id = {
        "P1": _toy_product("P1", "flour", nutrition_per_100g={"energy_kcal": 1500, "protein_g": 60, "fiber_g": 25, "sugars_g": 15, "sodium_mg": 100, "fat_g": 5}, grams_per_pick=100),
    }
    basket = [("P1", member.member_id)]
    result = grade_basket(household, basket, products_by_id)
    assert result.terminal_reward == pytest.approx(TERMINAL_FULL_PASS)
    assert not result.any_cap_violation


def test_cap_violation_returns_minus_half():
    """Sugars wildly above cap -> -0.5."""
    household = materialise_household(task_id=2)
    diabetic = next(m for m in household if "diabetes" in m.conditions)
    products_by_id = {
        "P1": _toy_product("P1", "biscuits", nutrition_per_100g={"sugars_g": 999, "sodium_mg": 0, "fat_g": 0, "energy_kcal": 100, "protein_g": 5, "fiber_g": 1}, grams_per_pick=100),
    }
    basket = [("P1", diabetic.member_id)]
    result = grade_basket(household, basket, products_by_id)
    assert result.terminal_reward == pytest.approx(TERMINAL_HARD_VIOLATION)
    assert result.any_cap_violation


def test_partial_credit_when_floors_fail_but_caps_ok():
    """Caps OK, floors not met (basket too sparse) -> +0.3."""
    household = materialise_household(task_id=1)
    member = household[0]
    products_by_id = {
        "P1": _toy_product("P1", "biscuits", nutrition_per_100g={"sugars_g": 1, "sodium_mg": 1, "fat_g": 1, "energy_kcal": 50, "protein_g": 1, "fiber_g": 0.1}, grams_per_pick=10),
    }
    basket = [("P1", member.member_id)]
    result = grade_basket(household, basket, products_by_id)
    assert result.terminal_reward == pytest.approx(TERMINAL_PARTIAL_CREDIT)
    assert result.any_floor_failure
    assert not result.any_cap_violation


def test_per_item_incompatibility_with_diabetic_yields_partial_or_violation():
    """Diabetic member tagged with carbonated_drinks -> per-item incompatibility.

    Per plan §4.5: that alone (without cap violation) yields partial credit.
    """
    household = materialise_household(task_id=2)
    diabetic = next(m for m in household if "diabetes" in m.conditions)
    products_by_id = {
        "P1": _toy_product(
            "P1",
            "carbonated_drinks",
            nutrition_per_100g={"sugars_g": 1, "sodium_mg": 1, "fat_g": 1, "energy_kcal": 100, "protein_g": 1, "fiber_g": 0.1},
            grams_per_pick=200,
        ),
    }
    basket = [("P1", diabetic.member_id)]
    result = grade_basket(household, basket, products_by_id)
    assert result.any_per_item_failure
    # No cap violation here (sugar very low), so partial credit fires
    assert result.terminal_reward in (TERMINAL_PARTIAL_CREDIT, TERMINAL_HARD_VIOLATION)


def test_grade_member_isolates_per_member_intake():
    """Items tagged to other members do not contribute to a member's grade."""
    household = materialise_household(task_id=3)
    diabetic = next(m for m in household if "diabetes" in m.conditions)
    senior = next(m for m in household if "hypertension" in m.conditions)
    products_by_id = {
        "P1": _toy_product("P1", "biscuits", nutrition_per_100g={"sugars_g": 999, "sodium_mg": 0, "fat_g": 0, "energy_kcal": 100, "protein_g": 5, "fiber_g": 1}, grams_per_pick=100),
        "P2": _toy_product("P2", "flour", nutrition_per_100g={"sugars_g": 1, "sodium_mg": 50, "fat_g": 5, "energy_kcal": 500, "protein_g": 30, "fiber_g": 10}, grams_per_pick=100),
    }
    basket = [("P1", senior.member_id), ("P2", diabetic.member_id)]
    diab_grade = grade_member(diabetic, basket, products_by_id)
    sen_grade = grade_member(senior, basket, products_by_id)
    # The big-sugar item is on senior; diabetic is unaffected by it
    assert diab_grade.cumulative_intake.get("sugars_g", 0.0) < 50
    assert sen_grade.cumulative_intake.get("sugars_g", 0.0) > 100


def test_grader_result_serializes_to_dict():
    household = materialise_household(task_id=1)
    products_by_id = {
        "P1": _toy_product("P1", "flour", grams_per_pick=100),
    }
    basket = [("P1", household[0].member_id)]
    result = grade_basket(household, basket, products_by_id)
    payload = result.as_dict()
    assert "terminal_reward" in payload
    assert "members" in payload
    assert isinstance(payload["members"], list)
