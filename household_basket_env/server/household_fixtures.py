# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Household fixtures for HouseholdBasketEnv.

Embeds per-member health profiles and ICMR-NIN-derived intake floors (plan
§4.5). All values are intentionally hard-coded constants, not loaded from
a JSON file, so the env image is fully self-contained and the same code works
in tests, locally, in Docker, and in HF Spaces with no extra data shipping.

ICMR-NIN references (per single basket episode, NOT a full day's RDA):
- Sedentary adult male, age 30-50:    energy 2110 kcal/day; protein 54 g/day; fiber 25 g/day
- Sedentary adult female, age 30-50:  energy 1660 kcal/day; protein 46 g/day; fiber 25 g/day
- Adolescent boy, age 13-15:          energy 2220 kcal/day; protein 45 g/day; fiber 25 g/day
- Sedentary senior male, age 65+:     energy 1800 kcal/day; protein 51 g/day; fiber 25 g/day

We approximate ONE BASKET as one full day's intake (the agent picks 3 / 5 / 7
items intended to *substantially* feed the household for a day). Caps /
floors below are scaled accordingly.

Caps (`thresholds_cap`) come from disease-specific guidance:
- Diabetes:        sugars  <= 25 g/day, sodium <= 1500 mg/day (ADA / ICMR)
- Hypertension:    sodium  <= 1500 mg/day (DASH), sugars <= 36 g/day, fat <= 65 g/day
- Healthy adult:   sugars  <= 50 g/day, sodium <= 2000 mg/day (WHO upper bounds)
- Healthy child:   sugars  <= 25 g/day, sodium <= 1200 mg/day, fat <= 50 g/day
- Healthy senior:  sugars  <= 50 g/day, sodium <= 2300 mg/day, fat <= 60 g/day

Floors (`thresholds_floor`) come from ICMR-NIN RDAs scaled to per-basket.
"""

from __future__ import annotations

from typing import Any

from ..models import HouseholdMember


# ---------------------------------------------------------------------------
# Member templates
# ---------------------------------------------------------------------------

# Each template renders to a HouseholdMember at reset time with a stable id.
# The dict shape lets curriculum.py compose households from templates.

ADULT_HEALTHY = {
    "id_suffix": "adult_healthy",
    "display_name": "Healthy adult (age 32)",
    "age": 32,
    "conditions": [],
    "dietary_restrictions": [],
    "thresholds_cap": {
        "sugars_g_max": 50.0,
        "sodium_mg_max": 2000.0,
        "fat_g_max": 70.0,
        "energy_kcal_max": 2200.0,
    },
    "thresholds_floor": {
        "energy_kcal_min": 1500.0,
        "protein_g_min": 45.0,
        "fiber_g_min": 20.0,
    },
    "watched_nutrients": ["sugars_g", "sodium_mg", "fat_g", "energy_kcal"],
}

ADULT_DIABETIC = {
    "id_suffix": "adult_diabetic",
    "display_name": "Diabetic adult (age 45)",
    "age": 45,
    "conditions": ["diabetes"],
    "dietary_restrictions": ["low_sugar"],
    "thresholds_cap": {
        "sugars_g_max": 25.0,        # ADA ≤25 g/day added sugar
        "sodium_mg_max": 1500.0,     # combined diabetes/CKD risk
        "fat_g_max": 65.0,
        "energy_kcal_max": 1900.0,
    },
    "thresholds_floor": {
        "energy_kcal_min": 1400.0,
        "protein_g_min": 50.0,       # ICMR-NIN +5g for diabetes management
        "fiber_g_min": 25.0,         # ADA recommends 25-30g/day for diabetics
    },
    "watched_nutrients": ["sugars_g", "sodium_mg", "fat_g", "energy_kcal"],
}

SENIOR_HYPERTENSIVE = {
    "id_suffix": "senior_hypertensive",
    "display_name": "Hypertensive senior (age 68)",
    "age": 68,
    "conditions": ["hypertension"],
    "dietary_restrictions": ["low_sodium"],
    "thresholds_cap": {
        "sugars_g_max": 36.0,
        "sodium_mg_max": 1500.0,     # DASH diet upper bound
        "fat_g_max": 60.0,
        "energy_kcal_max": 1800.0,
    },
    "thresholds_floor": {
        "energy_kcal_min": 1300.0,
        "protein_g_min": 51.0,       # ICMR-NIN sedentary senior male
        "fiber_g_min": 22.0,
    },
    "watched_nutrients": ["sodium_mg", "sugars_g", "fat_g", "energy_kcal"],
}

CHILD_GROWING = {
    "id_suffix": "child_growing",
    "display_name": "Growing child (age 9)",
    "age": 9,
    "conditions": [],
    "dietary_restrictions": [],
    "thresholds_cap": {
        "sugars_g_max": 25.0,
        "sodium_mg_max": 1200.0,     # AHA child upper bound
        "fat_g_max": 50.0,
        "energy_kcal_max": 1700.0,
    },
    "thresholds_floor": {
        "energy_kcal_min": 1300.0,
        "protein_g_min": 30.0,       # ICMR-NIN children 7-9
        "fiber_g_min": 18.0,
    },
    "watched_nutrients": ["sugars_g", "sodium_mg", "fat_g", "energy_kcal"],
}


def materialise_member(template: dict[str, Any], household_idx: int) -> HouseholdMember:
    """Convert a template + household index into a stable HouseholdMember."""
    return HouseholdMember(
        member_id=f"m{household_idx}_{template['id_suffix']}",
        display_name=template["display_name"],
        age=template["age"],
        conditions=list(template["conditions"]),
        dietary_restrictions=list(template["dietary_restrictions"]),
        thresholds_cap=dict(template["thresholds_cap"]),
        thresholds_floor=dict(template["thresholds_floor"]),
        watched_nutrients=list(template["watched_nutrients"]),
    )


# ---------------------------------------------------------------------------
# Tier -> household composition
# ---------------------------------------------------------------------------

# Plan §3:
#   Task 1: 1 healthy adult,                    3 valid steps, 20 curated, 500 INR
#   Task 2: 2 (healthy + diabetic),             5 valid steps, 50 catalog, 1000 INR
#   Task 3: 3 (diabetic + hypertensive + child), 7 valid steps, 492+adversarial, 1500 INR

TIER_MEMBER_TEMPLATES: dict[int, list[dict[str, Any]]] = {
    1: [ADULT_HEALTHY],
    2: [ADULT_HEALTHY, ADULT_DIABETIC],
    3: [ADULT_DIABETIC, SENIOR_HYPERTENSIVE, CHILD_GROWING],
}


def materialise_household(task_id: int, household_idx: int = 0) -> list[HouseholdMember]:
    templates = TIER_MEMBER_TEMPLATES.get(task_id) or TIER_MEMBER_TEMPLATES[1]
    return [materialise_member(t, household_idx) for t in templates]


# ---------------------------------------------------------------------------
# Member ↔ allergen / restriction filter
# ---------------------------------------------------------------------------

# Per-item sanity (plan §4.5 step 4): if a member's restrictions disqualify a
# product, that pick fails per-item sanity at terminal time. Conservative
# heuristic — categories the agent should avoid for a member with the listed
# condition.
INCOMPATIBLE_CATEGORIES_BY_CONDITION: dict[str, set[str]] = {
    "diabetes": {
        "carbonated_drinks",
        "fruit_drinks",
        "jams_spreads",
        "dairy_condensed",
        "flavoured_milk",
        "health_drinks",
    },
    "hypertension": {
        "namkeen_snacks",
        "papad",
        "sauces",
        "instant_noodles",
        "chips",
    },
}


def per_item_compatible(product: dict, member: HouseholdMember) -> bool:
    """True iff the (product, member) pair passes per-item sanity."""
    category = product.get("category", "")
    for cond in member.conditions:
        if category in INCOMPATIBLE_CATEGORIES_BY_CONDITION.get(cond, set()):
            return False
    return True
