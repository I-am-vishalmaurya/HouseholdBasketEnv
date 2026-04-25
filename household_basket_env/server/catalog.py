# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Catalog loader and deterministic augmenter.

Per the v3.1 plan, the env reuses module1's `food_label_auditor` 492-product catalog
verbatim. That catalog does NOT carry `meal_type`, `price_inr`, `protein_g`, or
`fiber_g`, all of which the basket env needs. We augment in-memory at server boot
using deterministic mappings so the agent sees a self-contained product record
without modifying the upstream JSON.

The augmentation is referentially transparent: same product_id always maps to the
same meal_type / price / protein / fiber, regardless of episode seed. This is
required for `seed_verifier.py` reproducibility.

Trade-off acknowledged honestly in README: protein / fiber values are
category-baseline estimates, not measured. They suffice for relative basket-level
comparisons, which is all the env needs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Catalog source path
# ---------------------------------------------------------------------------

# Default: module1 path, relative to this file. Can be overridden by env var
# HOUSEHOLD_BASKET_PRODUCTS_PATH (used in tests and Docker builds where module1
# is not co-located).
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_MODULE1_PATH = _THIS_DIR.parent.parent.parent / "module1" / "food_label_auditor" / "data" / "products.json"
_DEFAULT_LOCAL_PATH = _THIS_DIR.parent / "data" / "products.json"


def resolve_catalog_path() -> Path:
    override = os.environ.get("HOUSEHOLD_BASKET_PRODUCTS_PATH")
    if override:
        p = Path(override)
        if p.exists():
            return p
    if _DEFAULT_LOCAL_PATH.exists():
        return _DEFAULT_LOCAL_PATH
    if _DEFAULT_MODULE1_PATH.exists():
        return _DEFAULT_MODULE1_PATH
    raise FileNotFoundError(
        "Could not locate products.json. Set HOUSEHOLD_BASKET_PRODUCTS_PATH "
        f"or place the file at {_DEFAULT_LOCAL_PATH} or {_DEFAULT_MODULE1_PATH}."
    )


# ---------------------------------------------------------------------------
# Category -> meal_type mapping
# ---------------------------------------------------------------------------

# Plan §4.4 requires meal_type tags: staple, protein, vegetable, dairy, snack, beverage.
# Mapping derived from plain category semantics.
_CATEGORY_TO_MEAL_TYPE: dict[str, str] = {
    "instant_noodles": "staple",
    "biscuits": "snack",
    "dairy_milk": "dairy",
    "health_drinks": "beverage",
    "namkeen_snacks": "snack",
    "dairy_curd": "dairy",
    "ready_to_eat": "staple",
    "fruit_juice": "beverage",
    "spice_mix": "snack",  # used as flavour; not a meal-staple
    "chips": "snack",
    "dairy_cheese": "dairy",
    "breakfast_cereal": "staple",
    "dairy_butter": "dairy",
    "papad": "snack",
    "nut_butter": "protein",
    "popcorn": "snack",
    "carbonated_drinks": "beverage",
    "dairy_ghee": "dairy",
    "fruit_drinks": "beverage",
    "jams_spreads": "snack",
    "dairy_condensed": "dairy",
    "sauces": "snack",
    "dairy_yogurt": "dairy",
    "condiments": "snack",
    "cooking_oil": "staple",  # foundational, but not really a meal item
    "coffee": "beverage",
    "tea": "beverage",
    "flavoured_milk": "dairy",
    "probiotic_drinks": "beverage",
    "flour": "staple",
    "pulses": "protein",
    "oral_care": "snack",  # outlier; the lone toothpaste/etc gets a default; never selected
}

ALLOWED_MEAL_TYPES: tuple[str, ...] = ("staple", "protein", "vegetable", "dairy", "snack", "beverage")


# ---------------------------------------------------------------------------
# Category -> baseline price_inr (per pack), nutrition deltas
# ---------------------------------------------------------------------------

# Baseline pack prices in INR (rough Indian retail mid-2025). Final price is
# baseline ± deterministic per-product jitter (±25%). Used for R_budget and the
# hard budget check.
_CATEGORY_BASELINE_PRICE_INR: dict[str, float] = {
    "instant_noodles": 70.0,
    "biscuits": 35.0,
    "dairy_milk": 55.0,
    "health_drinks": 220.0,
    "namkeen_snacks": 45.0,
    "dairy_curd": 40.0,
    "ready_to_eat": 120.0,
    "fruit_juice": 110.0,
    "spice_mix": 60.0,
    "chips": 30.0,
    "dairy_cheese": 130.0,
    "breakfast_cereal": 250.0,
    "dairy_butter": 270.0,
    "papad": 50.0,
    "nut_butter": 320.0,
    "popcorn": 40.0,
    "carbonated_drinks": 40.0,
    "dairy_ghee": 600.0,
    "fruit_drinks": 90.0,
    "jams_spreads": 150.0,
    "dairy_condensed": 110.0,
    "sauces": 130.0,
    "dairy_yogurt": 75.0,
    "condiments": 80.0,
    "cooking_oil": 200.0,
    "coffee": 280.0,
    "tea": 240.0,
    "flavoured_milk": 35.0,
    "probiotic_drinks": 80.0,
    "flour": 70.0,
    "pulses": 130.0,
    "oral_care": 70.0,
}


# Per-100g protein and fiber estimates by category (rough Indian-context defaults).
# Used by R_terminal floor check (plan §4.5 step 3) since the upstream catalog
# does not record these. Honestly disclosed in README as approximations.
_CATEGORY_NUTRIENT_DEFAULTS: dict[str, dict[str, float]] = {
    "instant_noodles": {"protein_g": 9.0, "fiber_g": 2.5},
    "biscuits": {"protein_g": 6.0, "fiber_g": 2.0},
    "dairy_milk": {"protein_g": 3.2, "fiber_g": 0.0},
    "health_drinks": {"protein_g": 9.0, "fiber_g": 1.5},
    "namkeen_snacks": {"protein_g": 11.0, "fiber_g": 4.0},
    "dairy_curd": {"protein_g": 3.5, "fiber_g": 0.0},
    "ready_to_eat": {"protein_g": 8.0, "fiber_g": 3.0},
    "fruit_juice": {"protein_g": 0.5, "fiber_g": 0.5},
    "spice_mix": {"protein_g": 8.0, "fiber_g": 12.0},
    "chips": {"protein_g": 6.0, "fiber_g": 3.0},
    "dairy_cheese": {"protein_g": 22.0, "fiber_g": 0.0},
    "breakfast_cereal": {"protein_g": 8.0, "fiber_g": 6.0},
    "dairy_butter": {"protein_g": 0.9, "fiber_g": 0.0},
    "papad": {"protein_g": 18.0, "fiber_g": 4.0},
    "nut_butter": {"protein_g": 22.0, "fiber_g": 6.0},
    "popcorn": {"protein_g": 11.0, "fiber_g": 13.0},
    "carbonated_drinks": {"protein_g": 0.0, "fiber_g": 0.0},
    "dairy_ghee": {"protein_g": 0.3, "fiber_g": 0.0},
    "fruit_drinks": {"protein_g": 0.3, "fiber_g": 0.3},
    "jams_spreads": {"protein_g": 0.5, "fiber_g": 1.0},
    "dairy_condensed": {"protein_g": 7.0, "fiber_g": 0.0},
    "sauces": {"protein_g": 1.5, "fiber_g": 1.0},
    "dairy_yogurt": {"protein_g": 4.0, "fiber_g": 0.0},
    "condiments": {"protein_g": 5.0, "fiber_g": 4.0},
    "cooking_oil": {"protein_g": 0.0, "fiber_g": 0.0},
    "coffee": {"protein_g": 12.0, "fiber_g": 0.0},
    "tea": {"protein_g": 20.0, "fiber_g": 0.0},
    "flavoured_milk": {"protein_g": 3.0, "fiber_g": 0.0},
    "probiotic_drinks": {"protein_g": 1.5, "fiber_g": 0.0},
    "flour": {"protein_g": 11.0, "fiber_g": 4.0},
    "pulses": {"protein_g": 24.0, "fiber_g": 8.0},
    "oral_care": {"protein_g": 0.0, "fiber_g": 0.0},
}


# Indicative consumed grams per "pack pick" — maps a single basket pick to grams
# of food the tagged member consumes. Used to convert per-100g nutrition into
# absolute intake for cap/floor checks. Plan does not enumerate this; the choice
# is justified by typical per-pack serving sizes for Indian retail packs.
_CATEGORY_GRAMS_PER_PICK: dict[str, float] = {
    "instant_noodles": 70.0,
    "biscuits": 60.0,
    "dairy_milk": 200.0,
    "health_drinks": 25.0,        # mixed with milk, dry weight
    "namkeen_snacks": 50.0,
    "dairy_curd": 100.0,
    "ready_to_eat": 250.0,
    "fruit_juice": 200.0,
    "spice_mix": 10.0,
    "chips": 50.0,
    "dairy_cheese": 30.0,
    "breakfast_cereal": 40.0,
    "dairy_butter": 10.0,
    "papad": 15.0,
    "nut_butter": 25.0,
    "popcorn": 30.0,
    "carbonated_drinks": 250.0,
    "dairy_ghee": 10.0,
    "fruit_drinks": 200.0,
    "jams_spreads": 20.0,
    "dairy_condensed": 30.0,
    "sauces": 25.0,
    "dairy_yogurt": 100.0,
    "condiments": 15.0,
    "cooking_oil": 10.0,
    "coffee": 5.0,
    "tea": 3.0,
    "flavoured_milk": 200.0,
    "probiotic_drinks": 100.0,
    "flour": 60.0,
    "pulses": 50.0,
    "oral_care": 5.0,
}


def _stable_jitter(product_id: str, salt: str, low: float, high: float) -> float:
    """Deterministic jitter in [low, high) seeded by product_id + salt.

    Same (product_id, salt) always returns the same float. Used to give each
    product a stable price and stable per-product nutrient micro-variation.
    """
    digest = hashlib.sha256(f"{product_id}|{salt}".encode("utf-8")).digest()
    # Take first 4 bytes -> uint32 in [0, 2^32)
    bucket = int.from_bytes(digest[:4], "big")
    frac = bucket / 2**32
    return low + frac * (high - low)


def _augment_one(product: dict) -> dict:
    """Return a NEW dict with meal_type / price_inr / protein_g / fiber_g / grams_per_pick added.

    Does not mutate the input. Pure function. Same input -> same output.
    """
    category = product.get("category", "")
    pid = product["product_id"]

    meal_type = _CATEGORY_TO_MEAL_TYPE.get(category, "snack")

    base_price = _CATEGORY_BASELINE_PRICE_INR.get(category, 80.0)
    # ±25% deterministic jitter
    price = round(base_price * _stable_jitter(pid, "price", 0.75, 1.25), 2)

    cat_defaults = _CATEGORY_NUTRIENT_DEFAULTS.get(category, {"protein_g": 5.0, "fiber_g": 1.0})
    nutrition = dict(product.get("nutrition_per_100g") or {})
    if "protein_g" not in nutrition:
        nutrition["protein_g"] = round(
            cat_defaults["protein_g"] * _stable_jitter(pid, "protein", 0.85, 1.15), 2
        )
    if "fiber_g" not in nutrition:
        nutrition["fiber_g"] = round(
            cat_defaults["fiber_g"] * _stable_jitter(pid, "fiber", 0.85, 1.15), 2
        )

    grams = _CATEGORY_GRAMS_PER_PICK.get(category, 50.0)

    return {
        **product,
        "meal_type": meal_type,
        "price_inr": price,
        "nutrition_per_100g": nutrition,
        "grams_per_pick": grams,
    }


def load_augmented_catalog(path: Path | None = None) -> list[dict]:
    """Load module1's products.json and augment each product in-memory."""
    if path is None:
        path = resolve_catalog_path()
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [_augment_one(p) for p in raw]


def absolute_nutrients_for_pick(product: dict) -> dict[str, float]:
    """Convert per-100g nutrition into absolute consumption for one pack pick.

    Used by basket_grader / rewards to update cumulative_intake.
    """
    grams = float(product.get("grams_per_pick") or 50.0)
    factor = grams / 100.0
    nut = product.get("nutrition_per_100g") or {}
    out: dict[str, float] = {}
    for key in ("energy_kcal", "sugars_g", "sodium_mg", "fat_g", "protein_g", "fiber_g"):
        v = nut.get(key)
        if v is None:
            continue
        out[key] = round(float(v) * factor, 4)
    return out


def filter_catalog(
    catalog: Iterable[dict],
    *,
    include_adversarial: bool = True,
    only_categories: list[str] | None = None,
) -> list[dict]:
    out: list[dict] = []
    for p in catalog:
        if not include_adversarial and p.get("is_adversarial"):
            continue
        if only_categories is not None and p.get("category") not in only_categories:
            continue
        # Skip the lone oral_care outlier — never appropriate for a food basket.
        if p.get("category") == "oral_care":
            continue
        out.append(p)
    return out
