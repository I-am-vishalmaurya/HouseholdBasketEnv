# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pydantic data models for HouseholdBasketEnv.

Schema bindings (must agree with `openenv.yaml`):
    BasketAction       -> action class
    BasketObservation  -> observation class
    BasketState        -> internal state, never exposed to the agent
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation, State


class HouseholdMember(BaseModel):
    """A household member with health profile and per-nutrient caps + floors."""

    member_id: str = Field(..., description="Unique member id within the household, e.g. 'm_grandfather'")
    display_name: str = Field(..., description="Human-readable label, e.g. 'Grandfather (diabetic)'")
    age: int = Field(..., ge=1, le=120, description="Age in years")
    conditions: list[str] = Field(
        default_factory=list,
        description="Health conditions e.g. ['diabetes', 'hypertension']",
    )
    dietary_restrictions: list[str] = Field(
        default_factory=list,
        description="Dietary restrictions e.g. ['low_sugar', 'low_sodium', 'no_egg']",
    )
    thresholds_cap: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-nutrient hard caps for the *episode* tagged subset. Keys: "
            "sugars_g_max, sodium_mg_max, fat_g_max, energy_kcal_max."
        ),
    )
    thresholds_floor: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-nutrient minimum-intake floors (ICMR-NIN derived). Keys: "
            "energy_kcal_min, protein_g_min, fiber_g_min."
        ),
    )
    watched_nutrients: list[str] = Field(
        default_factory=lambda: ["sugars_g", "sodium_mg", "fat_g", "energy_kcal"],
        description="Subset of nutrients used for R_threshold dense reward.",
    )


class MemberSummary(BaseModel):
    """Per-member view exposed in BasketObservation.

    Distinct from HouseholdMember: this carries *running* state (cumulative_intake,
    items_tagged) so the agent can reason about margin to cap explicitly.
    """

    member_id: str = Field(..., description="Member id")
    display_name: str = Field(..., description="Human-readable label")
    conditions: list[str] = Field(default_factory=list)
    thresholds_cap: dict[str, float] = Field(default_factory=dict)
    thresholds_floor: dict[str, float] = Field(default_factory=dict)
    cumulative_intake: dict[str, float] = Field(
        default_factory=dict,
        description="Running per-nutrient sum across items tagged to this member.",
    )
    items_tagged: list[str] = Field(
        default_factory=list,
        description="Product ids tagged to this member so far.",
    )


class ProductSummary(BaseModel):
    """A product as the agent sees it in the candidate list."""

    product_id: str
    product_name: str
    brand: str = ""
    category: str
    meal_type: str = Field(..., description="One of: staple, protein, vegetable, dairy, snack, beverage")
    price_inr: float = Field(..., ge=0.0, description="Unit price in INR (deterministically derived from product_id)")
    nutrition_per_100g: dict[str, float] = Field(default_factory=dict)
    marketing_claims: list[str] = Field(default_factory=list)
    nutri_score: str = ""
    nova_group: int = 0
    is_adversarial: bool = False


class TaggedItem(BaseModel):
    """An item already added to the basket with its member tag."""

    product_id: str
    member_id: str
    price_inr: float
    meal_type: str


class BasketAction(Action):
    """Agent emits exactly one of these per step.

    Validation pipeline (server-side, in order, per plan §2.1):
      1. JSON parse           -> P_parse on failure
      2. Pydantic validation  -> P_parse on failure
      3. Membership checks    -> unknown_product=P_parse, unknown_member=P_unknown_member
      4. Duplicate check      -> P_duplicate
      5. Budget check         -> P_over_budget (terminates)
    """

    product_id: str = Field(
        ...,
        description=(
            "ID of the product to add to the basket. Must be present in the "
            "current observation's `candidates` list."
        ),
    )
    member_id: str = Field(
        ...,
        description=(
            "ID of the household member this item is tagged to. Must be present "
            "in the current observation's `household` list."
        ),
    )
    rationale: Optional[str] = Field(
        None,
        description=(
            "Optional one-line natural-language justification. Logged for "
            "inspection, never used in reward."
        ),
    )


class BasketObservation(Observation):
    """What the agent sees after each step (or after reset)."""

    prompt: str = Field(..., description="Task instructions + current situation in natural language.")
    household: list[MemberSummary] = Field(default_factory=list)
    basket_so_far: list[TaggedItem] = Field(default_factory=list)
    budget_remaining: float = Field(..., ge=0.0, description="INR budget left for the rest of the episode.")
    candidates: list[ProductSummary] = Field(default_factory=list)
    step_index: int = Field(default=0, description="Counts only valid steps that passed all 5 checks.")
    attempt_index: int = Field(
        default=0, description="Counts every step including parse / membership / duplicate failures."
    )
    max_steps: int = Field(default=3, description="Tier-dependent: 3 / 5 / 7.")
    seed: int = Field(default=0, description="Echoed for reproducibility debugging.")
    parse_error: Optional[str] = Field(default=None, description="Populated on invalid action.")
    terminated_reason: Optional[str] = Field(
        default=None,
        description=(
            "On done: 'max_steps' for normal terminal, 'attempt_cap' for "
            "attempt-cap exit, 'over_budget' for hard budget violation."
        ),
    )
    terminal_reward: Optional[float] = Field(
        default=None,
        description=(
            "Populated on done. Normal terminal: -0.5/+0.3/+1.0 from basket_grader. "
            "attempt_cap or over_budget: 0.0 (grader skipped)."
        ),
    )
    reward_breakdown: Optional[dict[str, float]] = Field(
        default=None,
        description="Per-component dense reward for the most recent step (debug / training logging).",
    )


class BasketState(State):
    """Internal env state. NEVER exposed to the agent."""

    seed: int = Field(default=0)
    task_id: int = Field(default=2, description="1, 2, or 3 (tier)")
    cumulative_spend: float = Field(default=0.0)
    attempt_index: int = Field(default=0)
    valid_step_index: int = Field(default=0)
    verified: bool = Field(
        default=False,
        description="True iff the seed is in the verified-seed list for this tier.",
    )
    terminated_reason: Optional[str] = Field(default=None)
