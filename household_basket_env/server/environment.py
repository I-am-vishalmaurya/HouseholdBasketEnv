# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
HouseholdBasketEnv environment implementation (plan §2, §3, §4).

Implements the 5-check action-validation pipeline:
    1. JSON parse              -> P_parse, attempt_index++
    2. Pydantic validation     -> P_parse, attempt_index++
    3. Membership checks       -> unknown_product (P_parse) | unknown_member (P_unknown_member)
    4. Duplicate check         -> P_duplicate
    5. Budget check            -> P_over_budget => terminate

Two terminal paths (plan §4.7):
    - normal:      step_index == max_steps        -> grader runs, terminal in {-0.5, +0.3, +1.0}
    - attempt_cap: attempt_index >= max_steps * 2 -> grader SKIPPED, terminal_reward = 0.0
"""

from __future__ import annotations

import json
import random
from typing import Any, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    class Environment:
        """Stub when openenv-core is not installed (notebook / standalone usage)."""
        pass

from pydantic import ValidationError

from ..models import (
    BasketAction,
    BasketObservation,
    BasketState,
    HouseholdMember,
    MemberSummary,
    ProductSummary,
    TaggedItem,
)
from .basket_grader import grade_basket
from .catalog import absolute_nutrients_for_pick, load_augmented_catalog
from .curriculum import (
    TIER_CONFIGS,
    TierConfig,
    is_seed_verified,
    jittered_budget,
    select_candidates_for_tier,
)
from .household_fixtures import materialise_household
from .rewards import (
    P_DUPLICATE,
    P_OVER_BUDGET,
    P_UNKNOWN_MEMBER,
    R_FORMAT_PARSE,
    StepRewardBreakdown,
    compose_valid_step_reward,
)


def _build_prompt(household: list[HouseholdMember], budget: float, max_steps: int) -> str:
    member_lines = []
    for m in household:
        cond = ", ".join(m.conditions) if m.conditions else "no conditions"
        caps = ", ".join(f"{k}={v}" for k, v in m.thresholds_cap.items())
        floors = ", ".join(f"{k}={v}" for k, v in m.thresholds_floor.items())
        member_lines.append(
            f"  - {m.member_id} | {m.display_name} | conditions: {cond} | caps: {caps} | floors: {floors}"
        )
    members_str = "\n".join(member_lines)

    return (
        "You are an Indian household grocery agent. You must compose a basket of "
        f"{max_steps} packaged-food items, drawn from the candidate list provided each step, "
        "subject to the following constraints:\n\n"
        "Household:\n"
        f"{members_str}\n\n"
        f"Budget: {budget:.2f} INR (hard cap; spending more terminates the episode).\n\n"
        "Each step you must emit ONE JSON object with two required fields and one optional "
        'field: {"product_id": "...", "member_id": "...", "rationale": "..."}. '
        "The product_id must come from the candidates list. The member_id must come from the "
        "household list above. The rationale field is optional and is logged but not graded.\n\n"
        "You earn dense reward for valid actions, items priced at-or-below the per-step "
        "allowance, picks that move each member's running intake into the healthy band "
        "(target ~60% of cap), and adding meal-type variety per member. At terminal you "
        "earn +1.0 if every member is within caps and meets minimum intake floors, +0.3 for "
        "partial credit, and -0.5 if any member exceeds a cap."
    )


class HouseholdBasketEnvironment(Environment):
    """Multi-stakeholder constraint-satisfaction grocery basket env."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        augmented_catalog: Optional[list[dict]] = None,
        *,
        enable_meal_type_coverage: bool = True,
    ) -> None:
        super().__init__()
        self._catalog: list[dict] = augmented_catalog if augmented_catalog is not None else load_augmented_catalog()
        self._catalog_by_id: dict[str, dict] = {p["product_id"]: p for p in self._catalog}

        self._tier: TierConfig = TIER_CONFIGS[2]
        self._household: list[HouseholdMember] = []
        self._candidates: list[dict] = []
        self._candidates_by_id: dict[str, dict] = {}
        self._budget_total: float = 0.0
        self._budget_remaining: float = 0.0
        self._basket: list[tuple[str, str]] = []
        self._tagged_items: list[TaggedItem] = []
        self._cumulative_intake_by_member: dict[str, dict[str, float]] = {}
        self._items_tagged_by_member: dict[str, list[str]] = {}
        self._meal_type_coverage: dict[str, set[str]] = {}
        self._enable_meal_type_coverage: bool = bool(enable_meal_type_coverage)
        self._prompt: str = ""
        self._rng: random.Random = random.Random(0)

        self._state = BasketState(
            episode_id=str(uuid4()),
            step_count=0,
            seed=0,
            task_id=2,
            cumulative_spend=0.0,
            attempt_index=0,
            valid_step_index=0,
            verified=False,
            terminated_reason=None,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: int = 2,
        enable_meal_type_coverage: Optional[bool] = None,
        **kwargs: Any,
    ) -> BasketObservation:
        effective_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        rng = random.Random(effective_seed)

        tier = TIER_CONFIGS.get(task_id, TIER_CONFIGS[2])
        self._tier = tier
        self._rng = rng
        if enable_meal_type_coverage is not None:
            self._enable_meal_type_coverage = bool(enable_meal_type_coverage)

        self._household = materialise_household(tier.task_id, household_idx=0)
        self._candidates = select_candidates_for_tier(self._catalog, tier, rng)
        self._candidates_by_id = {p["product_id"]: p for p in self._candidates}

        self._budget_total = jittered_budget(tier.base_budget_inr, rng)
        self._budget_remaining = self._budget_total

        self._basket = []
        self._tagged_items = []
        self._cumulative_intake_by_member = {m.member_id: {} for m in self._household}
        self._items_tagged_by_member = {m.member_id: [] for m in self._household}
        self._meal_type_coverage = {m.member_id: set() for m in self._household}

        self._prompt = _build_prompt(self._household, self._budget_total, tier.max_steps)

        verified = is_seed_verified(tier.task_id, effective_seed)

        self._state = BasketState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            seed=effective_seed,
            task_id=tier.task_id,
            cumulative_spend=0.0,
            attempt_index=0,
            valid_step_index=0,
            verified=verified,
            terminated_reason=None,
        )

        return self._build_observation(parse_error=None, breakdown=None)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        action: BasketAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> BasketObservation:
        # The OpenEnv HTTP server passes a Pydantic-validated BasketAction here,
        # so checks 1+2 (JSON parse + Pydantic) effectively already passed by the
        # time we are called. We still expose `apply_raw_action` for tests that
        # want to assert P_parse fires on bad strings.
        return self.apply_raw_action(action)

    def apply_raw_action(self, raw: Any) -> BasketObservation:
        """Apply an action that may be a BasketAction, dict, or raw string.

        Tests use this entry point to drive the 5-check pipeline end-to-end
        (the FastAPI layer handles checks 1+2 automatically; here we let
        non-BasketAction inputs trigger them too).
        """
        # Already-terminated guard.
        if self._state.terminated_reason is not None:
            return self._build_observation(parse_error="already_terminated", breakdown=None)

        action: Optional[BasketAction] = None
        parse_error: Optional[str] = None

        # Check 1: JSON parse (only relevant for strings)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, ValueError) as e:
                parse_error = f"invalid_json: {e}"

        # Check 2: Pydantic validation (only relevant for dicts)
        if parse_error is None:
            if isinstance(raw, BasketAction):
                action = raw
            elif isinstance(raw, dict):
                try:
                    action = BasketAction(**raw)
                except ValidationError as e:
                    parse_error = f"schema_mismatch: {e.errors()[0].get('msg', 'invalid')}"
            else:
                parse_error = "invalid_payload_type"

        if parse_error is not None or action is None:
            bd = StepRewardBreakdown()
            bd.p_parse = R_FORMAT_PARSE
            self._state.attempt_index += 1
            return self._maybe_terminate_attempt_cap(parse_error=parse_error or "parse_error", breakdown=bd)

        # Check 3a: unknown product
        if action.product_id not in self._candidates_by_id:
            bd = StepRewardBreakdown()
            bd.p_parse = R_FORMAT_PARSE
            self._state.attempt_index += 1
            return self._maybe_terminate_attempt_cap(
                parse_error=f"unknown_product: {action.product_id}", breakdown=bd
            )

        # Check 3b: unknown member  (uses P_unknown_member, not P_parse)
        member_ids = {m.member_id for m in self._household}
        if action.member_id not in member_ids:
            bd = StepRewardBreakdown()
            bd.p_unknown_member = P_UNKNOWN_MEMBER
            self._state.attempt_index += 1
            return self._maybe_terminate_attempt_cap(
                parse_error=f"unknown_member: {action.member_id}", breakdown=bd
            )

        # Check 4: duplicate
        already_in_basket = any(action.product_id == pid for pid, _ in self._basket)
        if already_in_basket:
            bd = StepRewardBreakdown()
            bd.p_duplicate = P_DUPLICATE
            self._state.attempt_index += 1
            return self._maybe_terminate_attempt_cap(
                parse_error=f"duplicate_product: {action.product_id}", breakdown=bd
            )

        product = self._candidates_by_id[action.product_id]
        price = float(product.get("price_inr", 0.0))

        # Check 5: budget
        if self._state.cumulative_spend + price > self._budget_total + 1e-6:
            bd = StepRewardBreakdown()
            bd.p_over_budget = P_OVER_BUDGET
            self._state.attempt_index += 1
            self._state.terminated_reason = "over_budget"
            return self._build_terminal_observation(
                parse_error=f"over_budget: spend={self._state.cumulative_spend + price:.2f}>budget={self._budget_total:.2f}",
                breakdown=bd,
                terminal_reward=0.0,
                run_grader=False,
            )

        # ------------------------------------------------------------------
        # Valid step. Compute dense reward, advance state.
        # ------------------------------------------------------------------
        member = next(m for m in self._household if m.member_id == action.member_id)
        absolute = absolute_nutrients_for_pick(product)
        pre_pick_intake = dict(self._cumulative_intake_by_member.get(member.member_id, {}))
        steps_remaining_before = self._tier.max_steps - self._state.valid_step_index

        bd = compose_valid_step_reward(
            pre_pick_intake=pre_pick_intake,
            pick_absolute_nutrients=absolute,
            member_caps=member.thresholds_cap,
            watched_nutrients=member.watched_nutrients,
            pick_price=price,
            budget_remaining_before=self._budget_remaining,
            steps_remaining_before=steps_remaining_before,
            member_id=member.member_id,
            pick_meal_type=product.get("meal_type", "snack"),
            coverage_so_far=self._meal_type_coverage,
            enable_meal_type_coverage=self._enable_meal_type_coverage,
        )

        # Mutate state.
        self._basket.append((action.product_id, action.member_id))
        self._tagged_items.append(
            TaggedItem(
                product_id=action.product_id,
                member_id=action.member_id,
                price_inr=price,
                meal_type=product.get("meal_type", "snack"),
            )
        )
        new_intake = dict(pre_pick_intake)
        for k, v in absolute.items():
            new_intake[k] = round(new_intake.get(k, 0.0) + v, 4)
        self._cumulative_intake_by_member[member.member_id] = new_intake
        self._items_tagged_by_member.setdefault(member.member_id, []).append(action.product_id)

        self._state.cumulative_spend = round(self._state.cumulative_spend + price, 2)
        self._budget_remaining = round(self._budget_total - self._state.cumulative_spend, 2)
        self._state.valid_step_index += 1
        self._state.step_count += 1
        self._state.attempt_index += 1

        # Normal terminal?
        if self._state.valid_step_index >= self._tier.max_steps:
            grader_result = grade_basket(self._household, self._basket, self._candidates_by_id)
            bd.terminal = grader_result.terminal_reward
            self._state.terminated_reason = "max_steps"
            return self._build_terminal_observation(
                parse_error=None,
                breakdown=bd,
                terminal_reward=grader_result.terminal_reward,
                run_grader=True,
                grader_dict=grader_result.as_dict(),
            )

        return self._build_observation(parse_error=None, breakdown=bd)

    # ------------------------------------------------------------------
    # State property (required by Environment ABC)
    # ------------------------------------------------------------------

    @property
    def state(self) -> BasketState:
        return self._state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _maybe_terminate_attempt_cap(
        self, *, parse_error: str, breakdown: StepRewardBreakdown
    ) -> BasketObservation:
        """If attempt_index >= max_steps * 2, terminate with grader skipped.

        Plan §4.7: dense penalties already accumulated are the cost; grader is
        not invoked because the basket is artificially short.
        """
        if self._state.attempt_index >= self._tier.max_steps * 2:
            self._state.terminated_reason = "attempt_cap"
            return self._build_terminal_observation(
                parse_error=parse_error,
                breakdown=breakdown,
                terminal_reward=0.0,
                run_grader=False,
            )
        return self._build_observation(parse_error=parse_error, breakdown=breakdown)

    def _build_observation(
        self,
        *,
        parse_error: Optional[str],
        breakdown: Optional[StepRewardBreakdown],
    ) -> BasketObservation:
        return BasketObservation(
            prompt=self._prompt,
            household=self._build_member_summaries(),
            basket_so_far=list(self._tagged_items),
            budget_remaining=self._budget_remaining,
            candidates=self._build_candidate_summaries(),
            step_index=self._state.valid_step_index,
            attempt_index=self._state.attempt_index,
            max_steps=self._tier.max_steps,
            seed=self._state.seed,
            parse_error=parse_error,
            terminated_reason=None,
            terminal_reward=None,
            reward=breakdown.total if breakdown is not None else 0.0,
            reward_breakdown=breakdown.as_dict() if breakdown is not None else None,
            done=False,
        )

    def _build_terminal_observation(
        self,
        *,
        parse_error: Optional[str],
        breakdown: StepRewardBreakdown,
        terminal_reward: float,
        run_grader: bool,
        grader_dict: Optional[dict] = None,
    ) -> BasketObservation:
        meta = {"grader": grader_dict} if grader_dict is not None else None
        return BasketObservation(
            prompt=self._prompt,
            household=self._build_member_summaries(),
            basket_so_far=list(self._tagged_items),
            budget_remaining=self._budget_remaining,
            candidates=self._build_candidate_summaries(),
            step_index=self._state.valid_step_index,
            attempt_index=self._state.attempt_index,
            max_steps=self._tier.max_steps,
            seed=self._state.seed,
            parse_error=parse_error,
            terminated_reason=self._state.terminated_reason,
            terminal_reward=terminal_reward,
            reward=breakdown.total,
            reward_breakdown=breakdown.as_dict(),
            done=True,
            metadata=meta or {},
        )

    def _build_member_summaries(self) -> list[MemberSummary]:
        return [
            MemberSummary(
                member_id=m.member_id,
                display_name=m.display_name,
                conditions=list(m.conditions),
                thresholds_cap=dict(m.thresholds_cap),
                thresholds_floor=dict(m.thresholds_floor),
                cumulative_intake=dict(self._cumulative_intake_by_member.get(m.member_id, {})),
                items_tagged=list(self._items_tagged_by_member.get(m.member_id, [])),
            )
            for m in self._household
        ]

    def _build_candidate_summaries(self) -> list[ProductSummary]:
        return [
            ProductSummary(
                product_id=p["product_id"],
                product_name=p.get("product_name", ""),
                brand=p.get("brand", ""),
                category=p.get("category", ""),
                meal_type=p.get("meal_type", "snack"),
                price_inr=float(p.get("price_inr", 0.0)),
                nutrition_per_100g=dict(p.get("nutrition_per_100g") or {}),
                marketing_claims=list(p.get("marketing_claims") or []),
                nutri_score=p.get("nutri_score", ""),
                nova_group=int(p.get("nova_group", 0) or 0),
                is_adversarial=bool(p.get("is_adversarial", False)),
            )
            for p in self._candidates
        ]
