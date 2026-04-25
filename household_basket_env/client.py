# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HouseholdBasketEnv client — async-first, with `.sync()` wrapper available."""

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import (
    BasketAction,
    BasketObservation,
    BasketState,
    MemberSummary,
    ProductSummary,
    TaggedItem,
)


class HouseholdBasketEnv(EnvClient[BasketAction, BasketObservation, BasketState]):
    """Client for HouseholdBasketEnv.

    Example (async, recommended):
        >>> client = await HouseholdBasketEnv.from_docker_image(
        ...     "household-basket-env:latest"
        ... )
        >>> async with client:
        ...     result = await client.reset(seed=42, task_id=2)
        ...     for cand in result.observation.candidates[:3]:
        ...         print(cand.product_id, cand.price_inr, cand.meal_type)
        ...     result = await client.step(BasketAction(
        ...         product_id=result.observation.candidates[0].product_id,
        ...         member_id=result.observation.household[0].member_id,
        ...         rationale="cheapest staple",
        ...     ))

    Example (sync):
        >>> with HouseholdBasketEnv(base_url="http://localhost:8000").sync() as client:
        ...     result = client.reset(seed=42, task_id=2)
        ...     result = client.step(BasketAction(
        ...         product_id=result.observation.candidates[0].product_id,
        ...         member_id=result.observation.household[0].member_id,
        ...     ))
    """

    def _step_payload(self, action: BasketAction) -> dict[str, Any]:
        return {
            "product_id": action.product_id,
            "member_id": action.member_id,
            "rationale": action.rationale,
        }

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[BasketObservation]:
        obs_data = payload.get("observation", {}) or {}

        def _members(raw: list[dict] | None) -> list[MemberSummary]:
            return [MemberSummary(**m) for m in (raw or [])]

        def _candidates(raw: list[dict] | None) -> list[ProductSummary]:
            return [ProductSummary(**p) for p in (raw or [])]

        def _basket(raw: list[dict] | None) -> list[TaggedItem]:
            return [TaggedItem(**t) for t in (raw or [])]

        observation = BasketObservation(
            prompt=obs_data.get("prompt", ""),
            household=_members(obs_data.get("household")),
            basket_so_far=_basket(obs_data.get("basket_so_far")),
            budget_remaining=float(obs_data.get("budget_remaining", 0.0) or 0.0),
            candidates=_candidates(obs_data.get("candidates")),
            step_index=int(obs_data.get("step_index", 0) or 0),
            attempt_index=int(obs_data.get("attempt_index", 0) or 0),
            max_steps=int(obs_data.get("max_steps", 0) or 0),
            seed=int(obs_data.get("seed", 0) or 0),
            parse_error=obs_data.get("parse_error"),
            terminated_reason=obs_data.get("terminated_reason"),
            terminal_reward=obs_data.get("terminal_reward"),
            reward=payload.get("reward"),
            reward_breakdown=obs_data.get("reward_breakdown"),
            done=bool(payload.get("done", False)),
            metadata=obs_data.get("metadata", {}) or {},
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict[str, Any]) -> BasketState:
        return BasketState(
            episode_id=payload.get("episode_id"),
            step_count=int(payload.get("step_count", 0) or 0),
            seed=int(payload.get("seed", 0) or 0),
            task_id=int(payload.get("task_id", 2) or 2),
            cumulative_spend=float(payload.get("cumulative_spend", 0.0) or 0.0),
            attempt_index=int(payload.get("attempt_index", 0) or 0),
            valid_step_index=int(payload.get("valid_step_index", 0) or 0),
            verified=bool(payload.get("verified", False)),
            terminated_reason=payload.get("terminated_reason"),
        )
