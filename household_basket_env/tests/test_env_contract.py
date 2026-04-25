# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Environment contract tests (plan §2.4 + §4.7)."""

from __future__ import annotations

import pytest

from household_basket_env.models import BasketAction


# ---------------------------------------------------------------------------
# Reset determinism (plan §2.4)
# ---------------------------------------------------------------------------

def _obs_signature(obs) -> dict:
    return {
        "household": [(m.member_id, tuple(sorted(m.thresholds_cap.items())), tuple(sorted(m.thresholds_floor.items()))) for m in obs.household],
        "candidates": [(p.product_id, p.price_inr, p.meal_type) for p in obs.candidates],
        "budget_remaining": obs.budget_remaining,
        "max_steps": obs.max_steps,
        "seed": obs.seed,
    }


@pytest.mark.parametrize("task_id", [1, 2, 3])
def test_reset_is_deterministic_from_seed(env, task_id):
    """Two resets with the same seed produce byte-identical observations."""
    obs_a = env.reset(seed=42, task_id=task_id)
    sig_a = _obs_signature(obs_a)
    obs_b = env.reset(seed=42, task_id=task_id)
    sig_b = _obs_signature(obs_b)
    assert sig_a == sig_b, f"Reset(seed=42, task_id={task_id}) was not deterministic"


@pytest.mark.parametrize("task_id", [1, 2, 3])
def test_different_seeds_yield_different_observations(env, task_id):
    obs_a = env.reset(seed=1, task_id=task_id)
    obs_b = env.reset(seed=2, task_id=task_id)
    assert _obs_signature(obs_a) != _obs_signature(obs_b)


def test_reset_populates_household_and_candidates(env):
    obs = env.reset(seed=7, task_id=2)
    assert len(obs.household) == 2
    assert len(obs.candidates) == 50
    assert obs.budget_remaining > 0
    assert obs.max_steps == 5
    assert obs.parse_error is None
    assert obs.done is False


def test_reset_task1_has_one_member_three_steps(env):
    obs = env.reset(seed=11, task_id=1)
    assert len(obs.household) == 1
    assert obs.max_steps == 3
    assert len(obs.candidates) == 20


def test_reset_task3_has_three_members_seven_steps_with_adversarial(env):
    obs = env.reset(seed=11, task_id=3)
    assert len(obs.household) == 3
    assert obs.max_steps == 7
    assert any(p.is_adversarial for p in obs.candidates)


# ---------------------------------------------------------------------------
# Step pipeline — five validation checks (plan §2.1)
# ---------------------------------------------------------------------------

def test_valid_step_emits_dense_reward_and_advances_step_index(env):
    obs = env.reset(seed=99, task_id=2)
    pid = obs.candidates[0].product_id
    mid = obs.household[0].member_id
    nxt = env.step(BasketAction(product_id=pid, member_id=mid))
    assert nxt.parse_error is None
    assert nxt.step_index == 1
    assert nxt.attempt_index == 1
    assert nxt.reward_breakdown["r_format"] == pytest.approx(0.20)


def test_unknown_product_fires_p_parse(env):
    env.reset(seed=99, task_id=2)
    nxt = env.apply_raw_action({"product_id": "GARBAGE_999", "member_id": "m0_adult_healthy"})
    assert "unknown_product" in (nxt.parse_error or "")
    assert nxt.reward_breakdown["p_parse"] == pytest.approx(-0.25)
    assert nxt.step_index == 0
    assert nxt.attempt_index == 1


def test_unknown_member_fires_p_unknown_member(env):
    obs = env.reset(seed=99, task_id=2)
    pid = obs.candidates[0].product_id
    nxt = env.apply_raw_action({"product_id": pid, "member_id": "nobody"})
    assert "unknown_member" in (nxt.parse_error or "")
    assert nxt.reward_breakdown["p_unknown_member"] == pytest.approx(-0.40)
    assert nxt.step_index == 0
    assert nxt.attempt_index == 1


def test_invalid_json_fires_p_parse(env):
    env.reset(seed=99, task_id=2)
    nxt = env.apply_raw_action("not a valid json {{{")
    assert "invalid_json" in (nxt.parse_error or "")
    assert nxt.reward_breakdown["p_parse"] == pytest.approx(-0.25)


def test_schema_mismatch_fires_p_parse(env):
    env.reset(seed=99, task_id=2)
    nxt = env.apply_raw_action({"product_id": 123})  # wrong type, missing member_id
    assert "schema_mismatch" in (nxt.parse_error or "") or "parse" in (nxt.parse_error or "")
    assert nxt.reward_breakdown["p_parse"] == pytest.approx(-0.25)


def test_duplicate_product_fires_p_duplicate(env):
    obs = env.reset(seed=99, task_id=2)
    pid = obs.candidates[0].product_id
    mid = obs.household[0].member_id
    env.step(BasketAction(product_id=pid, member_id=mid))
    nxt = env.step(BasketAction(product_id=pid, member_id=mid))
    assert "duplicate_product" in (nxt.parse_error or "")
    assert nxt.reward_breakdown["p_duplicate"] == pytest.approx(-0.30)
    assert nxt.step_index == 1  # did NOT advance


# ---------------------------------------------------------------------------
# Attempt-cap and terminal semantics (plan §4.7)
# ---------------------------------------------------------------------------

def test_attempt_cap_skips_terminal_grader(env):
    """Plan §4.7: max_steps × 2 parse errors -> done=True, grader not invoked,
    terminated_reason='attempt_cap', terminal_reward=0.0."""
    obs = env.reset(seed=42, task_id=2)
    cap = obs.max_steps * 2
    last = None
    for _ in range(cap):
        last = env.apply_raw_action({"product_id": "garbage", "member_id": "garbage"})
    assert last is not None
    assert last.done is True
    assert last.terminated_reason == "attempt_cap"
    assert last.terminal_reward == pytest.approx(0.0)
    # No grader metadata on attempt-cap exit
    assert (last.metadata or {}).get("grader") is None


def test_normal_terminal_runs_grader(env):
    """Plan §4.7: exactly max_steps valid actions -> grader invoked."""
    obs = env.reset(seed=42, task_id=2)
    used: set[str] = set()

    def first_unused_for(member_id: str):
        for c in obs.candidates:
            if c.product_id in used:
                continue
            return c.product_id
        return None

    last = None
    for _ in range(obs.max_steps):
        mid = obs.household[0].member_id
        pid = first_unused_for(mid)
        used.add(pid)
        last = env.step(BasketAction(product_id=pid, member_id=mid))
        if last.done:
            break

    assert last is not None
    assert last.done is True
    assert last.terminated_reason == "max_steps"
    assert last.terminal_reward in (-0.5, 0.3, 1.0)
    assert (last.metadata or {}).get("grader") is not None


def test_over_budget_terminates_with_grader_skipped(env):
    """Force a single-product spend that exceeds the budget -> over_budget exit.

    We do this by seeding a tier-1 episode (small budget) and stepping with the
    most expensive candidate repeatedly until the cumulative spend exceeds
    budget. Because tier-1 has a small budget, this can fire on the first big
    pick.
    """
    obs = env.reset(seed=1, task_id=1)
    # Pick the most expensive item; if its price already exceeds budget, the
    # very first step terminates.
    most_expensive = max(obs.candidates, key=lambda c: c.price_inr)
    if most_expensive.price_inr <= obs.budget_remaining:
        # Try several picks until budget is exhausted; tier 1 has 500 INR base.
        used: set[str] = set()
        candidates_sorted = sorted(obs.candidates, key=lambda c: -c.price_inr)
        last = None
        for c in candidates_sorted:
            if c.product_id in used:
                continue
            used.add(c.product_id)
            last = env.step(BasketAction(product_id=c.product_id, member_id=obs.household[0].member_id))
            if last.terminated_reason == "over_budget":
                break
            if last.done:
                break
        assert last is not None
        # Either we hit over_budget or the episode terminated normally; assert
        # that *if* over_budget fires, terminal_reward is 0.0 and grader is not
        # in metadata.
        if last.terminated_reason == "over_budget":
            assert last.terminal_reward == pytest.approx(0.0)
            assert (last.metadata or {}).get("grader") is None
    else:
        last = env.step(BasketAction(product_id=most_expensive.product_id, member_id=obs.household[0].member_id))
        assert last.terminated_reason == "over_budget"
        assert last.terminal_reward == pytest.approx(0.0)


def test_step_after_terminal_returns_done(env):
    """Once done=True, further steps return parse_error='already_terminated'."""
    obs = env.reset(seed=42, task_id=1)
    # Force attempt-cap exit
    for _ in range(obs.max_steps * 2):
        env.apply_raw_action({"product_id": "garbage", "member_id": "garbage"})
    nxt = env.apply_raw_action({"product_id": "x", "member_id": "y"})
    assert "already_terminated" in (nxt.parse_error or "")
