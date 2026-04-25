# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HouseholdBasketEnv: multi-stakeholder constraint-satisfaction OpenEnv environment.

A single agent must compose a basket of Indian packaged-food products that
simultaneously satisfies the conflicting nutritional constraints of multiple
household members under a hard INR budget. Theme #3.2 (Personalized Tasks).
"""

from .models import (
    BasketAction,
    BasketObservation,
    BasketState,
    HouseholdMember,
    MemberSummary,
    ProductSummary,
    TaggedItem,
)

try:
    from .client import HouseholdBasketEnv
except ImportError:
    HouseholdBasketEnv = None  # type: ignore[assignment,misc]

__all__ = [
    "HouseholdBasketEnv",
    "BasketAction",
    "BasketObservation",
    "BasketState",
    "HouseholdMember",
    "MemberSummary",
    "ProductSummary",
    "TaggedItem",
]
