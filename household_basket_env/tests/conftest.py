# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared pytest fixtures for HouseholdBasketEnv tests."""

import os
import sys
from pathlib import Path

import pytest


_PKG_ROOT = Path(__file__).resolve().parent.parent           # household_basket_env/
_PARENT_OF_PKG = _PKG_ROOT.parent                            # module2/

# Ensure `import household_basket_env...` works without installing the package.
for p in (str(_PARENT_OF_PKG), str(_PKG_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Point the catalog loader at the local copy in module2/household_basket_env/data/
os.environ.setdefault(
    "HOUSEHOLD_BASKET_PRODUCTS_PATH",
    str(_PKG_ROOT / "data" / "products.json"),
)


@pytest.fixture(scope="session")
def augmented_catalog():
    from household_basket_env.server.catalog import load_augmented_catalog

    return load_augmented_catalog()


@pytest.fixture()
def env(augmented_catalog):
    from household_basket_env.server.environment import HouseholdBasketEnvironment

    return HouseholdBasketEnvironment(augmented_catalog=augmented_catalog)
