"""Shared test configuration.

Tests that need a third-party checkout (see scripts/third_party.sh) are skipped
when it is not present, so the suite runs on a bare clone.
"""

from __future__ import annotations

import pytest

from dreamxcache.third_party import is_available

requires_aircheck_utils = pytest.mark.skipif(
    not is_available("aircheck_utils"), reason="aircheck_utils not found; run scripts/third_party.sh"
)
requires_bitbirch = pytest.mark.skipif(
    not is_available("bitbirch"), reason="bitbirch not found; run scripts/third_party.sh"
)
