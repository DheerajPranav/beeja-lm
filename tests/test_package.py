"""The package imports and exposes its identity."""

from __future__ import annotations

import beeja


def test_import_and_identity():
    assert beeja.MODEL_FAMILY == "Beeja"
    assert beeja.FIRST_CHECKPOINT == "Beeja-3M"
    assert isinstance(beeja.__version__, str) and beeja.__version__


def test_public_api_exposed():
    # The utilities the rest of the project relies on are importable from the top.
    assert callable(beeja.select_device)
    assert callable(beeja.set_seed)
    assert callable(beeja.environment_report)
