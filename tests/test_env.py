"""The environment report contains the fields the roadmap requires."""

from __future__ import annotations

from beeja.utils import environment_report

REQUIRED_KEYS = {
    "python",
    "platform",
    "machine",
    "memory_bytes",
    "torch",
    "cuda_available",
    "mps_available",
}


def test_report_has_required_keys():
    report = environment_report()
    assert REQUIRED_KEYS.issubset(report)


def test_report_is_json_serialisable():
    import json

    json.dumps(environment_report())
