import hashlib
import json

import pytest

from tests.training.test_backend_equivalence import FIXTURES, NUMERIC, assert_trace_close


def test_numeric_fixture_matches_the_independent_reference_hashes():
    reference = json.loads((FIXTURES / "backend_baseline_torch_2_9_1.json").read_text())["traces"]
    assert NUMERIC.keys() == reference.keys()
    for case, fields in NUMERIC.items():
        assert fields.keys() == reference[case].keys()
        for name, value in fields.items():
            assert hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest() == reference[case][name]


def test_near_rounding_boundary_uses_distance_not_decimal_rounding():
    a, b = 0.049707649, 0.049707651
    assert round(a, 7) != round(b, 7)
    assert_trace_close([a, 3.725290298461914e-9], [b, 0.0])


@pytest.mark.parametrize(
    "actual,expected",
    [
        (0.0501, 0.05),
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        ([1.0], [1.0, 2.0]),
        (2320, 2323),
        (True, 1),
        ({"loss": 0.0}, {"gradient": 0.0}),
        ("changed", "original"),
    ],
)
def test_comparison_rejects_regressions_and_discrete_changes(actual, expected):
    with pytest.raises(AssertionError):
        assert_trace_close(actual, expected)
