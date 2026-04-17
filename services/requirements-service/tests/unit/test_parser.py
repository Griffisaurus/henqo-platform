"""
Unit tests for requirements_service.domain.parser.

Targets ≥80% branch coverage on parser.py.
"""
from __future__ import annotations

import pytest

from requirements_service.domain.parser import ParseError, ParsedRequirement, parse


# ---------------------------------------------------------------------------
# Upper-bound patterns
# ---------------------------------------------------------------------------

class TestUpperBound:
    def test_less_than_symbol(self) -> None:
        r = parse("< 2 mm")
        assert r.direction == "upper_bound"
        assert r.value == 2.0
        assert r.unit == "mm"
        assert r.is_qualitative is False
        assert r.value_upper is None

    def test_shall_not_exceed(self) -> None:
        r = parse("shall not exceed 85 degC")
        assert r.direction == "upper_bound"
        assert r.value == 85.0
        assert r.unit == "degC"

    def test_not_exceed_no_shall(self) -> None:
        r = parse("not exceed 100 kPa")
        assert r.direction == "upper_bound"
        assert r.value == 100.0
        assert r.unit == "kPa"

    def test_no_more_than(self) -> None:
        r = parse("no more than 50 N")
        assert r.direction == "upper_bound"
        assert r.value == 50.0
        assert r.unit == "N"

    def test_at_most(self) -> None:
        r = parse("at most 200 MPa")
        assert r.direction == "upper_bound"
        assert r.value == 200.0
        assert r.unit == "MPa"

    def test_maximum_keyword(self) -> None:
        r = parse("maximum 30 degC")
        assert r.direction == "upper_bound"
        assert r.value == 30.0
        assert r.unit == "degC"

    def test_float_value(self) -> None:
        r = parse("< 2.5 mm")
        assert r.direction == "upper_bound"
        assert r.value == 2.5


# ---------------------------------------------------------------------------
# Lower-bound patterns
# ---------------------------------------------------------------------------

class TestLowerBound:
    def test_greater_than_symbol(self) -> None:
        r = parse("> 500 N")
        assert r.direction == "lower_bound"
        assert r.value == 500.0
        assert r.unit == "N"
        assert r.is_qualitative is False

    def test_at_least(self) -> None:
        r = parse("at least 1000 kN")
        assert r.direction == "lower_bound"
        assert r.value == 1000.0
        assert r.unit == "kN"

    def test_minimum_keyword(self) -> None:
        r = parse("minimum 9.81 m/s^2")
        assert r.direction == "lower_bound"
        assert r.value == 9.81
        assert r.unit == "m/s^2"

    def test_velocity_unit(self) -> None:
        r = parse("> 10 m/s")
        assert r.direction == "lower_bound"
        assert r.unit == "m/s"

    def test_kelvin_unit(self) -> None:
        r = parse("at least 293 K")
        assert r.direction == "lower_bound"
        assert r.unit == "K"


# ---------------------------------------------------------------------------
# Range patterns
# ---------------------------------------------------------------------------

class TestRange:
    def test_basic_range(self) -> None:
        r = parse("between 10 and 20 MPa")
        assert r.direction == "range"
        assert r.value == 10.0
        assert r.value_upper == 20.0
        assert r.unit == "MPa"
        assert r.is_qualitative is False

    def test_range_float(self) -> None:
        r = parse("between 1.5 and 3.5 mm")
        assert r.value == 1.5
        assert r.value_upper == 3.5
        assert r.unit == "mm"


# ---------------------------------------------------------------------------
# Qualitative patterns
# ---------------------------------------------------------------------------

class TestQualitative:
    def test_minimize_weight(self) -> None:
        r = parse("minimize weight")
        assert r.direction == "qualitative"
        assert r.is_qualitative is True
        assert r.value is None
        assert r.unit is None
        assert r.value_upper is None

    def test_maximize(self) -> None:
        r = parse("maximize stiffness during operation")
        assert r.direction == "qualitative"
        assert r.is_qualitative is True

    def test_reduce(self) -> None:
        r = parse("reduce thermal losses")
        assert r.direction == "qualitative"

    def test_avoid(self) -> None:
        r = parse("avoid resonance under load")
        assert r.direction == "qualitative"

    def test_ensure(self) -> None:
        r = parse("ensure uniform distribution")
        assert r.direction == "qualitative"


# ---------------------------------------------------------------------------
# Unit extraction coverage
# ---------------------------------------------------------------------------

class TestUnitExtraction:
    @pytest.mark.parametrize("unit", ["mm", "N", "kN", "MPa", "degC", "K", "m/s", "m/s^2"])
    def test_upper_bound_unit(self, unit: str) -> None:
        r = parse(f"< 1 {unit}")
        assert r.unit == unit

    def test_kelvin_lower(self) -> None:
        r = parse("> 0 K")
        assert r.unit == "K"


# ---------------------------------------------------------------------------
# Condition clause extraction
# ---------------------------------------------------------------------------

class TestConditionClause:
    def test_under_load(self) -> None:
        r = parse("< 5 mm under load")
        assert r.condition_clause is not None
        assert "under" in r.condition_clause.lower()

    def test_at_rated_speed(self) -> None:
        r = parse("> 500 N at rated speed")
        assert r.condition_clause is not None
        assert "at" in r.condition_clause.lower()

    def test_during_operation(self) -> None:
        r = parse("shall not exceed 85 degC during operation")
        assert r.condition_clause is not None
        assert "during" in r.condition_clause.lower()

    def test_no_condition(self) -> None:
        r = parse("< 2 mm")
        # "mm" alone doesn't trigger a condition pattern
        assert r.condition_clause is None


# ---------------------------------------------------------------------------
# ParseError cases
# ---------------------------------------------------------------------------

class TestParseError:
    def test_empty_string(self) -> None:
        with pytest.raises(ParseError):
            parse("")

    def test_pure_prose_no_keyword(self) -> None:
        with pytest.raises(ParseError):
            parse("the component shall be acceptable")

    def test_number_only_no_unit_no_keyword(self) -> None:
        # No unit → no bare number match → ParseError
        with pytest.raises(ParseError):
            parse("some value 42")

    def test_ambiguous_direction(self) -> None:
        # A bare number+unit with no direction keyword is ambiguous
        with pytest.raises(ParseError):
            parse("the gap is 3 mm")

    def test_no_qualitative_no_number(self) -> None:
        with pytest.raises(ParseError):
            parse("the design shall meet requirements")
