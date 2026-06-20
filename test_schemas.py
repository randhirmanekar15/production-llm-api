"""Validation tests for the request schema (no model load required)."""

import pytest
from pydantic import ValidationError

from schemas import GenerationRequest


def test_valid_request():
    req = GenerationRequest(prompt="hello")
    assert req.max_tokens == 256
    assert 0.0 <= req.temperature <= 1.0


def test_empty_prompt_rejected():
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="")


def test_max_tokens_bounds_enforced():
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="hi", max_tokens=5000)


def test_temperature_bounds_enforced():
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="hi", temperature=2.0)
