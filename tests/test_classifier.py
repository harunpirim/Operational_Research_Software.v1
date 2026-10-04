"""
Tests for Problem Classifier
"""

import json
import os

import pytest
from unittest.mock import Mock, patch
from src.agents.problem_classifier import ProblemClassifier


@pytest.fixture
def classifier():
    """Create a classifier instance for testing."""
    # Mock the API key
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
        return ProblemClassifier()


def test_classifier_initialization():
    """Test that classifier initializes correctly."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
        classifier = ProblemClassifier()
        assert classifier.api_client.api_key == "test_key"
        assert classifier.model is not None


def test_classifier_no_api_key():
    """Test that classifier raises error without API key."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="No API key found"):
            ProblemClassifier()


def test_problem_types_defined(classifier):
    """Test that problem types are defined."""
    assert len(classifier.PROBLEM_TYPES) > 0
    assert "linear_programming" in classifier.PROBLEM_TYPES
    assert "transportation" in classifier.PROBLEM_TYPES


def test_get_problem_template(classifier):
    """Test getting problem templates."""
    template = classifier.get_problem_template("transportation")
    assert isinstance(template, dict)

    # Transportation should have specific fields
    template = classifier.get_problem_template("transportation")
    assert "sources" in template or len(template) > 0


def test_classification_basic(classifier):
    """Test basic classification with a mocked API response (no network)."""
    response_json = json.dumps(
        {
            "problem_type": "linear_programming",
            "objective": "maximize",
            "objective_description": "profit from production",
            "decision_variables": [
                {"name": "x", "description": "units produced", "type": "continuous"}
            ],
            "constraints": [
                {"name": "capacity", "expression": "x <= 10", "sense": "<="}
            ],
            "parameters": {"profit": [5]},
            "confidence": 0.9,
            "assumptions": [],
            "warnings": [],
            "notes": "",
        }
    )
    classifier.api_client.create_message = Mock(
        return_value={
            "content": response_json,
            "model": "mock-model",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )

    problem = "Maximize profit from production"
    result = classifier.classify(problem)

    assert result["problem_type"] == "linear_programming"
    assert "confidence" in result
    classifier.api_client.create_message.assert_called_once()


def test_build_classification_prompt(classifier):
    """Test that prompt is built correctly."""
    problem = "Test problem"
    prompt = classifier._build_classification_prompt(problem)

    assert "Operations Research" in prompt
    assert problem in prompt
    assert "JSON" in prompt


# Integration tests (require actual API key)
@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No API key available")
def test_classify_transportation_problem():
    """Integration test with real API - transportation problem."""
    classifier = ProblemClassifier()

    problem = """
    I need to minimize transportation costs between 3 warehouses and 4 stores.
    Warehouse supplies: [100, 150, 200]
    Store demands: [80, 120, 90, 110]
    """

    result = classifier.classify(problem)

    assert "problem_type" in result
    assert result["problem_type"] in ["transportation", "linear_programming"]
    assert "confidence" in result


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No API key available")
def test_classify_production_problem():
    """Integration test with real API - production problem."""
    classifier = ProblemClassifier()

    problem = """
    Maximize profit from producing 3 products using 2 machines.
    Machine 1 has 480 hours, Machine 2 has 380 hours.
    Product A needs 8 hours on M1, 4 hours on M2, profit $50
    """

    result = classifier.classify(problem)

    assert "problem_type" in result
    assert result["problem_type"] in ["linear_programming", "production"]
