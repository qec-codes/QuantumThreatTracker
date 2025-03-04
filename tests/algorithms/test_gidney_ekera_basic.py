"""Tests for the `GidneyEkeraBasic` class."""

import pytest
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms import (
    CryptParams,
    GidneyEkeraBasic,
    QuantumAlgorithm,
)


@pytest.fixture()
def default_algorithm() -> GidneyEkeraBasic:
    """Get a default instance of `BaselineShor`."""
    return GidneyEkeraBasic(CryptParams("ECDH", 64))


def test_alg_sum(default_algorithm: QuantumAlgorithm) -> None:
    """Test that the algorithm outputs a valid `AlgorithmSummary`."""
    alg_sum = default_algorithm.get_algorithm_summary()
    assert isinstance(alg_sum, AlgorithmSummary)
    assert alg_sum.n_algo_qubits > 0
