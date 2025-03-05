"""Tests for the `BaselineShor` class."""

import pytest
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms import BaselineShor, CryptParams, QuantumAlgorithm


@pytest.fixture()
def default_algorithm() -> BaselineShor:
    """Get a default instance of `BaselineShor`."""
    return BaselineShor(CryptParams("RSA", 64))


def test_alg_sum(default_algorithm: QuantumAlgorithm) -> None:
    """Test that the algorithm outputs a valid `AlgorithmSummary`."""
    alg_sum = default_algorithm.get_algorithm_summary()
    assert isinstance(alg_sum, AlgorithmSummary)
    assert alg_sum.n_algo_qubits > 0
