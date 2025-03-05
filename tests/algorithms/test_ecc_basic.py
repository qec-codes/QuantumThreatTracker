"""Tests for the `ECCBasic` class."""

import pytest
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms import CryptParams, ECCBasic, QuantumAlgorithm


@pytest.fixture()
def default_algorithm() -> ECCBasic:
    """Get a default instance of `BaselineShor`."""
    return ECCBasic(CryptParams("ECDH", 64))


def test_alg_sum(default_algorithm: QuantumAlgorithm) -> None:
    """Test that the algorithm outputs a valid `AlgorithmSummary`."""
    alg_sum = default_algorithm.get_algorithm_summary()
    assert isinstance(alg_sum, AlgorithmSummary)
    assert alg_sum.n_algo_qubits > 0
