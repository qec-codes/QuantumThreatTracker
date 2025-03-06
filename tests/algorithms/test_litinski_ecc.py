"""Tests for the `LitinskiECC` class."""

import pytest
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms import CryptParams, LitinskiECC, QuantumAlgorithm


@pytest.fixture()
def default_algorithm() -> LitinskiECC:
    """Get a default instance of `LitinskiECC`."""
    return LitinskiECC(CryptParams("ECDH", 64))


def test_alg_sum(default_algorithm: QuantumAlgorithm) -> None:
    """Test that the algorithm outputs a valid `AlgorithmSummary`."""
    alg_sum = default_algorithm.get_algorithm_summary()
    assert isinstance(alg_sum, AlgorithmSummary)
    assert alg_sum.n_algo_qubits > 0
    assert alg_sum.n_logical_gates.toffoli > 0


def test_wrong_protocol_raises_error() -> None:
    """Test that using a non-ECDH protocol raises an error."""
    algorithm = LitinskiECC(CryptParams("RSA", 64))
    with pytest.raises(NameError, match='Protocol must be "ECDH"'):
        algorithm.get_algorithm_summary()


def test_window_size_affects_output(default_algorithm: LitinskiECC) -> None:
    """Test that changing window size affects the algorithm summary."""
    default_sum = default_algorithm.get_algorithm_summary(window_size=22)
    different_sum = default_algorithm.get_algorithm_summary(window_size=10)

    # Different window sizes should result in different resource estimates
    assert default_sum.n_logical_gates.toffoli != different_sum.n_logical_gates.toffoli
