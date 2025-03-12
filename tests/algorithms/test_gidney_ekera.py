"""Tests for the `GidneyEkera` class."""

import pytest
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms import (
    CryptParams,
    GidneyEkera,
    GidneyEkeraParams,
    QuantumAlgorithm,
)


@pytest.fixture()
def default_algorithm() -> GidneyEkera:
    """Get a default instance of `GidneyEkera`."""
    return GidneyEkera(CryptParams("RSA", 64), GidneyEkeraParams(96, 5, 5))


def test_alg_sum(default_algorithm: QuantumAlgorithm) -> None:
    """Test that the algorithm outputs a valid `AlgorithmSummary`."""
    alg_sum = default_algorithm.get_algorithm_summary()
    assert isinstance(alg_sum, AlgorithmSummary)
    assert alg_sum.n_algo_qubits > 0


def test_wrong_protocol_raises_error() -> None:
    """Test that using a non-RSA protocol raises an error."""
    algorithm = GidneyEkera(CryptParams("ECDH", 64), GidneyEkeraParams(96, 5, 5))
    with pytest.raises(NameError):
        algorithm.get_algorithm_summary()
