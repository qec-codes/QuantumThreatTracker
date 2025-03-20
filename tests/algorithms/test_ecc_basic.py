"""Tests for the `ECCBasic` class."""

import pytest
from qsharp.estimator import EstimatorParams
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms import CryptParams, QuantumAlgorithm
from quantumthreattracker.algorithms.ecc.ecc_basic import ECCBasic, ECCBasicParams


@pytest.fixture()
def default_params() -> ECCBasicParams:
    """Get default algorithm parameters for tests.

    Returns
    -------
        ECCBasicParams: The default parameters for the ECCBasic algorithm.
    """
    return ECCBasicParams()


@pytest.fixture()
def default_algorithm(default_params: ECCBasicParams) -> ECCBasic:
    """Get a default instance of `ECCBasic`.

    Returns
    -------
        ECCBasic: An instance of the ECCBasic class with default parameters.
    """
    return ECCBasic(CryptParams("ECDH", 64), default_params)


def test_alg_sum(default_algorithm: QuantumAlgorithm) -> None:
    """Test that the algorithm outputs a valid `AlgorithmSummary`."""
    alg_sum = default_algorithm.get_algorithm_summary()
    assert isinstance(alg_sum, AlgorithmSummary)
    assert alg_sum.n_algo_qubits > 0
    assert alg_sum.n_logical_gates.toffoli > 0


def test_wrong_protocol_raises_error() -> None:
    """Test that using a non-ECDH protocol raises an error."""
    algorithm = ECCBasic(CryptParams("RSA", 64))
    with pytest.raises(NameError, match='The protocol for this class must be "ECDH"'):
        algorithm.get_algorithm_summary()


def test_key_size_affects_output() -> None:
    """Test that changing key size affects the algorithm summary."""
    algorithm_small = ECCBasic(CryptParams("ECDH", 64))
    algorithm_large = ECCBasic(CryptParams("ECDH", 128))

    small_sum = algorithm_small.get_algorithm_summary()
    large_sum = algorithm_large.get_algorithm_summary()

    # Larger key size should result in more qubits and gates
    assert large_sum.n_algo_qubits > small_sum.n_algo_qubits
    assert large_sum.n_logical_gates.toffoli > small_sum.n_logical_gates.toffoli


def test_alg_summary_works_with_no_alg_params() -> None:
    """Test that not providing params at init or estimation works fine."""
    # Create algorithm without params
    algorithm = ECCBasic(CryptParams("ECDH", 128))

    estimator_params = EstimatorParams()

    # This should raise a ValueError because no params are provided
    azure_result = algorithm.estimate_resources_azure(estimator_params)
    assert "physicalCounts" in azure_result
    assert "physicalQubits" in azure_result["physicalCounts"]
