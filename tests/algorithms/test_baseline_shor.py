"""Tests for the `BaselineShor` class."""

import pytest
from qsharp.estimator import EstimatorParams
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms import CryptParams, QuantumAlgorithm
from quantumthreattracker.algorithms.rsa.baseline_shor import (
    BaselineShor,
    BaselineShorParams,
)


@pytest.fixture()
def default_params() -> BaselineShorParams:
    """Get default algorithm parameters for tests."""
    return BaselineShorParams()


@pytest.fixture()
def default_algorithm(default_params) -> BaselineShor:
    """Get a default instance of `BaselineShor`."""
    return BaselineShor(CryptParams("RSA", 64), default_params)


def test_alg_sum(default_algorithm: QuantumAlgorithm) -> None:
    """Test that the algorithm outputs a valid `AlgorithmSummary`."""
    alg_sum = default_algorithm.get_algorithm_summary()
    assert isinstance(alg_sum, AlgorithmSummary)
    assert alg_sum.n_algo_qubits > 0
    assert alg_sum.n_logical_gates.clifford > 0


def test_wrong_protocol_raises_error() -> None:
    """Test that using a non-RSA protocol raises an error."""
    algorithm = BaselineShor(CryptParams("ECDH", 64))
    with pytest.raises(NameError, match='The protocol for this class must be "RSA"'):
        algorithm.get_algorithm_summary()


def test_key_size_affects_output() -> None:
    """Test that changing key size affects the algorithm summary."""
    algorithm_small = BaselineShor(CryptParams("RSA", 64))
    algorithm_large = BaselineShor(CryptParams("RSA", 128))

    small_sum = algorithm_small.get_algorithm_summary()
    large_sum = algorithm_large.get_algorithm_summary()

    # Larger key size should result in more qubits and gates
    assert large_sum.n_algo_qubits > small_sum.n_algo_qubits
    assert (
        large_sum.n_logical_gates.clifford >
        small_sum.n_logical_gates.clifford
    )

def test_params_at_estimation_time() -> None:
    """Test providing params during estimation instead of initialization."""
    # Create algorithm without params
    algorithm = BaselineShor(CryptParams("RSA", 256))

    # Create params for estimation
    params = BaselineShorParams()

    estimator_params = EstimatorParams()

    # This should succeed because we're providing params at estimation time
    azure_result = algorithm.estimate_resources_azure(estimator_params, params)
    assert 'physicalCounts' in azure_result
    assert 'physicalQubits' in azure_result['physicalCounts']

def test_alg_summary_works_with_no_alg_params() -> None:
    """Test that not providing params at initialization or estimation works fine."""
    # Create algorithm without params
    algorithm = BaselineShor(CryptParams("RSA", 256))

    estimator_params = EstimatorParams()

    # This should raise a ValueError because no params are provided
    azure_result = algorithm.estimate_resources_azure(estimator_params)
    assert 'physicalCounts' in azure_result
    assert 'physicalQubits' in azure_result['physicalCounts']

