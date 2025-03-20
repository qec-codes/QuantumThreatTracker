"""Tests for the `LitinskiECC` class."""

import pytest
from qsharp.estimator import EstimatorParams
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms import AlgParams, CryptParams, QuantumAlgorithm
from quantumthreattracker.algorithms.ecc.litinski_ecc import (
    LitinskiECC,
    LitinskiECCParams,
)


@pytest.fixture()
def default_params() -> LitinskiECCParams:
    """Get default algorithm parameters for tests.

    Returns
    -------
        LitinskiECCParams: A default set of parameters for the LitinskiECC algorithm.
    """
    return LitinskiECCParams(window_size=22, classical_bits=48)


@pytest.fixture()
def default_algorithm(default_params: LitinskiECCParams) -> LitinskiECC:
    """Get a default instance of `LitinskiECC`.

    Returns
    -------
        LitinskiECC: A default instance of the LitinskiECC algorithm.
    """
    return LitinskiECC(CryptParams("ECDH", 64), default_params)


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
        algorithm.get_algorithm_summary(LitinskiECCParams(window_size=22))


def test_missing_params_raises_error() -> None:
    """Test that missing algorithm parameters raises an error."""
    algorithm = LitinskiECC(CryptParams("ECDH", 64))
    with pytest.raises(ValueError, match="Algorithm parameters must be provided"):
        algorithm.get_algorithm_summary()


def test_wrong_params_type_raises_error() -> None:
    """Test that providing wrong parameter type raises an error."""
    algorithm = LitinskiECC(CryptParams("ECDH", 64))

    class WrongParams(AlgParams):
        pass

    with pytest.raises(TypeError, match="Expected LitinskiECCParams"):
        algorithm.get_algorithm_summary(WrongParams())


def test_window_size_affects_output(default_algorithm: LitinskiECC) -> None:
    """Test that changing window size affects the algorithm summary."""
    default_sum = default_algorithm.get_algorithm_summary()

    different_params = LitinskiECCParams(window_size=10, classical_bits=48)
    different_sum = default_algorithm.get_algorithm_summary(different_params)

    # Different window sizes should result in different resource estimates
    assert default_sum.n_logical_gates.toffoli != different_sum.n_logical_gates.toffoli


def test_params_at_estimation_time() -> None:
    """Test providing params during estimation instead of initialization."""
    # Create algorithm without params
    algorithm = LitinskiECC(CryptParams("ECDH", 256))

    # Create params for estimation
    params = LitinskiECCParams(window_size=22, classical_bits=48)

    # Test Azure resource estimation with params at estimation time
    estimator_params = EstimatorParams()

    # This should succeed because we're providing params at estimation time
    azure_result = algorithm.estimate_resources_azure(estimator_params, params)
    assert "physicalCounts" in azure_result
    assert "physicalQubits" in azure_result["physicalCounts"]


def test_no_params_anywhere_raises_error() -> None:
    """Test that not providing params at init or estimation raises an error."""
    # Create algorithm without params
    algorithm = LitinskiECC(CryptParams("ECDH", 256))

    # Test Azure resource estimation without params
    estimator_params = EstimatorParams()

    # This should raise a ValueError because no params are provided
    with pytest.raises(ValueError, match="Algorithm parameters must be provided"):
        algorithm.estimate_resources_azure(estimator_params)
