"""Tests for the `GidneyEkera` class."""

import pytest
from qsharp.estimator import EstimatorParams
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms import AlgParams, CryptParams, QuantumAlgorithm
from quantumthreattracker.algorithms.rsa.gidney_ekera import (
    GidneyEkera,
    GidneyEkeraParams,
)


@pytest.fixture()
def default_params() -> GidneyEkeraParams:
    """Get default algorithm parameters for tests.

    Returns
    -------
        GidneyEkeraParams: Default parameters for the Gidney-Ekera algorithm.
    """
    return GidneyEkeraParams(num_exp_qubits=1536, window_size_exp=4, window_size_mul=4)


@pytest.fixture()
def default_algorithm(default_params: GidneyEkeraParams) -> GidneyEkera:
    """Get a default instance of `GidneyEkera`.

    Returns
    -------
        GidneyEkera: A default instance of the Gidney-Ekera algorithm.
    """
    return GidneyEkera(CryptParams("RSA", 1024), default_params)


def test_alg_sum(default_algorithm: QuantumAlgorithm) -> None:
    """Test that the algorithm outputs a valid `AlgorithmSummary`."""
    alg_sum = default_algorithm.get_algorithm_summary()
    assert isinstance(alg_sum, AlgorithmSummary)
    assert alg_sum.n_algo_qubits > 0
    assert alg_sum.n_logical_gates.toffoli > 0


def test_wrong_protocol_raises_error() -> None:
    """Test that using a non-RSA protocol raises an error."""
    algorithm = GidneyEkera(CryptParams("ECDH", 64))
    with pytest.raises(NameError, match='The protocol for this class must be "RSA"'):
        algorithm.get_algorithm_summary(
            GidneyEkeraParams(num_exp_qubits=96, window_size_exp=4, window_size_mul=4)
        )


def test_missing_params_raises_error() -> None:
    """Test that missing algorithm parameters raises an error."""
    algorithm = GidneyEkera(CryptParams("RSA", 1024))
    with pytest.raises(ValueError, match="Algorithm parameters must be provided"):
        algorithm.get_algorithm_summary()


def test_wrong_params_type_raises_error() -> None:
    """Test that providing wrong parameter type raises an error."""
    algorithm = GidneyEkera(CryptParams("RSA", 1024))

    class WrongParams(AlgParams):
        pass

    with pytest.raises(TypeError, match="Expected GidneyEkeraParams"):
        algorithm.get_algorithm_summary(WrongParams())


def test_window_size_affects_output(default_algorithm: GidneyEkera) -> None:
    """Test that changing window sizes affects the algorithm summary."""
    default_sum = default_algorithm.get_algorithm_summary()

    different_params = GidneyEkeraParams(
        num_exp_qubits=1536, window_size_exp=6, window_size_mul=6
    )
    different_sum = default_algorithm.get_algorithm_summary(different_params)

    # Different window sizes should result in different resource estimates
    assert default_sum.n_logical_gates.toffoli != different_sum.n_logical_gates.toffoli


def test_params_at_estimation_time() -> None:
    """Test providing params during estimation instead of initialization."""
    # Create algorithm without params
    algorithm = GidneyEkera(CryptParams("RSA", 1024))

    # Create params for estimation
    params = GidneyEkeraParams(
        num_exp_qubits=1536, window_size_exp=5, window_size_mul=5
    )

    # Test Azure resource estimation with params at estimation time
    estimator_params = EstimatorParams()

    # This should succeed because we're providing params at estimation time
    azure_result = algorithm.estimate_resources_azure(estimator_params, params)
    assert "physicalCounts" in azure_result
    assert "physicalQubits" in azure_result["physicalCounts"]


def test_no_params_anywhere_raises_error() -> None:
    """Test that not providing params at init or estimation raises an error."""
    # Create algorithm without params
    algorithm = GidneyEkera(CryptParams("RSA", 1024))

    # Test Azure resource estimation without params
    estimator_params = EstimatorParams()

    # This should raise a ValueError because no params are provided
    with pytest.raises(ValueError, match="Algorithm parameters must be provided"):
        algorithm.estimate_resources_azure(estimator_params)
