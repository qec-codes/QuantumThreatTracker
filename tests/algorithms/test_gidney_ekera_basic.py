"""Tests for the `GidneyEkera` class."""

import pytest
from qsharp.estimator import EstimatorParams
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms import CryptParams, QuantumAlgorithm
from quantumthreattracker.algorithms.rsa.gidney_ekera_basic import (
    GidneyEkeraBasic,
    GidneyEkeraBasicParams,
)


@pytest.fixture()
def default_params() -> GidneyEkeraBasicParams:
    """Get default algorithm parameters for tests.

    Returns
    -------
        GidneyEkeraBasicParams: A default set of parameters for the GidneyEkeraBasic
        algorithm.
    """
    return GidneyEkeraBasicParams()


@pytest.fixture()
def default_algorithm(default_params: GidneyEkeraBasicParams) -> GidneyEkeraBasic:
    """Get a default instance of `GidneyEkeraBasic`.

    Returns
    -------
        GidneyEkeraBasic: A default instance of the `GidneyEkeraBasic` class.
    """
    return GidneyEkeraBasic(CryptParams("RSA", 1024), default_params)


def test_alg_sum(default_algorithm: QuantumAlgorithm) -> None:
    """Test that the algorithm outputs a valid `AlgorithmSummary`."""
    alg_sum = default_algorithm.get_algorithm_summary()
    assert isinstance(alg_sum, AlgorithmSummary)
    assert alg_sum.n_algo_qubits > 0
    assert alg_sum.n_logical_gates.toffoli > 0


def test_wrong_protocol_raises_error() -> None:
    """Test that using a non-RSA protocol raises an error."""
    algorithm = GidneyEkeraBasic(CryptParams("ECDH", 64))
    with pytest.raises(NameError, match='The protocol for this class must be "RSA"'):
        algorithm.get_algorithm_summary(GidneyEkeraBasicParams())


def test_alg_summary_works_with_no_alg_params() -> None:
    """Test that not providing params at initialization or estimation works fine."""
    # Create algorithm without params
    algorithm = GidneyEkeraBasic(CryptParams("RSA", 128))

    estimator_params = EstimatorParams()

    # This should raise a ValueError because no params are provided
    azure_result = algorithm.estimate_resources_azure(estimator_params)
    assert "physicalCounts" in azure_result
    assert "physicalQubits" in azure_result["physicalCounts"]
