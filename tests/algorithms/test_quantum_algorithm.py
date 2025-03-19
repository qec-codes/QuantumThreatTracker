"""Tests for the quantum algorithm base class."""

from typing import Optional

import pytest
from qualtran.resource_counting import GateCounts
from qualtran.surface_code import AlgorithmSummary, PhysicalCostModel

from quantumthreattracker.algorithms import AlgParams, CryptParams, QuantumAlgorithm


class SampleQuantumAlgorithm(QuantumAlgorithm):
    """Sample instance of the `QuantumAlgorithm` class."""

    def get_algorithm_summary(self, alg_params: Optional[AlgParams] = None) -> AlgorithmSummary:
        """Compute logical resource estimates for the circuit.

        Returns
        -------
        AlgorithmSummary
            Logical resource estimates.
        """
        return AlgorithmSummary(n_algo_qubits=1, n_logical_gates=GateCounts(t=1))


@pytest.fixture()
def default_crypt_params() -> CryptParams:
    """Get a default set of `CryptParams`."""
    return CryptParams("RSA", 2048)


@pytest.fixture()
def quantum_algorithm(default_crypt_params: CryptParams) -> QuantumAlgorithm:
    """Get a default `QuantumAlgorithm`."""
    return SampleQuantumAlgorithm(crypt_params=default_crypt_params)


def test_setup_crypt_params(quantum_algorithm: QuantumAlgorithm) -> None:
    """Test `_crypt_params` is assigned on `QuantumAlgorithm` initialisation."""
    assert quantum_algorithm._crypt_params is not None


def test_algorithm_summary(quantum_algorithm: QuantumAlgorithm) -> None:
    """Test that an algorithm summary is successfully returned."""
    assert quantum_algorithm.get_algorithm_summary() is not None


def test_resource_estimation_qualtran(quantum_algorithm: QuantumAlgorithm) -> None:
    """Test that physical resource estimation via Qualtran is successful."""
    assert (
        quantum_algorithm.estimate_resources_qualtran(
            PhysicalCostModel.make_gidney_fowler(data_d=33)
        )
        is not None
    )


def test_resource_estimation_azure(quantum_algorithm: QuantumAlgorithm) -> None:
    """Test that physical resource estimation via Azure is successful."""
    assert (
        quantum_algorithm.estimate_resources_azure(
            {"qubitParams": {"name": "qubit_gate_ns_e3"}}
        )
        is not None
    )


def test_physical_counts(quantum_algorithm: QuantumAlgorithm) -> None:
    """Test that the physical counts from the resource estimates are non-zero."""
    estimator_result = quantum_algorithm.estimate_resources_azure(
        {"qubitParams": {"name": "qubit_gate_ns_e3"}}
    )
    assert estimator_result["physicalCounts"]["physicalQubits"] > 0
    assert estimator_result["physicalCounts"]["runtime"] > 0
