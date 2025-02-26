"""Base class for quantum algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union

from qsharp.estimator import EstimatorParams, EstimatorResult, LogicalCounts
from qualtran.surface_code import AlgorithmSummary, PhysicalCostModel


@dataclass
class CryptParams(ABC):
    """Dataclass describing the cryptographic protocol.

    Parameters
    ----------
    protocol: str
        Cryptographic protocol. Can be:
            - 'RSA' (Rivest-Shamir-Adleman; factoring)
            - 'DH' (Diffie-Hellman; discrete log)
            - 'ECDH' (Elliptic Curve Diffie-Hellman; discrete log over elliptic curves)
    key_size: int
        Cryptographic key size.
    """

    protocol: str
    key_size: int


class QuantumAlgorithm(ABC):
    """Base class for quantum algorithms."""

    def __init__(self, crypt_params: CryptParams):
        """Initialise the `QuantumAlgorithm`.

        Parameters
        ----------
        crypt_params : CryptParams
            Cryptographic protocol parameters.
        """
        self._crypt_params = crypt_params

    @abstractmethod
    def success_probability(self) -> float:
        """Calculate the algorithmic success probability.

        Returns
        -------
        float
            Algorithmic success probability.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def get_algorithm_summary(self) -> AlgorithmSummary:
        """Compute logical resource estimates for the circuit.

        Returns
        -------
        AlgorithmSummary
            Logical resource estimates.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented.
        """
        raise NotImplementedError

    def estimate_resources_qualtran(self, cost_model: PhysicalCostModel) -> dict:
        """Create a physical resource estimate using Qualtran.

        Parameters
        ----------
        cost_model : PhysicalCostModel
            Cost model used by Qualtran to estimate physical resources.

        Returns
        -------
        dict
            Physical resource estimates.
        """
        algorithm_summary = self.get_algorithm_summary()
        resource_estimate = {
            "duration_hr": cost_model.duration_hr(algorithm_summary),
            "n_phys_qubits": cost_model.n_phys_qubits(algorithm_summary),
            "error": cost_model.error(algorithm_summary),
        }
        return resource_estimate

    def estimate_resources_azure(
        self, estimator_params: Union[dict, List, EstimatorParams]
    ) -> EstimatorResult:
        """Create a physical resource estimate using Azure.

        Parameters
        ----------
        estimator_params : Union[dict, List, EstimatorParams]
            Parameters for the Microsoft Azure Quantum Resource Estimator.

        Returns
        -------
        EstimatorResult
            Physical resource estimates.
        """
        algorithm_summary = self.get_algorithm_summary()
        t_and_ccz_count = algorithm_summary.n_logical_gates.total_t_and_ccz_count()
        logical_counts = LogicalCounts(
            {
                "numQubits": algorithm_summary.n_algo_qubits,
                "tCount": t_and_ccz_count["n_t"],
                "rotationCount": algorithm_summary.n_logical_gates.rotation,
                "cczCount": t_and_ccz_count["n_ccz"],
                "measurementCount": algorithm_summary.n_logical_gates.measurement,
            }
        )

        return logical_counts.estimate(estimator_params)
