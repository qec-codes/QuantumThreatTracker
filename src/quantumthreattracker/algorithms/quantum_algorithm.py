"""Base class for quantum algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union

from qsharp.estimator import EstimatorParams, EstimatorResult, LogicalCounts
from qualtran.surface_code import AlgorithmSummary


@dataclass
class AlgorithmParams(ABC):
    """Base dataclass for quantum algorithm parameters.

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

    @abstractmethod
    def __init__(self, algorithm_params: AlgorithmParams):
        """Initialise the `QuantumAlgorithm`.

        Parameters
        ----------
        algorithm_params : AlgorithmParams
            Quantum algorithm parameters.
        """
        self._algorithm_params = algorithm_params

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

    @abstractmethod
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
                "cczCount": t_and_ccz_count["n_ccz"],
            }
        )

        return logical_counts.estimate(estimator_params)
