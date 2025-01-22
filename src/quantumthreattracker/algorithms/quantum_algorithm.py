"""Base class for quantum algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union

from qsharp.estimator import EstimatorParams, EstimatorResult, LogicalCounts
from qualtran import Bloq


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
    def create_subroutine(self) -> tuple[Bloq, int]:
        """Construct the subroutine which dominates the resource cost of the algorithm.

        Returns
        -------
        tuple[Bloq, int]
            Qualtran Bloq, together with the number of sequential repetitions needed in
            the circuit.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def get_logical_counts(self) -> LogicalCounts:
        """Create the logical counts to pass to the Azure Quantum Resource Estimator.

        Returns
        -------
        LogicalCounts
            Logical counts derived from the Qualtran Bloqs.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def estimate_resources(
        self, estimator_params: Union[dict, List, EstimatorParams]
    ) -> EstimatorResult:
        """Estimate the physical resources required for a single circuit run.

        Parameters
        ----------
        estimator_params : Union[dict, List, EstimatorParams]
            Parameters for the Microsoft Azure Quantum Resource Estimator.

        Returns
        -------
        EstimatorResult
            Physical resource estimates.
        """
        return self.get_logical_counts().estimate(estimator_params)
