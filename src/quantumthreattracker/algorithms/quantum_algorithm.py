"""Base class for quantum algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

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


@dataclass
class AlgParams:
    """Base class for algorithm parameters."""

    pass


class QuantumAlgorithm(ABC):
    """Abstract base class for quantum algorithms."""

    def __init__(
        self, crypt_params: CryptParams, alg_params: Optional[AlgParams] = None
    ):
        """Initialize the quantum algorithm.

        Parameters
        ----------
        crypt_params : CryptParams
            Cryptographic parameters.
        alg_params : Optional[AlgParams], optional
            Algorithmic parameters, by default None
        """
        self._crypt_params = crypt_params
        self._alg_params = alg_params

    @abstractmethod
    def get_algorithm_summary(
        self, alg_params: Optional[AlgParams] = None
    ) -> AlgorithmSummary:
        """Compute logical resource estimates for the circuit.

        This method must be implemented by all concrete algorithm classes.

        Parameters
        ----------
        alg_params : Optional[AlgParams], optional
            Algorithm parameters to use for the summary. If None, uses the parameters
            stored in the instance (self._alg_params).

        Returns
        -------
        AlgorithmSummary
            Logical resource estimates.

        Raises
        ------
        ValueError
            If no alg params are provided either at init or to this method.
        """
        pass

    def generate_search_space(self) -> list[AlgParams]:
        """Generate a search space for algorithm parameters.

        Returns
        -------
        list[AlgParams]
            List of algorithm parameters to search over.
        """
        return []

    def estimate_resources_qualtran(
        self, cost_model: PhysicalCostModel, alg_params: Optional[AlgParams] = None
    ) -> dict:
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
        algorithm_summary = self.get_algorithm_summary(alg_params)
        resource_estimate = {
            "duration_hr": cost_model.duration_hr(algorithm_summary),
            "n_phys_qubits": cost_model.n_phys_qubits(algorithm_summary),
            "error": cost_model.error(algorithm_summary),
        }
        return resource_estimate

    def estimate_resources_azure(
        self, estimator_params: EstimatorParams, alg_params: Optional[AlgParams] = None
    ) -> EstimatorResult:
        """Estimate resources using Azure Quantum Resource Estimator.

        Parameters
        ----------
        estimator_params : EstimatorParams
            Parameters for the resource estimator.
        alg_params : Optional[AlgParams], optional
            Algorithm parameters to use for the estimation. If None, uses the
            parameters stored in the instance (self._alg_params).

        Returns
        -------
        EstimatorResult
            Results from the Azure Quantum Resource Estimator.

        Raises
        ------
        TypeError
            If the estimator parameters are not given as an EstimatorParams instance or
            a dictionary.
        """
        algorithm_summary = self.get_algorithm_summary(alg_params)
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

        if isinstance(estimator_params, EstimatorParams):
            estimator_params = estimator_params.as_dict()
        elif not isinstance(estimator_params, dict):
            raise TypeError(
                f"{type(estimator_params)} is the wrong type for estimator parameters. "
                + "It must be given as either an EstimatorParams instance or a dictionary."
            )

        if "errorBudget" not in estimator_params:
            estimator_params["errorBudget"] = 0.9

        return logical_counts.estimate(estimator_params)
