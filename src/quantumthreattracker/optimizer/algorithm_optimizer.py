"""Module for optimizing quantum algorithm parameters based on resource estimates."""
from typing import List, Tuple

from qsharp.estimator import EstimatorParams, EstimatorResult

from quantumthreattracker.algorithms import AlgParams, CryptParams, QuantumAlgorithm


class AlgorithmOptimizer:
    """Class for optimizing quantum algorithm parameters based on resource estimates."""

    # TODO: Enforce that the alg params should correspond a specific quantum algorithm
    # TODO: or, just generate the search space here
    @staticmethod
    def find_min_estimate(
        algorithm: QuantumAlgorithm,
        crypt_params: CryptParams,
        search_space: List[AlgParams],
        estimator_params: EstimatorParams,
        minimize_metric: str = 'physicalQubits'
    ) -> Tuple[AlgParams, EstimatorResult]:
        """
        Optimize algorithm parameters to minimize a specific resource metric.

        Args:
            algorithm: The quantum algorithm instance to optimize.
            crypt_params: Cryptographic parameters for the algorithm.
            search_space: List of algorithm parameters to search over.
            estimator_params: Parameters for the resource estimator.
            minimize_metric: The resource metric to minimize (default: 'n_phys_qubits').
                             Options include: 'n_phys_qubits', 'runtime_in_seconds',
                             'toffoli_count', etc.

        Returns
        -------
            Tuple containing (optimal_parameters, optimal_resource_estimate).
        """
        # TODO: Change docstring for minimize_metric
        # Check if the search space is empty
        if not search_space:
            raise ValueError("The search space is empty.")

        # Initialize variables to track minimum estimate
        min_estimate_params = search_space[0]
        min_estimate = algorithm(crypt_params, search_space[0]).estimate_resources_azure(
            estimator_params
        )
        # Iterate through the rest of the search space
        for alg_params in search_space[1:]:
            # Configure algorithm with current parameters
            current_alg = algorithm(crypt_params, alg_params)

            # Get resource estimate for current parameters
            current_estimate = current_alg.estimate_resources_azure(estimator_params)

            # Extract current metric value
            current_metric_value = current_estimate['physicalCounts'].get(minimize_metric)
            min_metric_value = min_estimate['physicalCounts'].get(minimize_metric)

            # Skip if metric is not available
            if current_metric_value is None:
                continue

            # Update minimum if current is better
            if current_metric_value < min_metric_value:
                min_estimate = current_estimate
                min_estimate_params = alg_params

        return min_estimate_params, min_estimate
