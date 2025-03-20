"""Module for optimizing quantum algorithm parameters based on resource estimates."""

from typing import List, Tuple

from qsharp.estimator import EstimatorParams, EstimatorResult

from quantumthreattracker.algorithms import AlgParams, QuantumAlgorithm


class AlgorithmOptimizer:
    """Class for optimizing quantum algorithm parameters based on resource estimates."""

    @staticmethod
    def find_min_estimate(
        algorithm: QuantumAlgorithm,
        estimator_params: EstimatorParams,
        minimize_metric: str = "physicalQubits",
        search_space: List[AlgParams] | None = None,
    ) -> Tuple[AlgParams, EstimatorResult]:
        """Optimize algorithm parameters to minimize a specific resource metric.

        Parameters
        ----------
        algorithm : QuantumAlgorithm
            The quantum algorithm instance to optimize.
        estimator_params : EstimatorParams
            Parameters for the resource estimator.
        minimize_metric : str, optional
            Resource metric to minimize (default: 'physicalQubits'). Options include:
            'physicalQubits', 'runtimeInSeconds', 'toffoli_count', etc.
        search_space : List[AlgParams] | None, optional
            List of algorithm parameters to search over. If None or empty, will attempt
            to generate from algorithm.

        Returns
        -------
        Tuple[AlgParams, EstimatorResult]
            Tuple containing (optimal_parameters, optimal_resource_estimate).

        Raises
        ------
        ValueError
            If no search space is provided and the algorithm doesn't have a defined
            search space.
        """
        # If search_space is None or empty, try to generate it from the algorithm
        if not search_space:
            search_space = algorithm.generate_search_space()

            # If search_space is still empty, raise an error
            if not search_space:
                raise ValueError(
                    "No search space provided and the algorithm doesn't have a defined search space"
                )

        # Initialize variables to track minimum estimate
        min_estimate_params = search_space[0]
        min_estimate = algorithm.estimate_resources_azure(
            estimator_params, search_space[0]
        )
        # Iterate through the rest of the search space
        for alg_params in search_space[1:]:
            # Configure algorithm with current parameters
            current_estimate = algorithm.estimate_resources_azure(
                estimator_params, alg_params
            )

            # Extract current metric value
            current_metric_value = current_estimate["physicalCounts"].get(
                minimize_metric
            )
            min_metric_value = min_estimate["physicalCounts"].get(minimize_metric)

            # Skip if metric is not available
            if current_metric_value is None:
                continue

            # Update minimum if current is better
            if current_metric_value < min_metric_value:
                min_estimate = current_estimate
                min_estimate_params = alg_params

        return min_estimate_params, min_estimate
