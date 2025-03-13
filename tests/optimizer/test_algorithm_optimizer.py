"""Tests for the AlgorithmOptimizer class."""

import pytest
from qsharp.estimator import EstimatorParams

from quantumthreattracker.algorithms import CryptParams
from quantumthreattracker.algorithms.rsa.gidney_ekera import (
    GidneyEkera,
    GidneyEkeraParams,
)
from quantumthreattracker.optimizer.algorithm_optimizer import AlgorithmOptimizer


class TestGidneyEkeraOptimizer:
    """Test class specifically for optimizing GidneyEkera parameters."""

    def setup_method(self):
        """Set up test resources for GidneyEkera optimization tests."""
        # Create RSA cryptographic parameters
        self.crypt_params = CryptParams(protocol="RSA", key_size=1024)

        # Create estimator params
        self.estimator_params = EstimatorParams()

        # Create several GidneyEkera parameter sets with different window sizes
        self.params_small_windows = GidneyEkeraParams(
            num_exp_qubits=1536,
            window_size_exp=1,
            window_size_mul=1
        )
        self.params_medium_windows = GidneyEkeraParams(
            num_exp_qubits=1536,
            window_size_exp=4,
            window_size_mul=4
        )
        self.params_large_windows = GidneyEkeraParams(
            num_exp_qubits=1536,
            window_size_exp=10,
            window_size_mul=10
        )

    def test_optimize_gidney_ekera_qubit_count(self):
        """Test that optimizer finds parameters minimizing qubit count."""
        # Create search space
        search_space = [
            self.params_small_windows,
            self.params_medium_windows,
            self.params_large_windows
        ]

        # Run the optimizer
        best_params, _ = AlgorithmOptimizer.find_min_estimate(
            GidneyEkera,
            self.crypt_params,
            search_space,
            self.estimator_params,
            minimize_metric='physicalQubits'
        )

        # Verify the optimizer selected the parameters with minimum qubit count
        assert best_params == self.params_medium_windows

    def test_single_element_search_space(self):
        """Test that a value is returned when search space has only one element."""
        # Create a single parameter set
        single_param = GidneyEkeraParams(
            num_exp_qubits=1536,
            window_size_exp=3,
            window_size_mul=3
        )

        # Create a search space with a single element
        search_space = [single_param]

        # Run the optimizer
        best_params, best_estimate = AlgorithmOptimizer.find_min_estimate(
            GidneyEkera,
            self.crypt_params,
            search_space,
            self.estimator_params,
            minimize_metric='physicalQubits'
        )

        # Verify that the optimizer returns a non-None value
        assert best_estimate is not None, "Best estimate should not be None"

        # Verify that the optimizer returns the only element in the search space
        assert best_params == single_param

    def test_empty_search_space(self):
        """Test that an error is raised when the search space is empty."""
        # Create an empty search space
        search_space = []

        # Verify that the optimizer raises a ValueError
        with pytest.raises(ValueError, match="The search space is empty."):
            AlgorithmOptimizer.find_min_estimate(
                GidneyEkera,
                self.crypt_params,
                search_space,
                self.estimator_params,
                minimize_metric='physicalQubits'
            )

    def test_gidney_ekera_generate_search_space(self):
        """Test that a search space can be generated for GidneyEkera."""
        # Create an instance of GidneyEkera
        algorithm = GidneyEkera(self.crypt_params, self.params_medium_windows)

        # Generate search space
        search_space = algorithm.generate_search_space()

        # Verify search space properties
        assert len(search_space) > 0, "Search space should not be empty"
        assert all(isinstance(params, GidneyEkeraParams) for params in search_space)
