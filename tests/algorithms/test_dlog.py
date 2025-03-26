"""Tests for the DLog algorithm implementations."""

import pytest
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms import CryptParams, QuantumAlgorithm
from quantumthreattracker.algorithms.dlog.dlog_safe_prime_shor import (
    DLogSafePrimeShor,
    DLogSafePrimeShorParams,
)
from quantumthreattracker.algorithms.dlog.dlog_schnorr_shor import (
    DLogSchnorrShor,
    DLogSchnorrShorParams,
)


@pytest.fixture()
def default_safe_prime_params() -> DLogSafePrimeShorParams:
    """Get default algorithm parameters for safe prime tests.

    Returns
    -------
        DLogSafePrimeShorParams: The default parameters for safe prime DLog algorithm.
    """
    return DLogSafePrimeShorParams(window_size_exp=4, window_size_mul=4)


@pytest.fixture()
def default_schnorr_params() -> DLogSchnorrShorParams:
    """Get default algorithm parameters for Schnorr tests.

    Returns
    -------
        DLogSchnorrShorParams: The default parameters for Schnorr DLog algorithm.
    """
    return DLogSchnorrShorParams(window_size_exp=4, window_size_mul=4)


@pytest.fixture()
def default_safe_prime_algorithm(default_safe_prime_params: DLogSafePrimeShorParams) -> DLogSafePrimeShor:
    """Get a default instance of `DLogSafePrimeShor`.

    Returns
    -------
        DLogSafePrimeShor: An instance of the DLogSafePrimeShor class.
    """
    return DLogSafePrimeShor(CryptParams("DH-SP", 1024), default_safe_prime_params)


@pytest.fixture()
def default_schnorr_algorithm(default_schnorr_params: DLogSchnorrShorParams) -> DLogSchnorrShor:
    """Get a default instance of `DLogSchnorrShor`.

    Returns
    -------
        DLogSchnorrShor: An instance of the DLogSchnorrShor class.
    """
    return DLogSchnorrShor(CryptParams("DH-SCH", 1024), default_schnorr_params)


def test_safe_prime_alg_sum(default_safe_prime_algorithm: QuantumAlgorithm) -> None:
    """Test that the safe prime algorithm outputs a valid `AlgorithmSummary`."""
    alg_sum = default_safe_prime_algorithm.get_algorithm_summary()
    assert isinstance(alg_sum, AlgorithmSummary)
    assert alg_sum.n_algo_qubits > 0
    assert alg_sum.n_logical_gates.toffoli > 0


def test_schnorr_alg_sum(default_schnorr_algorithm: QuantumAlgorithm) -> None:
    """Test that the Schnorr algorithm outputs a valid `AlgorithmSummary`."""
    alg_sum = default_schnorr_algorithm.get_algorithm_summary()
    assert isinstance(alg_sum, AlgorithmSummary)
    assert alg_sum.n_algo_qubits > 0
    assert alg_sum.n_logical_gates.toffoli > 0


def test_safe_prime_wrong_protocol_raises_error() -> None:
    """Test that using a non-DH-SP protocol for safe prime raises an error."""
    algorithm = DLogSafePrimeShor(CryptParams("RSA", 1024))
    params = DLogSafePrimeShorParams(window_size_exp=4, window_size_mul=4)
    with pytest.raises(NameError, match='The protocol for this class must be "DH-SP"'):
        algorithm.get_algorithm_summary(params)


def test_schnorr_wrong_protocol_raises_error() -> None:
    """Test that using a non-DH-SCH protocol for Schnorr raises an error."""
    algorithm = DLogSchnorrShor(CryptParams("RSA", 1024))
    params = DLogSchnorrShorParams(window_size_exp=4, window_size_mul=4)
    with pytest.raises(NameError, match='The protocol for this class must be "DH-SCH"'):
        algorithm.get_algorithm_summary(params)


def test_key_size_affects_output() -> None:
    """Test that changing key size affects the algorithm summary."""
    params = DLogSafePrimeShorParams(window_size_exp=4, window_size_mul=4)

    algorithm_small = DLogSafePrimeShor(CryptParams("DH-SP", 1024), params)
    algorithm_large = DLogSafePrimeShor(CryptParams("DH-SP", 2048), params)

    small_sum = algorithm_small.get_algorithm_summary()
    large_sum = algorithm_large.get_algorithm_summary()

    # Larger key size should result in more qubits
    assert large_sum.n_algo_qubits > small_sum.n_algo_qubits
    # Larger key size should result in more gates
    assert large_sum.n_logical_gates.toffoli > small_sum.n_logical_gates.toffoli


def test_window_size_affects_output() -> None:
    """Test that changing window sizes affects the algorithm summary."""
    small_params = DLogSafePrimeShorParams(window_size_exp=3, window_size_mul=3)
    large_params = DLogSafePrimeShorParams(window_size_exp=6, window_size_mul=6)

    algorithm = DLogSafePrimeShor(CryptParams("DH-SP", 1024))

    small_sum = algorithm.get_algorithm_summary(small_params)
    large_sum = algorithm.get_algorithm_summary(large_params)

    # Different window sizes should result in different resource estimates
    assert small_sum.n_logical_gates.toffoli != large_sum.n_logical_gates.toffoli
