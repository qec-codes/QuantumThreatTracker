"""Tests for the algorithm lister."""

import pytest

from quantumthreattracker.algorithms import AlgorithmLister, CryptParams


@pytest.mark.parametrize(
    "crypt_params, expected_algorithms",
    [
        (CryptParams("RSA", 1024), ["GidneyEkera"]),
        (CryptParams("DH-SP", 1024), ["DLogSafePrimeEH", "DLogSafePrimeShor"]),
        (CryptParams("DH-SCH", 1024), ["DLogSchnorrEH", "DLogSchnorrShor"]),
        (CryptParams("ECDH", 256), ["LitinskiECC"]),
    ],
)
def test_list_algorithms(
    crypt_params: CryptParams,
    expected_algorithms: list[str],
) -> None:
    """Test that the algorithm lister returns the correct algorithms."""
    algorithms = AlgorithmLister.list_algorithms(crypt_params)
    algorithm_names = [type(alg).__name__ for alg in algorithms]
    assert set(algorithm_names) == set(expected_algorithms)
