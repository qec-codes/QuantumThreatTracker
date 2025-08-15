"""Tests for the `GidneyBasic` class."""

import pytest

from quantumthreattracker.algorithms.quantum_algorithm import CryptParams
from quantumthreattracker.algorithms.rsa.gidney_basic import (
    GidneyBasic,
    GidneyBasicParams,
)


def test_gidney_basic_rsa_supported_key_sizes() -> None:
    """Test key size support and algorithm summary generation."""
    key_sizes = [1024, 1536, 2048, 3072, 4096, 6144, 8192]
    for key_size in key_sizes:
        crypt_params = CryptParams(protocol="RSA", key_size=key_size)
        algo = GidneyBasic(crypt_params)
        summary = algo.get_algorithm_summary()
        assert summary.n_algo_qubits > 0
        assert summary.n_logical_gates.toffoli > 0


def test_gidney_basic_rsa_unsupported_key_size_raises() -> None:
    """Test that GidneyBasic raises ValueError for unsupported RSA key sizes."""
    crypt_params = CryptParams(protocol="RSA", key_size=10000)
    algo = GidneyBasic(crypt_params)
    with pytest.raises(ValueError):
        algo.get_algorithm_summary()


def test_gidney_basic_non_rsa_protocol_raises() -> None:
    """Test that GidneyBasic raises NameError when a non-RSA protocol is provided."""
    crypt_params = CryptParams(protocol="ECC", key_size=2048)
    algo = GidneyBasic(crypt_params)
    with pytest.raises(NameError):
        algo.get_algorithm_summary()


def test_gidney_basic_generate_search_space() -> None:
    """Test search space generation."""
    params_list = GidneyBasic.generate_search_space()
    assert isinstance(params_list, list)
    assert len(params_list) == 1
    assert isinstance(params_list[0], GidneyBasicParams)
