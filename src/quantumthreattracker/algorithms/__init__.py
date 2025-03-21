"""Algorithms module."""

from .ecc.ecc_basic import ECCBasic, ECCBasicParams
from .ecc.litinski_ecc import LitinskiECC, LitinskiECCParams
from .quantum_algorithm import AlgParams, CryptParams, QuantumAlgorithm
from .rsa.baseline_shor import BaselineShor, BaselineShorParams
from .rsa.gidney_ekera import GidneyEkera, GidneyEkeraParams
from .rsa.gidney_ekera_basic import GidneyEkeraBasic, GidneyEkeraBasicParams

__all__ = [
    "AlgParams",
    "AlgorithmLister",
    "BaselineShor",
    "BaselineShorParams",
    "CryptParams",
    "ECCBasic",
    "ECCBasicParams",
    "GidneyEkera",
    "GidneyEkeraBasic",
    "GidneyEkeraBasicParams",
    "GidneyEkeraParams",
    "LitinskiECC",
    "LitinskiECCParams",
    "QuantumAlgorithm",
]


class AlgorithmLister:
    """Class to list available quantum algorithms."""

    @classmethod
    def list_algorithms(
        cls,
        crypt_params: CryptParams,
    ) -> list[QuantumAlgorithm]:
        """List the quantum algorithms eligible for breaking a given crypt protocol.

        Parameters
        ----------
        crypt_params : CryptParams
            Cryptographic parameters.

        Returns
        -------
        list[QuantumAlgorithm]
            List of quantum algorithms eligible for breaking the given crypt protocol.

        Raises
        ------
        ValueError
            If the protocol is not recognized.
        """
        if crypt_params.protocol == "RSA":
            algorithms = [BaselineShor, GidneyEkera]
        elif crypt_params.protocol == "DLDH":
            algorithms = []
        elif crypt_params.protocol == "ECDH":
            algorithms = [ECCBasic, LitinskiECC]
        else:
            raise ValueError(f"Unknown protocol: {crypt_params.protocol}")
        return algorithms
