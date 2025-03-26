"""Class for listing the algorithms eligible for breaking a given crypt protocol."""

from .dlog.dlog_safe_prime_eh import DLogSafePrimeEH
from .dlog.dlog_safe_prime_shor import DLogSafePrimeShor
from .dlog.dlog_schnorr_eh import DLogSchnorrEH
from .dlog.dlog_schnorr_shor import DLogSchnorrShor
from .ecc.litinski_ecc import LitinskiECC
from .quantum_algorithm import (
    CryptParams,
    QuantumAlgorithm,
)
from .rsa.gidney_ekera import GidneyEkera


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
            algorithms = [GidneyEkera(crypt_params)]
        elif crypt_params.protocol == "DH-SP":
            algorithms = [DLogSafePrimeEH(crypt_params), DLogSafePrimeShor(crypt_params)]
        elif crypt_params.protocol == "DH-SCH":
            algorithms = [DLogSchnorrEH(crypt_params), DLogSchnorrShor(crypt_params)]
        elif crypt_params.protocol == "ECDH":
            algorithms = [LitinskiECC(crypt_params)]
        else:
            raise ValueError(f"Unknown protocol: {crypt_params.protocol}")
        return algorithms
