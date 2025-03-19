"""Class for a basic implementation of ECC.

Logical resource estimates taken from [1], Table VI.

[1] https://doi.org/10.48550/arXiv.2409.04643
"""

from dataclasses import dataclass
from typing import Optional

from qualtran.resource_counting import GateCounts
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms.quantum_algorithm import (
    AlgParams,
    CryptParams,
    QuantumAlgorithm,
)


@dataclass
class ECCBasicParams(AlgParams):
    """Parameters for the basic implementation of ECC.

    This implementation doesn't have configurable parameters,
    but this class is provided for consistency with the interface.
    """

    pass


class ECCBasic(QuantumAlgorithm):
    """Class for a basic implementation of ECC."""

    def __init__(self, crypt_params: CryptParams, alg_params: Optional[ECCBasicParams] = None):
        """Initialize the quantum algorithm.

        Parameters
        ----------
        crypt_params : CryptParams
            Cryptographic parameters.
        alg_params : Optional[ECCBasicParams], optional
            Algorithmic parameters. For ECCBasic, these have no effect but are
            included for consistency.
        """
        super().__init__(crypt_params, alg_params)

    def get_algorithm_summary(self, alg_params: Optional[AlgParams] = None) -> AlgorithmSummary:
        """Compute logical resource estimates for the circuit.

        Parameters
        ----------
        alg_params : Optional[AlgParams], optional
            Algorithm parameters (not used by ECCBasic)

        Returns
        -------
        AlgorithmSummary
            Logical resource estimates.

        Raises
        ------
        NameError
            If the protocol is not "ECDH".
        """
        # Parameters check is unnecessary for ECCBasic as it doesn't use parameters
        # But we'll include it for consistency with the interface
        # effective_alg_params = alg_params or self._alg_params

        if self._crypt_params.protocol != "ECDH":
            raise NameError(
                'The protocol for this class must be "ECDH". '
                + f'"{self._crypt_params.protocol}" was given.'
            )

        key_size = self._crypt_params.key_size

        qubit_count = 9 * key_size
        toffoli_count = 43 * key_size**3

        alg_sum = AlgorithmSummary(
            n_algo_qubits=qubit_count, n_logical_gates=GateCounts(toffoli=toffoli_count)
        )

        return alg_sum
