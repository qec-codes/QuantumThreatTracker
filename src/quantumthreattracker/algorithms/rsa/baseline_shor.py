"""Class for the baseline implementation of Shor's algorithm."""

from typing import Optional

from qualtran import QUInt
from qualtran.bloqs.mod_arithmetic import CModMulK
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms.quantum_algorithm import (
    AlgParams,
    CryptParams,
    QuantumAlgorithm,
)


class BaselineShorParams(AlgParams):
    """Dataclass describing the parameters for baseline Shor's algorithm.

    Note: The basic implementation doesn't have configurable parameters,
    but this class is provided for consistency with the interface.
    """

    pass


class BaselineShor(QuantumAlgorithm):
    """Class for the baseline implementation of Shor's algorithm."""

    def __init__(self, crypt_params: CryptParams, alg_params: Optional[BaselineShorParams] = None):
        """Initialize the quantum algorithm.

        Parameters
        ----------
        crypt_params : CryptParams
            Cryptographic parameters.
        alg_params : Optional[BaselineShorParams], optional
            Algorithmic parameters. For BaselineShor, these have no effect but are
            included for consistency.
        """
        super().__init__(crypt_params, alg_params)

    def get_algorithm_summary(self, alg_params: Optional[AlgParams] = None) -> AlgorithmSummary:
        """Compute logical resource estimates for the circuit.

        Parameters
        ----------
        alg_params : Optional[AlgParams], optional
            Algorithm parameters (not used by BaselineShor)

        Returns
        -------
        AlgorithmSummary
            Logical resource estimates.

        Raises
        ------
        NameError
            If the protocol is not "RSA".
        """
        # Parameters check is unnecessary for Baseline Shor as it doesn't use parameters
        # But we'll include it for consistency with the interface
        if self._crypt_params.protocol != "RSA":
            raise NameError(
                'The protocol for this class must be "RSA". '
                + f'"{self._crypt_params.protocol}" was given.'
            )

        # Original computation logic
        key_size = self._crypt_params.key_size
        modulus = 2**key_size - 1
        multiplicand = 2**key_size - 2

        mod_mul = CModMulK(dtype=QUInt(key_size), k=multiplicand, mod=modulus)
        alg_sum_controlled_mod_mul = AlgorithmSummary.from_bloq(mod_mul)

        reps = int(1.5 * key_size)
        alg_sum_mod_exp = AlgorithmSummary(
            n_algo_qubits=alg_sum_controlled_mod_mul.n_algo_qubits,
            n_logical_gates=alg_sum_controlled_mod_mul.n_logical_gates * reps,
        )

        return alg_sum_mod_exp
