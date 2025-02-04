"""Class for the baseline implementation of Shor's algorithm."""

from dataclasses import dataclass

from qualtran import QUInt
from qualtran.bloqs.mod_arithmetic import CModMulK
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms.quantum_algorithm import (
    AlgorithmParams,
    QuantumAlgorithm,
)


@dataclass
class BaselineShorParams(AlgorithmParams):
    """Parameters for the baseline implementation of Shor's algorithm.

    Parameters
    ----------
    protocol: str
        Cryptographic protocol. Can be:
            - 'RSA' (Rivest-Shamir-Adleman; factoring)
            - 'DH' (Diffie-Hellman; discrete log)
            - 'ECDH' (Elliptic Curve Diffie-Hellman; discrete log over elliptic curves)
    key_size: int
        Cryptographic key size.
    """

    protocol: str
    key_size: int


class BaselineShor(QuantumAlgorithm):
    """Class for the baseline implementation of Shor's algorithm."""

    def success_probability(self) -> float:
        """Calculate the algorithmic success probability.

        Returns
        -------
        float
            Algorithmic success probability.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented.
        """
        return super().success_probability()

    def get_algorithm_summary(self) -> AlgorithmSummary:
        """Compute logical resource estimates for the circuit.

        Returns
        -------
        AlgorithmSummary
            Logical resource estimates.

        Raises
        ------
        NotImplementedError
            If the method has not been implemented.
        """
        key_size = self._algorithm_params.key_size
        modulus = 2**key_size - 1
        multiplicand = 2**key_size - 2
        reps = int(1.5 * key_size)

        mod_mul = CModMulK(dtype=QUInt(key_size), k=multiplicand, mod=modulus)

        alg_sum_controlled_mod_mul = AlgorithmSummary.from_bloq(mod_mul)
        alg_sum_mod_exp = AlgorithmSummary(
            n_algo_qubits=alg_sum_controlled_mod_mul.n_algo_qubits,
            n_logical_gates=alg_sum_controlled_mod_mul.n_logical_gates * reps,
        )

        return alg_sum_mod_exp
