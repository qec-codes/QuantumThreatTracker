"""Litinski's implementation of Elliptic Curve Cryptography (ECC).

Circuit construction from [Lit23], Fig. 4:
[How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates]
(https://arxiv.org/abs/2306.08585)
"""

import math

from qualtran.bloqs.factoring.ecc import ECAdd
from qualtran.resource_counting import GateCounts, QECGatesCost, get_cost_value
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms.quantum_algorithm import QuantumAlgorithm


class LitinskiECC(QuantumAlgorithm):
    """Implements Litinski's ECC algorithm."""

    def get_algorithm_summary(self, window_size: int = 22) -> AlgorithmSummary:
        """Compute logical resource estimates for Litinski's ECC algorithm.

        Uses windowed approach with lookup, point addition, and unlookup operations.
        The algorithm uses (key_size-48)/window_size Lookup Additions.

        Args:
            window_size: Lookup table window size

        Returns
        -------
            AlgorithmSummary with logical resource estimates

        Raises
        ------
            NameError: If protocol is not "ECDH"
        """
        if self._crypt_params.protocol != "ECDH":
            raise NameError(
                f'Protocol must be "ECDH", got "{self._crypt_params.protocol}"'
            )

        key_size = self._crypt_params.key_size
        mod = 2**key_size - 3
        # Litinski assumes that 48 bits of the key can be bruteforced classically
        num_reps = math.ceil((key_size - 48) / window_size)

        # Calculate costs of individual operations
        ecc_circ = ECAdd(n=key_size, mod=mod)
        windowing_cost = GateCounts(
            toffoli=int(2**window_size + 2**(window_size / 2))
        )

        # From [Lit23]: Only window_size bits needed in memory at once
        logical_qubit_count = 10 * key_size + 2*window_size + 5

        # Total cost across all repetitions
        total_gate_count = num_reps * (
            get_cost_value(ecc_circ, QECGatesCost()) + windowing_cost
        )

        return AlgorithmSummary(
            n_algo_qubits=logical_qubit_count, n_logical_gates=total_gate_count
        )
