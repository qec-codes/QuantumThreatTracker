"""Litinski's implementation of Elliptic Curve Cryptography (ECC).

Circuit construction from [Lit23], Fig. 4:
[How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates]
(https://arxiv.org/abs/2306.08585)
"""

import math
from functools import cached_property
from typing import Optional

from attrs import frozen
from qualtran import Bloq, QBit, QUInt, Register, Signature
from qualtran.bloqs.basic_gates import CNOT, TGate, Toffoli
from qualtran.bloqs.factoring.ecc import ECAdd
from qualtran.resource_counting import (
    BloqCountDictT,
    GateCounts,
    SympySymbolAllocator,
)
from qualtran.resource_counting.generalizers import _ignore_wrapper
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms.quantum_algorithm import QuantumAlgorithm


@frozen
class CHalfCustom(Bloq):
    """A custom implementation of the CHalf bloq that overrides call graph."""

    n: int

    @cached_property
    def signature(self) -> 'Signature':
        """Bloq signature."""
        return Signature([Register('ctrl', QBit()), Register('x', QUInt(self.n))])

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        """Call graph construction."""
        return {Toffoli(): self.n, CNOT(): 2*self.n}

def generalize_c_half_decomp(b: Bloq) -> Optional[Bloq]:
    """Override the default CHalf Bloq of qualtran."""
    from qualtran.bloqs.arithmetic._shims import CHalf

    if isinstance(b, CHalf):
        return CHalfCustom(b.n)

    return _ignore_wrapper(generalize_c_half_decomp, b)

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
        # TODO: Change this to use the algorithm summary method once qualtran has a
        # decompositon for CHalf. This is a round about way of getting the cost of CHalf
        ecc_circ_bloq_counts = ecc_circ.call_graph(max_depth=10, generalizer=generalize_c_half_decomp)[1]
        ecc_add_cost = GateCounts(
            toffoli=ecc_circ_bloq_counts[Toffoli()],
            clifford=ecc_circ_bloq_counts[CNOT()],
            t=ecc_circ_bloq_counts[TGate()],
        )
        windowing_cost = GateCounts(
            toffoli=int(2**window_size + 2**(window_size / 2))
        )

        # From [Lit23]: Only window_size bits needed in memory at once
        logical_qubit_count = 10 * key_size + 2*window_size + 5
        # Total cost across all repetitions
        total_gate_count = num_reps * (ecc_add_cost + windowing_cost)

        return AlgorithmSummary(
            n_algo_qubits=logical_qubit_count, n_logical_gates=total_gate_count
        )
