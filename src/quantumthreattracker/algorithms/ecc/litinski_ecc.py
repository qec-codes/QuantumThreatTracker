"""Litinski's implementation of Elliptic Curve Cryptography (ECC).

Circuit construction from [Lit23], Fig. 4:
[How to compute a 256-bit elliptic curve private key with only 50 million Toffoli gates]
(https://arxiv.org/abs/2306.08585)
"""

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

from attrs import frozen
from qualtran import Bloq, QBit, QUInt, Register, Signature
from qualtran.bloqs.arithmetic._shims import CHalf  # noqa: PLC2701
from qualtran.bloqs.basic_gates import CNOT, TGate, Toffoli
from qualtran.bloqs.factoring.ecc import ECAdd
from qualtran.resource_counting import (
    BloqCountDictT,
    GateCounts,
    SympySymbolAllocator,
)
from qualtran.resource_counting.generalizers import _ignore_wrapper  # noqa: PLC2701
from qualtran.surface_code import AlgorithmSummary

from quantumthreattracker.algorithms.quantum_algorithm import (
    AlgParams,
    CryptParams,
    QuantumAlgorithm,
)


@frozen
class CHalfCustom(Bloq):
    """A custom implementation of the CHalf bloq that overrides call graph."""

    n: int

    @cached_property
    def signature(self) -> "Signature":
        """Bloq signature."""
        return Signature([Register("ctrl", QBit()), Register("x", QUInt(self.n))])

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:
        """Call graph construction.

        Returns
        -------
        BloqCountDictT
            Custom Bloq counts.
        """
        return {Toffoli(): self.n, CNOT(): 2 * self.n}


def generalize_c_half_decomp(b: Bloq) -> Optional[Bloq]:
    """Override the default CHalf Bloq of qualtran.

    Returns
    -------
    Optional[Bloq]
        A custom CHalf Bloq if the input is an instance of CHalf, otherwise the result
        of _ignore_wrapper.
    """
    if isinstance(b, CHalf):
        return CHalfCustom(b.n)

    return _ignore_wrapper(generalize_c_half_decomp, b)


@dataclass
class LitinskiECCParams(AlgParams):
    """Parameters for the Litinski ECC algorithm.

    Parameters
    ----------
    window_size: int
        Window size for point addition in the ECC implementation.
    classical_bits: int
        Number of classical bits to be bruteforced.
    """

    window_size: int
    classical_bits: int = 48


class LitinskiECC(QuantumAlgorithm):
    """Implementation of Litinski's algorithm for elliptic curve cryptography."""

    def __init__(
        self, crypt_params: CryptParams, alg_params: Optional[LitinskiECCParams] = None
    ):
        """Initialize the Litinski ECC algorithm.

        Parameters
        ----------
        crypt_params: CryptParams
            Cryptographic parameters, including key size.
        alg_params: Optional[LitinskiECCParams], optional
            Algorithm parameters, by default None
        """
        super().__init__(crypt_params, alg_params)

    def get_algorithm_summary(
        self, alg_params: Optional[AlgParams] = None
    ) -> AlgorithmSummary:
        """Compute logical resource estimates for Litinski's ECC algorithm.

        Uses windowed approach with lookup, point addition, and unlookup operations.
        The algorithm uses (key_size-48)/window_size Lookup Additions.

        Parameters
        ----------
        alg_params : Optional[AlgParams], optional
            Algorithm parameters to use for the summary. If None, uses the parameters
            stored in the instance (self._alg_params).

        Returns
        -------
            AlgorithmSummary with logical resource estimates

        Raises
        ------
        NameError
            If protocol is not "ECDH".
        ValueError
            If no algorithm parameters are provided.
        TypeError
            If alg_params is not LitinskiECCParams.
        """
        if self._crypt_params.protocol not in {"ECDH"}:
            raise NameError(
                f'Protocol must be "ECDH", got "{self._crypt_params.protocol}"'
            )

        # Use provided alg_params or instance alg_params
        effective_alg_params = alg_params or self._alg_params

        if effective_alg_params is None:
            raise ValueError(
                "Algorithm parameters must be provided either at initialization or to this method."
            )

        # Type checking
        if not isinstance(effective_alg_params, LitinskiECCParams):
            raise TypeError(
                f"Expected LitinskiECCParams, got {type(effective_alg_params).__name__}"
            )

        key_size = self._crypt_params.key_size
        window_size = effective_alg_params.window_size
        classical_bits = effective_alg_params.classical_bits
        mod = 2**key_size - 3
        # Litinski assumes that 48 bits of the key can be bruteforced classically
        num_reps = math.ceil((key_size - classical_bits) / window_size)

        # Calculate costs of individual operations
        ecc_circ = ECAdd(n=key_size, mod=mod)
        # TODO: Change this to use the algorithm summary method once qualtran has a
        # decompositon for CHalf. This is a round about way of getting the cost of CHalf
        ecc_circ_bloq_counts = ecc_circ.call_graph(
            max_depth=10, generalizer=generalize_c_half_decomp
        )[1]
        ecc_add_cost = GateCounts(
            toffoli=ecc_circ_bloq_counts[Toffoli()],
            clifford=ecc_circ_bloq_counts[CNOT()],
            t=ecc_circ_bloq_counts[TGate()],
        )
        windowing_cost = GateCounts(
            toffoli=int(2**window_size + 2 ** (window_size / 2))
        )

        # From [Lit23]: Only window_size bits needed in memory at once
        logical_qubit_count = 10 * key_size + 2 * window_size + 5
        # Total cost across all repetitions
        total_gate_count = num_reps * (ecc_add_cost + windowing_cost)

        return AlgorithmSummary(
            n_algo_qubits=logical_qubit_count, n_logical_gates=total_gate_count
        )

    def generate_search_space(self) -> list[LitinskiECCParams]:
        """Generate a search space for algorithm parameters.

        Creates a range of window sizes to search over. For smaller key sizes (≤256),
        it explores smaller window sizes. For larger key sizes, it explores larger
        window sizes which are more computationally efficient.

        Returns
        -------
        list[LitinskiECCParams]
            List of LitinskiECCParams with different window sizes.
        """
        key_size = self._crypt_params.key_size
        search_space = []

        # Default classical bits to bruteforce
        classical_bits = 48

        # Choose window sizes based on key size
        if key_size <= 256:
            # For smaller key sizes, smaller window sizes may be optimal
            window_sizes = [4, 8, 12, 16, 20, 24, 28]
        else:
            # For larger key sizes, larger window sizes are generally better
            window_sizes = [16, 20, 24, 28, 32, 36, 40]

        for window_size in window_sizes:
            params = LitinskiECCParams(
                window_size=window_size,
                classical_bits=classical_bits
            )
            search_space.append(params)

        return search_space
