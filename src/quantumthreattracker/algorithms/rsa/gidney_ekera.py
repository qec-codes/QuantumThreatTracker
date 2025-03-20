"""Class for a parameterised implementation of Gidney-Ekera.

[1] https://doi.org/10.22331/q-2021-04-15-433
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Optional

from attrs import field, frozen
from qualtran import Bloq, QInt, QMontgomeryUInt, QUInt, Register, Signature
from qualtran.bloqs.arithmetic import Add
from qualtran.bloqs.basic_gates import CNOT, Toffoli
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
class AddCustom(Bloq):
    """A custom implementation of the `Add` bloq that overrides the call graph."""

    a_dtype: QInt | QUInt | QMontgomeryUInt = field()
    b_dtype: QInt | QUInt | QMontgomeryUInt = field()

    @cached_property
    def signature(self) -> "Signature":
        """Bloq signature."""
        return Signature([Register("a", self.a_dtype), Register("b", self.b_dtype)])

    def build_call_graph(self, ssa: "SympySymbolAllocator") -> "BloqCountDictT":
        """Call graph construction.

        Returns
        -------
        BloqCountDictT
            Custom Bloq counts.
        """
        n = self.b_dtype.bitsize
        n_cnot = (n - 2) * 6 + 3
        return {Toffoli(): 2 * (n - 1), CNOT(): n_cnot}


def generalize_and_decomp(bloq: Bloq) -> Optional[Bloq]:
    """Override the default And Bloq of qualtran.

    Returns
    -------
    Optional[Bloq]
        A custom And Bloq if the input is an instance of And, otherwise the result
        of _ignore_wrapper.
    """
    if isinstance(bloq, Add):
        return AddCustom(a_dtype=bloq.a_dtype, b_dtype=bloq.b_dtype)

    return _ignore_wrapper(generalize_and_decomp, bloq)


@dataclass
class GidneyEkeraParams(AlgParams):
    """Dataclass describing the parameters for Gidney-Ekera.

    Parameters
    ----------
    num_exp_qubits: int
        Number of exponent qubits. Denoted $n_e$ in [1].
    window_size_exp: int
        Exponentiation windows size. Denoted $n_{exp}$ in [1].
    window_size_mul: int
        Multiplication window size. Denoted $n_{mul}$ in [1].
    """

    num_exp_qubits: int
    window_size_exp: int
    window_size_mul: int


class GidneyEkera(QuantumAlgorithm):
    """Class for a parameterised implementation of Gidney-Ekera."""

    def __init__(
        self, crypt_params: CryptParams, alg_params: Optional[GidneyEkeraParams] = None
    ):
        """Initialise the quantum algorithm.

        Parameters
        ----------
        crypt_params : CryptParams
            Cryptographic parameters.
        alg_params : Optional[GidneyEkeraParams], optional
            Algorithmic parameters, by default None
        """
        super().__init__(crypt_params, alg_params)

    def get_algorithm_summary(
        self, alg_params: Optional[AlgParams] = None
    ) -> AlgorithmSummary:
        """Compute logical resource estimates for the circuit.

        Parameters
        ----------
        alg_params : Optional[AlgParams], optional
            Algorithm parameters to use for the summary. If None, uses the parameters
            stored in the instance (self._alg_params).

        Returns
        -------
        AlgorithmSummary
            Logical resource estimates.

        Raises
        ------
        NameError
            If the protocol is not "RSA".
        ValueError
            If no algorithm parameters are provided.
        TypeError
            If alg_params is not GidneyEkeraParams.
        """
        if self._crypt_params.protocol != "RSA":
            raise NameError(
                'The protocol for this class must be "RSA". '
                + f'"{self._crypt_params.protocol}" was given.'
            )

        # Use provided alg_params or instance alg_params
        effective_alg_params = alg_params or self._alg_params

        if effective_alg_params is None:
            raise ValueError(
                "Algorithm parameters must be provided either at initialization or to this method."
            )

        # Type checking
        if not isinstance(effective_alg_params, GidneyEkeraParams):
            raise TypeError(
                f"Expected GidneyEkeraParams, got {type(effective_alg_params).__name__}"
            )

        key_size = self._crypt_params.key_size
        num_exp_qubits = effective_alg_params.num_exp_qubits
        window_size_exp = effective_alg_params.window_size_exp
        window_size_mul = effective_alg_params.window_size_mul

        # TODO: remove the custom overriding of the `Add` bloqs once the Qualtran
        # implementation is fixed.
        adder = Add(a_dtype=QUInt(key_size), b_dtype=QUInt(key_size))
        adder_bloq_counts = adder.call_graph(
            max_depth=10, generalizer=generalize_and_decomp
        )[1]
        adder_cost = GateCounts(
            toffoli=adder_bloq_counts[Toffoli()],
            clifford=adder_bloq_counts[CNOT()],
        )

        lookup_cost = GateCounts(toffoli=int(2 ** (window_size_exp + window_size_mul)))

        num_lookup_additions = int(
            2 * key_size * num_exp_qubits / (window_size_exp * window_size_mul)
        )
        total_gate_count = num_lookup_additions * (lookup_cost + adder_cost)

        logical_qubit_count = 3 * key_size

        return AlgorithmSummary(
            n_algo_qubits=logical_qubit_count, n_logical_gates=total_gate_count
        )

    def generate_search_space(self) -> list[GidneyEkeraParams]:
        """Generate a search space for algorithm parameters.

        Creates a comprehensive range of algorithm parameters to search over, including:
        - Fixed number of exponent qubits (typically 1.5x key size)
        - Various window sizes for exponentiation (2-7)
        - Various window sizes for multiplication (2-7)

        Returns
        -------
        list[GidneyEkeraParams]
            List of GidneyEkeraParams with various parameter combinations.
        """
        key_size = self._crypt_params.key_size
        search_space = []

        # Standard choice for exponent qubits
        num_exp_qubits = int(1.5 * key_size)

        # Create parameters for various window size combinations
        for window_size_exp in [2, 3, 4, 5, 6, 7]:
            for window_size_mul in [2, 3, 4, 5, 6, 7]:
                params = GidneyEkeraParams(
                    num_exp_qubits=num_exp_qubits,
                    window_size_exp=window_size_exp,
                    window_size_mul=window_size_mul,
                )
                search_space.append(params)

        return search_space
